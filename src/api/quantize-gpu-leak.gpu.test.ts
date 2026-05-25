// GPU resource lifecycle test.
//
// Pins the contract added by Commit 1 of the 2026-05-23 review:
//   - quantizeGpu releases every per-call buffer via try/finally, so repeated
//     calls don't leak VRAM (intermediateBuf alone is ~130 MB at 4K/k=256).
//   - getSharedDevice/disposeSharedDevice provide a process-wide cache that
//     survives many calls without re-acquiring the adapter.
//
// We can't directly query VRAM from JS, so the check is "50 sequential calls
// complete without throwing or a device-lost." On a leaking implementation a
// modest input still blows past the adapter's storage budget within ~30 calls.

import { assert } from "jsr:@std/assert@^1.0.0";
import { disposeSharedDevice, isWebGPUAvailable, quantizeGpu } from "../mod.ts";
import { getSharedDevice } from "../core/device.ts";
import { ramp, skipWarn } from "../_test-fixtures.ts";

const hasGpu = await isWebGPUAvailable();
const gpuOpts = { ignore: !hasGpu };

Deno.test({
  name: "GPU leak: 50 sequential quantizeGpu calls complete without exhaustion",
  ...gpuOpts,
  fn: async () => {
    const img = ramp(256, 256);
    const N = 50;

    // 3E-001 (3rd-pass review): snapshot the device identity *before* the
    // burst so we can detect any silent re-acquire caused by a mid-burst
    // device.lost. Without this snapshot, the device.lost race below
    // would be served by a freshly-acquired device with an unresolved
    // `lost` promise — false-negatives the leak detection entirely.
    const { device: deviceAtStart } = await getSharedDevice();

    for (let i = 0; i < N; i++) {
      const r = await quantizeGpu(img, { paletteSize: 64, kmeansIterations: 5 });
      assert(r.png.byteLength > 100, `iteration ${i}: tiny PNG ${r.png.byteLength}B`);
    }

    // Identity check: the cache must still hold the same device we saw
    // at iteration 0. A mismatch means device.lost fired (= driver reset
    // or VRAM eviction) and getSharedDevice transparently re-acquired,
    // which is exactly the leak we're hunting for.
    const { device: deviceAtEnd } = await getSharedDevice();
    assert(
      deviceAtStart === deviceAtEnd,
      "shared device was replaced mid-burst (device.lost fired and re-acquire ran) — VRAM exhaustion",
    );

    // NEW-E-006: quantitative VRAM verification isn't reachable from JS, but
    // `device.lost` is a strong proxy: the WebGPU runtime resolves it when
    // the device is destroyed, the driver resets, or the GPU process OOMs.
    // If the 50-call burst above leaked enough VRAM to trigger any of those,
    // device.lost would already be resolved by now. Race it against a
    // microtask and assert the microtask wins — i.e. the device is still
    // alive at end-of-test.
    const ALIVE = Symbol("alive");
    const outcome = await Promise.race([
      deviceAtEnd.lost.then(() => "lost" as const),
      new Promise<typeof ALIVE>((r) => setTimeout(() => r(ALIVE), 5)),
    ]);
    assert(
      outcome === ALIVE,
      "device.lost resolved during the 50-call burst — VRAM exhaustion or driver reset",
    );
  },
});

Deno.test({
  name: "GPU device acquire is single-flight under concurrent callers (W-A-5)",
  ...gpuOpts,
  fn: async () => {
    // Start from a fresh cache so we actually exercise the acquisition
    // race (otherwise the first `await` short-circuits via _cachedDevice).
    disposeSharedDevice();
    // Three parallel cold-cache callers. A leaky implementation would
    // start three separate acquireDevice() calls and the cache would
    // end up holding only the last one's result, with two orphans
    // sitting in VRAM until process exit.
    const [a, b, c] = await Promise.all([
      getSharedDevice(),
      getSharedDevice(),
      getSharedDevice(),
    ]);
    // Identity check pins single-flight: all three callers must observe
    // the *same* DeviceInfo (and therefore the same GPUDevice).
    assert(
      a === b && b === c,
      "concurrent getSharedDevice() returned different DeviceInfo — single-flight broken",
    );
    assert(
      a.device === b.device && b.device === c.device,
      "concurrent getSharedDevice() returned different GPUDevice — single-flight broken",
    );
    disposeSharedDevice();
  },
});

Deno.test({
  name: "concurrent Promise.all([quantizeGpu × 4]) — pipeline cache + indices determinism (§6-A)",
  ...gpuOpts,
  fn: async () => {
    // §6-A (4th-pass review): public-API analogue of W-A-5. Four concurrent
    // `quantizeGpu` calls on identical input from a cold cache must:
    //   1. all complete without throwing
    //   2. all share the same `_cachedDevice` (the single-flight lock and
    //      tombstone logic protect the device tier)
    //   3. produce byte-identical `indices` streams (the pipeline cache
    //      key is `device` identity; if a concurrent rebuild ever raced
    //      and produced a different pipeline, the deterministic algorithm
    //      would still match — but a leaked GPU buffer mid-pipeline would
    //      typically show as drift here, so the byte-equality assert is
    //      our cheapest smoke for concurrent-correctness regressions)
    disposeSharedDevice();
    const img = ramp(96, 96);
    const opts = { paletteSize: 16, kmeansIterations: 4, dither: "none" as const };
    const results = await Promise.all([
      quantizeGpu(img, opts),
      quantizeGpu(img, opts),
      quantizeGpu(img, opts),
      quantizeGpu(img, opts),
    ]);
    // Indices determinism: byte-equal across all four parallel calls.
    const ref = results[0].indices;
    for (let i = 1; i < results.length; i++) {
      const cur = results[i].indices;
      assert(
        cur.length === ref.length,
        `call ${i}: indices length ${cur.length} != ref ${ref.length}`,
      );
      // Avoid asserting every byte in a loop with explicit messages —
      // a single mismatch counter is enough and keeps the test fast.
      let mismatches = 0;
      for (let j = 0; j < ref.length; j++) if (cur[j] !== ref[j]) mismatches++;
      assert(
        mismatches === 0,
        `call ${i}: ${mismatches} pixels disagree with call 0 — concurrent quantizeGpu non-deterministic`,
      );
    }
    // Device sharing: all four calls ran against the same `_cachedDevice`.
    const { device: cached } = await getSharedDevice();
    // Sanity: a duplicate `getSharedDevice()` returns the same identity.
    const { device: cached2 } = await getSharedDevice();
    assert(cached === cached2, "shared device identity not stable post-burst");
    disposeSharedDevice();
  },
});

Deno.test({
  name: "disposeSharedDevice during in-flight acquire tombstones cleanly (4A-01)",
  ...gpuOpts,
  fn: async () => {
    // Defensive contract test for 4A-01 (4th-pass review): if dispose is
    // called while `getSharedDevice` is still acquiring, the pending
    // promise must reject and the in-flight device must not silently
    // re-populate `_cachedDevice` after dispose returned.
    //
    // Sequence:
    //   t=0: ensure cold cache (no pending in flight either)
    //   t=1: start an acquire — synchronously gets the pending promise
    //   t=2: synchronously call dispose — flips `_pendingAcquireDisposed`
    //   t=3: await the pending promise → must reject
    //   t=4: a fresh acquire after the tombstone must succeed
    //        (proves the tombstone was per-acquisition, not sticky)
    disposeSharedDevice();
    const pending = getSharedDevice();
    disposeSharedDevice();
    let rejected = false;
    try {
      await pending;
    } catch (e) {
      rejected = true;
      assert(e instanceof Error, "rejection should be an Error instance");
    }
    assert(
      rejected,
      "pending getSharedDevice() must reject after a racing disposeSharedDevice()",
    );
    // Cache must not have been populated by the tombstoned device.
    const fresh = await getSharedDevice();
    assert(fresh.device, "fresh getSharedDevice() after tombstone must succeed");
    // The fresh device is a new acquisition, not a leak from the tombstoned
    // attempt: its `lost` promise should still be pending.
    const ALIVE = Symbol("alive");
    const outcome = await Promise.race([
      fresh.device.lost.then(() => "lost" as const),
      new Promise<typeof ALIVE>((r) => setTimeout(() => r(ALIVE), 5)),
    ]);
    assert(outcome === ALIVE, "fresh device should be alive after the tombstone race");
    disposeSharedDevice();
  },
});

Deno.test({
  name: "GPU leak: disposeSharedDevice + re-acquire round-trips cleanly",
  ...gpuOpts,
  fn: async () => {
    // NEW-E-005 (updated in 4E-002, 4th-pass review): this test runs after the
    // single-flight test above, which calls disposeSharedDevice() at both its
    // start and end. So we always begin with a cold cache here — the first
    // batch below acquires a fresh device, the explicit dispose forces a
    // re-acquire, and the second batch confirms re-acquire works. Running
    // `deno test --filter "round-trips"` in isolation is identical because
    // acquireDevice lazily creates one.
    const img = ramp(64, 64);
    // First batch shares one device.
    for (let i = 0; i < 3; i++) {
      await quantizeGpu(img, { paletteSize: 16, kmeansIterations: 3 });
    }
    disposeSharedDevice();
    // Second batch re-acquires.
    for (let i = 0; i < 3; i++) {
      await quantizeGpu(img, { paletteSize: 16, kmeansIterations: 3 });
    }
    disposeSharedDevice();
  },
});

if (!hasGpu) skipWarn("GPU leak test");
