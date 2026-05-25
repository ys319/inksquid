// Unit tests for the device-acquisition helpers `_acquireWithRetry` (A-a)
// and `_acquireWithTimeout` (A-j) added in the 5th-pass session.
//
// These cover the production code paths that don't need a real adapter:
// transient-retry policy, NoWebGPUError fast-path, timeout behaviour, and
// the wrapper composition (`_acquireWithTimeout` → `_acquireWithRetry` →
// `acquireFn`). Real-adapter integration is covered by tests/gpu/leak.test.ts
// (which exercises `getSharedDevice` end-to-end). §6-B "rejection then
// retry" coverage notes are at the bottom of this file.

import { assert, assertEquals, assertRejects, assertStringIncludes } from "jsr:@std/assert@^1.0.0";
import {
  _acquireWithRetry,
  _acquireWithTimeout,
  type DeviceInfo,
  NoWebGPUError,
} from "./device.ts";

// Minimal fake DeviceInfo. The helpers under test only pass it through —
// no method on the contained `device` is called inside their code paths,
// so a structural placeholder is sufficient. The downstream consumer
// (`getSharedDevice`) attaches the `device.lost.then` listener, but that
// runs in its caller, not here.
function fakeDeviceInfo(): DeviceInfo {
  return {
    // deno-lint-ignore no-explicit-any
    device: { lost: new Promise(() => {}), destroy: () => {} } as any,
    // deno-lint-ignore no-explicit-any
    adapter: {} as any,
    limits: { maxBufferSize: 1024 * 1024 },
  };
}

// ----- A-a: _acquireWithRetry -----

Deno.test("A-a: _acquireWithRetry resolves on first-call success without retry", async () => {
  let calls = 0;
  const di = await _acquireWithRetry(() => {
    calls++;
    return Promise.resolve(fakeDeviceInfo());
  });
  assertEquals(calls, 1, "no retry should fire when first call succeeds");
  assert(di.device);
});

Deno.test("A-a: _acquireWithRetry retries once on transient (non-NoWebGPU) failure", async () => {
  let calls = 0;
  const di = await _acquireWithRetry(() => {
    calls++;
    if (calls === 1) return Promise.reject(new Error("transient driver blip"));
    return Promise.resolve(fakeDeviceInfo());
  });
  assertEquals(calls, 2, "should retry exactly once after transient failure");
  assert(di.device);
});

Deno.test("A-a: _acquireWithRetry does NOT retry on NoWebGPUError", async () => {
  // Permanent: no adapter exists, retrying is pointless. Surface the error
  // immediately so the caller (mode: "auto") can fall back to CPU on the
  // very first attempt instead of waiting 50 ms for a second pointless
  // adapter probe.
  let calls = 0;
  await assertRejects(
    () =>
      _acquireWithRetry(() => {
        calls++;
        return Promise.reject(new NoWebGPUError("no adapter"));
      }),
    NoWebGPUError,
  );
  assertEquals(calls, 1, "NoWebGPUError must propagate without retry");
});

Deno.test("A-a: _acquireWithRetry propagates after two consecutive transient failures", async () => {
  // Single-shot retry. If both attempts fail, the second rejection
  // propagates so the higher layer can either re-attempt later (a fresh
  // `_pendingAcquire`) or surface the error.
  let calls = 0;
  await assertRejects(
    () =>
      _acquireWithRetry(() => {
        calls++;
        return Promise.reject(new Error(`fail ${calls}`));
      }),
    Error,
    "fail 2",
  );
  assertEquals(calls, 2, "should attempt exactly twice before giving up");
});

// ----- A-j: _acquireWithTimeout -----

Deno.test("A-j: _acquireWithTimeout resolves before timeout fires", async () => {
  // Fast resolve. The timer must be cleared via the .finally hook so the
  // event loop doesn't hold a pending timer for the full TTL after
  // resolution. Deno's test runner panics on dangling timers, so a
  // missing clearTimeout would surface as a test failure regardless of
  // this explicit assertion.
  const di = await _acquireWithTimeout(
    () => Promise.resolve(fakeDeviceInfo()),
    {},
    5000,
  );
  assert(di.device);
});

Deno.test("A-j: _acquireWithTimeout rejects on hung acquire", async () => {
  // Acquire never resolves → TTL race wins. The error message should
  // mention the actual timeout used so consumers can distinguish "this
  // call took too long" from other Error types reaching them. Use a
  // short ttlMs so the test completes quickly.
  let neverResolveStarted = false;
  const start = performance.now();
  const err = await assertRejects(
    () =>
      _acquireWithTimeout(
        () => {
          neverResolveStarted = true;
          return new Promise(() => {}); // never resolves
        },
        {},
        80,
      ),
    Error,
  );
  const elapsed = performance.now() - start;
  assert(neverResolveStarted, "acquireFn must have been invoked");
  assert(elapsed >= 70, `should take ~ttlMs, got ${elapsed}ms`);
  assert(elapsed < 500, `should not block much past ttlMs, got ${elapsed}ms`);
  assertStringIncludes(err.message, "80ms");
  assertStringIncludes(err.message, "hung adapter");
});

Deno.test("A-j: _acquireWithTimeout composes with A-a retry (transient → retry → success within TTL)", async () => {
  // Wrapper composition test. The retry path (50 ms backoff + 2nd
  // attempt) must complete within the timeout so a transient blip
  // doesn't cascade into a TTL-driven rejection. The default TTL of
  // 10 s is way over this budget, but we still want the composition
  // pinned with a shorter TTL to keep the test fast and to catch
  // regressions where the timer somehow doesn't get cleared by the
  // retry's intermediate resolution.
  let calls = 0;
  const di = await _acquireWithTimeout(
    () => {
      calls++;
      if (calls === 1) return Promise.reject(new Error("blip"));
      return Promise.resolve(fakeDeviceInfo());
    },
    {},
    1000,
  );
  assertEquals(calls, 2);
  assert(di.device);
});

// ----- §6-B coverage note -----

// §6-B "_pendingAcquire rejection 後の retry を pin": this scenario is
// already pinned end-to-end by `disposeSharedDevice during in-flight
// acquire tombstones cleanly (4A-01)` in tests/gpu/leak.test.ts — that
// test makes `_pendingAcquire` reject (via the tombstone path), then
// asserts a subsequent `getSharedDevice` call kicks off a *fresh*
// acquire and succeeds. The above A-a / A-j unit tests cover the
// helper layer; together they pin the full retry-after-rejection
// behaviour without needing a new GPU integration test.
//
// §6-C "device.lost 中の concurrent re-acquire を pin" is not testable
// without a way to forcibly resolve `device.lost` from JS (the WebGPU
// API has no such trigger). The natural recovery path — `device.lost`
// → `_cachedDevice = null` → next `getSharedDevice` re-acquires — is
// already covered by `GPU leak: disposeSharedDevice + re-acquire
// round-trips cleanly` in leak.test.ts, which exercises the same code
// path via explicit dispose.
