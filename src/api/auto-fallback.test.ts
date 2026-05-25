// Behaviour pins for `quantize({ mode: "auto" })`.
//
// "auto" should treat the GPU as an optimisation — if the adapter is missing
// (NoWebGPUError) or the image won't fit in a single GPU buffer
// (GpuBufferOverflowError), the call quietly falls through to the CPU path.
// Everything else (option-validation errors, unexpected GPU runtime errors)
// must still propagate; otherwise misconfigurations vanish into a silent
// CPU run.

import { assert, assertEquals, assertRejects, assertStringIncludes } from "jsr:@std/assert@^1.0.0";
import { GpuBufferOverflowError, NoWebGPUError, quantize } from "../mod.ts";
import { _testHooks } from "./quantize.ts";
import { flat as flatHelper } from "../_test-fixtures.ts";

// Local wrapper: this suite specifically wants the (64,128,192) colour
// (one of the few tests that depends on a non-neutral RGB triple — the
// auto-fallback flow's PNG-size sanity check `r.png.byteLength > 50`
// relies on enough chroma to not collapse to a single palette entry).
function flat(w = 8, h = 8): ImageData {
  return flatHelper(w, h, [64, 128, 192, 255]);
}

Deno.test("auto: option errors propagate (do not silently fall back)", async () => {
  // paletteSize=0 should throw RangeError regardless of mode. If auto
  // swallowed it and ran the CPU path, this assertion would fail with a
  // returned QuantizeResult instead of the expected RangeError.
  await assertRejects(
    () => quantize(flat(), { mode: "auto", paletteSize: 0 }),
    RangeError,
    "paletteSize",
  );
  await assertRejects(
    // deno-lint-ignore no-explicit-any
    () => quantize(flat(), { mode: "auto", dither: "off" as any }),
    TypeError,
    "dither",
  );
});

Deno.test("auto: produces output in CPU-only sandbox (no GPU adapter)", async () => {
  // NEW-E-017: This test runs in two distinct environments with different
  // `navigator.gpu` shapes:
  //   - `deno task test:cpu` (no --unstable-webgpu): navigator.gpu is
  //     undefined entirely. `isWebGPUAvailable()` short-circuits to false
  //     in its first `typeof navigator !== "undefined" &&
  //     "gpu" in navigator` check; auto goes to CPU.
  //   - `deno task test` / `deno task test:gpu` (--unstable-webgpu on a
  //     machine with Metal/Vulkan/D3D12): navigator.gpu exists,
  //     requestAdapter returns a real adapter; auto picks GPU.
  // Both must succeed and produce a valid PNG — that's what this test
  // pins. The gpu-side decision is additionally pinned in tests/gpu/
  // smoke.test.ts via `meta.pipeline === "gpu"`.
  const r = await quantize(flat(16, 16), { mode: "auto", paletteSize: 8 });
  assertEquals(r.preview.width, 16);
  assertEquals(r.preview.height, 16);
  assert(r.png.byteLength > 50, `tiny PNG ${r.png.byteLength}`);
  // Pipeline must be one of the known values (the GPU-side smoke test
  // additionally asserts "gpu" when an adapter is present).
  //
  // NEW-E-002: the "cpu-tiled" branch in this OR is never exercised by
  // this 16×16 input — `pickCpuPath` only routes to tiled at ≥ 2 MP. The
  // auto → cpu-tiled handoff is pinned separately in
  // tests/cpu/tiled.test.ts ('tiled exposes meta.pipeline === "cpu-tiled"')
  // so we don't need to repeat the (slow) 2 MP allocation here just to
  // touch a third arm. Leaving the OR loose because adding a "cpu-tiled"
  // branch to pickCpuPath's selection logic should still be considered a
  // valid known value at this layer.
  assert(
    r.meta.pipeline === "cpu" || r.meta.pipeline === "cpu-tiled" || r.meta.pipeline === "gpu",
    `unexpected meta.pipeline=${r.meta.pipeline}`,
  );
});

Deno.test('quantize(mode=cpu): meta.pipeline === "cpu"', async () => {
  const r = await quantize(flat(16, 16), { mode: "cpu", paletteSize: 8 });
  assertEquals(r.meta.pipeline, "cpu");
});

// A-f (5th-pass): `meta.fallbackReason` is set on every CPU-via-dispatcher
// code path so callers can distinguish *why* CPU ran (explicit request vs
// missing adapter vs GPU-side limit). The four arms below are the same
// branches the dispatcher itself splits on — keep this in sync if the
// dispatcher gains a new fallback condition (extend the union in
// QuantizeResult.meta.fallbackReason at the same time).
Deno.test("fallbackReason: mode=cpu sets cpu-forced", async () => {
  const r = await quantize(flat(16, 16), { mode: "cpu", paletteSize: 8 });
  assertEquals(r.meta.fallbackReason, "cpu-forced");
});

Deno.test("fallbackReason: mode=auto + no adapter sets no-gpu-adapter", async () => {
  const r = await withHooks(
    { isWebGPUAvailable: () => Promise.resolve(false) },
    () => quantize(flat(16, 16), { mode: "auto", paletteSize: 8 }),
  );
  assertEquals(r.meta.fallbackReason, "no-gpu-adapter");
});

Deno.test("fallbackReason: mode=auto + NoWebGPUError thrown sets no-gpu-adapter", async () => {
  const r = await withHooks(
    {
      isWebGPUAvailable: () => Promise.resolve(true),
      quantizeGpu: () => Promise.reject(new NoWebGPUError("adapter vanished")),
    },
    () => quantize(flat(16, 16), { mode: "auto", paletteSize: 8 }),
  );
  assertEquals(r.meta.fallbackReason, "no-gpu-adapter");
});

Deno.test("fallbackReason: mode=auto + GpuBufferOverflowError sets gpu-buffer-overflow", async () => {
  const r = await withHooks(
    {
      isWebGPUAvailable: () => Promise.resolve(true),
      quantizeGpu: () => Promise.reject(new GpuBufferOverflowError(1_000_000_000, 100)),
    },
    () => quantize(flat(16, 16), { mode: "auto", paletteSize: 8 }),
  );
  assertEquals(r.meta.fallbackReason, "gpu-buffer-overflow");
});

// Direct entry points must NOT carry fallbackReason — they bypass the
// dispatcher entirely, so there's no decision to report. A future
// refactor that accidentally sets it would silently inflate telemetry
// counts; this assertion pins the absence.
Deno.test("fallbackReason: direct quantizeCpu call has no fallbackReason", async () => {
  const { quantizeCpu } = await import("../../src/api/quantize-cpu.ts");
  const r = await quantizeCpu(flat(16, 16), { paletteSize: 8 });
  assertEquals(r.meta.fallbackReason, undefined);
});

// Type-only sanity check: the new error classes are reachable from mod.ts
// so callers can `instanceof`-check them. This catches an accidental
// export removal in a refactor.
Deno.test("auto: NoWebGPUError and GpuBufferOverflowError are exported", () => {
  assertEquals(typeof NoWebGPUError, "function");
  assertEquals(typeof GpuBufferOverflowError, "function");
  const e = new GpuBufferOverflowError(100, 50);
  assert(e instanceof Error);
  assert(e instanceof GpuBufferOverflowError);
  assertEquals(e.name, "GpuBufferOverflowError");
  assertEquals(e.required, 100);
  assertEquals(e.maxBufferSize, 50);
  // 3E-015 (3rd-pass review): pin enough of the human-readable message
  // that consumers (logs, error reporters) can rely on the format.
  // Partial-match so we don't tie ourselves to the exact phrasing —
  // just the two numeric fields and the "auto" hint should both be
  // present so the user knows what to try.
  assertStringIncludes(e.message, "100");
  assertStringIncludes(e.message, "50");
  assertStringIncludes(e.message, 'mode: "auto"');
});

// Exercising the catch branches via the test-only hook on quantize.ts.
// Without these, the sandbox-only assertion ("isWebGPUAvailable=false →
// goes straight to CPU") never actually entered the try block, so the
// `catch (e)` arms were dead code from the test suite's perspective.

// §6-E (4th-pass review): re-entrancy guard. `withHooks` mutates the
// module-scoped `_testHooks` object and restores it in `finally`, so a
// nested `withHooks(...)` call (i.e. another `withHooks` invocation
// before the outer one's `finally` ran) would: (a) save the *outer*'s
// overrides as the "saved" baseline, (b) restore those overrides on
// inner-finally, leaving the outer's overrides in place even though the
// outer hasn't returned, and (c) on outer-finally restore the original
// overrides — but with timing entanglement that's confusing to debug.
// Today no test nests withHooks calls; this flag makes the assumption
// explicit so a future test reaching for nesting fails loudly instead
// of getting silently wrong hook state.
let _withHooksInUse = false;

function withHooks<T>(
  overrides: Partial<typeof _testHooks>,
  fn: () => Promise<T>,
): Promise<T> {
  if (_withHooksInUse) {
    throw new Error(
      "withHooks: nested invocation is not supported — _testHooks is a single " +
        "module-scoped object and overlapping overrides would clobber each other. " +
        "Compose your overrides into a single withHooks call instead.",
    );
  }
  _withHooksInUse = true;
  // 3E-004 (3rd-pass review): restore is keyed on the *override* set,
  // not on the full snapshot of `_testHooks`. If a future change adds
  // hooks to `_testHooks` and a test passes an override for them,
  // the shallow `{ ..._testHooks }` approach would leave the new
  // override in place after `finally` (because `Object.assign(target,
  // saved)` doesn't *delete* keys absent from `saved`). Restoring only
  // the keys we overrode keeps the function robust as `_testHooks`
  // grows.
  type Hooks = typeof _testHooks;
  const overrideKeys = Object.keys(overrides) as Array<keyof Hooks>;
  const saved: Partial<Hooks> = {};
  for (const k of overrideKeys) {
    // deno-lint-ignore no-explicit-any
    (saved as any)[k] = _testHooks[k];
  }
  Object.assign(_testHooks, overrides);
  return fn().finally(() => {
    Object.assign(_testHooks, saved);
    _withHooksInUse = false;
  });
}

Deno.test("auto: GpuBufferOverflowError falls through to CPU", async () => {
  let gpuStubCalled = false;
  const r = await withHooks(
    {
      isWebGPUAvailable: () => Promise.resolve(true),
      quantizeGpu: () => {
        gpuStubCalled = true;
        return Promise.reject(new GpuBufferOverflowError(1_000_000_000, 100));
      },
    },
    () => quantize(flat(16, 16), { mode: "auto", paletteSize: 8 }),
  );
  assert(gpuStubCalled, "GPU stub never invoked — auto bypassed the try block");
  // The returned result must come from the CPU pipeline.
  assertEquals(r.meta.pipeline, "cpu");
  assert(r.png.byteLength > 50);
});

Deno.test("auto: NoWebGPUError falls through to CPU", async () => {
  // Even when isWebGPUAvailable reports true, a downstream NoWebGPUError
  // (e.g. adapter vanishing between probe and use) must still get caught.
  const r = await withHooks(
    {
      isWebGPUAvailable: () => Promise.resolve(true),
      quantizeGpu: () => Promise.reject(new NoWebGPUError("adapter vanished")),
    },
    () => quantize(flat(16, 16), { mode: "auto", paletteSize: 8 }),
  );
  assertEquals(r.meta.pipeline, "cpu");
});

Deno.test("auto: concurrent fallback is deterministic — all N callers complete on CPU (§6-D)", async () => {
  // §6-D (4th-pass review): when the GPU stub rejects with a fallback-
  // eligible error and N concurrent `quantize({ mode: "auto" })` calls
  // race the same hook state, each must independently catch the
  // rejection and complete on CPU. A regression here would look like:
  //   - one or more calls escaping the catch (`assertRejects` would fail
  //     to find a meta.pipeline)
  //   - non-deterministic pipeline labels across the N results
  //   - hook state mutated mid-flight by one call corrupting another
  //     (would surface as one of the calls suddenly throwing the wrong
  //     error type or hanging).
  const N = 6;
  let gpuCalls = 0;
  const results = await withHooks(
    {
      isWebGPUAvailable: () => Promise.resolve(true),
      quantizeGpu: () => {
        gpuCalls++;
        return Promise.reject(new GpuBufferOverflowError(1_000_000_000, 100));
      },
    },
    () =>
      Promise.all(
        Array.from(
          { length: N },
          () => quantize(flat(16, 16), { mode: "auto", paletteSize: 8 }),
        ),
      ),
  );
  assertEquals(gpuCalls, N, `GPU stub should be hit N=${N} times, got ${gpuCalls}`);
  assertEquals(results.length, N);
  for (let i = 0; i < N; i++) {
    assertEquals(
      results[i].meta.pipeline,
      "cpu",
      `call ${i}: pipeline=${results[i].meta.pipeline} (expected cpu)`,
    );
    assert(results[i].png.byteLength > 50, `call ${i}: tiny PNG`);
  }
});

Deno.test("withHooks: nested invocation throws (§6-E re-entrancy guard)", async () => {
  // Sanity check on the guard itself. The inner withHooks must throw
  // synchronously before mutating `_testHooks`, so the outer scope's
  // overrides remain intact across the failed nesting attempt.
  let innerError: unknown;
  await withHooks(
    {
      isWebGPUAvailable: () => Promise.resolve(false),
    },
    async () => {
      try {
        await withHooks(
          { isWebGPUAvailable: () => Promise.resolve(true) },
          () => Promise.resolve(null),
        );
      } catch (e) {
        innerError = e;
      }
      // After the failed nesting attempt, the outer override (false)
      // is still in place — quantize(auto) goes to CPU.
      const r = await quantize(flat(8, 8), { mode: "auto", paletteSize: 4 });
      assertEquals(r.meta.pipeline, "cpu");
    },
  );
  assert(innerError instanceof Error, "nested withHooks should have thrown");
  assertStringIncludes((innerError as Error).message, "nested invocation");
});

Deno.test("auto: unexpected GPU errors propagate (do not silently fall back)", async () => {
  // A non-WebGPU error from the GPU pipeline must surface so users notice
  // misconfiguration / driver bugs instead of getting a quiet CPU result.
  await assertRejects(
    () =>
      withHooks(
        {
          isWebGPUAvailable: () => Promise.resolve(true),
          quantizeGpu: () => Promise.reject(new Error("simulated driver crash")),
        },
        () => quantize(flat(16, 16), { mode: "auto", paletteSize: 8 }),
      ),
    Error,
    "simulated driver crash",
  );
});
