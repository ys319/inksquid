// A-e (5th-pass): AbortSignal coverage.
//
// Pins the checkpoint behaviour at every public entry point: a pre-aborted
// signal must throw before any meaningful work runs, and an abort that
// fires *during* a long-ish call must surface at the next stage / iter /
// tile boundary. Per-pixel granularity is intentionally not promised —
// abort latency is "one stage / one iter / one tile", which keeps the hot
// path branch-free.

import { assert, assertEquals, assertRejects } from "jsr:@std/assert@^1.0.0";
import { quantize, quantizeCpu, quantizeTiled } from "../mod.ts";
import { flat, ramp } from "../_test-fixtures.ts";

// Helper: assert that the rejection is an "abort" — DOMException with
// name "AbortError" is what `AbortSignal.throwIfAborted()` throws by
// default. Custom reasons are also valid (we test that path
// separately).
function isAbortError(e: unknown): boolean {
  if (e instanceof DOMException) return e.name === "AbortError";
  // Some runtimes may surface as plain Error with name "AbortError"
  if (e instanceof Error && e.name === "AbortError") return true;
  return false;
}

Deno.test("A-e: pre-aborted signal throws from quantize(mode=cpu) without doing work", async () => {
  const ctrl = new AbortController();
  ctrl.abort();
  const err = await assertRejects(
    () => quantize(flat(16, 16), { mode: "cpu", paletteSize: 8, signal: ctrl.signal }),
  );
  assert(isAbortError(err), `expected AbortError, got ${err}`);
});

Deno.test("A-e: pre-aborted signal throws from quantizeCpu directly", async () => {
  const ctrl = new AbortController();
  ctrl.abort();
  const err = await assertRejects(
    () => quantizeCpu(flat(16, 16), { paletteSize: 8, signal: ctrl.signal }),
  );
  assert(isAbortError(err), `expected AbortError, got ${err}`);
});

Deno.test("A-e: pre-aborted signal throws from quantizeTiled directly", async () => {
  // quantizeTiled doesn't gate on size — directly call it. The 2 MP
  // threshold is only enforced by `pickCpuPath` in quantize().
  const ctrl = new AbortController();
  ctrl.abort();
  const err = await assertRejects(
    () => quantizeTiled(ramp(64, 64), { paletteSize: 8, signal: ctrl.signal }),
  );
  assert(isAbortError(err), `expected AbortError, got ${err}`);
});

Deno.test("A-e: pre-aborted signal also throws from quantize(mode=auto) before CPU/GPU dispatch", async () => {
  // mode: "auto" probes for an adapter before falling back to CPU. A
  // pre-aborted signal should short-circuit that probe so we don't do
  // a wasted ~10-100ms round-trip just to throw.
  const ctrl = new AbortController();
  ctrl.abort();
  const err = await assertRejects(
    () => quantize(flat(16, 16), { mode: "auto", paletteSize: 8, signal: ctrl.signal }),
  );
  assert(isAbortError(err), `expected AbortError, got ${err}`);
});

Deno.test("A-e: abort reason propagates (custom Error)", async () => {
  // Callers can pass a custom reason to `abort(reason)`. The pipeline
  // must surface that exact reason via `throwIfAborted()`, not wrap it
  // or swallow it.
  const ctrl = new AbortController();
  const customReason = new Error("user cancelled by clicking X");
  ctrl.abort(customReason);
  const err = await assertRejects(
    () => quantizeCpu(flat(16, 16), { paletteSize: 8, signal: ctrl.signal }),
  );
  assertEquals(err, customReason);
});

Deno.test("A-e: kmeansRefine top-of-iter checkpoint throws on pre-aborted signal", () => {
  // Direct kmeansRefine call. The CPU pipeline is entirely synchronous
  // between entry and the final `await encodePng8(...)`, so we can't
  // race a microtask abort against an in-progress quantizeCpu call —
  // the microtask only fires after the iter loop completes. Instead
  // pin the checkpoint at the iter-loop level: a pre-aborted signal
  // must throw before iter 0 runs its assignment pass.
  //
  // Tested via the same code path the public entry points use, so a
  // regression that removes the iter-boundary poll would surface here
  // even if the pipeline-level checkpoint stayed intact.
  return import("../../src/palette/kmeans.ts").then(({ kmeansRefine }) => {
    const ctrl = new AbortController();
    ctrl.abort();
    const oklab = new Float32Array([0.5, 0, 0, 1, 0.6, 0, 0, 1]);
    const initial = { oklab: new Float32Array([0.5, 0, 0]), count: 1 };
    try {
      kmeansRefine({
        oklab,
        initial,
        iterations: 5,
        signal: ctrl.signal,
      });
      throw new Error("kmeansRefine did not throw on aborted signal");
    } catch (e) {
      assert(isAbortError(e), `expected AbortError, got ${e}`);
    }
  });
});

Deno.test("A-e: unaborted signal does not affect a successful run", async () => {
  // Smoke test: passing a non-aborted signal must be a no-op for the
  // happy path. A bug where the checkpoint accidentally throws on any
  // non-null signal would surface here.
  const ctrl = new AbortController();
  const r = await quantizeCpu(flat(16, 16), { paletteSize: 8, signal: ctrl.signal });
  assertEquals(r.meta.pipeline, "cpu");
  assert(r.png.byteLength > 50);
});
