import { GpuBufferOverflowError, isWebGPUAvailable, NoWebGPUError } from "../core/device.ts";
import type { QuantizeOptions, QuantizeResult, RawImage } from "../core/types.ts";
import { quantizeCpu } from "./quantize-cpu.ts";
import { quantizeGpu } from "./quantize-gpu.ts";
import { quantizeTiled } from "./quantize-tiled.ts";

/**
 * Test seam: lets tests swap the GPU implementation and adapter probe so
 * the auto-fallback branches (catching `NoWebGPUError` /
 * `GpuBufferOverflowError`, re-throwing everything else) can be exercised
 * without needing a too-large GPU input or a real WebGPU adapter.
 *
 * Production code never reads or writes these fields; the dispatcher
 * below resolves them on every call so mutations take effect immediately.
 * Tests are expected to restore the original values via try/finally
 * (see `withHooks` in tests/cpu/auto-fallback.test.ts, which is keyed on
 * the override set to stay robust as this object grows — D-A-5 / 3E-004
 * in the 3rd-pass review). If you add a field here, make sure tests
 * using `withHooks` continue to round-trip cleanly: pass only the
 * fields you actually want to override.
 *
 * **Not safe under parallel test execution**: `_testHooks` is a single
 * module-scoped object, so two tests running concurrently (`deno test
 * --parallel`) would clobber each other's overrides. The Deno test
 * runner is serial by default and this library's `deno task test`
 * relies on that; if `--parallel` is ever enabled, this object must be
 * moved behind an `AsyncLocalStorage` first (4A-05, 4th-pass review).
 *
 * @internal
 */
export const _testHooks = {
  quantizeGpu,
  isWebGPUAvailable,
};

/**
 * Pipeline selector for {@link quantize}:
 * - `"gpu"` (default): WebGPU pipeline. Throws {@link NoWebGPUError} when
 *   no adapter is available, or {@link GpuBufferOverflowError} when the
 *   image exceeds `device.limits.maxBufferSize`.
 * - `"cpu"`: CPU pipeline (auto-tiles above ~2 megapixels). Always works.
 * - `"auto"`: try GPU first; catch {@link NoWebGPUError} /
 *   {@link GpuBufferOverflowError} and fall back to CPU. Other GPU
 *   errors still propagate so misconfigurations aren't silently hidden.
 */
export type QuantizeMode = "gpu" | "cpu" | "auto";

/**
 * Caller-facing options for {@link quantize}. Extends
 * {@link QuantizeOptions} with the {@link QuantizeMode} selector and
 * makes every field optional — {@link normalizeOptions} fills in
 * {@link DEFAULT_OPTIONS} for anything omitted.
 */
export interface QuantizeOptionsExt extends Partial<QuantizeOptions> {
  /** Pipeline selector. Defaults to `"gpu"`. */
  mode?: QuantizeMode;
}

/**
 * Pixel-count threshold above which the CPU path switches to
 * `quantizeTiled`. Sized to keep the non-tiled OkLab f32 working set
 * (n × 16 bytes) under ~32 MB so we don't allocate a huge buffer for
 * a one-off large image. Exported so tests can probe the boundary
 * without hard-coding a duplicate copy of the value; not re-exported
 * from `src/mod.ts`, so it isn't part of the public JSR surface.
 * L-A-7 (3rd-pass review) made this public-to-tests instead of
 * file-private.
 *
 * 4A-03 (4th-pass review): this threshold is **CPU memory budget**,
 * unrelated to `device.limits.maxBufferSize` (which gates the GPU
 * single-pass via `GpuBufferOverflowError` at ~250 MP on M-series).
 * An earlier proposal suggested linking the two; that conflated CPU and
 * GPU constraints. On bench data (M5, HEAD≈3a243f3) CPU at 512² takes
 * ~390 ms — linear extrapolation puts 2 MP at ~3 s, which is the right
 * "tile or not tile" interactive boundary. Raising the threshold makes
 * single-pass CPU calls block for longer; lowering it makes the tile
 * subsample step kick in earlier with no working-set benefit. Leave at
 * 2 MP unless a future bench probes the >2 MP region directly.
 */
export const TILE_PIXEL_THRESHOLD = 2_000_000;

/**
 * Pick `quantizeCpu` vs `quantizeTiled` based on input pixel count.
 * `quantizeTiled` keeps the working set bounded by processing the image in
 * 1024-row strips; `quantizeCpu` is faster on smaller inputs because it
 * skips the per-tile palette subsampling step.
 */
function pickCpuPath(
  input: ImageBitmap | ImageData | RawImage,
  userOpts: Partial<QuantizeOptions>,
): Promise<QuantizeResult> {
  if (input.width * input.height >= TILE_PIXEL_THRESHOLD) {
    return quantizeTiled(input, userOpts);
  }
  return quantizeCpu(input, userOpts);
}

/**
 * Quantize an image to a PNG-8 palette using WebGPU.
 *
 * The default mode is "gpu" — the function throws NoWebGPUError if the
 * environment lacks a WebGPU adapter. Pass `{ mode: "auto" }` to fall back
 * to the CPU pipeline when (a) no adapter is available or (b) the input
 * exceeds the GPU's single-buffer limit; option-validation errors and
 * other GPU runtime failures still propagate so they don't get silently
 * masked. `{ mode: "cpu" }` forces the CPU path (useful for SSR or testing).
 *
 * The CPU path automatically switches to a tile-based variant for images
 * above ~2 megapixels to keep memory usage bounded.
 */
export async function quantize(
  input: ImageBitmap | ImageData | RawImage,
  options: QuantizeOptionsExt = {},
): Promise<QuantizeResult> {
  const { mode = "gpu", ...userOpts } = options;

  // A-e: pre-check before doing anything else, even adapter probes. A
  // signal that's already aborted should reject immediately rather than
  // do the (~10-100 ms) `isWebGPUAvailable` round-trip first.
  userOpts.signal?.throwIfAborted();

  if (mode === "gpu") return _testHooks.quantizeGpu(input, userOpts);

  // Track *why* the CPU pipeline ends up running, so we can surface it
  // via `result.meta.fallbackReason` for callers who care (telemetry,
  // UI spinners that say "falling back to CPU"). The value is set on
  // every code path that reaches `pickCpuPath` below — the TS compiler
  // verifies that with definite-assignment analysis.
  let fallbackReason: "no-gpu-adapter" | "gpu-buffer-overflow" | "cpu-forced";

  if (mode === "cpu") {
    fallbackReason = "cpu-forced";
  } else {
    // mode === "auto"
    const hasGpu = await _testHooks.isWebGPUAvailable();
    if (!hasGpu) {
      fallbackReason = "no-gpu-adapter";
    } else {
      try {
        return await _testHooks.quantizeGpu(input, userOpts);
      } catch (e) {
        // Only swallow conditions that mean "GPU can't service this call";
        // RangeError/TypeError from normalizeOptions or unexpected GPU
        // failures should still surface so users notice them.
        if (e instanceof NoWebGPUError) {
          fallbackReason = "no-gpu-adapter";
        } else if (e instanceof GpuBufferOverflowError) {
          fallbackReason = "gpu-buffer-overflow";
        } else {
          throw e;
        }
        // Intentionally no console.warn: SSR/CI logs would get spammed
        // for the *expected* large-image case. Callers can inspect
        // `result.meta.fallbackReason` instead.
      }
    }
  }

  const result = await pickCpuPath(input, userOpts);
  result.meta.fallbackReason = fallbackReason;
  return result;
}
