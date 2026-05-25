/**
 * @module
 *
 * `@ys319/inksquid` — WebGPU-based image color quantization → PNG-8 encoder.
 *
 * Pipeline:
 *   RGBA → OkLab (GPU) → DoG importance (GPU) → Wu init (CPU) →
 *   weighted k-means refine (GPU) → blue-noise dither + assign (GPU) →
 *   PNG-8 encode (CPU: chunks + CompressionStream)
 *
 * @example
 * ```ts
 * import { quantize } from "@ys319/inksquid";
 *
 * const result = await quantize(myImageBitmap, {
 *   paletteSize: 128,
 *   dither: "blue-noise",
 *   ditherStrength: 1.0,
 * });
 *
 * // result.png is a Uint8Array containing a PNG-8 file
 * // result.preview is an ImageData of the quantized image
 * // result.meta.elapsedMs is the wall-clock time taken
 * ```
 */

export { quantize, type QuantizeMode, type QuantizeOptionsExt } from "./api/quantize.ts";
export { quantizeCpu } from "./api/quantize-cpu.ts";
export { quantizeGpu } from "./api/quantize-gpu.ts";
export { quantizeTiled } from "./api/quantize-tiled.ts";
export {
  DEFAULT_OPTIONS,
  normalizeOptions,
  type QuantizeOptions,
  type QuantizeResult,
  type RawImage,
} from "./core/types.ts";
export {
  disposeSharedDevice,
  GpuBufferOverflowError,
  isWebGPUAvailable,
  NoWebGPUError,
} from "./core/device.ts";
