// CPU-only quantization pipeline. Used as:
//   - reference for tests
//   - actual fallback in environments without WebGPU (when mode='cpu' or
//     mode='auto' and no adapter)
//
// The pipeline mirrors the GPU one stage-for-stage, so output between the two
// is broadly comparable (subject to float-precision differences in k-means).

import { imageToOklabF32 } from "../colorspace/oklab.ts";
import { detailMap } from "../detail/dog.ts";
import { oklabPaletteToSrgb, wuQuantizeOklab } from "../palette/wu.ts";
import { kmeansRefine } from "../palette/kmeans.ts";
import { scolorqQuantize } from "../palette/scolorq.ts";
import { blueNoiseDither } from "../dither/blue-noise.ts";
import { floydSteinberg } from "../dither/floyd-steinberg.ts";
import { encodePng8 } from "../encode-png/mod.ts";
import {
  normalizeOptions,
  type QuantizeOptions,
  type QuantizeResult,
  type RawImage,
  toRawImage,
} from "../core/types.ts";

/**
 * CPU-only quantization pipeline. Reference implementation that mirrors
 * the GPU pipeline stage-for-stage; output stays within sub-perceptual
 * distance of `quantizeGpu` on the same input (pinned by the parity
 * tests). Always works regardless of WebGPU availability — use this
 * directly when you need deterministic CPU behaviour (SSR, sandboxed
 * runtimes, tests), or when an `ImageBitmap` isn't worth the GPU
 * dispatch overhead.
 *
 * For images above ~2 megapixels prefer {@link quantizeTiled} (or
 * `quantize({ mode: "cpu" })`, which routes to it automatically) — this
 * function keeps the whole working set resident.
 */
export async function quantizeCpu(
  input: ImageBitmap | ImageData | RawImage,
  optionsIn: Partial<QuantizeOptions> = {},
): Promise<QuantizeResult> {
  const options = normalizeOptions(optionsIn);
  // A-e: cancellation checkpoint at entry, then at each major-stage
  // boundary below. The per-pixel inner loops are intentionally not
  // polled — abort latency is bounded by "one stage" instead of
  // "instantaneous", which keeps the hot path branch-free.
  options.signal?.throwIfAborted();
  const t0 = performance.now();
  const img = toRawImage(input);
  const { width, height, data } = img;

  // 1. RGBA -> OkLab.
  const oklab = imageToOklabF32(data);
  options.signal?.throwIfAborted();

  // 2. DoG importance map.
  const importance = options.detailWeight > 0 ? detailMap({ width, height, oklab }) : undefined;

  // 3. Per-pixel weights = alpha * lerp(1, importance, detailWeight).
  const weights = new Float32Array(width * height);
  for (let i = 0; i < weights.length; i++) {
    const alpha = oklab[i * 4 + 3];
    if (alpha <= 0) {
      weights[i] = 0;
      continue;
    }
    if (importance) {
      const w = 1 - options.detailWeight + options.detailWeight * (importance[i] * 4 + 0.25);
      weights[i] = alpha * w;
    } else {
      weights[i] = alpha;
    }
  }

  // 4. Wu init.
  const wu = wuQuantizeOklab({ oklab, weights, paletteSize: options.paletteSize });
  options.signal?.throwIfAborted();

  // 5. Palette refinement + dither — branched on the dither option.
  // For `"scolorq"` the palette refinement and the dither indices are
  // jointly optimised by `scolorqQuantize`, replacing the historical
  // `Wu + kmeansRefine + dither` sequence. For every other dither
  // option, we still run `kmeansRefine` and then dispatch to the
  // appropriate dither (or fall through to `kmeansRefine`'s own
  // indices for `"none"`).
  let finalCentroids: Float32Array;
  let finalCount: number;
  let indices: Uint8Array;
  if (options.dither === "scolorq") {
    const sq = scolorqQuantize({
      width,
      height,
      oklab,
      weights,
      paletteSize: options.paletteSize,
      initialPalette: wu.oklab,
      signal: options.signal,
    });
    finalCentroids = sq.centroids;
    finalCount = sq.count;
    indices = sq.indices;
  } else {
    const km = kmeansRefine({
      oklab,
      weights,
      initial: wu,
      iterations: options.kmeansIterations,
      signal: options.signal,
    });
    finalCentroids = km.centroids;
    finalCount = km.count;
    if (options.dither === "floyd-steinberg") {
      indices = floydSteinberg({
        width,
        height,
        oklab,
        palette: km.centroids,
        paletteCount: km.count,
        strength: options.ditherStrength,
        importance,
      });
    } else if (options.dither === "blue-noise") {
      indices = blueNoiseDither({
        width,
        height,
        oklab,
        palette: km.centroids,
        paletteCount: km.count,
        strength: options.ditherStrength,
        importance,
      });
    } else {
      indices = km.indices;
    }
  }

  // 6. Palette → sRGB and alpha table.
  const paletteSrgb = oklabPaletteToSrgb({ oklab: finalCentroids, count: finalCount });

  // Per-palette-entry alpha: take the rounded mean alpha of pixels mapped to it.
  // For fully-opaque inputs this is a no-op (all 255).
  let alphaTable: Uint8Array | undefined;
  let needsAlpha = false;
  for (let i = 0; i < data.length / 4; i++) {
    if (data[i * 4 + 3] !== 255) {
      needsAlpha = true;
      break;
    }
  }
  if (needsAlpha) {
    const sumA = new Float64Array(finalCount);
    const countA = new Uint32Array(finalCount);
    for (let i = 0; i < indices.length; i++) {
      sumA[indices[i]] += data[i * 4 + 3];
      countA[indices[i]]++;
    }
    alphaTable = new Uint8Array(finalCount);
    for (let j = 0; j < finalCount; j++) {
      alphaTable[j] = countA[j] === 0 ? 255 : Math.round(sumA[j] / countA[j]);
    }
  }

  // 8. PNG-8 encode. Skip index validation: indices come from Wu →
  // k-means whose count is the same we hand to PLTE, so the bounds
  // check would just be busy work on a multi-MP image.
  options.signal?.throwIfAborted();
  const png = await encodePng8({
    width,
    height,
    indices,
    palette: { rgb: paletteSrgb, alpha: alphaTable },
    validate: false,
  });

  // 9. Build preview ImageData. We splat directly via paletteSrgb (the
  // post-oklabPaletteToSrgb byte table) — no separate OkLab→sRGB cache
  // here. Identical results to the older `cache`-based approach
  // because `oklabPaletteToSrgb` uses the same `linearToSrgbU8` helper
  // the cache used; removing the duplicate keeps the rounding policy
  // in one place (B-4 helper consolidation extends naturally here).
  const previewData = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < indices.length; i++) {
    const j = indices[i];
    previewData[i * 4] = paletteSrgb[j * 3];
    previewData[i * 4 + 1] = paletteSrgb[j * 3 + 1];
    previewData[i * 4 + 2] = paletteSrgb[j * 3 + 2];
    previewData[i * 4 + 3] = alphaTable ? alphaTable[j] : 255;
  }
  const preview: ImageData = typeof ImageData !== "undefined"
    ? new ImageData(previewData, width, height)
    : ({ data: previewData, width, height, colorSpace: "srgb" } as ImageData);

  const elapsedMs = performance.now() - t0;
  return {
    png,
    preview,
    palette: paletteSrgb,
    indices,
    meta: {
      outputBytes: png.byteLength,
      paletteSize: finalCount,
      elapsedMs,
      pipeline: "cpu",
    },
  };
}
