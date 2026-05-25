// Tile-based variant for large images.
//
// Strategy:
//   1. Downsample globally → 512×512 max, run full pipeline → get palette
//      (Wu + k-means refine).
//   2. For each tile, run only the per-pixel stages with the shared palette:
//      OkLab convert → DoG (optional, for adaptive dither only) → dither/assign.
//   3. Concatenate tile indices into a single image-sized array.
//   4. PNG-8 encode at the end.
//
// This avoids ever holding the full OkLab buffer (16 B/pixel) in memory: a
// 3840×2160 image's OkLab buffer is ~130 MB which exceeds wgpu's typical
// maxStorageBufferBindingSize of 128 MB. By keeping per-tile work to
// e.g. 1024×1024 = 16 MB we stay well under the limit.
//
// Dither caveat: Floyd-Steinberg propagates the residual error to the next
// row, so dithering tile-by-tile would reset the error at every seam and
// produce visible streaks. Callers that ask for FS on a tiled input get
// blue-noise instead (warned once); blue-noise is purely point-wise once
// we pass the tile's absolute origin as the mask phase offset.

import { imageToOklabF32 } from "../colorspace/oklab.ts";
import { oklabPaletteToSrgb, wuQuantizeOklab } from "../palette/wu.ts";
import { assignNearestOklab, kmeansRefine } from "../palette/kmeans.ts";
import { blueNoiseDither } from "../dither/blue-noise.ts";
import { detailMap } from "../detail/dog.ts";
import { encodePng8 } from "../encode-png/mod.ts";
import {
  normalizeOptions,
  type QuantizeOptions,
  type QuantizeResult,
  type RawImage,
  toRawImage,
} from "../core/types.ts";

const PALETTE_SUBSAMPLE_MAX = 512;

/**
 * Strip height in pixels. Exposed for tests so seam-quality checks can
 * align their probe rows with the actual tile boundary instead of
 * hard-coding `1024`. Production callers shouldn't depend on the exact
 * value — it's tuned to keep the per-tile working set under ~32 MB and
 * may change if memory characteristics shift. JSR consumers can't reach
 * this symbol because `quantize-tiled.ts` isn't registered in the
 * `deno.json` `exports` map.
 */
export const TILE = 1024;

// Single-shot warning for the FS→blue-noise downgrade (see header comment).
// Module-level so we don't spam the console when callers process a batch.
let _warnedAboutFsDowngrade = false;
function warnFsDowngradeOnce(): void {
  if (_warnedAboutFsDowngrade) return;
  _warnedAboutFsDowngrade = true;
  console.warn(
    '[inksquid] dither: "floyd-steinberg" is not supported on the ' +
      "tiled CPU path (>2 MP images); falling back to blue-noise to avoid " +
      'tile-boundary streaking. Pass dither: "blue-noise" explicitly to ' +
      "silence this warning.",
  );
}

// Exposed for tests. Production callers shouldn't need this.
export function _resetTiledFsDowngradeWarning(): void {
  _warnedAboutFsDowngrade = false;
}

// Same one-shot warning shape for scolorq's tiled downgrade.
let _warnedAboutScolorqDowngrade = false;
function warnScolorqDowngradeOnce(): void {
  if (_warnedAboutScolorqDowngrade) return;
  _warnedAboutScolorqDowngrade = true;
  console.warn(
    '[inksquid] dither: "scolorq" is not supported on the ' +
      "tiled CPU path (>2 MP images); falling back to blue-noise. " +
      "scolorq's spatial filter and soft-assignment matrix don't tile " +
      'cleanly. Pass dither: "blue-noise" explicitly to silence this warning.',
  );
}

export function _resetTiledScolorqDowngradeWarning(): void {
  _warnedAboutScolorqDowngrade = false;
}

function subsample(img: RawImage, maxDim: number): RawImage {
  if (img.width <= maxDim && img.height <= maxDim) return img;
  const scale = Math.min(maxDim / img.width, maxDim / img.height);
  const newW = Math.max(1, Math.floor(img.width * scale));
  const newH = Math.max(1, Math.floor(img.height * scale));
  const out = new Uint8ClampedArray(newW * newH * 4);
  // Simple box downsampling — fine for palette estimation.
  for (let y = 0; y < newH; y++) {
    const sy0 = Math.floor(y * img.height / newH);
    const sy1 = Math.max(sy0 + 1, Math.floor((y + 1) * img.height / newH));
    for (let x = 0; x < newW; x++) {
      const sx0 = Math.floor(x * img.width / newW);
      const sx1 = Math.max(sx0 + 1, Math.floor((x + 1) * img.width / newW));
      let r = 0, g = 0, b = 0, a = 0, n = 0;
      for (let yy = sy0; yy < sy1; yy++) {
        for (let xx = sx0; xx < sx1; xx++) {
          const i = (yy * img.width + xx) * 4;
          r += img.data[i];
          g += img.data[i + 1];
          b += img.data[i + 2];
          a += img.data[i + 3];
          n++;
        }
      }
      // L-A-2 (2nd-pass review): the `Math.max(sx0+1, ...)` / `Math.max(sy0+1, ...)`
      // clamps above guarantee sx1 > sx0 and sy1 > sy0 whenever img.width
      // and img.height are ≥ 1, so n ≥ 1 here by construction. Assert it
      // anyway: if a future refactor changes the box bounds and lets n
      // hit 0, dividing by zero produces NaN → Math.round(NaN) === 0 → a
      // black sample silently corrupts the subsampled palette init. A
      // loud failure is much easier to debug than a black pixel.
      if (n === 0) {
        throw new Error(
          `subsample: zero-sample box at (x=${x}, y=${y}); sx0=${sx0} sx1=${sx1} sy0=${sy0} sy1=${sy1}`,
        );
      }
      const j = (y * newW + x) * 4;
      out[j] = Math.round(r / n);
      out[j + 1] = Math.round(g / n);
      out[j + 2] = Math.round(b / n);
      out[j + 3] = Math.round(a / n);
    }
  }
  return { width: newW, height: newH, data: out };
}

/**
 * Tiled CPU quantization pipeline for images larger than ~2 megapixels.
 * Processes the image in 1024-row strips so the per-tile working set
 * stays bounded; `quantize({ mode: "cpu" })` routes here automatically
 * above the threshold. Always works regardless of WebGPU availability.
 *
 * The tiled path force-downgrades `dither: "floyd-steinberg"` and
 * `"scolorq"` to `"blue-noise"` (with a single stderr warning per
 * process). FS would reset its row-to-row error diffusion at every tile
 * seam and show visible streaks; scolorq's soft-assignment matrix and
 * spatial filter don't tile cleanly either.
 */
export async function quantizeTiled(
  input: ImageBitmap | ImageData | RawImage,
  optionsIn: Partial<QuantizeOptions> = {},
): Promise<QuantizeResult> {
  const options = normalizeOptions(optionsIn);
  // A-e: cancellation checkpoint at entry and at each tile boundary
  // inside the main loop (the longest part of the tiled path). Inner
  // per-pixel loops aren't polled — abort latency is bounded by "one
  // tile" (~16 MB working set, ~50-100 ms).
  options.signal?.throwIfAborted();
  const t0 = performance.now();
  const img = toRawImage(input);
  const { width, height } = img;

  // 1. Subsample to derive palette.
  const small = subsample(img, PALETTE_SUBSAMPLE_MAX);
  const smallOklab = imageToOklabF32(small.data);
  const smallImportance = options.detailWeight > 0
    ? detailMap({ width: small.width, height: small.height, oklab: smallOklab })
    : undefined;
  const weights = new Float32Array(small.width * small.height);
  for (let i = 0; i < weights.length; i++) {
    const alpha = smallOklab[i * 4 + 3];
    if (alpha <= 0) {
      weights[i] = 0;
      continue;
    }
    if (smallImportance) {
      weights[i] = alpha *
        (1 - options.detailWeight + options.detailWeight * (smallImportance[i] * 4 + 0.25));
    } else {
      weights[i] = alpha;
    }
  }
  const wu = wuQuantizeOklab({ oklab: smallOklab, weights, paletteSize: options.paletteSize });
  const km = kmeansRefine({
    oklab: smallOklab,
    weights,
    initial: wu,
    iterations: options.kmeansIterations,
    signal: options.signal,
  });
  const paletteSrgb = oklabPaletteToSrgb({ oklab: km.centroids, count: km.count });

  // 2. Apply palette tile-by-tile.
  const indices = new Uint8Array(width * height);
  const sumA = new Float64Array(km.count);
  const countA = new Uint32Array(km.count);
  let needsAlpha = false;
  for (let i = 0; i < img.data.length / 4; i++) {
    if (img.data[i * 4 + 3] !== 255) {
      needsAlpha = true;
      break;
    }
  }

  // Force-route both FS and scolorq to blue-noise on the tiled path:
  // FS would streak across tile seams (file header), and scolorq's
  // spatial filter and soft-assignment matrix don't tile cleanly
  // either. Same downgrade warning machinery for both.
  const effectiveDither: QuantizeOptions["dither"] = (() => {
    if (options.dither === "floyd-steinberg") {
      warnFsDowngradeOnce();
      return "blue-noise";
    }
    if (options.dither === "scolorq") {
      warnScolorqDowngradeOnce();
      return "blue-noise";
    }
    return options.dither;
  })();

  for (let ty = 0; ty < height; ty += TILE) {
    const th = Math.min(TILE, height - ty);
    for (let tx = 0; tx < width; tx += TILE) {
      // A-e: poll the signal at every tile boundary. A 3840×2160 image
      // has 2×4 tiles at TILE=1024, so worst-case abort latency is
      // ~one-tile-of-work (~50-100 ms on CPU).
      options.signal?.throwIfAborted();
      const tw = Math.min(TILE, width - tx);
      const tile = new Uint8ClampedArray(tw * th * 4);
      for (let y = 0; y < th; y++) {
        const srcStart = ((ty + y) * width + tx) * 4;
        tile.set(img.data.subarray(srcStart, srcStart + tw * 4), y * tw * 4);
      }
      const tileOklab = imageToOklabF32(tile);
      const tileImportance = options.detailWeight > 0
        ? detailMap({ width: tw, height: th, oklab: tileOklab })
        : undefined;
      let tileIndices: Uint8Array;
      if (effectiveDither === "blue-noise") {
        // Pass the tile's absolute origin so the mask phase is continuous
        // across tile seams regardless of TILE (no longer relies on
        // TILE % 64 == 0 — see blue-noise.ts).
        tileIndices = blueNoiseDither({
          width: tw,
          height: th,
          oklab: tileOklab,
          palette: km.centroids,
          paletteCount: km.count,
          strength: options.ditherStrength,
          importance: tileImportance,
          offsetX: tx,
          offsetY: ty,
        });
      } else {
        // dither === "none" — straight nearest-centroid assignment, no jitter.
        tileIndices = assignNearestOklab(tileOklab, km.centroids, km.count);
      }
      // Splat into the global indices buffer + accumulate alpha stats.
      for (let y = 0; y < th; y++) {
        const dstStart = (ty + y) * width + tx;
        indices.set(tileIndices.subarray(y * tw, (y + 1) * tw), dstStart);
      }
      if (needsAlpha) {
        for (let i = 0; i < tw * th; i++) {
          const y = i / tw | 0;
          const x = i % tw;
          const srcIdx = ((ty + y) * width + tx + x) * 4;
          const c = tileIndices[i];
          sumA[c] += img.data[srcIdx + 3];
          countA[c]++;
        }
      }
    }
  }

  let alphaTable: Uint8Array | undefined;
  if (needsAlpha) {
    alphaTable = new Uint8Array(km.count);
    for (let j = 0; j < km.count; j++) {
      alphaTable[j] = countA[j] === 0 ? 255 : Math.round(sumA[j] / countA[j]);
    }
  }

  options.signal?.throwIfAborted();
  const png = await encodePng8({
    width,
    height,
    indices,
    palette: { rgb: paletteSrgb, alpha: alphaTable },
    // Indices come from blueNoiseDither / assignNearestOklab; both clamp
    // to km.count. Skip the O(n) revalidation on the tiled hot path —
    // for a 4 MP image this saves ~15-20 ms of pure busy work.
    validate: false,
  });

  // Preview: just splat palette colours via indices.
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

  return {
    png,
    preview,
    palette: paletteSrgb,
    indices,
    meta: {
      outputBytes: png.byteLength,
      paletteSize: km.count,
      elapsedMs: performance.now() - t0,
      pipeline: "cpu-tiled",
    },
  };
}
