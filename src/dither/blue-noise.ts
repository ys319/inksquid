/**
 * @module
 *
 * Blue-noise threshold dither in OkLab space — the default dither used
 * by the rest of inksquid, and importable standalone via the
 * `./blue-noise` sub-module entry point.
 *
 * The mask is a baked 64×64 void-and-cluster pattern (
 * {@link BLUE_NOISE_64_SIZE} / {@link getBlueNoise64}); sampling it three
 * times per pixel at coprime offsets gives decorrelated L / a / b
 * jitter without per-pixel RNG. The GPU shader at
 * `dither/blue-noise.wgsl.ts` mirrors this logic for the non-tiled
 * (whole-image) case. `offsetX` / `offsetY` are CPU-only — they're used
 * by the tiled CPU path to keep the mask phase continuous across tile
 * boundaries; the GPU path doesn't tile, so the shader has no
 * equivalent input.
 *
 * @example
 * ```ts
 * import { blueNoiseDither } from "@ys319/inksquid/blue-noise";
 *
 * const indices = blueNoiseDither({
 *   width, height,
 *   oklab,                 // Float32Array, stride-4 [L, a, b, alpha]
 *   palette,               // Float32Array, stride-3 [L, a, b]
 *   paletteCount,
 *   strength: 1.0,
 * });
 * ```
 */

import { BLUE_NOISE_64_SIZE, getBlueNoise64 } from "./blue-noise-data.ts";

export { BLUE_NOISE_64_SIZE, getBlueNoise64 } from "./blue-noise-data.ts";

/** Input record for {@link blueNoiseDither}. */
export interface BlueNoiseInput {
  /** Image width in pixels. */
  width: number;
  /** Image height in pixels. */
  height: number;
  /**
   * Stride-4 OkLab values `[L, a, b, alpha]` per pixel, row-major.
   * Produced by {@link imageToOklabF32} or an equivalent.
   */
  oklab: Float32Array;
  /** Stride-3 OkLab palette `[L, a, b]` per palette entry. */
  palette: Float32Array;
  /** Number of palette entries (i.e. `palette.length / 3`). */
  paletteCount: number;
  /**
   * Dither strength in `[0, 1]`. `0` collapses to nearest-palette
   * assignment (no jitter); `1` is the maximum perturbation amplitude
   * before chroma fringing becomes visible.
   */
  strength: number;
  /**
   * Optional per-pixel importance in `[0, 1]` (e.g. a DoG response).
   * Higher importance dampens the dither amplitude so detailed regions
   * keep their structure instead of being smeared by jitter.
   */
  importance?: Float32Array;
  /**
   * X-axis phase offset for the blue-noise mask, in absolute pixel
   * coordinates. Callers that dither a sub-region (e.g. the tiled CPU
   * path, which passes one tile at a time) should set this to the tile
   * origin so neighbouring tiles continue the same mask phase across
   * the seam. Defaults to `0` (mask aligned with the tile-local origin).
   */
  offsetX?: number;
  /** Y-axis phase offset; see {@link BlueNoiseInput.offsetX}. */
  offsetY?: number;
}

// We sample the 64x64 blue-noise mask three times — once per OkLab channel —
// at three large, mutually-different offsets so the L/a/b jitter is
// decorrelated. The chroma channels (a, b) get a smaller multiplier than L:
// hue jitter at palette boundaries reads as "rough colour edges" to the eye,
// while luminance jitter looks like ordinary stippling, which is what we want.
//
// stepSize is heuristic — roughly the average distance between palette
// entries along each axis. For a 256-color palette in unit OkLab, that's
// about 1/cbrt(256) ≈ 0.16, scaled by ditherStrength.
//
// Offsets within the 64x64 mask. Any odd dx/dy is coprime with 64, so the
// three samples come from cells that are far apart relative to the mask's
// spectral structure. The exact numbers below were picked to spread the three
// sample positions across the mask diagonally. L uses (0,0) implicitly —
// there's no offset to apply because L is the "anchor" sample everything
// else differs from. Internal-only; the WGSL shader has its own copies.
const NOISE_OFFSET_A = { dx: 17, dy: 23 };
const NOISE_OFFSET_B = { dx: 37, dy: 11 };

// Chroma channels get less noise than luminance so the dither reads as
// luminance stippling rather than colour fringing at boundaries.
const CHROMA_NOISE_SCALE = 0.6;

/**
 * Apply blue-noise threshold dither in OkLab space, returning a row-major
 * `Uint8Array` of palette indices (length = `width * height`, each byte
 * in `[0, paletteCount)`).
 *
 * Transparency contract (mirrors `floydSteinberg`, intentionally simpler):
 * `oklab` is stride-4 with the alpha lane present but unread inside the
 * inner loop. Transparent pixels (alpha = 0) still get a deterministic
 * palette index — they're just not observable in the output, so the
 * index value is harmless. Blue-noise is point-based and doesn't
 * propagate error, so there's no per-pixel "skip" needed.
 */
export function blueNoiseDither(input: BlueNoiseInput): Uint8Array {
  const { width, height, oklab, palette, paletteCount } = input;
  const strength = Math.max(0, Math.min(1, input.strength));
  const indices = new Uint8Array(width * height);
  const noise = getBlueNoise64();
  const noiseSize = BLUE_NOISE_64_SIZE;
  const stepSize = 0.18 / Math.cbrt(paletteCount) * strength;
  // Normalize the offset modulo noiseSize once so every pixel can use a
  // small positive addition; this keeps the inner-loop modulo a single
  // step regardless of how large the absolute tile origin is.
  const ox = ((input.offsetX ?? 0) % noiseSize + noiseSize) % noiseSize;
  const oy = ((input.offsetY ?? 0) % noiseSize + noiseSize) % noiseSize;

  // 3C-01 (3rd-pass review) + 4C-01 (4th-pass review): ay/ax max values are
  //   ay: (height-1) + (noiseSize-1) + max_dy = (height-1) + 63 + 23 = height + 85
  //   ax: (width-1)  + (noiseSize-1) + max_dx = (width-1)  + 63 + 37 = width  + 99
  // (max_dy = NOISE_OFFSET_A.dy = 23 acts on ay; max_dx = NOISE_OFFSET_B.dx = 37
  // acts on ax — keep the axes separate so a future noiseSize bump doesn't
  // accidentally apply the wrong worst-case offset.)
  // For any realistic image (width, height < 2^20) both values are comfortably
  // within Number's safe integer range. The `% noiseSize` inside the inner
  // loop then folds them back into [0, noiseSize). Modulo is distributive
  // over addition, so a future `(y + oy) % noiseSize` precompute optimisation
  // would still be correct.
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const ay = y + oy;
      const ax = x + ox;
      const niL = (ay % noiseSize) * noiseSize + (ax % noiseSize);
      const niA = ((ay + NOISE_OFFSET_A.dy) % noiseSize) * noiseSize +
        ((ax + NOISE_OFFSET_A.dx) % noiseSize);
      const niB = ((ay + NOISE_OFFSET_B.dy) % noiseSize) * noiseSize +
        ((ax + NOISE_OFFSET_B.dx) % noiseSize);
      const tL = (noise[niL] / 255) - 0.5; // [-0.5, 0.5)
      const tA = (noise[niA] / 255) - 0.5;
      const tB = (noise[niB] / 255) - 0.5;
      const scale = input.importance ? (1 - input.importance[i] * 0.5) : 1;
      const offsetL = tL * stepSize * scale;
      const offsetA = tA * stepSize * scale * CHROMA_NOISE_SCALE;
      const offsetB = tB * stepSize * scale * CHROMA_NOISE_SCALE;
      const L = oklab[i * 4] + offsetL;
      const a = oklab[i * 4 + 1] + offsetA;
      const b = oklab[i * 4 + 2] + offsetB;
      let best = 0;
      let bestDist = Infinity;
      for (let j = 0; j < paletteCount; j++) {
        const dL = L - palette[j * 3];
        const da = a - palette[j * 3 + 1];
        const db = b - palette[j * 3 + 2];
        const d = dL * dL + da * da + db * db;
        if (d < bestDist) {
          bestDist = d;
          best = j;
        }
      }
      indices[i] = best;
    }
  }
  return indices;
}
