/**
 * @module
 *
 * OkLab colour space conversions — sRGB ↔ linear-RGB ↔ OkLab — plus the
 * byte-level helpers (`linearToSrgbU8`, `oklabToSrgbU8`, `imageToOklabF32`)
 * the rest of the pipeline uses.
 *
 * Forward and inverse matrices come from Björn Ottosson's reference
 * (https://bottosson.github.io/posts/oklab/). The sRGB ↔ linear transfer
 * uses the official IEC 61966-2-1 piecewise curve (threshold 0.04045), not
 * gamma 2.2 — the small piecewise difference matters in the dark region
 * where k-means assignment decisions live.
 *
 * Importable standalone via the `./oklab` sub-module entry point.
 *
 * @example
 * ```ts
 * import { oklabToSrgbU8, srgbU8ToOklab } from "@ys319/inksquid/oklab";
 *
 * const lab = srgbU8ToOklab(128, 200, 64);
 * const [r, g, b] = oklabToSrgbU8(lab);
 * ```
 */

/** sRGB → linear-RGB for a single channel value in `[0, 1]`. */
export function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** Linear-RGB → sRGB for a single channel value in `[0, 1]`. */
export function linearToSrgb(c: number): number {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
}

/** Triplet in OkLab space. */
export interface Oklab {
  /** Perceptual lightness in `[0, 1]`. */
  L: number;
  /** Green ↔ red opponent axis, ≈ `[-0.4, 0.4]`. */
  a: number;
  /** Blue ↔ yellow opponent axis, ≈ `[-0.4, 0.4]`. */
  b: number;
}

/** Triplet in linear-RGB space. Each channel is in `[0, 1]` for in-gamut colours. */
export interface LinearRgb {
  /** Red channel, `[0, 1]` in gamut. */
  r: number;
  /** Green channel, `[0, 1]` in gamut. */
  g: number;
  /** Blue channel, `[0, 1]` in gamut. */
  b: number;
}

/** Linear-RGB → OkLab using Ottosson's reference matrices. */
export function linearRgbToOklab(rgb: LinearRgb): Oklab {
  const l_ = 0.4122214708 * rgb.r + 0.5363325363 * rgb.g + 0.0514459929 * rgb.b;
  const m_ = 0.2119034982 * rgb.r + 0.6806995451 * rgb.g + 0.1073969566 * rgb.b;
  const s_ = 0.0883024619 * rgb.r + 0.2817188376 * rgb.g + 0.6299787005 * rgb.b;

  const l = Math.cbrt(l_);
  const m = Math.cbrt(m_);
  const s = Math.cbrt(s_);

  return {
    L: 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
    a: 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
    b: 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
  };
}

/**
 * OkLab → linear-RGB using Ottosson's reference matrices. The returned
 * triplet may fall outside `[0, 1]` for OkLab values that lie outside the
 * sRGB gamut; callers that need byte output should clamp through
 * {@link linearToSrgbU8} or {@link oklabToSrgbU8}.
 */
export function oklabToLinearRgb(lab: Oklab): LinearRgb {
  const l_ = lab.L + 0.3963377774 * lab.a + 0.2158037573 * lab.b;
  const m_ = lab.L - 0.1055613458 * lab.a - 0.0638541728 * lab.b;
  const s_ = lab.L - 0.0894841775 * lab.a - 1.2914855480 * lab.b;

  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;

  return {
    r: 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    g: -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    b: -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
  };
}

/** sRGB byte triple `(r, g, b)` (each in `[0, 255]`) → {@link Oklab}. */
export function srgbU8ToOklab(r: number, g: number, b: number): Oklab {
  return linearRgbToOklab({
    r: srgbToLinear(r / 255),
    g: srgbToLinear(g / 255),
    b: srgbToLinear(b / 255),
  });
}

/**
 * Convert a single linear-RGB channel value to an sRGB byte, applying
 * gamma + clamp + the standard CPU rounding policy. Use this instead of
 * inlining `Math.round(Math.max(0, Math.min(1, linearToSrgb(c))) * 255)`
 * so the rounding behaviour stays consistent across every CPU palette-
 * conversion site (B-4 consolidation).
 *
 * Math.round is round-half-up (toward +∞: `Math.round(0.5) === 1`,
 * `Math.round(-0.5) === 0`). The matching WGSL `round()` in
 * `oklab.wgsl.ts` uses round-half-to-even (banker's rounding:
 * `round(0.5) === 0`, `round(2.5) === 2`). For sRGB byte outputs from
 * in-gamut OkLab values, half-integer outputs are rare, but this is
 * one of the documented sub-byte sources of CPU/GPU palette divergence
 * absorbed by the parity test gradient case's 0.15 ΔE ceiling
 * (alongside f32 vs Float64 accumulation order and `pow(x,1/3)` vs
 * `Math.cbrt`). Keeping CPU on `Math.round` so standalone JS callers
 * see standard JS rounding semantics.
 */
export function linearToSrgbU8(c: number): number {
  return Math.round(Math.max(0, Math.min(1, linearToSrgb(c))) * 255);
}

/**
 * {@link Oklab} → sRGB byte triple `[r, g, b]`. Each channel is clamped
 * to `[0, 255]` via {@link linearToSrgbU8}, so out-of-gamut OkLab values
 * are projected onto the sRGB cube rather than producing negative bytes.
 */
export function oklabToSrgbU8(lab: Oklab): [number, number, number] {
  const lin = oklabToLinearRgb(lab);
  return [linearToSrgbU8(lin.r), linearToSrgbU8(lin.g), linearToSrgbU8(lin.b)];
}

/**
 * Convert an RGBA byte image into a per-pixel OkLab + alpha stride-4
 * Float32Array, layout: [L, a, b, alphaNorm] per pixel.
 *
 * `alphaNorm` is the original byte alpha divided by 255, so it lands in
 * [0, 1]. Downstream weighting (wu.ts, kmeans.ts, dither/*.ts) multiplies
 * by this alpha, which means fully-transparent pixels (`alphaNorm === 0`)
 * contribute zero weight to centroid placement and to the importance
 * map — they're effectively ignored. Partially-transparent pixels are
 * weighted proportionally. This is the contract the rest of the pipeline
 * expects; bypassing imageToOklabF32 (e.g. constructing your own
 * Float32Array) means you must reproduce the alpha normalisation
 * yourself or the weight calculations will all explode by 255×.
 */
export function imageToOklabF32(rgba: Uint8ClampedArray): Float32Array {
  const n = rgba.length / 4;
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const lab = srgbU8ToOklab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]);
    out[i * 4] = lab.L;
    out[i * 4 + 1] = lab.a;
    out[i * 4 + 2] = lab.b;
    out[i * 4 + 3] = rgba[i * 4 + 3] / 255;
  }
  return out;
}
