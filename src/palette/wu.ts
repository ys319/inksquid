// Wu's variance-minimization color quantization (Graphics Gems II, 1991),
// adapted to OkLab space.
//
// Pipeline:
//   1. Quantize each (L, a, b) sample to one of 32 bins per axis, then store
//      it into a 33×33×33 cumulative-moment grid (index 0 acts as the
//      "before any bin" sentinel so range queries become subtractions).
//   2. Build the cumulative moments so box sums become O(1).
//   3. Recursively split the box with maximum weighted variance until we
//      have `paletteSize` boxes. Each split picks the axis that maximizes
//      the sum of resulting sub-box variances.
//   4. The centroid of each final box is the palette entry.
//
// Sample weights (e.g. from the DoG importance map) are supported by passing
// per-sample weights; centroids and box variance use weighted sums.

import {
  linearRgbToOklab,
  linearToSrgbU8,
  type Oklab,
  oklabToLinearRgb,
  srgbToLinear,
} from "../colorspace/oklab.ts";
import { assignNearestOklab } from "./kmeans.ts";

const BINS = 32;
const SIZE = BINS + 1;
const VOX = SIZE * SIZE * SIZE;
const IDX = (l: number, a: number, b: number) => l * SIZE * SIZE + a * SIZE + b;

// OkLab axis bounds. L: [0,1]. a,b: roughly [-0.4, 0.4] for in-gamut sRGB.
const A_MIN = -0.4;
const A_MAX = 0.4;
const B_MIN = -0.4;
const B_MAX = 0.4;

function binL(L: number): number {
  return Math.min(BINS - 1, Math.max(0, Math.floor(L * BINS)));
}
function binA(a: number): number {
  return Math.min(BINS - 1, Math.max(0, Math.floor((a - A_MIN) / (A_MAX - A_MIN) * BINS)));
}
function binB(b: number): number {
  return Math.min(BINS - 1, Math.max(0, Math.floor((b - B_MIN) / (B_MAX - B_MIN) * BINS)));
}
function unbinL(bin: number): number {
  return (bin + 0.5) / BINS;
}
function unbinA(bin: number): number {
  return A_MIN + (bin + 0.5) / BINS * (A_MAX - A_MIN);
}
function unbinB(bin: number): number {
  return B_MIN + (bin + 0.5) / BINS * (B_MAX - B_MIN);
}

interface Box {
  l0: number;
  l1: number;
  a0: number;
  a1: number;
  b0: number;
  b1: number;
}

interface Moments {
  // wt: sum of weights
  // mL, mA, mB: weighted sums of L, a, b
  // m2: weighted sum of (L^2 + a^2 + b^2)
  wt: Float64Array;
  mL: Float64Array;
  mA: Float64Array;
  mB: Float64Array;
  m2: Float64Array;
}

export interface WuInput {
  oklab: Float32Array;
  weights?: Float32Array;
  paletteSize: number;
}

export interface WuPalette {
  /**
   * Palette in OkLab space, **stride 3** (not 4): `[L0, a0, b0, L1, a1, b1, ...]`,
   * length = `count * 3`. Note the contrast with the per-pixel format produced
   * by `imageToOklabF32` which is stride 4 ([L, a, b, alpha] per pixel).
   * `kmeansRefine`/`assignNearestOklab`/`blueNoiseDither`/`floydSteinberg`
   * all expect the stride-3 layout for palette inputs.
   */
  oklab: Float32Array;
  /** Number of palette entries actually returned. May be less than the requested
   * `paletteSize` when the input has fewer unique-after-binning colours. */
  count: number;
}

function buildMoments(input: WuInput): Moments {
  const wt = new Float64Array(VOX);
  const mL = new Float64Array(VOX);
  const mA = new Float64Array(VOX);
  const mB = new Float64Array(VOX);
  const m2 = new Float64Array(VOX);

  const n = input.oklab.length / 4;
  for (let i = 0; i < n; i++) {
    const alpha = input.oklab[i * 4 + 3];
    if (alpha <= 0) continue;
    const w = input.weights ? input.weights[i] : alpha;
    if (w <= 0) continue;
    const L = input.oklab[i * 4];
    const a = input.oklab[i * 4 + 1];
    const b = input.oklab[i * 4 + 2];
    const idx = IDX(binL(L) + 1, binA(a) + 1, binB(b) + 1);
    wt[idx] += w;
    mL[idx] += w * L;
    mA[idx] += w * a;
    mB[idx] += w * b;
    m2[idx] += w * (L * L + a * a + b * b);
  }

  // Cumulative moments via 3D prefix sums.
  const accumulate = (arr: Float64Array) => {
    for (let l = 1; l < SIZE; l++) {
      const area = new Float64Array(SIZE);
      for (let a = 1; a < SIZE; a++) {
        let line = 0;
        for (let b = 1; b < SIZE; b++) {
          line += arr[IDX(l, a, b)];
          area[b] += line;
          arr[IDX(l, a, b)] = arr[IDX(l - 1, a, b)] + area[b];
        }
      }
    }
  };
  accumulate(wt);
  accumulate(mL);
  accumulate(mA);
  accumulate(mB);
  accumulate(m2);

  return { wt, mL, mA, mB, m2 };
}

// Inclusion-exclusion box sum over moments.
function boxSum(m: Float64Array, box: Box): number {
  return (
    m[IDX(box.l1, box.a1, box.b1)] -
    m[IDX(box.l1, box.a1, box.b0)] -
    m[IDX(box.l1, box.a0, box.b1)] +
    m[IDX(box.l1, box.a0, box.b0)] -
    m[IDX(box.l0, box.a1, box.b1)] +
    m[IDX(box.l0, box.a1, box.b0)] +
    m[IDX(box.l0, box.a0, box.b1)] -
    m[IDX(box.l0, box.a0, box.b0)]
  );
}

function boxVariance(mom: Moments, box: Box): number {
  const w = boxSum(mom.wt, box);
  if (w <= 0) return 0;
  const sL = boxSum(mom.mL, box);
  const sA = boxSum(mom.mA, box);
  const sB = boxSum(mom.mB, box);
  const s2 = boxSum(mom.m2, box);
  return s2 - (sL * sL + sA * sA + sB * sB) / w;
}

// Returns the variance of the best split along the given axis, or -1 if no
// valid split exists. On success, mutates `outCut` to record the split index.
function maximize(
  mom: Moments,
  box: Box,
  axis: 0 | 1 | 2,
  outCut: { pos: number; varL: number; varR: number },
): number {
  const halfA: Box = { ...box };
  const halfB: Box = { ...box };
  let best = -1;
  outCut.pos = -1;
  const lo = axis === 0 ? box.l0 + 1 : axis === 1 ? box.a0 + 1 : box.b0 + 1;
  const hi = axis === 0 ? box.l1 : axis === 1 ? box.a1 : box.b1;
  for (let i = lo; i < hi; i++) {
    if (axis === 0) {
      halfA.l1 = i;
      halfB.l0 = i;
    } else if (axis === 1) {
      halfA.a1 = i;
      halfB.a0 = i;
    } else {
      halfA.b1 = i;
      halfB.b0 = i;
    }
    const wA = boxSum(mom.wt, halfA);
    const wB = boxSum(mom.wt, halfB);
    if (wA <= 0 || wB <= 0) continue;
    const vA = boxVariance(mom, halfA);
    const vB = boxVariance(mom, halfB);
    const sum = vA + vB;
    if (sum > best) {
      best = sum;
      outCut.pos = i;
      outCut.varL = vA;
      outCut.varR = vB;
    }
  }
  return best;
}

function volume(box: Box): number {
  return (box.l1 - box.l0) * (box.a1 - box.a0) * (box.b1 - box.b0);
}

// Try every axis and keep the best.
function cutBox(mom: Moments, box: Box): { axis: 0 | 1 | 2; pos: number } | null {
  if (volume(box) <= 1) return null;
  const cuts = [
    { pos: -1, varL: 0, varR: 0 },
    { pos: -1, varL: 0, varR: 0 },
    { pos: -1, varL: 0, varR: 0 },
  ];
  const scores = [
    maximize(mom, box, 0, cuts[0]),
    maximize(mom, box, 1, cuts[1]),
    maximize(mom, box, 2, cuts[2]),
  ];
  // Reuse total variance to convert "sum of sub-variances" into something we can compare:
  // a sub-split is preferred when its sub-variance sum is *smaller* (i.e. the box's
  // own variance dropped the most after the split). But Wu's original maximize-of-(va+vb)
  // is equivalent up to a constant, so we use the larger score directly.
  let bestAxis: 0 | 1 | 2 = 0;
  let bestScore = scores[0];
  for (let i = 1 as 0 | 1 | 2; i < 3; i = (i + 1) as 0 | 1 | 2) {
    if (scores[i] > bestScore) {
      bestScore = scores[i];
      bestAxis = i;
    }
  }
  if (cuts[bestAxis].pos < 0) return null;
  return { axis: bestAxis, pos: cuts[bestAxis].pos };
}

export function wuQuantizeOklab(input: WuInput): WuPalette {
  const k = Math.max(1, Math.min(256, input.paletteSize | 0));
  const mom = buildMoments(input);

  const boxes: Box[] = [];
  const variances: number[] = [];
  boxes.push({ l0: 0, l1: BINS, a0: 0, a1: BINS, b0: 0, b1: BINS });
  variances.push(boxVariance(mom, boxes[0]));

  while (boxes.length < k) {
    // Pick the box with maximum variance.
    let target = 0;
    for (let i = 1; i < boxes.length; i++) {
      if (variances[i] > variances[target]) target = i;
    }
    if (variances[target] <= 0) break;
    const cut = cutBox(mom, boxes[target]);
    if (!cut) {
      variances[target] = 0;
      continue;
    }
    const parent = boxes[target];
    const left: Box = { ...parent };
    const right: Box = { ...parent };
    if (cut.axis === 0) {
      left.l1 = cut.pos;
      right.l0 = cut.pos;
    } else if (cut.axis === 1) {
      left.a1 = cut.pos;
      right.a0 = cut.pos;
    } else {
      left.b1 = cut.pos;
      right.b0 = cut.pos;
    }
    boxes[target] = left;
    variances[target] = boxVariance(mom, left);
    boxes.push(right);
    variances.push(boxVariance(mom, right));
  }

  const oklab = new Float32Array(boxes.length * 3);
  for (let i = 0; i < boxes.length; i++) {
    const w = boxSum(mom.wt, boxes[i]);
    if (w <= 0) {
      // Use the box centre as a fallback so we still emit a defined color.
      //
      // Box convention (from boxSum/inclusion-exclusion above): `l0` is the
      // exclusive lower sentinel in 1-based shifted-bin space, `l1` is the
      // inclusive upper. So the box covers original bins [l0, l1-1] and the
      // centre original bin is floor((l0 + (l1-1)) / 2).
      //
      // The previous expression `((l0 + l1) >> 1) - 1` was off by one for
      // small boxes (e.g. a single-bin box l0=5, l1=6 produced bin 4, one
      // below the box's only valid bin). Fixed below; impact in practice is
      // limited to k > unique_colors edge cases that hit the empty-box
      // fallback at all.
      //
      // NEW-B-02 (3rd-pass review): the `(l0 + l1 - 1) >> 1` expression
      // floor-rounds, so for odd-width boxes the chosen centre sits one bin
      // toward the lower end of the box. This is a sub-bin bias on a
      // 33-bin axis (~3 % per axis = ~9 % cube volume off-centre), but it
      // only matters in the empty-box fallback — i.e. when the user asked
      // for more palette entries than the image has unique colours, where
      // the choice of centre vs slightly-off-centre is well within "best-
      // effort" anyway. Not worth a more accurate rounding scheme.
      const box = boxes[i];
      oklab[i * 3] = unbinL((box.l0 + box.l1 - 1) >> 1);
      oklab[i * 3 + 1] = unbinA((box.a0 + box.a1 - 1) >> 1);
      oklab[i * 3 + 2] = unbinB((box.b0 + box.b1 - 1) >> 1);
      continue;
    }
    oklab[i * 3] = boxSum(mom.mL, boxes[i]) / w;
    oklab[i * 3 + 1] = boxSum(mom.mA, boxes[i]) / w;
    oklab[i * 3 + 2] = boxSum(mom.mB, boxes[i]) / w;
  }
  return { oklab, count: boxes.length };
}

// Convert the OkLab palette to sRGB [0..255] palette bytes. The per-channel
// gamma + clamp + round policy lives in `linearToSrgbU8` (B-4 consolidation)
// so this and `oklabToSrgbU8` can't drift apart on the half-integer rounding
// asymmetry vs WGSL `round()` (banker's rounding).
export function oklabPaletteToSrgb(palette: WuPalette): Uint8Array {
  const out = new Uint8Array(palette.count * 3);
  for (let i = 0; i < palette.count; i++) {
    const lab: Oklab = {
      L: palette.oklab[i * 3],
      a: palette.oklab[i * 3 + 1],
      b: palette.oklab[i * 3 + 2],
    };
    const lin = oklabToLinearRgb(lab);
    out[i * 3] = linearToSrgbU8(lin.r);
    out[i * 3 + 1] = linearToSrgbU8(lin.g);
    out[i * 3 + 2] = linearToSrgbU8(lin.b);
  }
  return out;
}

// Helper for tests: takes RGBA u8 and runs the full Wu pipeline.
export function wuQuantizeFromSrgbU8(
  rgba: Uint8ClampedArray | Uint8Array,
  paletteSize: number,
): { palette: WuPalette; paletteSrgb: Uint8Array } {
  const n = rgba.length / 4;
  const oklab = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const r = srgbToLinear(rgba[i * 4] / 255);
    const g = srgbToLinear(rgba[i * 4 + 1] / 255);
    const b = srgbToLinear(rgba[i * 4 + 2] / 255);
    const lab = linearRgbToOklab({ r, g, b });
    oklab[i * 4] = lab.L;
    oklab[i * 4 + 1] = lab.a;
    oklab[i * 4 + 2] = lab.b;
    oklab[i * 4 + 3] = rgba[i * 4 + 3] / 255;
  }
  const palette = wuQuantizeOklab({ oklab, paletteSize });
  return { palette, paletteSrgb: oklabPaletteToSrgb(palette) };
}

// For tests: nearest-color assignment in OkLab. Thin wrapper around
// `assignNearestOklab` that unpacks the WuPalette shape.
//
// kmeans.ts only imports `WuPalette` as a type from this file, so the
// runtime import below doesn't create an evaluation-order cycle.
export function nearestOklabIndex(
  oklab: Float32Array,
  palette: WuPalette,
): Uint8Array {
  return assignNearestOklab(oklab, palette.oklab, palette.count);
}
