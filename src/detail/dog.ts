// Multi-scale Difference-of-Gaussians importance map (CPU reference).
//
// The combine loop treats `blurred[0]` as the unblurred L channel itself
// (Gaussian with σ=0). Sequence: blurred = [L, G_1(L), G_2(L), G_4(L)].
// Detail at a pixel = Σ_k |blurred[k] - blurred[k+1]| across k = 0..2,
// summing three DoG residuals (raw-vs-σ1, σ1-vs-σ2, σ2-vs-σ4). After
// multiplying by gain and clamping to [0, 1] you get a normalised "is
// this pixel near an edge?" map.
//
// We only DoG the luminance channel (L) — chroma changes that aren't tied
// to luminance changes are not the kind of "detail" we want to preserve.

const SIGMAS = [1, 2, 4];

// Shared between the CPU DoG path here and the GPU DoG path in
// `src/api/quantize-gpu.ts`. Both pipelines must use bit-identical kernel
// values so the DoG importance map matches across CPU/GPU; keeping a single
// source of truth here makes that guarantee structural rather than
// per-formula. (4C-03, 4th-pass review.)
export function gaussianKernel1D(
  sigma: number,
): { kernel: Float32Array; radius: number } {
  const radius = Math.max(1, Math.ceil(3 * sigma));
  const k = new Float32Array(2 * radius + 1);
  const inv2s2 = 1 / (2 * sigma * sigma);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-i * i * inv2s2);
    k[i + radius] = v;
    sum += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return { kernel: k, radius };
}

// Robust mirror-without-repeat reflection for any |i|, any n.
//
// Previous version applied at most one fold (`if xx<0: -xx`, then
// `if xx>=n: 2n-2-xx`), which breaks when radius >= n: e.g. n=5, i=12 →
// single fold lands at -4, the read drops off the array (CPU returns
// `undefined` → NaN propagates; WGSL `u32(-4)` is undefined behaviour).
// This bit at width/height < 13 because σ=4 produces radius=12.
//
// Period of the mirrored signal is 2*(n-1); index modulo the period puts
// us in one full reflected cycle, and the final if-flip folds the upper
// half back into [0, n).
function mirrorIdx(i: number, n: number): number {
  if (n <= 1) return 0;
  const period = 2 * (n - 1);
  let m = ((i % period) + period) % period;
  if (m >= n) m = period - m;
  return m;
}

function blurSeparable(
  src: Float32Array,
  width: number,
  height: number,
  sigma: number,
): Float32Array {
  if (sigma <= 0) return new Float32Array(src);
  const { kernel: k, radius } = gaussianKernel1D(sigma);
  const tmp = new Float32Array(src.length);
  const dst = new Float32Array(src.length);

  // Horizontal pass.
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let s = 0;
      for (let t = -radius; t <= radius; t++) {
        const xx = mirrorIdx(x + t, width);
        s += src[y * width + xx] * k[t + radius];
      }
      tmp[y * width + x] = s;
    }
  }
  // Vertical pass.
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let s = 0;
      for (let t = -radius; t <= radius; t++) {
        const yy = mirrorIdx(y + t, height);
        s += tmp[yy * width + x] * k[t + radius];
      }
      dst[y * width + x] = s;
    }
  }
  return dst;
}

export interface DoGInput {
  width: number;
  height: number;
  oklab: Float32Array;
  gain?: number;
}

export function detailMap(input: DoGInput): Float32Array {
  const { width, height, oklab } = input;
  const gain = input.gain ?? 6;

  const L = new Float32Array(width * height);
  for (let i = 0; i < L.length; i++) L[i] = oklab[i * 4];

  const blurred: Float32Array[] = [L];
  for (const s of SIGMAS) blurred.push(blurSeparable(L, width, height, s));
  const detail = new Float32Array(width * height);
  for (let i = 0; i < detail.length; i++) {
    let d = 0;
    for (let k = 0; k < blurred.length - 1; k++) {
      d += Math.abs(blurred[k][i] - blurred[k + 1][i]);
    }
    detail[i] = Math.max(0, Math.min(1, d * gain));
  }
  return detail;
}
