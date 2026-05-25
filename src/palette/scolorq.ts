// Soft k-means with deterministic annealing in OkLab space —
// inspired by Puzicha 1998's spatial colour quantization but
// shipping WITHOUT the spatial-filter term that the original paper
// pairs with the annealing.
//
// Empirical behaviour on the 10-image Kodak suite at paletteSize=128
// after the T0 retune from 0.01 → 0.001 (mode-collapse fix, see the
// `initialTemp` field docs and the `bench/scolorq-t0-sweep.ts`
// telemetry that drove the retune):
//   - PSNR: 36.08 dB  (baseline blue-noise: 34.14 dB → +1.94 dB,
//                      pngquant: 36.80 → −0.72 dB)
//   - SSIM: 0.980     (baseline blue-noise: 0.931 → +0.049,
//                      pngquant: 0.981 → −0.001)
//   - time: ~5× slower than baseline (~5.3 sec/Kodak photo on M5)
//
// Both PSNR and SSIM now sit within a sub-dB / sub-SSIM-point of
// pngquant — the earlier "scolorq trades PSNR for SSIM" framing was
// an artefact of the over-large T0=0.01. The retune is a Pareto
// improvement (both metrics up, weird-colour bad-pixel rate dropped
// 10× across the Kodak suite). For users who want the best
// perceptual quality from the library (`dither: "scolorq"`),
// scolorq is now the recommended option on natural photos.
//
// Spatial filter status: the `spatialWeight` knob and the 3×3
// Gaussian-blur convolution code remain in place but default to 0
// (disabled). A Kodak-suite sweep showed monotonic PSNR/SSIM
// regression as λ rose, because the simple
// "target = (1-λ)·oklab + λ·filteredColor" formulation just
// smooths the assignment — it does NOT reproduce Puzicha's
// dither-emergence behaviour, which requires the full
// cross-covariance energy `e_c = w·|c|² + 2c·(p + b_middle·
// weighted_color)`. The plumbing is preserved so a future port of
// rscolorq's actual energy can re-use it.
//
// Algorithm (Phase B.1 / no spatial):
//
//   centroids ← Wu init
//   T ← T0
//   for sweep = 0..N-1:
//     for each pixel i:
//       d_c = ||oklab[i] - centroids[c]||²    for all c
//       prob_c = exp(-d_c / T) / sum_d exp(-d_d / T)   (numerically
//         stabilised by subtracting min(d) before the exp)
//       accumulate: palette_sum[c] += prob_c * weights[i] * oklab[i]
//                    palette_weight[c] += prob_c * weights[i]
//     centroids[c] = palette_sum[c] / palette_weight[c]
//     T *= α   where α = (Tf / T0)^(1/N)
//
//   indices ← assignNearestOklab(oklab, centroids, k)
//
// As T → 0 the softmax becomes argmax, recovering hard k-means. The
// annealing schedule lets the optimiser explore the energy landscape
// at high T (where many clusters compete for each pixel) and harden
// the choice as T drops — the deterministic-annealing rationale from
// Rose 1998.
//
// Avoiding mode-collapse: when T is too large for the OkLab distance
// scale, each centroid's update receives non-negligible cumulative
// weight from every distant pixel (even with exp(-(d−d_min)/T) per
// pixel small, summing across 100k+ pixels makes the tail dominate).
// Minority-colour centroids then drift toward the majority colour
// region — empirically visible as "weird-colour" artefacts: e.g.
// leaf-green centroids in an orange-bird photo end up rendering as
// yellow-green or neutral. The T0=0.001 default is in the sharp-
// softmax regime where this tail mass is suppressed.
//
// Memory: no full M matrix. We stream the soft assignments through
// the centroid accumulator each sweep, allocating only `Float32Array
// (k)` scratch for the distance and softmax vectors. Memory cost is
// O(k), not O(N · k) — matters for Kodak (N=393k, k=128 → 200 MB if
// stored densely).
//
// Determinism: Wu init is deterministic; the sweep order is row-
// major; there is no RNG; floating-point sums use Float64 accumulators
// so the order of additions is fixed. Re-running on the same input
// produces byte-identical output.

import { wuQuantizeOklab } from "./wu.ts";
import { assignNearestOklab } from "./kmeans.ts";

export interface ScolorqInput {
  width: number;
  height: number;
  /** Per-pixel OkLab, stride 4 (`[L, a, b, alpha]`). */
  oklab: Float32Array;
  /** Per-pixel perceptual weight (importance map × alpha). Length = W·H. */
  weights: Float32Array;
  paletteSize: number;
  /**
   * Optional seed palette. If absent, Wu is run internally to derive
   * one (the canonical caller — `quantize-cpu.ts` — has already
   * derived a Wu palette and passes it through, but `scolorqQuantize`
   * is self-sufficient).
   */
  initialPalette?: Float32Array;
  /**
   * Default 0.001 — calibrated to typical OkLab squared-distance
   * scale. OkLab L is in [0,1] and a,b are in roughly [-0.4, 0.4],
   * so typical squared distances between adjacent palette entries
   * are ~0.0005-0.005. At T0=0.001 the softmax exp(-(d−d_min)/T) is
   * sharply concentrated on each pixel's 2-3 nearest centroids —
   * sharp enough that distant centroids don't accumulate
   * mode-collapse-pull from majority-colour pixels (the
   * "weird colours" failure mode seen at T0=0.01 — Kodak suite
   * sweep showed minority-colour centroids drifting toward majority
   * regions, dropping 8-20 green centroids relative to BN). Lower
   * values (T0=0.0005) trade a touch of SSIM for slightly tighter
   * convergence; higher values (T0=0.005+) re-introduce the drift.
   * Cf. T0=1.0 from rscolorq's RGB default which collapses all
   * centroids toward the global weighted mean (PSNR −9.5 dB on
   * earlier OkLab-port attempts).
   */
  initialTemp?: number;
  /** Default 0.00001 — softmax effectively becomes argmax. */
  finalTemp?: number;
  /** Default 15 sweeps over the temperature range. */
  itersPerLevel?: number;
  /**
   * Spatial weight λ ∈ [0, 1] for the 3×3-Gaussian-blurred
   * filtered-colour term. The soft-assignment target is
   * `(1-λ) * oklab[i] + λ * filteredColor[i]`, where `filteredColor`
   * is the blur of last sweep's soft-assigned colours.
   *
   * **Default: 0** (i.e. spatial term disabled).
   *
   * Why: a Kodak-suite sweep (10 photos, λ ∈ {0, 0.05, 0.1, 0.2,
   * 0.3, 0.4}) showed monotonic regression in both PSNR and SSIM as
   * λ rose. The simple "target = blend" formulation does NOT
   * recover the dither-emergence behaviour described in Puzicha
   * 1998 — that requires the full cross-covariance energy
   * `e_c = w·|c|² + 2c·(p + b_middle·weighted_color)` which is a
   * meaningfully different objective. The simpler formulation
   * implemented here just smooths the assignment toward the
   * neighbourhood blur, which loses pixel detail without producing
   * useful dither patterns.
   *
   * The field and the spatial code path are kept so a future
   * port of rscolorq's actual energy can re-use the buffer
   * infrastructure. With the current code, leave at 0.
   */
  spatialWeight?: number;
  signal?: AbortSignal;
}

export interface ScolorqResult {
  /** OkLab palette, stride 3. */
  centroids: Float32Array;
  count: number;
  /** Final hard assignment via `assignNearestOklab` against `centroids`. */
  indices: Uint8Array;
  meta: {
    sweepsRun: number;
    /** Final temperature when the loop ended. */
    finalTemp: number;
  };
}

export function scolorqQuantize(input: ScolorqInput): ScolorqResult {
  input.signal?.throwIfAborted();
  const { width, height, oklab, weights, paletteSize: k } = input;
  const N = width * height;

  // 1. Seed palette: caller-provided (typical) or Wu (self-contained
  // path so this function is independently usable).
  let centroids: Float32Array;
  let count: number;
  if (input.initialPalette) {
    centroids = new Float32Array(input.initialPalette);
    count = centroids.length / 3;
  } else {
    const wu = wuQuantizeOklab({ oklab, weights, paletteSize: k });
    centroids = new Float32Array(wu.oklab);
    count = wu.count;
  }

  const T0 = input.initialTemp ?? 0.001;
  const Tf = input.finalTemp ?? 0.00001;
  const totalSweeps = input.itersPerLevel ?? 15;
  const lambda = input.spatialWeight ?? 0;
  // Geometric decay so we visit log-spaced temperatures: at sweep n,
  // T = T0 * α^n with α chosen so T_n=Tf after `totalSweeps` steps.
  const alpha = Math.pow(Tf / T0, 1 / Math.max(1, totalSweeps));

  // Scratch buffers reused every sweep.
  const dists = new Float32Array(count);
  const probs = new Float32Array(count);
  const accumLAB = new Float64Array(count * 3);
  const accumW = new Float64Array(count);

  // Spatial-filter buffers (Phase B.2). `assignedColor[i]` is the
  // soft-assigned colour at pixel i; updated at the END of every
  // sweep using the current sweep's softmax probabilities.
  // `filteredColor[i]` is the 3×3-Gaussian-blurred view of
  // assignedColor; recomputed at the START of every sweep and used
  // in that sweep's softmax target. The two buffers form a
  // Jacobi-style update (read previous, write next) so the spatial
  // term doesn't depend on intra-sweep update order — important for
  // determinism and future GPU portability.
  //
  // Initialised from the original OkLab image, which is the "what
  // each pixel should look like" target before any quantisation
  // pressure. The filter at sweep 0 is therefore a Gaussian blur of
  // the original image — a smoothness prior on the first
  // assignment, which the soft k-means then refines as the system
  // converges to the actual palette.
  const useSpatial = lambda > 0;
  let assignedColor: Float32Array | null = null;
  let filteredColor: Float32Array | null = null;
  if (useSpatial) {
    assignedColor = new Float32Array(N * 3);
    filteredColor = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      assignedColor[i * 3] = oklab[i * 4];
      assignedColor[i * 3 + 1] = oklab[i * 4 + 1];
      assignedColor[i * 3 + 2] = oklab[i * 4 + 2];
    }
  }

  let T = T0;
  let sweepsRun = 0;

  for (let sweep = 0; sweep < totalSweeps; sweep++) {
    input.signal?.throwIfAborted();
    sweepsRun++;

    // Phase B.2 spatial pass: 3×3 Gaussian blur of `assignedColor`
    // into `filteredColor`. Kernel weights are the standard
    // separable Gaussian [1,2,1]/4 × [1,2,1]/4 = [[1,2,1],[2,4,2],
    // [1,2,1]]/16. Edge pixels use a "duplicate-clamp" sampler
    // (sample (x, y) outside the image returns the nearest valid
    // pixel) so the filter doesn't darken edges.
    if (useSpatial) {
      const ac = assignedColor!;
      const fc = filteredColor!;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let sL = 0, sA = 0, sB = 0;
          let sumW2 = 0;
          for (let dy = -1; dy <= 1; dy++) {
            const sy = Math.max(0, Math.min(height - 1, y + dy));
            for (let dx = -1; dx <= 1; dx++) {
              const sx = Math.max(0, Math.min(width - 1, x + dx));
              // Gaussian weights: corners=1, edges=2, centre=4.
              const wk = (1 + (dx === 0 ? 1 : 0)) * (1 + (dy === 0 ? 1 : 0));
              const ni = (sy * width + sx) * 3;
              sL += wk * ac[ni];
              sA += wk * ac[ni + 1];
              sB += wk * ac[ni + 2];
              sumW2 += wk;
            }
          }
          const inv = 1 / sumW2;
          const oi = (y * width + x) * 3;
          fc[oi] = sL * inv;
          fc[oi + 1] = sA * inv;
          fc[oi + 2] = sB * inv;
        }
      }
    }

    accumLAB.fill(0);
    accumW.fill(0);

    for (let i = 0; i < N; i++) {
      const alphaPx = oklab[i * 4 + 3];
      if (alphaPx <= 0) continue;
      const w = weights[i];
      if (w <= 0) continue;

      // Target for soft assignment: blend of pixel colour with
      // filtered-neighbourhood colour. Reduces to plain pixel
      // colour when lambda=0 (B.1 behaviour). The maths: minimising
      //   (1-λ)|oklab[i] - c|² + λ|filteredColor[i] - c|²
      // over c is equivalent to minimising |target - c|² where
      //   target = (1-λ) * oklab[i] + λ * filteredColor[i].
      let L = oklab[i * 4];
      let a = oklab[i * 4 + 1];
      let b = oklab[i * 4 + 2];
      if (useSpatial) {
        const fi = i * 3;
        const fL = filteredColor![fi];
        const fA = filteredColor![fi + 1];
        const fB = filteredColor![fi + 2];
        L = (1 - lambda) * L + lambda * fL;
        a = (1 - lambda) * a + lambda * fA;
        b = (1 - lambda) * b + lambda * fB;
      }

      // Squared OkLab distance to every centroid, tracking the min
      // for numerical stability before the exp.
      let minD = Infinity;
      for (let j = 0; j < count; j++) {
        const dL = L - centroids[j * 3];
        const da = a - centroids[j * 3 + 1];
        const db = b - centroids[j * 3 + 2];
        const d = dL * dL + da * da + db * db;
        dists[j] = d;
        if (d < minD) minD = d;
      }

      // Numerically-stable softmax: exp(-(d - minD) / T). Subtracting
      // minD ensures the largest exp(...) is 1.0; without this, at
      // small T (≤0.01) every term underflows to 0 and Z is 0.
      let Z = 0;
      for (let j = 0; j < count; j++) {
        const p = Math.exp(-(dists[j] - minD) / T);
        probs[j] = p;
        Z += p;
      }
      if (Z <= 0) continue; // pathological — skip pixel rather than divide by 0

      // Update `assignedColor` for next sweep's spatial pass:
      // soft-assigned colour = sum_c prob[c] * centroid[c].
      // Accumulate centroid contributions (palette update) using the
      // *original* pixel colour (oklab[i*4..]) not the blended
      // target — the palette should match real pixels, not the
      // spatial blur. The blend only affects assignment, not what
      // the centroid converges to.
      const invZ = 1 / Z;
      const origL = oklab[i * 4];
      const origA = oklab[i * 4 + 1];
      const origB = oklab[i * 4 + 2];
      let acL = 0, acA = 0, acB = 0;
      for (let j = 0; j < count; j++) {
        const p = probs[j] * invZ;
        const pw = p * w;
        accumLAB[j * 3] += pw * origL;
        accumLAB[j * 3 + 1] += pw * origA;
        accumLAB[j * 3 + 2] += pw * origB;
        accumW[j] += pw;
        if (useSpatial) {
          acL += p * centroids[j * 3];
          acA += p * centroids[j * 3 + 1];
          acB += p * centroids[j * 3 + 2];
        }
      }
      if (useSpatial) {
        const ai = i * 3;
        assignedColor![ai] = acL;
        assignedColor![ai + 1] = acA;
        assignedColor![ai + 2] = acB;
      }
    }

    // Closed-form palette update: centroid ← weighted mean of pixel
    // contributions weighted by soft probabilities. Mirrors the
    // hard-k-means update with probs in place of an indicator.
    // Dead clusters (accumW=0) keep their old centroid — they
    // typically come back to life at lower T as their basin
    // sharpens. (If we end the run with persistent dead clusters,
    // the final hard-assign just won't pick them; no NaN.)
    for (let j = 0; j < count; j++) {
      if (accumW[j] > 0) {
        const inv = 1 / accumW[j];
        centroids[j * 3] = accumLAB[j * 3] * inv;
        centroids[j * 3 + 1] = accumLAB[j * 3 + 1] * inv;
        centroids[j * 3 + 2] = accumLAB[j * 3 + 2] * inv;
      }
    }

    T *= alpha;
  }

  // Final hard assignment so the returned `indices` align byte-for-
  // byte with `centroids` (the same invariant `kmeansRefine`
  // maintains via its post-loop assign).
  const indices = assignNearestOklab(oklab, centroids, count);

  return {
    centroids,
    count,
    indices,
    meta: { sweepsRun, finalTemp: T },
  };
}
