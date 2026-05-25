// CPU reference implementation of weighted k-means in OkLab space.
// The same algorithm is mirrored in WGSL (palette/kmeans.wgsl.ts) for GPU use.
//
// Iteration steps per round:
//   1. Assignment: each sample → nearest centroid (squared L2 in OkLab).
//   2. Update: each centroid ← weighted mean of its assigned samples.
//
// Weights come from the importance map (and alpha) so high-detail pixels
// pull centroids toward them.

import type { WuPalette } from "./wu.ts";

export interface KMeansInput {
  oklab: Float32Array; // [L, a, b, alpha] per sample
  weights?: Float32Array; // one weight per sample
  initial: WuPalette;
  iterations: number;
  /**
   * Optional cancellation signal. Polled once per iter boundary (not in
   * the per-pixel inner loop). Aborting mid-iteration aborts at the
   * *next* iter's top, so latency is bounded by one iteration's cost.
   * A-e (5th-pass review).
   */
  signal?: AbortSignal;
}

export interface KMeansResult {
  centroids: Float32Array; // [L, a, b] per palette entry
  count: number;
  indices: Uint8Array;
}

/**
 * Single-pass nearest-centroid assignment in OkLab space.
 *
 * Extracted from the inner loop of `kmeansRefine` and the historical
 * `dither: "none"` branch of `quantizeTiled` so all three callers share
 * one definition of "nearest". Behavioural contract:
 *   - oklab is stride-4 ([L, a, b, alpha]); palette is stride-3 ([L, a, b]).
 *   - Transparent samples (alpha ≤ 0) map to index 0 (the convention used
 *     elsewhere — pixels with alpha=0 aren't observable anyway).
 *   - Distance is squared L2 (no sqrt). Cheap, monotonic with L2.
 */
export function assignNearestOklab(
  oklab: Float32Array,
  palette: Float32Array,
  paletteCount: number,
): Uint8Array {
  const n = oklab.length / 4;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const alpha = oklab[i * 4 + 3];
    if (alpha <= 0) {
      out[i] = 0;
      continue;
    }
    const L = oklab[i * 4];
    const a = oklab[i * 4 + 1];
    const b = oklab[i * 4 + 2];
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
    out[i] = best;
  }
  return out;
}

export function kmeansRefine(input: KMeansInput): KMeansResult {
  const n = input.oklab.length / 4;
  const k = input.initial.count;
  const centroids = new Float32Array(input.initial.oklab);
  // Declared with the wider Uint8Array (= Uint8Array<ArrayBufferLike>) so
  // the post-loop reassignment from `assignNearestOklab` (which returns
  // the same widened type) typechecks under TS 5.7's tightened generics.
  let indices: Uint8Array = new Uint8Array(n);

  for (let iter = 0; iter < input.iterations; iter++) {
    // Cancellation checkpoint — polled per iteration boundary. Adding
    // this to the per-pixel inner loop would be measurable on small
    // images so we accept "abort latency = one iteration" instead.
    input.signal?.throwIfAborted();
    // Assignment. Also track the worst-residual pixel (the one whose
    // assigned centroid is farthest, in OkLab squared distance) so the
    // dead-cluster re-seed below has a candidate ready without a
    // second pass.
    let worstDist = -Infinity;
    let worstIdx = -1;
    for (let i = 0; i < n; i++) {
      const alpha = input.oklab[i * 4 + 3];
      if (alpha <= 0) {
        indices[i] = 0;
        continue;
      }
      const L = input.oklab[i * 4];
      const a = input.oklab[i * 4 + 1];
      const b = input.oklab[i * 4 + 2];
      let best = 0;
      let bestDist = Infinity;
      for (let j = 0; j < k; j++) {
        const dL = L - centroids[j * 3];
        const da = a - centroids[j * 3 + 1];
        const db = b - centroids[j * 3 + 2];
        const d = dL * dL + da * da + db * db;
        if (d < bestDist) {
          bestDist = d;
          best = j;
        }
      }
      indices[i] = best;
      if (bestDist > worstDist) {
        worstDist = bestDist;
        worstIdx = i;
      }
    }

    // Update: accumulate weighted sums per cluster. Alpha is used as the
    // weight when `weights` is absent (matches Wu's convention).
    const sumL = new Float64Array(k);
    const sumA = new Float64Array(k);
    const sumB = new Float64Array(k);
    const sumW = new Float64Array(k);
    const weights = input.weights;
    for (let i = 0; i < n; i++) {
      const alpha = input.oklab[i * 4 + 3];
      if (alpha <= 0) continue;
      const w = weights ? weights[i] : alpha;
      if (w <= 0) continue;
      const c = indices[i];
      sumL[c] += w * input.oklab[i * 4];
      sumA[c] += w * input.oklab[i * 4 + 1];
      sumB[c] += w * input.oklab[i * 4 + 2];
      sumW[c] += w;
    }
    let moved = 0;
    for (let j = 0; j < k; j++) {
      // "Dead" clusters (sumW == 0 in this iter's assign) used to be
      // frozen at their previous centroid value (B-09), wasting a
      // palette slot for the rest of the run. Phase 3.1 (5th-pass):
      // re-seed them instead — see the loop below this one. The skip
      // here keeps the *live* centroid update untouched; dead clusters
      // are handled separately so the squared-displacement
      // accumulator `moved` doesn't include the discrete re-seed jump
      // (which would prevent convergence forever).
      if (sumW[j] <= 0) continue;
      const newL = sumL[j] / sumW[j];
      const newA = sumA[j] / sumW[j];
      const newB = sumB[j] / sumW[j];
      const dL = newL - centroids[j * 3];
      const da = newA - centroids[j * 3 + 1];
      const db = newB - centroids[j * 3 + 2];
      moved += dL * dL + da * da + db * db;
      centroids[j * 3] = newL;
      centroids[j * 3 + 1] = newA;
      centroids[j * 3 + 2] = newB;
    }

    // Dead-cluster re-seeding (Phase 3.1 / 5th-pass). When a cluster
    // ends an iteration with no assigned weight, its slot is wasted:
    // the post-loop `assignNearestOklab` will never pick it because
    // no pixel was closer to it than to *any* other centroid. Re-seed
    // those slots onto the highest-residual pixel found during the
    // assignment pass above — a pixel that's poorly served by the
    // current palette is the best candidate to anchor a fresh
    // cluster. The next iter's assignment will draw nearby pixels in
    // and the cluster comes back to life.
    //
    // Limit to one re-seed per iter: re-seeding multiple dead clusters
    // to the same worst pixel collapses the colours, and tracking the
    // top-N worst residuals would need a heap or sort that isn't
    // worth the complexity for N typically ≤ 5. Across `iterations`
    // iters, multiple dead clusters get a re-seed each.
    //
    // The re-seed counts as a discrete jump, not gradient descent, so:
    // (a) we don't add it to `moved`, and (b) we suppress the
    // convergence break this iter so the freshly-seeded cluster gets
    // at least one iter to attract pixels before we'd stop.
    let reseeded = false;
    if (worstIdx >= 0) {
      for (let j = 0; j < k; j++) {
        if (sumW[j] > 0) continue;
        centroids[j * 3] = input.oklab[worstIdx * 4];
        centroids[j * 3 + 1] = input.oklab[worstIdx * 4 + 1];
        centroids[j * 3 + 2] = input.oklab[worstIdx * 4 + 2];
        reseeded = true;
        break;
      }
    }

    // `moved` is sum of squared centroid displacements in OkLab units (L ∈ [0,1],
    // a/b ≈ [-0.4, 0.4]). 1e-7 corresponds to ~3e-4 RMS movement per centroid
    // — well below sRGB byte rounding (~1/255 ≈ 4e-3), so additional iterations
    // would only chase ULP noise. This is what lets the bench show k-means
    // settling in ~10 iters on most inputs.
    if (!reseeded && moved < 1e-7) break;
  }

  // Post-loop assign so the returned `indices` always match the returned
  // `centroids`. Without this, callers reading `indices` directly (the
  // dither="none" pipeline, palette-swap consumers) would see a stream
  // that lags the centroids by one update step — and for `iterations=0`
  // would see all-zero indices instead of nearest-to-Wu-init assignment.
  // Cost: one extra assign pass (typically 5-10 % of refine time), but
  // it's the only thing that gives `kmeansRefine` a clean leaf contract:
  // "indices is argmin over centroids". N-B-04 / W-A-4 in the 2nd-pass
  // review; the GPU mirror is in src/api/quantize-gpu.ts (gated to the
  // dither="none" branch there since blue-noise / FS overwrite indices
  // anyway, and the GPU dispatch is more expensive than the CPU loop).
  indices = assignNearestOklab(input.oklab, centroids, k);

  return { centroids, count: k, indices };
}
