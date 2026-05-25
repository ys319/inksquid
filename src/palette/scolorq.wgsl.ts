// Scolorq soft-assignment + accumulate pass (WGSL).
//
// GPU mirror of `src/palette/scolorq.ts`'s annealing-only soft k-means.
// Per sweep: this shader is dispatched once over the whole image, then
// `KMEANS_REDUCE_WGSL` (re-used unchanged from kmeans) divides the
// partial sums to produce new centroids. The host loop schedules the
// temperature decay between dispatches.
//
// Differences vs `KMEANS_ACCUMULATE_WGSL`:
//   1. No `indices` buffer is read or written. The soft probability
//      distribution over k centroids is computed on the fly per pixel
//      from the current centroids and temperature, then folded
//      directly into the workgroup-local accumulators. The final hard
//      `indices` (for PNG-8 output) come from a separate
//      KMEANS_ASSIGN dispatch after the annealing loop terminates.
//   2. A `Tinv` uniform (= 1 / T) sets the softmax sharpness for this
//      sweep. The host loop steps T geometrically across sweeps.
//   3. Each pixel contributes to ALL k centroids (weighted by its
//      softmax probability), not just one. The same atomic-add
//      accumulator works because every per-thread per-cell
//      contribution is `prob[c] * weight * SCALE`, bounded by
//      `1.0 * weight * SCALE` — the same per-cell upper bound as
//      hard k-means.
//
// Memory layout matches kmeans (and `KMEANS_REDUCE_WGSL` consumes the
// same `intermediate` shape): [num_workgroups][k][4: L,a,b,W].
//
// Implementation note on the centroid cache (the bug fix that brought
// this shader from 13 dB CPU↔GPU parity to passing): an earlier
// version stored per-pixel squared distances in a *private*
// `array<f32, 256>`, planning to reuse the storage across two passes
// (find minD, then compute exp/Z). That worked on paper but the
// 1 KB/thread × 256 threads/workgroup = 256 KB private memory
// footprint exceeded the Apple M-series register-file budget. The
// compiler silently spilled the array to slow scratch memory, and
// — more importantly — produced wrong centroids (PSNR 13 dB vs
// original on the parity test). Switching to a 4 KB *workgroup*-
// memory cache of the k centroids + on-the-fly recomputation of d
// (three loop passes through centroids: minD, Z, accumulate) brought
// the shader into agreement with the CPU reference. The cache stays
// fast because workgroup memory is on-chip and shared across all
// 256 threads in the workgroup.

import { KMEANS_LOCAL_SCALE, KMEANS_MAX_K, KMEANS_WG_SIZE } from "./kmeans.wgsl.ts";

const LOCAL_CELLS = KMEANS_MAX_K * 4;

export const SCOLORQ_ACCUMULATE_WGSL = /* wgsl */ `
struct Dims { n: u32, k: u32 };
struct Params { Tinv: f32 };

@group(0) @binding(0) var<storage, read> oklab: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> centroids: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> intermediate: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;
@group(0) @binding(5) var<uniform> params: Params;

const WG_SIZE: u32 = ${KMEANS_WG_SIZE}u;
const LOCAL_CELLS: u32 = ${LOCAL_CELLS}u;
const SCALE: f32 = ${KMEANS_LOCAL_SCALE}.0;
const MAX_K: u32 = ${KMEANS_MAX_K}u;

// Workgroup-local centroid cache. Loaded once per workgroup; reused
// by all 256 threads across the three centroid-loop passes below.
// Size: KMEANS_MAX_K * 16 = 4 KB workgroup memory (vs 4 KB for
// local_sums; total ~8 KB, well under any sane GPU's workgroup-
// memory limit).
var<workgroup> centroid_cache: array<vec4<f32>, ${KMEANS_MAX_K}>;
var<workgroup> local_sums: array<atomic<u32>, ${LOCAL_CELLS}>;

@compute @workgroup_size(${KMEANS_WG_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  // 1. Zero workgroup-local sums AND populate the centroid cache.
  // Both are cooperative ops across the workgroup; do them together
  // so we share one barrier afterward.
  for (var c: u32 = lid.x; c < LOCAL_CELLS; c = c + WG_SIZE) {
    atomicStore(&local_sums[c], 0u);
  }
  for (var c: u32 = lid.x; c < dims.k; c = c + WG_SIZE) {
    centroid_cache[c] = centroids[c];
  }
  workgroupBarrier();

  // 2. Per-pixel softmax + accumulate. Three loops through the cached
  // centroids, no per-thread scratch:
  //   (a) compute min squared-distance for numerical stability
  //   (b) compute Z = sum exp(-(d-minD) * Tinv)
  //   (c) atomic-add prob[c] * weight * (L, a+0.5, b+0.5) into
  //       local_sums[c]
  // Each pass re-reads centroid_cache[c] (4 KB workgroup memory =
  // single-cycle latency on Apple/Adreno/Mali; cheap).
  let i = gid.x;
  if (i < dims.n) {
    let p = oklab[i];
    if (p.w > 0.0) {
      let w = weights[i];
      if (w > 0.0) {
        // Pass (a): minD.
        var minD: f32 = 1e30;
        for (var c: u32 = 0u; c < dims.k; c = c + 1u) {
          let cc = centroid_cache[c];
          let dL = p.x - cc.x;
          let dA = p.y - cc.y;
          let dB = p.z - cc.z;
          let d = dL * dL + dA * dA + dB * dB;
          if (d < minD) { minD = d; }
        }
        // Pass (b): Z.
        var Z: f32 = 0.0;
        for (var c: u32 = 0u; c < dims.k; c = c + 1u) {
          let cc = centroid_cache[c];
          let dL = p.x - cc.x;
          let dA = p.y - cc.y;
          let dB = p.z - cc.z;
          let d = dL * dL + dA * dA + dB * dB;
          Z = Z + exp(-(d - minD) * params.Tinv);
        }
        // Pass (c): accumulate prob * weight * pixel into local sums.
        if (Z > 0.0) {
          let invZ_w = w / Z;
          let px_shifted_a = p.y + 0.5;
          let px_shifted_b = p.z + 0.5;
          for (var c: u32 = 0u; c < dims.k; c = c + 1u) {
            let cc = centroid_cache[c];
            let dL = p.x - cc.x;
            let dA = p.y - cc.y;
            let dB = p.z - cc.z;
            let d = dL * dL + dA * dA + dB * dB;
            let e = exp(-(d - minD) * params.Tinv);
            let pw = e * invZ_w; // prob[c] * w
            if (pw > 0.0) {
              let base = c * 4u;
              atomicAdd(&local_sums[base + 0u], u32(max(0.0, p.x) * pw * SCALE));
              atomicAdd(&local_sums[base + 1u], u32(max(0.0, px_shifted_a) * pw * SCALE));
              atomicAdd(&local_sums[base + 2u], u32(max(0.0, px_shifted_b) * pw * SCALE));
              atomicAdd(&local_sums[base + 3u], u32(pw * SCALE));
            }
          }
        }
      }
    }
  }
  workgroupBarrier();

  // 3. Non-atomic flush — identical layout / shift-undo logic to
  // KMEANS_ACCUMULATE_WGSL. KMEANS_REDUCE_WGSL consumes this buffer
  // unchanged.
  for (var c: u32 = lid.x; c < dims.k; c = c + WG_SIZE) {
    let base_local = c * 4u;
    let lw_fp = atomicLoad(&local_sums[base_local + 3u]);
    let lw = f32(lw_fp) / SCALE;
    let lsumL = f32(atomicLoad(&local_sums[base_local + 0u])) / SCALE;
    let lsumA = f32(atomicLoad(&local_sums[base_local + 1u])) / SCALE - 0.5 * lw;
    let lsumB = f32(atomicLoad(&local_sums[base_local + 2u])) / SCALE - 0.5 * lw;
    let dst = (wid.x * dims.k + c) * 4u;
    intermediate[dst + 0u] = lsumL;
    intermediate[dst + 1u] = lsumA;
    intermediate[dst + 2u] = lsumB;
    intermediate[dst + 3u] = lw;
  }
}
`;
