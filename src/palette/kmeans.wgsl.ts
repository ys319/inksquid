// Weighted k-means in OkLab space (WGSL).
//
// Three passes per iteration:
//   1. assign:     per-pixel → nearest centroid                          (writes indices[i])
//   2. accumulate: workgroup-local atomic-add into shared u32 cells,
//                  then non-atomic write of (sumL, sumA, sumB, sumW) per
//                  (workgroup, centroid) into a flat intermediate buffer
//   3. reduce:     per-centroid scan over the workgroup dimension of the
//                  intermediate buffer, divide sum/weight, write new centroid
//
// Rationale: this avoids global atomics entirely. Two earlier attempts
// (per-pixel f32 CAS, then workgroup-local + global f32 CAS) produced wrong
// centroids under high cluster-assignment contention — neighbouring pixels in
// smooth images all map to the same centroid, hammering one atomic slot.
// Tree reduction sidesteps the atomic correctness questions entirely.
//
// Memory cost: num_workgroups × k × 16 bytes for the intermediate buffer.
// For 512² / WG_SIZE=256 / k=64: 1024 × 64 × 16 = 1 MB. For 4K / k=256
// (assuming we don't tile): ~130 MB, well under M5's ≈4 GB max buffer size.

export const KMEANS_WG_SIZE = 256;
export const KMEANS_MAX_K = 256;
export const KMEANS_LOCAL_SCALE = 1_000_000;

const LOCAL_CELLS = KMEANS_MAX_K * 4;

// Compile-time-ish sanity check: the workgroup-local u32 accumulator must not
// overflow. Worst case is WG_SIZE threads all landing on the same centroid,
// each contributing max(p.{x,y+0.5,z+0.5}) * weight * SCALE. In-gamut sRGB
// gives L ∈ [0, 1] and a, b ∈ [-0.5, 0.5], so the shifted axis values are
// bounded by 1. Weight is alpha * (1 - dW + dW * (importance*4 + 0.25))
// where alpha, detailWeight (dW), importance ∈ [0, 1] after normalizeOptions,
// hitting 4.25 when alpha=dW=importance=1.
//
// NEW-B-04 (3rd-pass review): the L ≤ 1 / a,b ≥ -0.5 bound depends on the
// input arriving through `imageToOklabF32`, which only processes byte sRGB
// (in-gamut by construction). A future caller that constructs a Float32Array
// directly with values from a wider gamut (P3, Rec.2020, scRGB) could feed
// L > 1 or a/b outside [-0.5, 0.5], silently overflowing this accumulator
// and producing meaningless centroids. If that path becomes a thing, either
// clamp at the input boundary or bump the safety factor here.
//
// If you bump SCALE, WG_SIZE, or the weight formula, recompute and update
// the cell-max comment inside KMEANS_ACCUMULATE_WGSL below.
const _MAX_WEIGHT = 4.25;
const _CELL_MAX = KMEANS_WG_SIZE * _MAX_WEIGHT * KMEANS_LOCAL_SCALE;
if (_CELL_MAX >= 2 ** 32) {
  throw new Error(
    `kmeans accumulate may overflow u32: cell_max=${_CELL_MAX} >= 2^32. ` +
      `Lower KMEANS_LOCAL_SCALE or KMEANS_WG_SIZE, or cap weight in quantize-gpu.`,
  );
}

export const KMEANS_ASSIGN_WGSL = /* wgsl */ `
struct Dims { n: u32, k: u32 };

@group(0) @binding(0) var<storage, read> oklab: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> centroids: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(${KMEANS_WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.n) { return; }
  let p = oklab[i];
  if (p.w <= 0.0) {
    indices[i] = 0u;
    return;
  }
  var best: u32 = 0u;
  var best_d: f32 = 1e30;
  for (var j: u32 = 0u; j < dims.k; j = j + 1u) {
    let c = centroids[j];
    let d = (p.x - c.x) * (p.x - c.x)
          + (p.y - c.y) * (p.y - c.y)
          + (p.z - c.z) * (p.z - c.z);
    if (d < best_d) { best_d = d; best = j; }
  }
  indices[i] = best;
}
`;

// Accumulate pass writes `intermediate` of shape [num_workgroups][k][4] (f32).
// Layout: intermediate[wg * k * 4 + c * 4 + channel] where channel ∈ {0:L, 1:A, 2:B, 3:W}.
export const KMEANS_ACCUMULATE_WGSL = /* wgsl */ `
struct Dims { n: u32, k: u32 };

@group(0) @binding(0) var<storage, read> oklab: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> intermediate: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

const WG_SIZE: u32 = ${KMEANS_WG_SIZE}u;
const LOCAL_CELLS: u32 = ${LOCAL_CELLS}u;
const SCALE: f32 = ${KMEANS_LOCAL_SCALE}.0;

var<workgroup> local_sums: array<atomic<u32>, ${LOCAL_CELLS}>;

@compute @workgroup_size(${KMEANS_WG_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  // 1. Zero workgroup-local sums.
  for (var c: u32 = lid.x; c < LOCAL_CELLS; c = c + WG_SIZE) {
    atomicStore(&local_sums[c], 0u);
  }
  workgroupBarrier();

  // 2. Each thread accumulates its pixel into local sums.
  //    Local cell max (worst case: all WG_SIZE threads to one centroid,
  //    w=4.25, post-shift axis value=1.0):
  //      256 * 4.25 * 1.0 * 1e6 = 1.088e9 < 2^32 (~3.94x headroom).
  //    TS-side sanity check above re-derives this from KMEANS_WG_SIZE,
  //    KMEANS_LOCAL_SCALE, and the weight formula in quantize-gpu.ts.
  let i = gid.x;
  if (i < dims.n) {
    let p = oklab[i];
    if (p.w > 0.0) {
      let w = weights[i];
      if (w > 0.0) {
        let c = indices[i];
        let base = c * 4u;
        // L is in [0, 1]; a, b are in roughly [-0.4, 0.4]. Shift a/b by +0.5
        // so the fixed-point conversion to u32 never sees a negative value.
        atomicAdd(&local_sums[base + 0u], u32(max(0.0, p.x) * w * SCALE));
        atomicAdd(&local_sums[base + 1u], u32(max(0.0, p.y + 0.5) * w * SCALE));
        atomicAdd(&local_sums[base + 2u], u32(max(0.0, p.z + 0.5) * w * SCALE));
        atomicAdd(&local_sums[base + 3u], u32(w * SCALE));
      }
    }
  }
  workgroupBarrier();

  // 3. Non-atomic flush to intermediate[wg, c, 0..3]. Each (wg, c) is written
  //    by exactly one thread (the one whose lid.x == c mod WG_SIZE), so no
  //    race. Slots not touched by any pixel get zero (since we wrote 0 above).
  //
  //    The 0.5 * lw subtraction undoes the (p.y + 0.5) shift applied during
  //    accumulate. This relies on the shift never clamping to zero — i.e.
  //    a, b ≥ -0.5, which holds for any in-gamut sRGB input (imageToOklab
  //    on byte input is well inside this bound).
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

// Reduce pass: per-centroid linear scan over num_workgroups partial sums,
// then divide and write to centroids. No atomics needed.
//
// The `(w * dims.k + c) * 4u` index expression below is the read-side match
// for the accumulate pass's `(wid.x * dims.k + c) * 4u` write. Both index
// the same [num_workgroups][k][4 channels: L, a, b, W] layout — if one
// changes (e.g. swapping dimensions to [k][num_workgroups][4]), the other
// must too.
export const KMEANS_REDUCE_WGSL = /* wgsl */ `
struct Dims { k: u32, num_workgroups: u32 };

@group(0) @binding(0) var<storage, read> intermediate: array<f32>;
@group(0) @binding(1) var<storage, read_write> centroids: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= dims.k) { return; }
  var sL: f32 = 0.0;
  var sA: f32 = 0.0;
  var sB: f32 = 0.0;
  var sW: f32 = 0.0;
  for (var w: u32 = 0u; w < dims.num_workgroups; w = w + 1u) {
    let off = (w * dims.k + c) * 4u;
    sL = sL + intermediate[off + 0u];
    sA = sA + intermediate[off + 1u];
    sB = sB + intermediate[off + 2u];
    sW = sW + intermediate[off + 3u];
  }
  // Dead cluster: no pixels (with weight > 0) were assigned to centroid c
  // this iteration. Skip the update to keep the previous centroid frozen
  // instead of dividing by zero. The matching CPU branch is in
  // src/palette/kmeans.ts (B-09 in the 2026-05-23 review tracks the design
  // rationale). Dead clusters are harmless because the post-loop
  // KMEANS_ASSIGN dispatch in quantize-gpu.ts (gated to dither none)
  // re-runs the assign with this iter centroids: if no pixel picked
  // this centroid in the iter assign, none will pick it in the post-loop
  // assign either, so the FINAL indices never reference a dead cluster.
  // (Mid-iter behaviour is not pinned by this argument; a centroid can
  // flip dead-to-alive across iterations.)
  if (sW <= 0.0) { return; }
  centroids[c] = vec4<f32>(sL / sW, sA / sW, sB / sW, 0.0);
}
`;
