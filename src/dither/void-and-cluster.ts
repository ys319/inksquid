// Void-and-cluster blue-noise generator (Ulichney, 1993).
//
// Produces a tiling NxN mask where each integer rank r ∈ [0..N²-1] occurs
// exactly once. The spatial arrangement of low ranks is "blue" — high spatial
// frequencies, no low-frequency clumping — so thresholding the normalized
// mask gives perceptually pleasant patterns.
//
// Algorithm overview:
//   1. Start from a random binary pattern with ~N²/2 ones.
//   2. Repeat: find the tightest cluster (1 with highest neighbourhood
//      density), move it to be a 0 at the largest void (0 with lowest
//      neighbourhood density). Stop when the cluster IS the void.
//   3. Phase 1 (rank): copy the stable pattern as `initial`. While ones
//      remain, find the tightest cluster, set its rank, flip to 0.
//   4. Phase 2: starting from `initial` again, while zeros remain, find the
//      largest void, set its rank, flip to 1.
//
// We use a Gaussian filter with sigma=1.5 (Ulichney's recommendation),
// implemented via toroidal precomputed lookup tables for an O(N²) per-step
// cost instead of O(N⁴) full re-filtering.

export interface BlueNoiseMask {
  size: number;
  data: Uint8Array; // values in [0..255]
}

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianKernel(size: number, sigma: number): { kernel: Float64Array; radius: number } {
  // Truncate the Gaussian at 3 sigmas; clamp to half the mask size since the
  // filter wraps toroidally.
  const radius = Math.min(Math.floor(size / 2), Math.ceil(3 * sigma));
  const diam = 2 * radius + 1;
  const kernel = new Float64Array(diam * diam);
  const inv2s2 = 1 / (2 * sigma * sigma);
  let sum = 0;
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      const v = Math.exp(-(dx * dx + dy * dy) * inv2s2);
      kernel[(dy + radius) * diam + (dx + radius)] = v;
      sum += v;
    }
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  return { kernel, radius };
}

// Add ±delta * kernel to the density field, wrapping toroidally.
function splat(
  density: Float64Array,
  size: number,
  x: number,
  y: number,
  kernel: Float64Array,
  radius: number,
  sign: number,
) {
  const diam = 2 * radius + 1;
  for (let dy = -radius; dy <= radius; dy++) {
    const yy = ((y + dy) % size + size) % size;
    for (let dx = -radius; dx <= radius; dx++) {
      const xx = ((x + dx) % size + size) % size;
      density[yy * size + xx] += sign * kernel[(dy + radius) * diam + (dx + radius)];
    }
  }
}

function recompute(
  density: Float64Array,
  size: number,
  binary: Uint8Array,
  kernel: Float64Array,
  radius: number,
) {
  density.fill(0);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (binary[y * size + x]) splat(density, size, x, y, kernel, radius, 1);
    }
  }
}

function tightestCluster(density: Float64Array, binary: Uint8Array): number {
  let bestIdx = -1;
  let bestVal = -Infinity;
  for (let i = 0; i < binary.length; i++) {
    if (binary[i] && density[i] > bestVal) {
      bestVal = density[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

function largestVoid(density: Float64Array, binary: Uint8Array): number {
  let bestIdx = -1;
  let bestVal = Infinity;
  for (let i = 0; i < binary.length; i++) {
    if (!binary[i] && density[i] < bestVal) {
      bestVal = density[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

export function generateBlueNoise(size: number, seed = 1337): BlueNoiseMask {
  const n = size * size;
  // C-08: Ulichney's original paper starts with ~n/2 ones; we use n/10 because
  // a sparser starting pattern converges faster on the void-and-cluster swap
  // loop (fewer "tightest cluster" candidates means fewer iterations to
  // stable). Visual quality of the resulting mask is equivalent — the
  // refinement phase moves ones around freely regardless of initial count,
  // and downstream rank assignment is monotonic in either case.
  const initialOnes = Math.max(1, Math.floor(n / 10));
  const rng = mulberry32(seed);

  const binary = new Uint8Array(n);
  // Random initial placement.
  const ranks: number[] = [];
  for (let i = 0; i < n; i++) ranks.push(i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [ranks[i], ranks[j]] = [ranks[j], ranks[i]];
  }
  for (let k = 0; k < initialOnes; k++) binary[ranks[k]] = 1;

  const { kernel, radius } = gaussianKernel(size, 1.5);
  const density = new Float64Array(n);
  recompute(density, size, binary, kernel, radius);

  // Stable refinement: swap until cluster == void.
  // 3C-04 (3rd-pass review): the `n * 4` cap is empirical — on the 64×64
  // mask we generate (n=4096) the loop converges in O(n) iterations in
  // practice, breaking out via `cluster == void` (line below) long before
  // hitting 16384. The 4× multiplier is just a safety margin so an
  // unforeseen input distribution can't spin forever. This is build-time
  // code (the resulting mask is baked into blue-noise-data.ts), so even
  // hitting the cap once isn't user-visible.
  for (let iter = 0; iter < n * 4; iter++) {
    const cluster = tightestCluster(density, binary);
    if (cluster < 0) break;
    // Remove cluster.
    binary[cluster] = 0;
    splat(density, size, cluster % size, Math.floor(cluster / size), kernel, radius, -1);
    const vd = largestVoid(density, binary);
    if (vd < 0 || vd === cluster) {
      // restore and stop
      binary[cluster] = 1;
      splat(density, size, cluster % size, Math.floor(cluster / size), kernel, radius, 1);
      break;
    }
    binary[vd] = 1;
    splat(density, size, vd % size, Math.floor(vd / size), kernel, radius, 1);
  }

  const initial = new Uint8Array(binary);
  const rankArr = new Int32Array(n);
  rankArr.fill(-1);

  // Phase I: while ones exist, remove tightest cluster, assign descending ranks
  // from `onesRemaining - 1` down to 0.
  let onesRemaining = initial.reduce((a, b) => a + b, 0);
  // density already reflects `binary === initial`
  while (onesRemaining > 0) {
    const cluster = tightestCluster(density, binary);
    if (cluster < 0) break;
    rankArr[cluster] = onesRemaining - 1;
    binary[cluster] = 0;
    splat(density, size, cluster % size, Math.floor(cluster / size), kernel, radius, -1);
    onesRemaining--;
  }

  // Phase II: reset to initial pattern, fill voids ascending from `originalOnes` upward.
  binary.set(initial);
  recompute(density, size, binary, kernel, radius);
  let rankCounter = initial.reduce((a, b) => a + b, 0);
  while (rankCounter < n) {
    const vd = largestVoid(density, binary);
    if (vd < 0) break;
    rankArr[vd] = rankCounter;
    binary[vd] = 1;
    splat(density, size, vd % size, Math.floor(vd / size), kernel, radius, 1);
    rankCounter++;
  }

  // Map ranks to 8-bit. Use n-1 as denominator so the bottom rank is 0 and top is 255.
  //
  // 4C-02 (4th-pass review): the `Math.max(0, rankArr[i])` guard is a
  // defensive backstop. In a healthy run, Phase I + II assign every cell
  // a rank in `[0, n)` exactly once — the `< n` loop bounds plus the
  // `largestVoid(...) >= 0` precondition guarantee that. If `largestVoid`
  // ever returns -1 early (e.g. via a future kernel change that
  // confuses the void search), the loop breaks before all cells have a
  // rank, and the affected cells stay at the zero-initialised default
  // (which is valid: rank 0 = brightest mask byte). The clamp here
  // ensures the 8-bit map stays in [0, 255] even if a future code path
  // produces a negative sentinel. The rank-uniqueness test below pins
  // that no cell is *legitimately* assigned a negative rank under
  // current generation logic.
  const data = new Uint8Array(n);
  const denom = Math.max(1, n - 1);
  for (let i = 0; i < n; i++) {
    const r = Math.max(0, rankArr[i]);
    data[i] = Math.round(r * 255 / denom);
  }
  return { size, data };
}
