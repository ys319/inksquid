import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1.0.0";
import { assignNearestOklab, kmeansRefine } from "./kmeans.ts";
import { wuQuantizeFromSrgbU8 } from "./wu.ts";
import { imageToOklabF32 } from "../colorspace/oklab.ts";
import { rampBytes } from "../_test-fixtures.ts";

// 4E-005 (4th-pass review): this file previously had a local
// `gradient` with R/G varying and B fixed at 64. The shared
// `rampBytes` helper varies B as well — broad-band assertions
// (no NaN, indices in range, deterministic re-run) are insensitive
// to the exact B value, so the consolidation is safe.
const gradient = rampBytes;

Deno.test("kmeansRefine converges (no NaN, indices in range)", () => {
  const data = gradient(32, 32);
  const { palette } = wuQuantizeFromSrgbU8(data, 16);
  const oklab = imageToOklabF32(data);
  const out = kmeansRefine({ oklab, initial: palette, iterations: 10 });
  for (let i = 0; i < out.centroids.length; i++) {
    assert(Number.isFinite(out.centroids[i]), `centroid value ${i} is non-finite`);
  }
  for (let i = 0; i < out.indices.length; i++) {
    assert(out.indices[i] < palette.count, `index ${out.indices[i]} >= palette ${palette.count}`);
  }
});

Deno.test("kmeansRefine does not increase total distortion across iterations", () => {
  const data = gradient(48, 48);
  const { palette } = wuQuantizeFromSrgbU8(data, 8);
  const oklab = imageToOklabF32(data);
  function distortion(centroids: Float32Array, indices: Uint8Array): number {
    let sum = 0;
    for (let i = 0; i < indices.length; i++) {
      const c = indices[i];
      const dL = oklab[i * 4] - centroids[c * 3];
      const da = oklab[i * 4 + 1] - centroids[c * 3 + 1];
      const db = oklab[i * 4 + 2] - centroids[c * 3 + 2];
      sum += dL * dL + da * da + db * db;
    }
    return sum;
  }
  const first = kmeansRefine({ oklab, initial: palette, iterations: 1 });
  const second = kmeansRefine({ oklab, initial: palette, iterations: 10 });
  assert(
    distortion(second.centroids, second.indices) <=
      distortion(first.centroids, first.indices) + 1e-9,
  );
});

Deno.test("kmeansRefine: total distortion is monotonically non-increasing iter-by-iter", () => {
  // E-006: only the endpoints (iter=1 vs iter=10) were checked above, leaving
  // open a bug where iteration N increases distortion only to recover by N+1.
  // Run every iteration count from 1..10 and assert the sequence is monotone.
  const data = gradient(48, 48);
  const { palette } = wuQuantizeFromSrgbU8(data, 8);
  const oklab = imageToOklabF32(data);
  function distortion(centroids: Float32Array, indices: Uint8Array): number {
    let sum = 0;
    for (let i = 0; i < indices.length; i++) {
      const c = indices[i];
      const dL = oklab[i * 4] - centroids[c * 3];
      const da = oklab[i * 4 + 1] - centroids[c * 3 + 1];
      const db = oklab[i * 4 + 2] - centroids[c * 3 + 2];
      sum += dL * dL + da * da + db * db;
    }
    return sum;
  }
  let prev = Infinity;
  for (let iters = 1; iters <= 10; iters++) {
    const r = kmeansRefine({ oklab, initial: palette, iterations: iters });
    const d = distortion(r.centroids, r.indices);
    assert(
      d <= prev + 1e-7,
      `distortion increased: iters=${iters} d=${d.toExponential(3)} > prev=${
        prev.toExponential(3)
      }`,
    );
    prev = d;
  }
});

Deno.test("kmeansRefine determinism: identical input -> identical output", () => {
  const data = gradient(24, 24);
  const { palette } = wuQuantizeFromSrgbU8(data, 8);
  const oklab = imageToOklabF32(data);
  const a = kmeansRefine({ oklab, initial: palette, iterations: 10 });
  const b = kmeansRefine({ oklab, initial: palette, iterations: 10 });
  assertEquals(a.count, b.count);
  for (let i = 0; i < a.centroids.length; i++) {
    assertAlmostEquals(a.centroids[i], b.centroids[i], 1e-9);
  }
});

Deno.test("kmeansRefine: iterations=0 returns nearest-to-Wu-init (not all zero)", () => {
  // N-B-04 / W-A-4: before the post-loop assign was added, kmeansRefine
  // with iterations=0 returned `new Uint8Array(n)` (all zeros), which
  // collapsed every pixel onto palette[0]. The fix is to always run the
  // assign pass once at the end so iter=0 yields nearest-to-Wu-init.
  const data = gradient(32, 32);
  const { palette } = wuQuantizeFromSrgbU8(data, 8);
  const oklab = imageToOklabF32(data);
  const r = kmeansRefine({ oklab, initial: palette, iterations: 0 });
  // Centroids are unchanged from Wu init.
  for (let i = 0; i < palette.oklab.length; i++) {
    assertAlmostEquals(r.centroids[i], palette.oklab[i], 1e-9);
  }
  // Indices must show multiple distinct values — a gradient with k=8
  // cannot legitimately collapse to a single palette slot.
  const unique = new Set(r.indices);
  assert(unique.size > 1, `iter=0 indices collapsed to ${unique.size} slot(s)`);
  // And the indices must equal assignNearestOklab against the same palette.
  const expected = assignNearestOklab(oklab, r.centroids, r.count);
  assertEquals(r.indices, expected);
});

Deno.test("kmeansRefine: dead clusters get re-seeded onto worst-residual pixel (Phase 3.1)", () => {
  // Closes B-09 / B-1 on the CPU side. The previous behaviour was to
  // leave a dead cluster (sumW=0 after assignment) frozen at its
  // initial position forever — the post-loop `assignNearestOklab`
  // would then return indices that ignored that slot, effectively
  // wasting a palette entry for the rest of the call.
  //
  // Construct a pathological initial: two centroids at the same OkLab
  // location (0.4, 0, 0). Without re-seeding, all input pixels go to
  // centroid 0 by tie-break-on-equal-distance (lower-index wins) and
  // centroid 1 stays dead through every iter. With re-seeding,
  // centroid 1 gets the worst-residual pixel after iter 0 — the one
  // at L=1.0 — and the two centroids diverge to cover the input
  // properly.
  //
  // This case can't be triggered via Wu init (Wu returns at most
  // unique-after-binning entries, all in distinct locations), so the
  // test bypasses Wu and constructs `initial` directly.
  const oklab = new Float32Array([
    0.0,
    0,
    0,
    1,
    0.5,
    0,
    0,
    1,
    1.0,
    0,
    0,
    1,
  ]);
  const initial = {
    oklab: new Float32Array([0.4, 0, 0, 0.4, 0, 0]),
    count: 2,
  };
  const r = kmeansRefine({ oklab, initial, iterations: 5 });
  // The two centroids must end up distinct — re-seeding gave centroid
  // 1 a home at the worst-residual pixel (L=1.0) and the subsequent
  // iter assigned the L=0/L=0.5 pixels to centroid 0 (so its centroid
  // settled around 0.25). The exact values depend on Float32
  // rounding; pin a generous separation that the buggy (no re-seed)
  // path would never produce.
  const c0L = r.centroids[0];
  const c1L = r.centroids[3];
  assert(
    Math.abs(c0L - c1L) > 0.3,
    `centroids should diverge after re-seed, got c0=${c0L} c1=${c1L}`,
  );
  // And both should be picked by some pixel — i.e. neither stays
  // dead at the end.
  const unique = new Set(r.indices);
  assertEquals(unique.size, 2, `expected 2 distinct indices, got ${unique.size}`);
});

Deno.test("kmeansRefine: final indices match assignNearestOklab(final centroids)", () => {
  // W-A-4: the historical contract had `indices` lagging `centroids` by
  // one update step (the last iter's assign used the pre-update centroids).
  // Post-loop assign closes that off-by-one. This test pins the fix:
  // re-running assignNearestOklab on the returned centroids must produce
  // the exact same byte stream.
  const data = gradient(48, 48);
  const { palette } = wuQuantizeFromSrgbU8(data, 16);
  const oklab = imageToOklabF32(data);
  for (const iterations of [1, 5, 10]) {
    const r = kmeansRefine({ oklab, initial: palette, iterations });
    const expected = assignNearestOklab(oklab, r.centroids, r.count);
    assertEquals(
      r.indices,
      expected,
      `iter=${iterations}: returned indices don't match final centroids' nearest`,
    );
  }
});
