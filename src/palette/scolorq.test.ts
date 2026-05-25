// Tests for `scolorqQuantize` (Phase B.1 — soft k-means + annealing,
// no spatial filter yet). Coverage: determinism, abort, edge cases,
// invariant checks. The "+1 dB on Kodak" check lives in the bench,
// not here.

import { assert, assertAlmostEquals, assertEquals, assertRejects } from "jsr:@std/assert@^1.0.0";
import { scolorqQuantize } from "./scolorq.ts";
import { wuQuantizeOklab } from "./wu.ts";
import { imageToOklabF32 } from "../colorspace/oklab.ts";
import { flat, ramp } from "../_test-fixtures.ts";

function makeWeights(oklab: Float32Array): Float32Array {
  const n = oklab.length / 4;
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const alpha = oklab[i * 4 + 3];
    w[i] = alpha > 0 ? alpha : 0;
  }
  return w;
}

Deno.test("scolorqQuantize: determinism — identical input -> identical output", () => {
  const img = ramp(32, 32);
  const oklab = imageToOklabF32(img.data);
  const weights = makeWeights(oklab);
  const wu = wuQuantizeOklab({ oklab, weights, paletteSize: 16 });

  const a = scolorqQuantize({
    width: 32,
    height: 32,
    oklab,
    weights,
    paletteSize: 16,
    initialPalette: wu.oklab,
  });
  const b = scolorqQuantize({
    width: 32,
    height: 32,
    oklab,
    weights,
    paletteSize: 16,
    initialPalette: wu.oklab,
  });

  assertEquals(a.count, b.count);
  assertEquals(a.meta.sweepsRun, b.meta.sweepsRun);
  for (let i = 0; i < a.centroids.length; i++) {
    assertAlmostEquals(a.centroids[i], b.centroids[i], 1e-9);
  }
  assertEquals(a.indices, b.indices);
});

Deno.test("scolorqQuantize: centroids have no NaN, indices are in range", () => {
  const img = ramp(64, 64);
  const oklab = imageToOklabF32(img.data);
  const weights = makeWeights(oklab);

  const r = scolorqQuantize({
    width: 64,
    height: 64,
    oklab,
    weights,
    paletteSize: 32,
  });

  for (let i = 0; i < r.centroids.length; i++) {
    assert(Number.isFinite(r.centroids[i]), `centroid value ${i} non-finite`);
  }
  for (let i = 0; i < r.indices.length; i++) {
    assert(r.indices[i] < r.count, `index ${r.indices[i]} >= count ${r.count}`);
  }
});

Deno.test("scolorqQuantize: single-colour image -> no NaN, count >= 1", () => {
  const img = flat(16, 16);
  const oklab = imageToOklabF32(img.data);
  const weights = makeWeights(oklab);

  const r = scolorqQuantize({
    width: 16,
    height: 16,
    oklab,
    weights,
    paletteSize: 8,
  });

  assert(r.count >= 1);
  for (let i = 0; i < r.centroids.length; i++) {
    assert(Number.isFinite(r.centroids[i]));
  }
});

Deno.test("scolorqQuantize: pre-aborted signal throws before sweep 0", async () => {
  const img = ramp(16, 16);
  const oklab = imageToOklabF32(img.data);
  const weights = makeWeights(oklab);
  const ctrl = new AbortController();
  ctrl.abort();

  await assertRejects(
    () =>
      Promise.resolve().then(() =>
        scolorqQuantize({
          width: 16,
          height: 16,
          oklab,
          weights,
          paletteSize: 8,
          signal: ctrl.signal,
        })
      ),
  );
});

Deno.test("scolorqQuantize: meta.sweepsRun matches the requested iter count", () => {
  const img = ramp(16, 16);
  const oklab = imageToOklabF32(img.data);
  const weights = makeWeights(oklab);

  const r = scolorqQuantize({
    width: 16,
    height: 16,
    oklab,
    weights,
    paletteSize: 8,
    itersPerLevel: 5,
  });
  assertEquals(r.meta.sweepsRun, 5);
});

Deno.test("scolorqQuantize: indices match assignNearestOklab(final centroids)", async () => {
  // Mirrors the kmeansRefine invariant: returned indices == argmin
  // over returned centroids.
  const img = ramp(48, 48);
  const oklab = imageToOklabF32(img.data);
  const weights = makeWeights(oklab);

  const r = scolorqQuantize({
    width: 48,
    height: 48,
    oklab,
    weights,
    paletteSize: 16,
  });

  const { assignNearestOklab } = await import("../../src/palette/kmeans.ts");
  const expected = assignNearestOklab(oklab, r.centroids, r.count);
  assertEquals(r.indices, expected);
});
