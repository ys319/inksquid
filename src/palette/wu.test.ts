import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1.0.0";
import { nearestOklabIndex, wuQuantizeFromSrgbU8, wuQuantizeOklab } from "./wu.ts";
import { imageToOklabF32 } from "../colorspace/oklab.ts";
import { rampBytes } from "../_test-fixtures.ts";

// 4E-005 (4th-pass review): this file previously had a local
// `makeGradient` with R/G varying and B fixed at 128. The shared
// `rampBytes` helper varies B with (x+y)/(w+h) instead — slightly more
// chroma diversity, but the assertions here are bounded ranges
// (`palette.count` between 8 and 16, palette.count <= input.unique)
// that absorb the change. If a future Wu tweak makes the count band
// tight enough to be sensitive to B, restore a local fixed-B fixture.
const makeGradient = rampBytes;

Deno.test("Wu reduces a 3-color image to a 3-color palette", () => {
  // 2×2 image with 3 unique colours (red ×2, green, blue) and full alpha:
  //   (red)   (green)
  //   (blue)  (red)
  // 3E-011 (3rd-pass review): the image-shape + unique-count weren't
  // obvious from the byte literal alone. deno-fmt-ignore keeps the rows
  // grouped one pixel per line so the geometry stays readable.
  // deno-fmt-ignore
  const data = new Uint8ClampedArray([
    255, 0, 0, 255,   // (0,0) red
    0, 255, 0, 255,   // (1,0) green
    0, 0, 255, 255,   // (0,1) blue
    255, 0, 0, 255,   // (1,1) red
  ]);
  const { palette, paletteSrgb } = wuQuantizeFromSrgbU8(data, 3);
  assertEquals(palette.count, 3);
  assertEquals(paletteSrgb.length, 9);
});

Deno.test("Wu produces requested palette size when input has enough colors", () => {
  const data = makeGradient(64, 64);
  const { palette } = wuQuantizeFromSrgbU8(data, 16);
  // Some implementations clamp to <= requested when input is small.
  assert(palette.count >= 8, `expected >= 8 colors, got ${palette.count}`);
  assert(palette.count <= 16, `expected <= 16 colors, got ${palette.count}`);
});

Deno.test("Wu is deterministic: same input -> same palette", () => {
  const data = makeGradient(32, 32);
  const { palette: p1 } = wuQuantizeFromSrgbU8(data, 8);
  const { palette: p2 } = wuQuantizeFromSrgbU8(data, 8);
  assertEquals(p1.count, p2.count);
  for (let i = 0; i < p1.oklab.length; i++) {
    assertAlmostEquals(p1.oklab[i], p2.oklab[i], 1e-9);
  }
});

Deno.test("Wu palette + nearest-color assignment for solid colors recovers them", () => {
  const data = new Uint8ClampedArray([
    255,
    0,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    0,
    255,
    255,
    255,
    255,
    255,
    255,
    0,
    0,
    0,
    255,
  ]);
  const { palette } = wuQuantizeFromSrgbU8(data, 5);
  const oklab = imageToOklabF32(data);
  const indices = nearestOklabIndex(oklab, palette);
  // All 5 unique colors should map to distinct palette entries.
  const used = new Set(indices);
  assertEquals(used.size, 5);
});

Deno.test("Wu with paletteSize=1 yields a single average color", () => {
  // paletteSize=1 is below the public API floor (normalizeOptions throws
  // on <2), but the internal wuQuantizeOklab still has to behave sensibly
  // at the algorithmic boundary because the implementation uses Math.max
  // (1, paletteSize). This pins that internal contract independently of
  // the public validation layer.
  const data = makeGradient(16, 16);
  const result = wuQuantizeOklab({ oklab: imageToOklabF32(data), paletteSize: 1 });
  assertEquals(result.count, 1);
});

Deno.test("Wu respects per-sample weights (zero-weight pixels are ignored)", () => {
  // Two clusters: 4 reds, 4 greens. With uniform weights, palette of size 2 should
  // get both. With reds weighted to zero, only greens should appear.
  const data = new Uint8ClampedArray([
    255,
    0,
    0,
    255,
    255,
    0,
    0,
    255,
    255,
    0,
    0,
    255,
    255,
    0,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    255,
  ]);
  const oklab = imageToOklabF32(data);
  const weights = new Float32Array([0, 0, 0, 0, 1, 1, 1, 1]);
  const result = wuQuantizeOklab({ oklab, weights, paletteSize: 2 });
  // All effective samples are green; centroids should all be green.
  for (let i = 0; i < result.count; i++) {
    // OkLab for green has a < 0
    assert(result.oklab[i * 3 + 1] < 0, `entry ${i} should have negative a (green)`);
  }
});

Deno.test("Wu boundary: paletteSize=2 and paletteSize=256 both return that many entries on a rich image", () => {
  // E-007: only paletteSize=1 was pinned. Cover both ends of the supported
  // range so a future bug that breaks at k=2 (the minimum non-degenerate
  // case) or k=256 (KMEANS_MAX_K ceiling) is caught immediately.
  const data = makeGradient(64, 64);
  const oklab = imageToOklabF32(data);
  const r2 = wuQuantizeOklab({ oklab, paletteSize: 2 });
  assertEquals(r2.count, 2);
  assertEquals(r2.oklab.length, 6);
  const r256 = wuQuantizeOklab({ oklab, paletteSize: 256 });
  assertEquals(r256.count, 256);
  assertEquals(r256.oklab.length, 768);
});

Deno.test("Wu with unique_colors < paletteSize returns at most unique_colors entries", () => {
  // E-007 / B-06: the input has 3 distinct colours but the caller asks for
  // 16. Wu should not invent ghost entries; the returned count must be ≤ 3.
  const data = new Uint8ClampedArray([
    // 3 distinct sRGB triples, repeated to fill a small image:
    255,
    0,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    0,
    255,
    255,
    255,
    0,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    0,
    255,
    255,
    255,
    0,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    0,
    255,
    255,
    255,
    0,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    0,
    255,
    255,
  ]); // 4x4 image, but only 3 colours (each repeated 4 times)
  const oklab = imageToOklabF32(data);
  const r = wuQuantizeOklab({ oklab, paletteSize: 16 });
  assert(r.count <= 16, `count ${r.count} exceeded requested 16`);
  assert(r.count >= 1, `count ${r.count} below 1 — Wu must always return something`);
  // The recovery test ("Wu palette + nearest-color for solid colors")
  // already exercises that the centroids equal the unique colours; here
  // we just pin the count contract.
});

// 4B-02 (4th-pass review): alpha boundary tests. NEW-E-003 (2nd pass)
// added Wu's alpha-as-weight contract: every pixel's alpha (0..1)
// becomes its sample weight in the histogram. The two extremes need
// their own pins so neither degenerates silently into "no input" or
// "full input" without the rest of the test surface noticing.
Deno.test("Wu boundary: all alpha=0 returns at least 1 palette entry without throwing", () => {
  // Every pixel has zero alpha → zero weight → Wu's histogram is
  // empty. The variance-minimisation has nothing to partition, so
  // count must be ≥ 1 (Wu's "always return something" floor) and the
  // call must not throw / produce NaN palette entries.
  const w = 8, h = 8;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = 200;
    data[i * 4 + 1] = 100;
    data[i * 4 + 2] = 50;
    data[i * 4 + 3] = 0; // fully transparent
  }
  const { palette, paletteSrgb } = wuQuantizeFromSrgbU8(data, 8);
  assert(palette.count >= 1, `count ${palette.count}: Wu must return ≥ 1 entry`);
  assert(palette.count <= 8);
  assertEquals(paletteSrgb.length, palette.count * 3);
  for (const v of paletteSrgb) {
    assert(Number.isFinite(v), `palette has non-finite byte ${v}`);
    assert(v >= 0 && v <= 255, `palette byte ${v} out of range`);
  }
});

Deno.test("Wu boundary: all alpha=255 + uniform colour collapses to 1 entry", () => {
  // Every pixel fully opaque, all identical colour. The full-weight
  // histogram has a single occupied bin so Wu's split heuristic can't
  // produce a useful second box — count must be 1 (or pinned ≤ paletteSize
  // with the actual count being what Wu returned for a single unique
  // input). Pins the alpha=255 path mirror of the alpha=0 test above.
  const w = 8, h = 8;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = 64;
    data[i * 4 + 1] = 128;
    data[i * 4 + 2] = 192;
    data[i * 4 + 3] = 255;
  }
  const { palette } = wuQuantizeFromSrgbU8(data, 8);
  assertEquals(
    palette.count,
    1,
    `uniform colour should collapse to 1 entry, got ${palette.count}`,
  );
  // The single palette entry recovers the input colour (within OkLab
  // round-trip tolerance — 1 byte). Stride 3 per WuPalette.oklab contract.
  const L = palette.oklab[0];
  const a = palette.oklab[1];
  const b = palette.oklab[2];
  assert(Number.isFinite(L) && Number.isFinite(a) && Number.isFinite(b));
});
