import { assert, assertEquals } from "jsr:@std/assert@^1.0.0";
import { _resetTiledFsDowngradeWarning, quantizeTiled, TILE } from "./quantize-tiled.ts";
import { quantizeCpu } from "../mod.ts";
import { ramp } from "../_test-fixtures.ts";

Deno.test("tiled path produces a valid PNG and preview at the expected dimensions", async () => {
  const img = ramp(2048, 1024);
  const r = await quantizeTiled(img, { paletteSize: 32 });
  assertEquals(r.preview.width, 2048);
  assertEquals(r.preview.height, 1024);
  // PNG signature.
  const sig = [0x89, 0x50, 0x4e, 0x47];
  for (let i = 0; i < sig.length; i++) assertEquals(r.png[i], sig[i]);
});

Deno.test("tiled output is broadly similar to single-pass on a smaller image", async () => {
  const img = ramp(256, 256);
  const a = await quantizeCpu(img, { paletteSize: 16, dither: "none" });
  const b = await quantizeTiled(img, { paletteSize: 16, dither: "none" });
  // Palette size should be close (Wu+k-means converges from the same input on
  // the subsampled image; for a smooth ramp it converges to similar centroids).
  assert(Math.abs(a.meta.paletteSize - b.meta.paletteSize) <= 2);
  // Both should be the same dimensions.
  assertEquals(a.preview.width, b.preview.width);
  assertEquals(a.preview.height, b.preview.height);
});

Deno.test("tiled output crosses tile boundary cleanly (no row of step artefacts)", async () => {
  // E-008: the tiled path used to dither each tile in isolation, so the
  // 64×64 blue-noise mask phase reset at every TILE-sized seam. With the
  // offsetX/Y plumbing added in C-A-9b that's no longer possible, but the
  // regression test wasn't there. Pin it.
  //
  // Construct an image taller than a single tile and look at the row just
  // below the seam vs the row just above. The number of palette-index
  // changes between adjacent pixels should not spike at the seam — if it
  // does (e.g. mask phase reset), the difference rate jumps measurably.
  // Heights are expressed relative to `TILE` (imported above) so that if
  // the tile strip size ever changes, the probe rows follow without a
  // silent pass (NEW-E-009 in the 2026-05-23 2nd-pass review).
  const W = TILE + 256;
  const H = TILE + 176; // > 1 tile in height, < 2 tiles
  assert(H > TILE && H < 2 * TILE, "test setup invariant broken: need 1 < H/TILE < 2");
  const img = ramp(W, H);
  const r = await quantizeTiled(img, { paletteSize: 32, dither: "blue-noise" });

  function rowChangeRate(y: number): number {
    let changes = 0;
    for (let x = 1; x < W; x++) {
      if (r.indices[y * W + x] !== r.indices[y * W + x - 1]) changes++;
    }
    return changes / (W - 1);
  }
  // Sample a band of rows above and below the y=TILE seam. If the seam
  // breaks the mask phase, you'd expect rows just below to have a
  // distinctly different change rate from the bulk.
  const aboveRows = [TILE - 6, TILE - 4, TILE - 2];
  const seamRows = [TILE, TILE + 1, TILE + 2];
  const aboveMean = aboveRows.reduce((s, y) => s + rowChangeRate(y), 0) / aboveRows.length;
  const seamMean = seamRows.reduce((s, y) => s + rowChangeRate(y), 0) / seamRows.length;
  // Tolerate ±15 % swing — natural variation from the underlying ramp,
  // not from the mask phase. A regression that resets the mask would
  // typically push this well past 30%.
  const ratio = Math.abs(seamMean - aboveMean) / Math.max(1e-9, aboveMean);
  assert(
    ratio < 0.15,
    `tile-seam row change rate diverges ${(ratio * 100).toFixed(1)}% from interior ` +
      `(above=${aboveMean.toFixed(3)}, seam=${seamMean.toFixed(3)})`,
  );
});

Deno.test('tiled exposes meta.pipeline === "cpu-tiled"', async () => {
  const img = ramp(2048, 1024);
  const r = await quantizeTiled(img, { paletteSize: 16 });
  assertEquals(r.meta.pipeline, "cpu-tiled");
});

Deno.test("tiled downgrades floyd-steinberg to blue-noise with a one-shot warning", async () => {
  // FS can't run tile-by-tile without error-row carryover, so quantizeTiled
  // routes it to blue-noise and warns once. Pin both behaviours.
  //
  // 3E-002 (3rd-pass review continued): the byte-equality assertion below
  // relies on the implicit invariant that the FS→BN downgrade computes
  // `tileImportance` identically to a direct BN call. If quantize-tiled
  // ever branches the importance-map construction by `dither`, this
  // equality will silently break in unrelated ways. The cross-reference
  // lives in quantize-tiled.ts (search for "downgrade").
  //
  // 3E-005 (3rd-pass review continued): the `console.warn` mock + the
  // module-scoped `_warnedAboutFsDowngrade` flag (reset via
  // `_resetTiledFsDowngradeWarning`) are *not* safe under
  // `deno test --parallel`. The Deno test runner is serial by default
  // and this suite relies on that; any future move to --parallel must
  // either gate this test or move the flag out of module scope.
  _resetTiledFsDowngradeWarning();
  const originalWarn = console.warn;
  const warnings: string[] = [];
  console.warn = (...args: unknown[]) => {
    warnings.push(args.map((a) => String(a)).join(" "));
  };
  try {
    const img = ramp(1280, 768);
    // First call: should warn once.
    const a = await quantizeTiled(img, { paletteSize: 16, dither: "floyd-steinberg" });
    // Second call: should NOT warn again (one-shot).
    const b = await quantizeTiled(img, { paletteSize: 16, dither: "floyd-steinberg" });
    assertEquals(warnings.length, 1, `expected one warning, got ${warnings.length}`);
    assert(
      warnings[0].includes("floyd-steinberg") && warnings[0].includes("blue-noise"),
      `warning text unexpected: ${warnings[0]}`,
    );
    // NEW-E-010: `c` is constructed inside the same try block (warn mock
    // still active) but with dither: "blue-noise" — that path does not
    // emit the downgrade warning, so the `warnings.length === 1` assertion
    // above stays bit-stable regardless of `c`'s position. The downgrade
    // contract is: FS input must yield byte-identical indices to blue-noise
    // input.
    //
    // 4E-001 (4th-pass review): this assertion covers the alpha=255-only
    // input path. The alphaTable branch in quantize-tiled (per-palette-
    // entry alpha computed from a non-uniform alpha channel) is exercised
    // separately by the alpha-aware ramp test below.
    const c = await quantizeTiled(img, { paletteSize: 16, dither: "blue-noise" });
    assertEquals(a.indices.length, c.indices.length);
    let mismatches = 0;
    for (let i = 0; i < a.indices.length; i++) {
      if (a.indices[i] !== c.indices[i]) mismatches++;
    }
    assertEquals(mismatches, 0, `FS→blue-noise downgrade not equal to blue-noise: ${mismatches}`);
    // Second call's output should also equal the first (deterministic).
    let drift = 0;
    for (let i = 0; i < a.indices.length; i++) {
      if (a.indices[i] !== b.indices[i]) drift++;
    }
    assertEquals(drift, 0, `repeat FS call drifted: ${drift}`);
  } finally {
    console.warn = originalWarn;
    _resetTiledFsDowngradeWarning();
  }
});

Deno.test(
  "tiled downgrade equivalence holds on alpha-varying input (4E-001)",
  async () => {
    // 4E-001 (4th-pass review): the original "FS → blue-noise downgrade
    // is byte-equal" assertion runs on a fully-opaque ramp, so the
    // alphaTable path inside quantizeTiled (per-palette-entry rounded
    // mean alpha computed from non-uniform alpha) was uncovered. Build
    // a ramp where alpha varies across pixels and pin the same byte-
    // equality contract: FS input must yield byte-identical indices to
    // blue-noise input even when alphaTable is non-trivial.
    _resetTiledFsDowngradeWarning();
    const originalWarn = console.warn;
    console.warn = () => {};
    try {
      const w = 1280;
      const h = 768;
      const data = new Uint8ClampedArray(w * h * 4);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = (y * w + x) * 4;
          data[i] = (x * 255 / w) | 0;
          data[i + 1] = (y * 255 / h) | 0;
          data[i + 2] = ((x + y) * 127 / (w + h)) | 0;
          // Alpha varies: 64, 128, 192, 255 cycling — forces alphaTable
          // to compute non-uniform means per palette slot.
          data[i + 3] = [64, 128, 192, 255][(x + y) & 3];
        }
      }
      const img = new ImageData(data, w, h);
      const a = await quantizeTiled(img, { paletteSize: 16, dither: "floyd-steinberg" });
      const b = await quantizeTiled(img, { paletteSize: 16, dither: "blue-noise" });
      assertEquals(a.indices.length, b.indices.length);
      let mismatches = 0;
      for (let i = 0; i < a.indices.length; i++) {
        if (a.indices[i] !== b.indices[i]) mismatches++;
      }
      assertEquals(
        mismatches,
        0,
        `FS→blue-noise downgrade not equal under alpha-varying input: ${mismatches}`,
      );
    } finally {
      console.warn = originalWarn;
      _resetTiledFsDowngradeWarning();
    }
  },
);
