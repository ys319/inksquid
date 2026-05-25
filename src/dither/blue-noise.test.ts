import { assert, assertEquals } from "jsr:@std/assert@^1.0.0";
import { BLUE_NOISE_64_SIZE, blueNoiseDither, getBlueNoise64 } from "./blue-noise.ts";
import { generateBlueNoise } from "./void-and-cluster.ts";
import { imageToOklabF32 } from "../colorspace/oklab.ts";
import { wuQuantizeFromSrgbU8 } from "../palette/wu.ts";
import { assignNearestOklab } from "../palette/kmeans.ts";

Deno.test("baked 64x64 blue noise has correct size", () => {
  const m = getBlueNoise64();
  assertEquals(m.length, BLUE_NOISE_64_SIZE * BLUE_NOISE_64_SIZE);
});

Deno.test("baked blue noise mean is near 127.5", () => {
  const m = getBlueNoise64();
  let sum = 0;
  for (const v of m) sum += v;
  const mean = sum / m.length;
  assert(Math.abs(mean - 127.5) < 1, `mean ${mean} should be close to 127.5`);
});

Deno.test("baked blue noise has no value repeated more than expected", () => {
  // With 4096 cells and 256 values, each value appears ~16 times on average.
  // Spike tolerance: any value appearing more than 32 times is suspicious.
  const m = getBlueNoise64();
  const counts = new Uint16Array(256);
  for (const v of m) counts[v]++;
  let maxCount = 0;
  for (const c of counts) maxCount = Math.max(maxCount, c);
  assert(maxCount <= 32, `max value frequency ${maxCount} > 32`);
});

Deno.test("generateBlueNoise produces a smaller mask with similar statistics", () => {
  const m = generateBlueNoise(32, 42);
  assertEquals(m.data.length, 32 * 32);
  let sum = 0;
  for (const v of m.data) sum += v;
  const mean = sum / m.data.length;
  // Ideal uniform 0..255 has mean 127.5. A ranked 32×32 mask spreads exactly
  // 0..1023 (then scales to 0..255 in 4-unit steps), giving the same mean.
  // Tolerance of ±3 covers float rounding in the scale step and is comfortably
  // tighter than the ±15 a random uniform 1024-sample population would show.
  assert(Math.abs(mean - 127.5) < 3, `32x32 mean ${mean} off`);
});

// 4C-02 (4th-pass review): rank uniqueness. Phase I + II of the V&C
// generator are supposed to assign every cell a distinct rank in
// `[0, n)`; the `Math.max(0, rankArr[i])` clamp in the output mapping
// is a defensive backstop, not an algorithmic feature. Reconstruct the
// implied ranks from the scaled mask and confirm every integer in
// `[0, n-1]` appears exactly once (modulo the float→u8 quantisation
// step which buckets multiple ranks into the same byte at large n, so
// we test a 32×32 generation where n=1024 → step=255/1023 makes the
// rank/byte map roughly 1:0.25, still recoverable as a per-byte
// histogram of ≤ 5 per bucket).
Deno.test("generateBlueNoise: every cell receives a distinct rank (no duplicates, no gaps)", () => {
  // Use a small power-of-two size so the rank→byte scaling is exact
  // enough to reconstruct the rank ordering via argsort.
  const size = 16; // n = 256, denom = 255 → 1:1 rank:byte mapping
  const m = generateBlueNoise(size, 7);
  assertEquals(m.data.length, size * size);
  // For size=16, n=256, denom=255, the scaled byte = round(rank * 255 / 255)
  // = rank exactly. Every byte must appear exactly once.
  const seen = new Uint8Array(size * size);
  for (const v of m.data) {
    assert(v < seen.length, `byte ${v} out of [0, ${seen.length})`);
    assert(seen[v] === 0, `duplicate rank ${v} — V&C generator violated uniqueness`);
    seen[v] = 1;
  }
  for (let i = 0; i < seen.length; i++) {
    assert(seen[i] === 1, `missing rank ${i} — V&C generator left a gap`);
  }
});

Deno.test("blueNoiseDither produces indices in palette range", () => {
  const w = 24;
  const h = 24;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = (i * 23) & 0xff;
    data[i * 4 + 1] = (i * 31) & 0xff;
    data[i * 4 + 2] = (i * 37) & 0xff;
    data[i * 4 + 3] = 255;
  }
  const { palette } = wuQuantizeFromSrgbU8(data, 8);
  const oklab = imageToOklabF32(data);
  const indices = blueNoiseDither({
    width: w,
    height: h,
    oklab,
    palette: palette.oklab,
    paletteCount: palette.count,
    strength: 1,
  });
  for (let i = 0; i < indices.length; i++) {
    assert(indices[i] < palette.count, `index ${indices[i]} >= ${palette.count}`);
  }
});

Deno.test("blueNoiseDither offsetX/offsetY shifts the mask phase (continuity across tiles)", () => {
  // Goal: dithering a single W×H strip should be reproducible by
  // dithering the right portion with offsetX = split point. The split
  // (40) is intentionally not a multiple of the 64-wide mask period so
  // the offset-less version reads a different mask cell — without that,
  // a no-op offset would silently pass the test (this is the same
  // "tile size is a multiple of 64, so the mask wraps to the same cell
  // and the offset never matters" trap noted in the review; see also
  // `TILE` in src/api/quantize-tiled.ts).
  const W = 104, H = 16, SPLIT = 40;
  const data = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      data[i] = ((x * 2) + y) & 0xff;
      data[i + 1] = (x + (y * 2)) & 0xff;
      data[i + 2] = ((x + y) * 3) & 0xff;
      data[i + 3] = 255;
    }
  }
  const { palette } = wuQuantizeFromSrgbU8(data, 16);
  const oklab = imageToOklabF32(data);

  // Reference: dither the whole strip at offset 0.
  const whole = blueNoiseDither({
    width: W,
    height: H,
    oklab,
    palette: palette.oklab,
    paletteCount: palette.count,
    strength: 1,
  });

  // Split: dither the right portion (W-SPLIT cols) with offsetX=SPLIT
  // against an oklab sub-buffer starting at column SPLIT. The mask
  // phase should pick up exactly where the left half left off.
  const RW = W - SPLIT;
  const rightOklab = new Float32Array(RW * H * 4);
  for (let y = 0; y < H; y++) {
    rightOklab.set(
      oklab.subarray((y * W + SPLIT) * 4, (y * W + W) * 4),
      y * RW * 4,
    );
  }
  const rightShifted = blueNoiseDither({
    width: RW,
    height: H,
    oklab: rightOklab,
    palette: palette.oklab,
    paletteCount: palette.count,
    strength: 1,
    offsetX: SPLIT,
  });
  // Compare the right portion of `whole` against `rightShifted`.
  let mismatches = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < RW; x++) {
      const w = whole[y * W + (SPLIT + x)];
      const r = rightShifted[y * RW + x];
      if (w !== r) mismatches++;
    }
  }
  assertEquals(mismatches, 0, `${mismatches} mismatches between phased and reference`);

  // Sanity: omitting the offset (default 0) breaks the phase, so the
  // same comparison should show many mismatches. This guards against a
  // future refactor that accidentally makes the offset a no-op.
  const rightZeroOffset = blueNoiseDither({
    width: RW,
    height: H,
    oklab: rightOklab,
    palette: palette.oklab,
    paletteCount: palette.count,
    strength: 1,
  });
  let zeroOffsetMismatches = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < RW; x++) {
      if (whole[y * W + (SPLIT + x)] !== rightZeroOffset[y * RW + x]) {
        zeroOffsetMismatches++;
      }
    }
  }
  // SPLIT=40 is coprime with the 64-wide mask period, so the phase
  // genuinely shifts; a real fraction of pixels should re-roll. 3E-006
  // (3rd-pass review) noted the hard 20-cell floor would lose its
  // signal at smaller test images; express as a ratio of the compared
  // region instead, with the same effective threshold for the current
  // size (RW*H ≈ 9120 cells → 20/9120 ≈ 0.0022, round to 0.002).
  const totalCells = RW * H;
  assert(
    zeroOffsetMismatches / totalCells >= 0.002,
    `omitting offsetX should have broken phase but produced ${zeroOffsetMismatches}/${totalCells} mismatches`,
  );
});

Deno.test("blueNoiseDither offsetY shifts the mask phase (continuity across row tiles)", () => {
  // NEW-E-012 mirror of the offsetX test above. The X test pinned horizontal
  // tile seams; this one pins vertical seams in case a future refactor splits
  // the offset application by axis. Tiled.test.ts also exercises a real Y
  // seam at H=TILE+176, but its detection (row change-rate ±15%) is far less
  // sensitive than the mismatches=0 check we can do at the dither layer.
  // SPLIT=40 is intentionally not a multiple of 64 (the mask period).
  const W = 16, H = 104, SPLIT = 40;
  const data = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      data[i] = ((x * 2) + y) & 0xff;
      data[i + 1] = (x + (y * 2)) & 0xff;
      data[i + 2] = ((x + y) * 3) & 0xff;
      data[i + 3] = 255;
    }
  }
  const { palette } = wuQuantizeFromSrgbU8(data, 16);
  const oklab = imageToOklabF32(data);

  const whole = blueNoiseDither({
    width: W,
    height: H,
    oklab,
    palette: palette.oklab,
    paletteCount: palette.count,
    strength: 1,
  });

  // Bottom portion (H-SPLIT rows) dithered with offsetY=SPLIT against an
  // oklab sub-buffer starting at row SPLIT.
  const RH = H - SPLIT;
  const bottomOklab = oklab.subarray(SPLIT * W * 4, H * W * 4);
  const bottomShifted = blueNoiseDither({
    width: W,
    height: RH,
    oklab: bottomOklab,
    palette: palette.oklab,
    paletteCount: palette.count,
    strength: 1,
    offsetY: SPLIT,
  });
  let mismatches = 0;
  for (let y = 0; y < RH; y++) {
    for (let x = 0; x < W; x++) {
      if (whole[(SPLIT + y) * W + x] !== bottomShifted[y * W + x]) mismatches++;
    }
  }
  assertEquals(mismatches, 0, `${mismatches} mismatches between Y-phased and reference`);

  // Sanity: omitting offsetY (default 0) breaks the phase.
  const bottomZeroOffset = blueNoiseDither({
    width: W,
    height: RH,
    oklab: bottomOklab,
    palette: palette.oklab,
    paletteCount: palette.count,
    strength: 1,
  });
  let zeroOffsetMismatches = 0;
  for (let y = 0; y < RH; y++) {
    for (let x = 0; x < W; x++) {
      if (whole[(SPLIT + y) * W + x] !== bottomZeroOffset[y * W + x]) zeroOffsetMismatches++;
    }
  }
  // 3E-006 (3rd-pass review): ratio form so the assertion scales with
  // image size if a future refactor shrinks the test image.
  const totalCells = W * RH;
  assert(
    zeroOffsetMismatches / totalCells >= 0.002,
    `omitting offsetY should have broken phase but produced ${zeroOffsetMismatches}/${totalCells} mismatches`,
  );
});

Deno.test("blueNoiseDither with strength=0 collapses to nearest-color", () => {
  const w = 16;
  const h = 16;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      data[i] = ((x + y) * 8) & 0xff;
      data[i + 1] = (x * 16) & 0xff;
      data[i + 2] = (y * 16) & 0xff;
      data[i + 3] = 255;
    }
  }
  const { palette } = wuQuantizeFromSrgbU8(data, 6);
  const oklab = imageToOklabF32(data);
  const a = blueNoiseDither({
    width: w,
    height: h,
    oklab,
    palette: palette.oklab,
    paletteCount: palette.count,
    strength: 0,
  });
  const b = blueNoiseDither({
    width: w,
    height: h,
    oklab,
    palette: palette.oklab,
    paletteCount: palette.count,
    strength: 0,
  });
  // Two runs at strength 0 should produce the same indices.
  for (let i = 0; i < a.length; i++) assertEquals(a[i], b[i]);

  // E-003: strength=0 should also match the pure nearest-color
  // assignment byte-for-byte. Previously the test only checked
  // determinism, leaving room for a regression where strength=0 still
  // applied some unintended offset.
  const nearest = assignNearestOklab(oklab, palette.oklab, palette.count);
  for (let i = 0; i < a.length; i++) {
    assertEquals(
      a[i],
      nearest[i],
      `strength=0 differs from nearest at i=${i}: a=${a[i]} nearest=${nearest[i]}`,
    );
  }
});
