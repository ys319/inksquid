import { assert, assertEquals } from "jsr:@std/assert@^1.0.0";
import { floydSteinberg } from "./floyd-steinberg.ts";
import { imageToOklabF32 } from "../colorspace/oklab.ts";
import { wuQuantizeFromSrgbU8 } from "../palette/wu.ts";

Deno.test("FS dither indices are all within palette range", () => {
  const w = 16;
  const h = 16;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = (i * 13) & 0xff;
    data[i * 4 + 1] = (i * 17) & 0xff;
    data[i * 4 + 2] = (i * 19) & 0xff;
    data[i * 4 + 3] = 255;
  }
  const { palette } = wuQuantizeFromSrgbU8(data, 8);
  const oklab = imageToOklabF32(data);
  const indices = floydSteinberg({
    width: w,
    height: h,
    oklab,
    palette: palette.oklab,
    paletteCount: palette.count,
    strength: 1.0,
  });
  assertEquals(indices.length, w * h);
  for (let i = 0; i < indices.length; i++) {
    assert(indices[i] < palette.count, `index ${indices[i]} out of range at ${i}`);
  }
});

Deno.test("FS dither produces more unique indices than no-dither on a smooth gradient", () => {
  const w = 32;
  const h = 32;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = ((x + y) * 255 / (w + h - 2)) | 0;
      const i = (y * w + x) * 4;
      data[i] = v;
      data[i + 1] = v;
      data[i + 2] = v;
      data[i + 3] = 255;
    }
  }
  const { palette } = wuQuantizeFromSrgbU8(data, 4);
  const oklab = imageToOklabF32(data);
  // No dither: nearest only.
  const noDither = floydSteinberg({
    width: w,
    height: h,
    oklab,
    palette: palette.oklab,
    paletteCount: palette.count,
    strength: 0,
  });
  const fs = floydSteinberg({
    width: w,
    height: h,
    oklab,
    palette: palette.oklab,
    paletteCount: palette.count,
    strength: 1.0,
  });
  // FS doesn't *add* palette entries, but on a gradient with 4 colours it should
  // exercise all of them (whereas no-dither may leave one unused due to banding).
  const usedNoDither = new Set(noDither).size;
  const usedFs = new Set(fs).size;
  assert(usedFs >= usedNoDither, `FS used ${usedFs}, no-dither used ${usedNoDither}`);

  // E-004: the original assertion (`>=`) is vacuously true if FS produces the
  // same output as no-dither — which would silently regress error diffusion.
  // Strengthen: FS should toggle between neighbouring palette entries far more
  // often than the banded no-dither output. Count adjacent-pixel index
  // transitions and require FS to have meaningfully more.
  function adjacentTransitions(arr: Uint8Array, width: number, height: number): number {
    let t = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 1; x < width; x++) if (arr[y * width + x] !== arr[y * width + x - 1]) t++;
    }
    return t;
  }
  const tNo = adjacentTransitions(noDither, w, h);
  const tFs = adjacentTransitions(fs, w, h);
  // On a smooth gradient with k=4, no-dither produces ~k-1 transitions per
  // row (one per band edge). FS should produce many more — stippling across
  // bands. Require at least 3× the transitions of the un-dithered baseline.
  assert(
    tFs >= 3 * tNo,
    `FS adjacent transitions ${tFs} should be ≥ 3× no-dither ${tNo} on smooth gradient`,
  );
});
