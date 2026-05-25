import { assert, assertEquals } from "jsr:@std/assert@^1.0.0";
import { detailMap } from "./dog.ts";
import { imageToOklabF32 } from "../colorspace/oklab.ts";

Deno.test("detailMap on a uniform image is ~0 everywhere", () => {
  const w = 16, h = 16;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = 128;
    data[i * 4 + 1] = 128;
    data[i * 4 + 2] = 128;
    data[i * 4 + 3] = 255;
  }
  const map = detailMap({ width: w, height: h, oklab: imageToOklabF32(data) });
  for (let i = 0; i < map.length; i++) {
    assert(map[i] < 1e-3, `uniform map[${i}] = ${map[i]}, expected ~0`);
  }
});

Deno.test("detailMap on a sharp edge is higher near the edge than far from it", () => {
  // Image must be large enough that the sigma=4 blur kernel can drop off; with
  // a 16-wide image the kernel covers the entire row. Use 64 instead.
  const w = 64, h = 16;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const v = x < w / 2 ? 0 : 255;
      data[i] = v;
      data[i + 1] = v;
      data[i + 2] = v;
      data[i + 3] = 255;
    }
  }
  const map = detailMap({ width: w, height: h, oklab: imageToOklabF32(data) });
  const center = (h / 2) | 0;
  const edgeVal = map[center * w + (w / 2 - 1)];
  const farVal = map[center * w + 0];
  assert(edgeVal > 0.5, `edge value ${edgeVal} should be > 0.5`);
  assert(farVal < 0.05, `far value ${farVal} should be < 0.05`);
  assert(edgeVal > farVal * 5, `edge ${edgeVal} should be >> far ${farVal}`);
});

Deno.test("detailMap length matches image size", () => {
  const w = 7, h = 11;
  const data = new Uint8ClampedArray(w * h * 4);
  const map = detailMap({ width: w, height: h, oklab: imageToOklabF32(data) });
  assertEquals(map.length, w * h);
});

Deno.test("detailMap on small images (width/height < kernel radius) stays finite", () => {
  // σ=4 → radius=12, so widths 1..12 used to drop reads off the array and
  // propagate NaN through the entire map. Sweep small sizes and assert every
  // entry is finite and in [0, 1].
  for (const w of [1, 2, 3, 5, 7, 12, 13]) {
    for (const h of [1, 2, 3, 5, 7, 12, 13]) {
      const data = new Uint8ClampedArray(w * h * 4);
      // Non-trivial pattern: a vertical gradient so a real Gaussian convolution
      // produces something non-zero. (A flat image would also exercise the
      // boundary handling but offers no signal.)
      for (let y = 0; y < h; y++) {
        const v = Math.round((y / Math.max(1, h - 1)) * 255);
        for (let x = 0; x < w; x++) {
          const i = (y * w + x) * 4;
          data[i] = v;
          data[i + 1] = v;
          data[i + 2] = v;
          data[i + 3] = 255;
        }
      }
      const map = detailMap({ width: w, height: h, oklab: imageToOklabF32(data) });
      assertEquals(map.length, w * h);
      for (let i = 0; i < map.length; i++) {
        assert(
          Number.isFinite(map[i]) && map[i] >= 0 && map[i] <= 1,
          `w=${w} h=${h} map[${i}]=${map[i]} (must be finite ∈ [0,1])`,
        );
      }
    }
  }
});
