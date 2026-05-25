import { assert, assertEquals } from "jsr:@std/assert@^1.0.0";
import { quantize, quantizeCpu } from "../mod.ts";
import { ramp } from "../_test-fixtures.ts";

const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

Deno.test("quantize(cpu) produces a valid PNG signature", async () => {
  const img = ramp(32, 32);
  const r = await quantizeCpu(img, { paletteSize: 16 });
  for (let i = 0; i < PNG_SIG.length; i++) assertEquals(r.png[i], PNG_SIG[i]);
});

Deno.test("quantize(cpu) returns meta with non-zero elapsedMs and outputBytes", async () => {
  const img = ramp(64, 64);
  const r = await quantizeCpu(img, { paletteSize: 32 });
  assert(r.meta.outputBytes === r.png.byteLength, "outputBytes != png.byteLength");
  assert(r.meta.elapsedMs > 0, "elapsedMs should be > 0");
});

Deno.test("quantize(cpu) palette has the requested size (within -2..0)", async () => {
  // Wu may merge near-identical bins when the input has < paletteSize
  // distinct colours after histogram quantisation; the implementation
  // guarantees `count <= paletteSize` but doesn't add ghost entries on
  // the low side, so the actual band is [requested-2, requested].
  const img = ramp(48, 48);
  const r = await quantizeCpu(img, { paletteSize: 8 });
  assert(r.meta.paletteSize >= 6 && r.meta.paletteSize <= 8, `paletteSize=${r.meta.paletteSize}`);
  assertEquals(r.palette.length, r.meta.paletteSize * 3);
});

Deno.test("quantize(cpu) honors dither option without throwing", async () => {
  const img = ramp(32, 32);
  for (const d of ["none", "blue-noise", "floyd-steinberg"] as const) {
    const r = await quantizeCpu(img, { paletteSize: 16, dither: d });
    assert(
      r.png.byteLength > 100,
      `dither=${d} produced suspiciously small PNG (${r.png.byteLength})`,
    );
  }
});

Deno.test("quantize(auto) falls back to cpu when no GPU and produces output", async () => {
  const img = ramp(24, 24);
  const r = await quantize(img, { mode: "auto", paletteSize: 8 });
  for (let i = 0; i < PNG_SIG.length; i++) assertEquals(r.png[i], PNG_SIG[i]);
});

Deno.test("quantize(cpu) preview dimensions match input", async () => {
  const img = ramp(40, 20);
  const r = await quantizeCpu(img, { paletteSize: 8 });
  assertEquals(r.preview.width, 40);
  assertEquals(r.preview.height, 20);
  assertEquals(r.preview.data.length, 40 * 20 * 4);
});

Deno.test("quantize(cpu) returns indices with width*height entries in [0, paletteSize)", async () => {
  const img = ramp(40, 20);
  const r = await quantizeCpu(img, { paletteSize: 8 });
  assertEquals(r.indices.length, 40 * 20);
  assert(r.indices instanceof Uint8Array, "indices must be Uint8Array");
  for (let i = 0; i < r.indices.length; i++) {
    assert(
      r.indices[i] < r.meta.paletteSize,
      `indices[${i}]=${r.indices[i]} out of range for paletteSize=${r.meta.paletteSize}`,
    );
  }
});

Deno.test("quantize(cpu) indices recover the preview through palette (all dithers — NEW-E-013)", async () => {
  // Spot-check that splatting palette[indices[i]] reproduces preview RGB
  // for every pixel — this pins the contract that `indices` and `preview`
  // describe the same final color choice. Originally pinned for dither
  // "none" only; NEW-E-013 widened the loop so blue-noise and FS regress
  // visibly if their preview ever drifts from indices (the two dithers
  // build preview from indices internally, so a future refactor that
  // double-writes preview from a separate code path would surface here).
  const img = ramp(16, 16);
  for (const d of ["none", "blue-noise", "floyd-steinberg"] as const) {
    const r = await quantizeCpu(img, { paletteSize: 8, dither: d });
    for (let i = 0; i < r.indices.length; i++) {
      const j = r.indices[i];
      assertEquals(
        r.preview.data[i * 4],
        r.palette[j * 3],
        `R drift at i=${i} (dither=${d})`,
      );
      assertEquals(
        r.preview.data[i * 4 + 1],
        r.palette[j * 3 + 1],
        `G drift at i=${i} (dither=${d})`,
      );
      assertEquals(
        r.preview.data[i * 4 + 2],
        r.palette[j * 3 + 2],
        `B drift at i=${i} (dither=${d})`,
      );
    }
  }
});
