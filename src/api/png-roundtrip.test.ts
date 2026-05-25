import { assertEquals } from "jsr:@std/assert@^1.0.0";
import { encodePng8 } from "../encode-png/mod.ts";

async function inflate(data: Uint8Array): Promise<Uint8Array> {
  const ds = new DecompressionStream("deflate");
  const w = ds.writable.getWriter();
  w.write(data as Uint8Array<ArrayBuffer>);
  w.close();
  const buf = await new Response(ds.readable).arrayBuffer();
  return new Uint8Array(buf);
}

function findChunkData(png: Uint8Array, type: string): Uint8Array {
  const target = new TextEncoder().encode(type);
  let i = 8;
  while (i < png.length - 8) {
    const view = new DataView(png.buffer, png.byteOffset);
    const len = view.getUint32(i, false);
    const matches = png[i + 4] === target[0] && png[i + 5] === target[1] &&
      png[i + 6] === target[2] && png[i + 7] === target[3];
    if (matches) return png.subarray(i + 8, i + 8 + len);
    i += 4 + 4 + len + 4;
  }
  throw new Error(`chunk ${type} not found`);
}

Deno.test("encodePng8 IDAT roundtrips: deflate(filtered_indices) recovers original", async () => {
  const width = 5;
  const height = 3;
  const indices = new Uint8Array([
    0,
    1,
    2,
    1,
    0,
    1,
    2,
    0,
    2,
    1,
    2,
    0,
    1,
    0,
    2,
  ]);
  const png = await encodePng8({
    width,
    height,
    indices,
    palette: { rgb: new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255]) },
  });

  const idat = findChunkData(png, "IDAT");
  const decompressed = await inflate(idat);

  // (width + 1) bytes per row: filter byte (0) followed by indices
  assertEquals(decompressed.length, (width + 1) * height);
  for (let y = 0; y < height; y++) {
    assertEquals(decompressed[y * (width + 1)], 0, `filter byte on row ${y} should be 0`);
    for (let x = 0; x < width; x++) {
      assertEquals(
        decompressed[y * (width + 1) + 1 + x],
        indices[y * width + x],
        `pixel (${x},${y})`,
      );
    }
  }
});

Deno.test("encodePng8 produces a chunk sequence ending with IEND", async () => {
  const png = await encodePng8({
    width: 1,
    height: 1,
    indices: new Uint8Array([0]),
    palette: { rgb: new Uint8Array([128, 128, 128]) },
  });
  const types: string[] = [];
  let i = 8;
  const view = new DataView(png.buffer, png.byteOffset);
  const dec = new TextDecoder();
  while (i < png.length - 8) {
    const len = view.getUint32(i, false);
    types.push(dec.decode(png.subarray(i + 4, i + 8)));
    i += 4 + 4 + len + 4;
  }
  assertEquals(types[0], "IHDR");
  assertEquals(types[1], "PLTE");
  assertEquals(types.includes("IDAT"), true);
  assertEquals(types[types.length - 1], "IEND");
});

Deno.test("encodePng8 tRNS appears after PLTE and before IDAT (PNG §11)", async () => {
  // E-015: PNG spec §11.3.2.1 requires tRNS to follow PLTE and precede the
  // first IDAT. Pin this ordering — a decoder that's strict about ancillary
  // chunk placement (e.g. PNG vault validators) would reject the file
  // otherwise. The previous tests checked tRNS *presence* but not position.
  const png = await encodePng8({
    width: 1,
    height: 1,
    indices: new Uint8Array([0]),
    palette: {
      rgb: new Uint8Array([255, 0, 0]),
      alpha: new Uint8Array([128]), // forces tRNS emission
    },
  });
  const types: string[] = [];
  let i = 8;
  const view = new DataView(png.buffer, png.byteOffset);
  const dec = new TextDecoder();
  while (i < png.length - 8) {
    const len = view.getUint32(i, false);
    types.push(dec.decode(png.subarray(i + 4, i + 8)));
    i += 4 + 4 + len + 4;
  }
  const plteIdx = types.indexOf("PLTE");
  const trnsIdx = types.indexOf("tRNS");
  const idatIdx = types.indexOf("IDAT");
  assertEquals(trnsIdx > plteIdx, true, `tRNS (${trnsIdx}) must come after PLTE (${plteIdx})`);
  assertEquals(trnsIdx < idatIdx, true, `tRNS (${trnsIdx}) must come before IDAT (${idatIdx})`);
});
