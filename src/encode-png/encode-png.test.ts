import { assertEquals } from "jsr:@std/assert@^1.0.0";
import { encodePng8 } from "./mod.ts";

const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

Deno.test("encodePng8 prefixes the PNG signature", async () => {
  const png = await encodePng8({
    width: 2,
    height: 2,
    indices: new Uint8Array([0, 1, 1, 0]),
    palette: { rgb: new Uint8Array([255, 0, 0, 0, 0, 255]) },
  });
  for (let i = 0; i < PNG_SIG.length; i++) assertEquals(png[i], PNG_SIG[i]);
});

Deno.test("encodePng8 produces IHDR with correct width, height, color type, bit depth", async () => {
  const png = await encodePng8({
    width: 13,
    height: 7,
    indices: new Uint8Array(13 * 7),
    palette: { rgb: new Uint8Array([0, 0, 0, 255, 255, 255]) },
  });
  // Bytes [8..11]: IHDR length (always 13 for color type 3)
  const view = new DataView(png.buffer, png.byteOffset);
  assertEquals(view.getUint32(8, false), 13);
  // Bytes [12..15]: chunk type "IHDR"
  assertEquals(png[12], 0x49);
  assertEquals(png[13], 0x48);
  assertEquals(png[14], 0x44);
  assertEquals(png[15], 0x52);
  // Bytes [16..19]: width
  assertEquals(view.getUint32(16, false), 13);
  // Bytes [20..23]: height
  assertEquals(view.getUint32(20, false), 7);
  // Byte 24: bit depth = 8, Byte 25: color type = 3 (indexed)
  assertEquals(png[24], 8);
  assertEquals(png[25], 3);
});

Deno.test("encodePng8 rejects an index that exceeds palette size", async () => {
  let threw = false;
  try {
    await encodePng8({
      width: 1,
      height: 1,
      indices: new Uint8Array([5]),
      palette: { rgb: new Uint8Array([0, 0, 0]) },
    });
  } catch {
    threw = true;
  }
  assertEquals(threw, true);
});

Deno.test("encodePng8 validate:false skips the index bounds check", async () => {
  // The same input that rejects under default validation passes under
  // validate:false. The library-internal callers (quantize-{cpu,gpu,tiled})
  // use this to skip an O(n) revalidation on the encode hot path; this
  // test pins that contract so a future refactor can't quietly remove the
  // opt-out and slow the hot path back down.
  const out = await encodePng8({
    width: 1,
    height: 1,
    indices: new Uint8Array([5]),
    palette: { rgb: new Uint8Array([0, 0, 0]) },
    validate: false,
  });
  // Output is a well-formed PNG byte stream (signature) — we don't claim
  // the resulting image is meaningful, just that encode didn't throw.
  const sig = [0x89, 0x50, 0x4e, 0x47];
  for (let i = 0; i < sig.length; i++) assertEquals(out[i], sig[i]);
});

Deno.test("encodePng8 emits a tRNS chunk only when some alpha < 255", async () => {
  const withAlpha = await encodePng8({
    width: 1,
    height: 1,
    indices: new Uint8Array([0]),
    palette: { rgb: new Uint8Array([0, 0, 0]), alpha: new Uint8Array([0]) },
  });
  const withoutAlpha = await encodePng8({
    width: 1,
    height: 1,
    indices: new Uint8Array([0]),
    palette: { rgb: new Uint8Array([0, 0, 0]), alpha: new Uint8Array([255]) },
  });
  // Look for "tRNS" substring in chunk type positions.
  const findChunk = (png: Uint8Array, type: string): boolean => {
    const target = new TextEncoder().encode(type);
    outer: for (let i = 8; i < png.length - 8; i++) {
      for (let j = 0; j < 4; j++) {
        if (png[i + j] !== target[j]) continue outer;
      }
      return true;
    }
    return false;
  };
  assertEquals(findChunk(withAlpha, "tRNS"), true);
  assertEquals(findChunk(withoutAlpha, "tRNS"), false);
});

Deno.test("encodePng8 ends with IEND", async () => {
  const png = await encodePng8({
    width: 2,
    height: 2,
    indices: new Uint8Array([0, 1, 1, 0]),
    palette: { rgb: new Uint8Array([0, 0, 0, 255, 255, 255]) },
  });
  // IEND chunk: 00 00 00 00 49 45 4E 44 AE 42 60 82 (length=0, type=IEND, CRC)
  const expected = [0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82];
  const tail = Array.from(png.slice(-12));
  assertEquals(tail, expected);
});
