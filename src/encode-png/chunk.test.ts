import { assertEquals } from "jsr:@std/assert@^1.0.0";
import { chunk } from "./chunk.ts";
import { crc32 } from "./crc32.ts";

Deno.test("chunk wraps data with length, type, payload, CRC", () => {
  const data = new Uint8Array([1, 2, 3, 4]);
  const out = chunk("IHDR", data);
  // 4 (len) + 4 (type) + 4 (data) + 4 (crc) = 16
  assertEquals(out.length, 16);
  // length field, big-endian
  assertEquals(out[0], 0);
  assertEquals(out[1], 0);
  assertEquals(out[2], 0);
  assertEquals(out[3], 4);
  // type field
  assertEquals(out[4], 0x49); // 'I'
  assertEquals(out[5], 0x48); // 'H'
  assertEquals(out[6], 0x44); // 'D'
  assertEquals(out[7], 0x52); // 'R'
  // data
  assertEquals(out[8], 1);
  assertEquals(out[9], 2);
  assertEquals(out[10], 3);
  assertEquals(out[11], 4);

  // E-010: the trailing 4 bytes are the CRC of (type ++ data). Pin the
  // actual bytes — previously the test only counted them. Catches a
  // regression where the CRC seed or feed order is wrong.
  const expected = crc32(out.subarray(4, 12)); // type+data
  assertEquals(out[12], (expected >>> 24) & 0xff);
  assertEquals(out[13], (expected >>> 16) & 0xff);
  assertEquals(out[14], (expected >>> 8) & 0xff);
  assertEquals(out[15], expected & 0xff);
});

Deno.test("chunk rejects non-4-char type", () => {
  let threw = false;
  try {
    chunk("X", new Uint8Array(0));
  } catch {
    threw = true;
  }
  assertEquals(threw, true);
});

Deno.test("chunk rejects non-ASCII chars in type", () => {
  let threw = false;
  try {
    chunk("IHÿR", new Uint8Array(0));
  } catch {
    threw = true;
  }
  assertEquals(threw, true);
});

Deno.test("chunk rejects control chars in type (D-009)", () => {
  // D-009 (2nd-pass review): before the guard widened to also reject
  // 0x00..0x1F, a type string like "IHR" passed silently and
  // produced a corrupt PNG. Pin the new behaviour so any future
  // loosening shows up as a test failure.
  for (const cc of [0x00, 0x01, 0x09, 0x0a, 0x1f]) {
    let threw = false;
    try {
      chunk("IH" + String.fromCharCode(cc) + "R", new Uint8Array(0));
    } catch {
      threw = true;
    }
    assertEquals(threw, true, `expected throw for control char 0x${cc.toString(16)}`);
  }
  // 0x20 (space) is still allowed — printable ASCII boundary.
  chunk("IH R", new Uint8Array(0));
});
