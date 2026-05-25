import { assertEquals } from "jsr:@std/assert@^1.0.0";
import { crc32, crc32Multi } from "./crc32.ts";

Deno.test("crc32 of '123456789' equals canonical 0xCBF43926", () => {
  const input = new TextEncoder().encode("123456789");
  assertEquals(crc32(input), 0xcbf43926);
});

Deno.test("crc32 of empty array is 0", () => {
  // Starts at the canonical 0xFFFFFFFF seed and never enters the loop body
  // when input is empty, so the final XOR-invert (`^ 0xFFFFFFFF`) returns 0.
  // PNG spec doesn't explicitly require this, but it's a well-defined
  // identity that downstream consumers (e.g. tests) implicitly depend on.
  assertEquals(crc32(new Uint8Array(0)), 0);
});

Deno.test("crc32Multi splits the same as crc32 over concatenation", () => {
  const a = new TextEncoder().encode("12345");
  const b = new TextEncoder().encode("6789");
  const combined = new Uint8Array(a.length + b.length);
  combined.set(a, 0);
  combined.set(b, a.length);
  assertEquals(crc32Multi(a, b), crc32(combined));
});

Deno.test("crc32 of IEND chunk type yields 0xAE426082", () => {
  // The CRC of the IEND chunk's [type+data] is a well-known constant
  // since IEND has no data.
  const iendType = new TextEncoder().encode("IEND");
  assertEquals(crc32(iendType), 0xae426082);
});
