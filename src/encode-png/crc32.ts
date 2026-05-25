const TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n >>> 0;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    }
    t[n] = c >>> 0;
  }
  return t;
})();

// Single-shot CRC32 (PNG polynomial 0xEDB88320).
//
// There is intentionally no `seed` argument: the previous signature
// returned the final XOR-inverted value, so `crc32(b, crc32(a))` did NOT
// equal `crc32(concat(a, b))` and the parameter could only mislead.
// Use `crc32Multi(a, b, ...)` to hash a sequence without allocating the
// concatenation; that is the only correct way to split the input.
export function crc32(data: Uint8Array): number {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < data.length; i++) {
    c = TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xFFFFFFFF) >>> 0;
}

export function crc32Multi(...parts: Uint8Array[]): number {
  let c = 0xFFFFFFFF;
  for (const p of parts) {
    for (let i = 0; i < p.length; i++) {
      c = TABLE[(c ^ p[i]) & 0xff] ^ (c >>> 8);
    }
  }
  return (c ^ 0xFFFFFFFF) >>> 0;
}
