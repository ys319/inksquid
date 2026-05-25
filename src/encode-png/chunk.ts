import { crc32Multi } from "./crc32.ts";

/**
 * Wrap an arbitrary payload as a PNG chunk: 4-byte big-endian length,
 * 4-byte ASCII type, payload, 4-byte CRC32 over `type + data`.
 *
 * `type` must be a 4-character ASCII string in the printable range
 * `[0x20, 0x7E]`. PNG spec §11.2 restricts chunk types to letters
 * (case-encoded semantic flags); the wider check here is intentionally
 * permissive — any future chunk type we ourselves emit (IHDR / PLTE /
 * tRNS / IDAT / IEND) is letters-only, and an external caller passing
 * digits / punctuation produces a structurally-valid PNG that lint tools
 * will flag rather than a silently-corrupt file.
 */
export function chunk(type: string, data: Uint8Array): Uint8Array {
  if (type.length !== 4) throw new Error(`chunk type must be 4 ASCII chars, got "${type}"`);
  // PNG spec §11.2: chunk types are 4 bytes, each restricted to A-Z or a-z
  // (case carries the ancillary/private/reserved/safe-to-copy semantics).
  // The guard below is the wider printable-ASCII range (0x20..0x7E) rather
  // than a strict letter check because the simpler test suffices: the 5
  // internal callers (IHDR / PLTE / tRNS / IDAT / IEND in encode-png/mod.ts
  // plus any future iCCP / pHYs etc.) are all letters anyway, so they're
  // covered. External `./png` consumers passing non-letter chunk types
  // (digits, punctuation) would technically violate §11.2 but we let them
  // through as a minor trade-off in favour of guard simplicity — they get
  // a well-formed-looking PNG that lints will flag, not a silent corrupt
  // file. D-009 (2nd-pass review): the earlier guard only rejected bytes
  // > 0x7E and let control characters (0x00..0x1F) through, producing
  // silently corrupt output. The lower bound here closes that hole.
  const typeBytes = new Uint8Array(4);
  for (let i = 0; i < 4; i++) {
    const cc = type.charCodeAt(i);
    if (cc > 0x7f || cc < 0x20) {
      throw new Error(
        `chunk type must be printable ASCII, got 0x${cc.toString(16).padStart(2, "0")} ` +
          `at index ${i} in "${type}"`,
      );
    }
    typeBytes[i] = cc;
  }
  const out = new Uint8Array(4 + 4 + data.length + 4);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length, false);
  out.set(typeBytes, 4);
  out.set(data, 8);
  const crc = crc32Multi(typeBytes, data);
  view.setUint32(8 + data.length, crc, false);
  return out;
}

/** Concatenate an ordered list of `Uint8Array`s into a single new buffer. */
export function concatBytes(parts: Uint8Array[]): Uint8Array {
  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}
