/**
 * @module
 *
 * PNG-8 (indexed-colour) encoder — `encodePng8` plus the byte-level
 * helpers (`chunk`, `concatBytes`) needed to compose additional chunks
 * around it. Importable standalone via the `./png` sub-module entry
 * point; useful for callers that already have indexed pixels and a
 * palette and just need a PNG-8 file out.
 *
 * @example
 * ```ts
 * import { encodePng8 } from "@ys319/inksquid/png";
 *
 * const png = await encodePng8({
 *   width, height, indices,
 *   palette: { rgb: paletteRgbBytes },
 * });
 * ```
 */

import { chunk, concatBytes } from "./chunk.ts";
import { deflate } from "./deflate.ts";

// crc32 / crc32Multi are intentionally NOT re-exported: they're PNG
// implementation details. Importers that need a CRC32 should reach for a
// general utility, not the PNG encoder subpath.
//
// chunk and concatBytes ARE re-exported so callers that need to compose
// additional PNG chunks (e.g. custom ancillary chunks not currently emitted
// by encodePng8) can reuse the same CRC-wrapping primitive without
// duplicating the type/length/CRC layout logic.
export { chunk, concatBytes } from "./chunk.ts";

const PNG_SIGNATURE = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function buildIHDR(width: number, height: number): Uint8Array {
  const data = new Uint8Array(13);
  const view = new DataView(data.buffer);
  view.setUint32(0, width, false);
  view.setUint32(4, height, false);
  data[8] = 8;
  data[9] = 3;
  data[10] = 0;
  data[11] = 0;
  data[12] = 0;
  return chunk("IHDR", data);
}

function buildPLTE(palette: Uint8Array): Uint8Array {
  if (palette.length % 3 !== 0) {
    throw new Error(`PLTE payload must be a multiple of 3 (RGB triplets), got ${palette.length}`);
  }
  const colors = palette.length / 3;
  if (colors < 1 || colors > 256) {
    throw new Error(`PLTE must contain 1-256 colors, got ${colors}`);
  }
  return chunk("PLTE", palette);
}

function buildTRNS(alphas: Uint8Array): Uint8Array {
  return chunk("tRNS", alphas);
}

function buildIEND(): Uint8Array {
  return chunk("IEND", new Uint8Array(0));
}

// PNG filter type is hard-coded to 0 (None). PNG-8 (color type 3) indexes
// into a palette, and the bytes encode arbitrary palette IDs whose numerical
// adjacency carries no signal; the difference-style filters (Sub/Up/Average/
// Paeth) typically produce no benefit and often slightly inflate the deflate
// stream for indexed color (a small win is possible only when the input
// happens to produce monotonic palette IDs, which quantization rarely does).
// The 30-50 % size win of per-row filter trial that you sometimes hear about
// applies to truecolor (color type 2/6), not indexed. See PNG-011 in
// `.claude/reviews/2026-05-23_initial/ROADMAP.md` for the deferral rationale.
function addFilterBytes(indices: Uint8Array, width: number, height: number): Uint8Array {
  if (indices.length !== width * height) {
    throw new Error(`indices.length ${indices.length} !== width*height ${width * height}`);
  }
  const out = new Uint8Array((width + 1) * height);
  for (let y = 0; y < height; y++) {
    out[y * (width + 1)] = 0;
    out.set(indices.subarray(y * width, (y + 1) * width), y * (width + 1) + 1);
  }
  return out;
}

/**
 * Palette payload for {@link encodePng8}. The encoder emits a PLTE chunk
 * from `rgb` and a tRNS chunk from `alpha` (when non-trivial).
 */
export interface PaletteRGBA {
  /** Stride-3 RGB byte values per palette entry; length = `entries * 3`. */
  rgb: Uint8Array;
  /**
   * Optional per-entry alpha in `[0, 255]`. When omitted (or every byte
   * is 255), `encodePng8` skips the tRNS chunk entirely — fully-opaque
   * PNGs stay one chunk smaller. When provided, trailing 255 entries are
   * trimmed (PNG spec: tRNS shorter than PLTE is legal; missing entries
   * are treated as opaque).
   */
  alpha?: Uint8Array;
}

/** Input record for {@link encodePng8}. */
export interface EncodePng8Input {
  /** Image width in pixels. Must be a positive integer. */
  width: number;
  /** Image height in pixels. Must be a positive integer. */
  height: number;
  /**
   * Row-major palette index per pixel. Length must equal `width * height`;
   * every byte must lie in `[0, palette.rgb.length / 3)` unless
   * {@link EncodePng8Input.validate} is set to `false`.
   */
  indices: Uint8Array;
  /** Palette to emit as PLTE (+ optional tRNS). See {@link PaletteRGBA}. */
  palette: PaletteRGBA;
  /**
   * If true (default), every byte of `indices` is checked against the
   * palette size. This is O(n) on 4 MP inputs (~10-20 ms of synchronous
   * work). Library-internal callers that already know the indices came
   * from a clamped source (Wu init → k-means → Uint8Array) pass `false`
   * to skip the check; the three internal pipelines that do this today
   * are `quantizeCpu`, `quantizeGpu`, and `quantizeTiled` — any future
   * fourth pipeline must also opt in to `validate: false` for the same
   * O(n) saving. External callers should leave the default on so a
   * stray bad index is surfaced as a clear error rather than silently
   * producing a corrupt PNG.
   *
   * **Contract when `validate: false`**: every byte of `indices` MUST
   * be in `[0, palette.rgb.length / 3)`. Passing an out-of-range index
   * is undefined behaviour — `encodePng8` will not throw and the
   * resulting bytes will still parse as a structurally valid PNG, but
   * decoders that respect the palette length will render the bad pixel
   * as either the wrong colour or transparent. There is no upstream
   * boundary check to fall back on once this knob is set, so the
   * guarantee must hold at the call site.
   */
  validate?: boolean;
}

/**
 * Encode an indexed-colour image as a PNG-8 file (PNG colour type 3).
 *
 * The returned `Uint8Array` is a complete, decoder-ready PNG: signature,
 * IHDR, PLTE, optional tRNS (when any palette alpha < 255), IDAT
 * (deflate-compressed via {@link CompressionStream}), IEND.
 *
 * Throws on:
 * - non-positive `width` / `height`
 * - `palette.rgb.length` not divisible by 3
 * - `palette.alpha.length > palette.rgb.length / 3`
 * - any byte of `indices` exceeding the palette size (only when
 *   `validate` is left at its default `true`)
 *
 * @param input — see {@link EncodePng8Input}.
 * @returns the complete PNG-8 byte stream.
 */
export async function encodePng8(input: EncodePng8Input): Promise<Uint8Array> {
  const { width, height, indices, palette } = input;
  const validate = input.validate ?? true;
  if (width <= 0 || height <= 0) {
    throw new Error(`width and height must be positive, got ${width}x${height}`);
  }
  if (palette.rgb.length % 3 !== 0) {
    throw new Error("palette.rgb length must be multiple of 3");
  }
  const colors = palette.rgb.length / 3;
  if (palette.alpha && palette.alpha.length > colors) {
    throw new Error(
      `palette.alpha length (${palette.alpha.length}) exceeds palette colors (${colors})`,
    );
  }
  if (validate) {
    for (let i = 0; i < indices.length; i++) {
      if (indices[i] >= colors) {
        throw new Error(`index ${indices[i]} at position ${i} exceeds palette size ${colors}`);
      }
    }
  }

  const filtered = addFilterBytes(indices, width, height);
  const compressed = await deflate(filtered);

  const parts: Uint8Array[] = [PNG_SIGNATURE, buildIHDR(width, height), buildPLTE(palette.rgb)];
  if (palette.alpha) {
    let lastOpaque = palette.alpha.length;
    while (lastOpaque > 0 && palette.alpha[lastOpaque - 1] === 255) lastOpaque--;
    if (lastOpaque > 0) {
      parts.push(buildTRNS(palette.alpha.subarray(0, lastOpaque)));
    }
  }
  parts.push(chunk("IDAT", compressed));
  parts.push(buildIEND());
  return concatBytes(parts);
}
