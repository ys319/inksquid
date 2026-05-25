// Shared test fixtures and image generators for colocated `*.test.ts` /
// `*.gpu.test.ts` files.
//
// Excluded from the JSR publish via `deno.json`'s `publish.exclude`. The
// underscore prefix is the convention this repo uses for files that live
// inside `src/` for test-discovery / colocation reasons but are not part
// of the public API surface.
//
// History:
//   - 4E-005 (4th-pass review) consolidated the per-file `ramp` / `flat`
//     copies from 6 test files into `tests/cpu/helpers.ts`.
//   - Cleanup (2026-05-25) folded `tests/gpu/helpers.ts` (re-export +
//     skipWarn) and `bench/synthesize.ts` (the gradient / photoNoise /
//     geometric image generators consumed by the GPU parity test) into
//     this single file when tests moved next to their source.

function fillRamp(data: Uint8ClampedArray, w: number, h: number): void {
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      data[i] = (x * 255 / w) | 0;
      data[i + 1] = (y * 255 / h) | 0;
      data[i + 2] = ((x + y) * 127 / (w + h)) | 0;
      data[i + 3] = 255;
    }
  }
}

/**
 * Standard ramp fixture: R varies with x (0..255), G with y (0..255),
 * B with (x+y)/(w+h) scaled to 0..127, alpha = 255. Returns an
 * `ImageData` — the most commonly used input shape across the suite.
 */
export function ramp(w: number, h: number): ImageData {
  const data = new Uint8ClampedArray(w * h * 4);
  fillRamp(data, w, h);
  return new ImageData(data, w, h);
}

/**
 * Same ramp as {@link ramp} but returns raw RGBA bytes. Use this when
 * the consumer accepts `Uint8ClampedArray` directly (e.g.
 * `wuQuantizeFromSrgbU8`, raw kmeans testing) and constructing an
 * `ImageData` would just be unwrapped immediately.
 */
export function rampBytes(w: number, h: number): Uint8ClampedArray {
  const data = new Uint8ClampedArray(w * h * 4);
  fillRamp(data, w, h);
  return data;
}

/**
 * Solid-colour fixture. Defaults to mid-grey (128, 128, 128, 255).
 * Pass an RGBA tuple to override the colour or alpha. Returns
 * `ImageData`.
 */
export function flat(
  w: number,
  h: number,
  rgba: [number, number, number, number] = [128, 128, 128, 255],
): ImageData {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = rgba[0];
    data[i * 4 + 1] = rgba[1];
    data[i * 4 + 2] = rgba[2];
    data[i * 4 + 3] = rgba[3];
  }
  return new ImageData(data, w, h);
}

/**
 * Emit a uniform "[skip] <suite>: no WebGPU adapter..." console warning
 * for GPU test files that can't run in the current environment. Lets
 * every skipped suite be grep'd with a single query.
 */
export function skipWarn(suite: string): void {
  console.warn(
    `\n[skip] ${suite}: no WebGPU adapter. ` +
      "Run with --unstable-webgpu on a machine that exposes Metal / Vulkan / D3D12.\n",
  );
}

function fract(x: number): number {
  return x - Math.floor(x);
}

/**
 * Smooth two-axis colour gradient with a sparse-band B channel. Used by
 * the GPU parity suite as the "k-means actually has to converge"
 * fixture — the f32 vs Float64 accumulator ordering between CPU and
 * GPU shows here.
 */
export function gradient(w: number, h: number): ImageData {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      data[i] = (x * 255 / w) | 0;
      data[i + 1] = (y * 255 / h) | 0;
      data[i + 2] = ((x ^ y) * 7) & 0xff;
      data[i + 3] = 255;
    }
  }
  return new ImageData(data, w, h);
}

/**
 * Deterministic photo-ish noise via layered fractal sine. Used as the
 * "CPU and GPU produce bit-identical output" parity case — the
 * fixture is high-entropy enough that any divergence shows up as
 * a real PSNR drop.
 */
export function photoNoise(w: number, h: number): ImageData {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      let v = 0;
      let amp = 1;
      let freq = 1 / 64;
      let total = 0;
      for (let oct = 0; oct < 5; oct++) {
        const n = Math.sin(x * freq * 17.31 + y * freq * 4.7) * 43758.5;
        v += fract(n) * amp;
        total += amp;
        amp *= 0.5;
        freq *= 2;
      }
      v = v / total;
      data[i] = Math.max(0, Math.min(255, (v * 240 + 16) | 0));
      data[i + 1] = Math.max(0, Math.min(255, ((1 - v) * 200 + 30) | 0));
      data[i + 2] = Math.max(0, Math.min(255, ((v * 0.5 + 0.4) * 255) | 0));
      data[i + 3] = 255;
    }
  }
  return new ImageData(data, w, h);
}

/**
 * Hard-edged radial/angular pattern with a fixed 8-colour palette.
 * Used as the "tiny palette, no dither" parity case — the bit-exact
 * baseline is the easy floor here.
 */
export function geometric(w: number, h: number): ImageData {
  const data = new Uint8ClampedArray(w * h * 4);
  const cx = w / 2;
  const cy = h / 2;
  const palette = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 255, 255],
    [0, 0, 0],
  ];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const r = Math.sqrt(dx * dx + dy * dy);
      const a = Math.atan2(dy, dx);
      const slice = ((a / Math.PI + 1) * 4) | 0;
      const ring = (r / 20) | 0;
      const idx = (slice + ring) % palette.length;
      const i = (y * w + x) * 4;
      data[i] = palette[idx][0];
      data[i + 1] = palette[idx][1];
      data[i + 2] = palette[idx][2];
      data[i + 3] = 255;
    }
  }
  return new ImageData(data, w, h);
}
