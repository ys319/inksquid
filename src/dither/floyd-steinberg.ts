// Floyd-Steinberg error diffusion in OkLab space.
// Distributes the quantization error of each pixel into its 4 unvisited
// neighbours with the standard 7/16, 3/16, 5/16, 1/16 weights.
// Boustrophedon (alternating left↔right scan) yields better visual results
// than always-left-to-right; we use it by default.

export interface FsInput {
  width: number;
  height: number;
  oklab: Float32Array; // [L, a, b, alpha] per pixel, row-major
  palette: Float32Array; // [L, a, b] per palette entry
  paletteCount: number;
  strength: number; // 0..1, fraction of error to diffuse
  importance?: Float32Array; // per-pixel; when present, scales diffusion *down* where detail is high
}

/**
 * Transparency contract (mirrors `blueNoiseDither`):
 * - All pixels — transparent included — get a nearest-palette index.
 *   Transparent pixels additionally skip error diffusion (`alpha <= 0` →
 *   `continue` below the `indices[i] = best` line), so the error from a
 *   never-observed pixel doesn't leak into its visible neighbours.
 * - 3C-03 (3rd-pass review): "skip" here means **outgoing** diffusion
 *   only. A transparent pixel still *consumes* any error that was
 *   diffused into its L/a/b lanes by earlier (visible) neighbours
 *   above and to the left — those values are read by the nearest
 *   search above the `continue`. The resulting index is unobservable
 *   (alpha=0 → the PNG palette's alpha table maps it to invisible),
 *   so this is just a precise statement of what the existing code
 *   does, not a behaviour change. C-07 in the 2026-05-23 review
 *   tracks any future API-surface alignment between the two dither
 *   paths.
 */
export function floydSteinberg(input: FsInput): Uint8Array {
  const { width, height, palette, paletteCount } = input;
  const strength = Math.max(0, Math.min(1, input.strength));
  const work = new Float32Array(input.oklab.length);
  work.set(input.oklab);
  const indices = new Uint8Array(width * height);

  for (let y = 0; y < height; y++) {
    const leftToRight = (y & 1) === 0;
    const xStart = leftToRight ? 0 : width - 1;
    const xEnd = leftToRight ? width : -1;
    const dx = leftToRight ? 1 : -1;
    for (let x = xStart; x !== xEnd; x += dx) {
      const i = y * width + x;
      const alpha = work[i * 4 + 3];
      const L = work[i * 4];
      const a = work[i * 4 + 1];
      const b = work[i * 4 + 2];
      let best = 0;
      let bestDist = Infinity;
      for (let j = 0; j < paletteCount; j++) {
        const dL = L - palette[j * 3];
        const da = a - palette[j * 3 + 1];
        const db = b - palette[j * 3 + 2];
        const d = dL * dL + da * da + db * db;
        if (d < bestDist) {
          bestDist = d;
          best = j;
        }
      }
      indices[i] = best;
      if (alpha <= 0) continue;

      const eL = (L - palette[best * 3]) * strength;
      const ea = (a - palette[best * 3 + 1]) * strength;
      const eb = (b - palette[best * 3 + 2]) * strength;
      const scale = input.importance ? (1 - input.importance[i] * 0.5) : 1;
      const elL = eL * scale;
      const ela = ea * scale;
      const elb = eb * scale;

      // Neighbours: (x+dx, y), (x-dx, y+1), (x, y+1), (x+dx, y+1)
      const diffuse = (xn: number, yn: number, w: number) => {
        if (xn < 0 || xn >= width || yn < 0 || yn >= height) return;
        const ni = yn * width + xn;
        work[ni * 4] += elL * w;
        work[ni * 4 + 1] += ela * w;
        work[ni * 4 + 2] += elb * w;
      };
      diffuse(x + dx, y, 7 / 16);
      diffuse(x - dx, y + 1, 3 / 16);
      diffuse(x, y + 1, 5 / 16);
      diffuse(x + dx, y + 1, 1 / 16);
    }
  }
  return indices;
}
