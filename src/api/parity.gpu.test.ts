// CPU↔GPU parity tests.
//
// Both pipelines are stage-for-stage equivalent, but the GPU path uses f32
// throughout while the CPU path uses Float64 for k-means accumulation. The
// remaining divergence is sub-perceptual; this test pins it numerically so
// regressions surface immediately.
//
// Run with:
//   deno test --unstable-webgpu --allow-read 'src/**/*.gpu.test.ts'
// Skipped automatically when no WebGPU adapter is present.

import { assert } from "jsr:@std/assert@^1.0.0";
import { isWebGPUAvailable, quantizeCpu, quantizeGpu } from "../mod.ts";
import type { QuantizeOptions } from "../mod.ts";
import { geometric, gradient, photoNoise, skipWarn } from "../_test-fixtures.ts";

const hasGpu = await isWebGPUAvailable();
const gpuOpts = { ignore: !hasGpu };

interface ParityCase {
  name: string;
  // Optional regression-guard anchor (e.g. "N-B-04", "W-A-5"). When set, the
  // Deno test name gets a `[<id>]` suffix so CI logs and `--filter` calls can
  // find the case by review ID rather than scanning the rationale comment.
  regression?: string;
  build: (w: number, h: number) => ImageData;
  width: number;
  height: number;
  options: Partial<QuantizeOptions>;
  // Floors derived from M5 measurements (see comment block on CASES) with a
  // wobble margin. Tighter than the original heuristics so the test catches
  // numeric drift without being affected by per-machine f32 ordering noise.
  minPreviewPsnr: number;
  minIndexMatchRate: number;
  // Maximum mean per-slot Euclidean distance between the CPU and GPU
  // palettes, measured in **sRGB byte space** (NOT a CIE ΔE colour
  // difference — that would require an OkLab/Lab distance, and the
  // numerical scale would be very different). 3E-013 (3rd-pass review)
  // renamed this and the helper away from "ΔE" so the metric isn't
  // mistaken for CIE ΔE; the value is still the right thing to compare
  // floors against because both pipelines start from the same Wu init,
  // so slot j ↔ slot j byte-by-byte.
  maxPaletteSrgbDist: number;
}

// Measured values (Deno 2.7.12, Apple M5, multiple consecutive runs all
// identical after the post-loop assign fix in 5ff117f):
//   gradient 256/64/blue-noise → PSNR 46.83 dB, idx 99.94 %, palette dist 0.038
//   noise    256/64/blue-noise → PSNR ∞,        idx 100.00 %, palette dist 0.000
//   geom     256/16/none       → PSNR ∞,        idx 100.00 %, palette dist 0.000
// 3E-003 tightening (3rd-pass review): the previous gradient-case floors
// (PSNR 42, idx 0.90) were calibrated when the observed values were 96.69 %.
// After the post-loop assign fix, the gradient case agrees on 99.94 % of
// pixels, so the old 0.90 floor leaves 9.94 pp of slack that would silently
// swallow a real 5-9 pp regression. Tighten to 0.95 (5 pp headroom for
// cross-machine f32 ordering variance, still well clear of any single-
// configuration regression). PSNR floor moves 42 → 44 in lockstep: 2.83 dB
// of slack vs the observed 46.83 dB. The bit-exact cases (3 of 4) keep
// `maxPaletteSrgbDist` at 0.01 (NEW-E-007 in 2nd-pass review). The gradient
// case keeps the looser 0.15 ceiling on palette distance because its
// CPU/GPU divergence is genuine (one byte per slot averaged), and 0.038
// can drift toward 0.1 on other adapters with different f32 ordering — we
// haven't observed that yet, just leave the headroom so the test doesn't
// flake the first time it runs on Vulkan / D3D12.
const CASES: ParityCase[] = [
  {
    // 3E-008 (3rd-pass review): `iterN` is anchored in every case name
    // so the iter0 N-B-04 case is no longer the lone outlier — the
    // naming convention "<fixture>/<width>/p<k>/iter<N>/<dither>" is
    // now uniform across the suite.
    name: "gradient/256/p64/iter15/blue-noise",
    build: gradient,
    width: 256,
    height: 256,
    options: { paletteSize: 64, dither: "blue-noise", kmeansIterations: 15 },
    minPreviewPsnr: 44,
    minIndexMatchRate: 0.95,
    maxPaletteSrgbDist: 0.15,
  },
  {
    name: "noise/256/p64/iter15/blue-noise",
    build: photoNoise,
    width: 256,
    height: 256,
    options: { paletteSize: 64, dither: "blue-noise", kmeansIterations: 15 },
    // CPU and GPU agree bit-for-bit on photoNoise: PSNR is Infinity in
    // practice. A real regression (a few pixels drifting by ≥ 5 bytes) drops
    // PSNR well below 60 dB, so this floor still surfaces them.
    minPreviewPsnr: 60,
    minIndexMatchRate: 0.99,
    maxPaletteSrgbDist: 0.01,
  },
  {
    name: "geometric/256/p16/iter15/none",
    build: geometric,
    width: 256,
    height: 256,
    options: { paletteSize: 16, dither: "none", kmeansIterations: 15 },
    minPreviewPsnr: 60,
    minIndexMatchRate: 0.99,
    maxPaletteSrgbDist: 0.01,
  },
  {
    // N-B-04 regression guard: with kmeansIterations=0 the post-loop assign
    // must run on both CPU and GPU and produce the same nearest-to-Wu-init
    // index stream. Before the fix, GPU returned all-zero indices (and CPU
    // returned a different all-zero stream) so the test would have caught
    // it by failing both PSNR (collapsed pixels render as palette[0]) and
    // index-match-rate. Pin the case at high thresholds since Wu init is
    // shared and the only difference is CPU/GPU float ordering.
    name: "gradient/128/p8/iter0/none",
    regression: "N-B-04",
    build: gradient,
    width: 128,
    height: 128,
    options: { paletteSize: 8, dither: "none", kmeansIterations: 0 },
    minPreviewPsnr: 60,
    minIndexMatchRate: 0.99,
    maxPaletteSrgbDist: 0.01,
  },
  {
    // Phase C: scolorq CPU↔GPU parity. Both pipelines run the same
    // annealing schedule (T0=0.001, Tf=0.00001, 15 sweeps) with the
    // same softmax math; the only remaining divergence is f32 (GPU)
    // vs Float64 (CPU) accumulator ordering inside the centroid
    // update.
    //
    // Measured values (Deno 2.7.12, Apple M5, multiple consecutive
    // runs identical):
    //   gradient/128/p16/scolorq → PSNR 64.18 dB, idx 100.00 %,
    //                              palette sRGB dist 0.063
    // The earlier looser floors (PSNR 30 / idx 0.50 / palette dist
    // 20) were calibrated for the over-large T0=0.01 era when the
    // soft assignment touched every centroid every sweep and
    // accumulated f32-ordering noise across 15 sweeps. After the
    // mode-collapse fix retuned T0 to 0.001 the softmax is sharp
    // enough that CPU and GPU converge to near-identical centroids;
    // the new floors take advantage of that to catch real
    // regressions instead of swallowing 30+ dB of slack.
    // 14 dB / 5 pp / ~3× headroom against the measured values, in
    // line with the gradient/256/p64/blue-noise case's headroom
    // policy (see top-of-CASES comment block on calibration).
    name: "gradient/128/p16/scolorq",
    build: gradient,
    width: 128,
    height: 128,
    options: { paletteSize: 16, dither: "scolorq" },
    minPreviewPsnr: 50,
    minIndexMatchRate: 0.95,
    maxPaletteSrgbDist: 0.2,
  },
];

function psnrSrgb(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
  // Per-channel MSE on the 3 colour bytes; alpha is ignored because the test
  // images are fully opaque.
  let sumSq = 0;
  let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const d = a[i + c] - b[i + c];
      sumSq += d * d;
      n++;
    }
  }
  if (sumSq === 0) return Infinity;
  const mse = sumSq / n;
  return 10 * Math.log10((255 * 255) / mse);
}

function indexMatchRate(a: Uint8Array, b: Uint8Array): number {
  // Wu init is shared, so slot j on CPU is the same slot j on GPU and a
  // byte-exact comparison of the index streams is a true "did both pipelines
  // pick the same palette entry?" measurement (not a rendered-color proxy).
  if (a.length !== b.length) return 0;
  let matched = 0;
  for (let i = 0; i < a.length; i++) if (a[i] === b[i]) matched++;
  return matched / a.length;
}

function paletteSrgbDistance(a: Uint8Array, b: Uint8Array): number {
  // Mean per-slot Euclidean distance between two palettes in **sRGB
  // byte space**, not a CIE ΔE colour difference. Both pipelines share
  // the same Wu init, so slot j ↔ slot j is the correct pairing and a
  // per-slot byte distance is exactly the quantity to floor for "did
  // k-means accumulation drift?" — the magnitude scales with f32
  // ordering noise, not with perceptual difference.
  const k = Math.min(a.length, b.length) / 3;
  if (k === 0) return 0;
  let sum = 0;
  for (let j = 0; j < k; j++) {
    const dr = a[j * 3] - b[j * 3];
    const dg = a[j * 3 + 1] - b[j * 3 + 1];
    const db = a[j * 3 + 2] - b[j * 3 + 2];
    sum += Math.sqrt(dr * dr + dg * dg + db * db);
  }
  return sum / k;
}

for (const c of CASES) {
  Deno.test({
    name: `parity: ${c.name}${c.regression ? ` [${c.regression}]` : ""}`,
    ...gpuOpts,
    fn: async () => {
      const img = c.build(c.width, c.height);
      const cpu = await quantizeCpu(img, c.options);
      const gpu = await quantizeGpu(img, c.options);

      const psnr = psnrSrgb(cpu.preview.data, gpu.preview.data);
      const idx = indexMatchRate(cpu.indices, gpu.indices);
      const sd = paletteSrgbDistance(cpu.palette, gpu.palette);

      // Surface measured values on every run so regressions show up as a
      // drifting margin against the floor, not only as a hard failure.
      console.log(
        `  [parity:${c.name}] PSNR ${Number.isFinite(psnr) ? psnr.toFixed(2) + " dB" : "∞"}, idx ${
          (idx * 100).toFixed(2)
        }%, palette sRGB dist ${sd.toFixed(3)}`,
      );

      assert(
        psnr >= c.minPreviewPsnr,
        `preview PSNR ${psnr.toFixed(2)} dB < floor ${c.minPreviewPsnr} dB`,
      );
      assert(
        idx >= c.minIndexMatchRate,
        `idx match ${(idx * 100).toFixed(2)} % < floor ${(c.minIndexMatchRate * 100).toFixed(0)} %`,
      );
      assert(
        sd <= c.maxPaletteSrgbDist,
        `palette sRGB dist ${sd.toFixed(3)} > ceiling ${c.maxPaletteSrgbDist}`,
      );
      // Output sizes should also be close — if they diverge by >30 % the
      // index streams have meaningfully different entropy, which deserves a
      // look.
      const sizeRatio = Math.max(cpu.png.length, gpu.png.length) /
        Math.min(cpu.png.length, gpu.png.length);
      assert(
        sizeRatio < 1.30,
        `PNG size ratio ${
          sizeRatio.toFixed(3)
        } > 1.30 (cpu=${cpu.png.length}, gpu=${gpu.png.length})`,
      );
    },
  });
}

if (!hasGpu) skipWarn("GPU parity test");
