// GPU smoke tests. Run with:
//   deno test --unstable-webgpu --allow-read 'src/**/*.gpu.test.ts'
// Skips automatically when no adapter is available (e.g. inside sandboxed CI
// or harness environments).

import { assert, assertEquals } from "jsr:@std/assert@^1.0.0";
import { disposeSharedDevice, isWebGPUAvailable, quantize, quantizeGpu } from "../mod.ts";
import { ramp, skipWarn } from "../_test-fixtures.ts";

const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

const hasGpu = await isWebGPUAvailable();
const gpuOpts = { ignore: !hasGpu };

Deno.test({
  name: "GPU: quantizeGpu produces a valid PNG-8 byte stream",
  ...gpuOpts,
  fn: async () => {
    const r = await quantizeGpu(ramp(64, 64), { paletteSize: 16, kmeansIterations: 5 });
    for (let i = 0; i < PNG_SIG.length; i++) assertEquals(r.png[i], PNG_SIG[i]);
    assertEquals(r.preview.width, 64);
    assertEquals(r.preview.height, 64);
    assert(r.meta.elapsedMs > 0);
  },
});

Deno.test({
  name: "GPU: quantize(mode=gpu) honors all dither options",
  ...gpuOpts,
  fn: async () => {
    const img = ramp(48, 48);
    for (const d of ["none", "blue-noise", "floyd-steinberg", "scolorq"] as const) {
      const r = await quantize(img, {
        mode: "gpu",
        paletteSize: 16,
        dither: d,
        kmeansIterations: 5,
      });
      assert(r.png.byteLength > 100, `dither=${d} gave tiny PNG (${r.png.byteLength}B)`);
    }
  },
});

Deno.test({
  name: "GPU: scolorq produces valid output (Phase C smoke)",
  ...gpuOpts,
  fn: async () => {
    // scolorq has a different shader pipeline (SCOLORQ_ACCUMULATE_WGSL +
    // KMEANS_REDUCE_WGSL with annealing in the host loop, plus a final
    // KMEANS_ASSIGN dispatch for hard indices). Pin the basics: PNG
    // signature, palette size, no NaN-equivalent (single-index
    // collapse), index range.
    const r = await quantizeGpu(ramp(64, 64), {
      paletteSize: 16,
      dither: "scolorq",
    });
    for (let i = 0; i < PNG_SIG.length; i++) assertEquals(r.png[i], PNG_SIG[i]);
    assertEquals(r.meta.pipeline, "gpu");
    assert(r.png.byteLength > 100);
    const unique = new Set(r.indices);
    assert(unique.size > 1, `expected multi-index scolorq output, got ${unique.size}`);
    for (let i = 0; i < r.indices.length; i++) {
      assert(r.indices[i] < r.meta.paletteSize, `index ${r.indices[i]} out of range`);
    }
  },
});

Deno.test({
  name: "GPU: quantize(mode=auto) picks GPU when adapter present",
  ...gpuOpts,
  fn: async () => {
    const r = await quantize(ramp(32, 32), { mode: "auto", paletteSize: 8, kmeansIterations: 3 });
    for (let i = 0; i < PNG_SIG.length; i++) assertEquals(r.png[i], PNG_SIG[i]);
    // E-009: pin which path actually ran. Earlier the test couldn't tell
    // whether auto picked GPU or silently fell to CPU; meta.pipeline now
    // makes the choice observable.
    assertEquals(r.meta.pipeline, "gpu");
  },
});

Deno.test({
  name: "GPU: output palette size is approximately the requested size",
  ...gpuOpts,
  fn: async () => {
    const r = await quantizeGpu(ramp(128, 128), { paletteSize: 32, kmeansIterations: 5 });
    assert(r.meta.paletteSize >= 24 && r.meta.paletteSize <= 32, `got ${r.meta.paletteSize}`);
    assertEquals(r.palette.length, r.meta.paletteSize * 3);
  },
});

Deno.test({
  name: "GPU: kmeansIterations=0 + dither=none does not collapse to palette[0]",
  ...gpuOpts,
  fn: async () => {
    // N-B-04: before the post-loop kmeansAssign dispatch was added,
    // dither="none" with iterations=0 read the zero-initialised indicesBuf
    // and mapped every pixel to palette[0]. A gradient with k=8 must hit
    // multiple slots to be useful.
    const r = await quantizeGpu(ramp(64, 64), {
      paletteSize: 8,
      kmeansIterations: 0,
      dither: "none",
    });
    const unique = new Set(r.indices);
    assert(unique.size > 1, `expected multi-index output, got ${unique.size} unique value(s)`);
  },
});

// NEW-E-014: dispose the shared device after smoke runs so test files loaded
// later (parity, leak) start from a clean cache instead of inheriting the
// pipelines + device this file warmed. Keeps per-file behaviour deterministic
// regardless of the order Deno discovers them in.
Deno.test({
  name: "GPU: teardown disposes shared device",
  ...gpuOpts,
  fn: () => {
    disposeSharedDevice();
  },
});

if (!hasGpu) skipWarn("GPU smoke test");
