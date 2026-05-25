# inksquid — Claude project notes

WebGPU-first image color quantizer. RGBA → PNG-8, with a CPU mirror for SSR /
sandboxed environments. Dither modes: `blue-noise` / `floyd-steinberg` /
`scolorq` / `none`.

## Public API (the minimum you need to know)

```ts
quantize(img, {
  mode,             // "gpu" | "cpu" | "auto" (default "gpu")
  paletteSize,      // 2..256, default 128
  dither,           // "blue-noise" | "floyd-steinberg" | "scolorq" | "none"
                    //   default "blue-noise"
  ditherStrength,   // 0..1, default 1.0
  detailWeight,     // 0..1, default 0.5  (DoG importance ↔ k-means weight)
  kmeansIterations, // default 15
  signal,           // optional AbortSignal
});
```

Use `quantizeCpu` / `quantizeGpu` / `quantizeTiled` to pin a specific pipeline.
`mode: "auto"` falls back to CPU only on `NoWebGPUError` and
`GpuBufferOverflowError`; option-validation errors still throw.

For the pipeline layout and CPU/GPU stage mapping see
[`docs/architecture.md`](./docs/architecture.md); for the CPU↔GPU parity
test floors and the rationale behind them see
[`docs/parity-test.md`](./docs/parity-test.md).

## Dev tasks

```sh
deno task test           # full suite (--unstable-webgpu; GPU tests self-skip)
deno task test:cpu       # CPU-only (no WebGPU permission needed)
deno task test:gpu       # *.gpu.test.ts only
deno task bundle         # rebuild examples/inksquid.bundle.js
deno task demo           # bundle + static server on http://localhost:8000
deno task check          # type-check src/mod.ts
deno task fmt / lint
deno task dry-publish    # JSR publish dry-run
```

Tests are colocated next to source: `<name>.test.ts` (CPU) and
`<name>.gpu.test.ts` (requires a WebGPU adapter; auto-skips otherwise).
Shared fixtures live in `src/_test-fixtures.ts` and are excluded from the
JSR publish.

## Traps to avoid

- **Don't reintroduce global atomics in k-means.** The current tree-reduction
  path is the stable one; the earlier workgroup-local + global f32 CAS
  variant collapsed under contention on smooth images (ΔE 60).
- **Rebuild the bundle after WGSL changes.** Run `deno task bundle` — the
  browser demo loads from the bundle, not from source.
- **`scolorq`'s `initialTemp` must stay at 0.001.** At 0.01 the soft-mean
  centroid update lets majority-colour pixels pull minority-hue centroids
  inward (mode collapse). See the header comment in
  `src/palette/scolorq.ts` for the full diagnosis.
- **`DEFAULT_OPTIONS.paletteSize` is 128** (it used to be 256).
- **The tiled CPU path downgrades `dither: "floyd-steinberg"` and
  `"scolorq"` to `"blue-noise"`** (with a single stderr warning). FS can't
  carry residual error across tile seams; scolorq's soft-assignment matrix
  and spatial filter don't tile cleanly either.
- **Adding `npm:` / `jsr:` imports may fail under sandboxed network.**
  `examples/bundle.ts` uses `npm:esbuild@^0.24.0`, but that task only runs
  in the user's environment.

## scripts/

`scripts/generate-blue-noise.ts` regenerates the baked 64×64 mask in
`src/dither/blue-noise-data.ts` via void-and-cluster (deterministic for a
given seed). The data is checked in; the script is only needed when you
want to re-seed or resize the mask.
