# Architecture

## Pipeline

```
[Image] → OkLab (GPU) → DoG importance (GPU) → Wu init (CPU)
        → k-means iter loop:
             assign (GPU, kNN, workgroup_size=256)
             accumulate (GPU, workgroup-local u32 atomic → non-atomic intermediate buffer)
             reduce (GPU, per-centroid linear scan, atomic-free)
        → dither: blue-noise (GPU) / floyd-steinberg (CPU) / scolorq (CPU+GPU) / none
        → preview render (CPU) + PNG-8 encode (CPU)
```

The CPU pipeline mirrors every GPU stage 1:1 (for SSR / sandboxed
environments). CPU output stays within sub-perceptual distance of the GPU
output; the numerical guarantees are in [`./parity-test.md`](./parity-test.md).

Above ~2 megapixels the CPU path automatically switches to a tiled
variant to bound the working set. The tiled path force-downgrades
`dither: "floyd-steinberg"` and `dither: "scolorq"` to `"blue-noise"`
(with a single stderr warning): FS can't carry residual error across
tile seams, and scolorq's soft-assignment matrix and spatial filter
can't be split tile-cleanly.

## k-means dead-cluster reseeding

When a centroid finishes an iteration with `sumW == 0` ("dead"), it
would otherwise waste a palette slot for the rest of the run. The CPU
path reseeds dead clusters onto the iteration's highest-residual pixel
(i.e. the pixel the current palette serves worst). One reseed per
iteration: multiple dead clusters reseeded to the same worst pixel
would collapse them onto the same colour. The reseed counts as a
discrete jump, not a gradient-descent step, so it isn't accumulated
into `moved` and the convergence break is suppressed for that
iteration — the freshly seeded centroid gets at least one iteration to
attract pixels.

## scolorq

Soft k-means with deterministic annealing, inspired by Puzicha 1998's
spatial colour quantization.

```
soft-assign:  p_ij = softmax_j(-d(x_i, c_j) / T)   // numerically stable
centroid:     c_j = Σ_i p_ij · x_i / Σ_i p_ij
T schedule:   geometric, T0 = 0.001 → Tf = 0.00001, 15 sweeps
final assign: hard nearest (assignNearestOklab)
```

**At `T0 = 0.01` the algorithm mode-collapses.** `exp(-d/T)` is small per
pixel, but summed across 100k+ majority-colour pixels it dominates the
few high-weight minority pixels and pulls minority-hue centroids
toward the majority cluster. `T0 = 0.001` keeps the softmax in the
sharp regime — each pixel distributes mass over only its 2-3 nearest
centroids — and avoids the collapse. See the header comment in
`src/palette/scolorq.ts` for the full mechanism.

The GPU port (`src/palette/scolorq.wgsl.ts`) is a soft-accumulate
shader that reuses k-means's reduce pass and the existing assign
shader. The final hard-assignment readout runs on the GPU.

## File layout

```
src/
  api/
    quantize.ts                       # mode dispatch
    quantize-cpu.ts                   # CPU pipeline
    quantize-gpu.ts                   # GPU pipeline
    quantize-tiled.ts                 # CPU tiled (> 2 MP)
    abort.test.ts
    auto-fallback.test.ts
    parity.gpu.test.ts                # CPU↔GPU parity (measured-value floors)
    png-roundtrip.test.ts
    quantize.test.ts
    quantize-gpu.gpu.test.ts          # smoke
    quantize-gpu-leak.gpu.test.ts     # device/buffer leak
    quantize-tiled.test.ts
  colorspace/
    oklab.ts, oklab.wgsl.ts, oklab.test.ts
  core/
    device.ts, types.ts, device.test.ts, options-validation.test.ts
  detail/
    dog.ts, dog.wgsl.ts, dog.test.ts
  dither/
    blue-noise.ts, blue-noise.wgsl.ts, blue-noise-data.ts, blue-noise.test.ts
    floyd-steinberg.ts, floyd-steinberg.test.ts
    void-and-cluster.ts               # mask generator (build-time only)
  encode-png/
    crc32.ts, chunk.ts, deflate.ts, mod.ts
    crc32.test.ts, chunk.test.ts, encode-png.test.ts
  palette/
    wu.ts, kmeans.ts, kmeans.wgsl.ts
    scolorq.ts, scolorq.wgsl.ts
    wu.test.ts, kmeans.test.ts, scolorq.test.ts
  _test-fixtures.ts                   # ramp / flat / gradient / photoNoise /
                                      # geometric / skipWarn (publish exclude)
  mod.ts
examples/
  browser.html, serve.ts, bundle.ts
  inksquid.bundle.js(.map)            # gitignored (output of `deno task bundle`)
scripts/
  generate-blue-noise.ts              # regenerate blue-noise-data.ts (rare)
```

## Sub-module exports

- `jsr:@ys319/inksquid/png` — `encodePng8` + byte-level helpers
  (`chunk`, `concatBytes`).
- `jsr:@ys319/inksquid/oklab` — pure-CPU OkLab ↔ linear-RGB / sRGB
  converters.
- `jsr:@ys319/inksquid/blue-noise` — standalone blue-noise dither.

Each entry resolves to `src/mod.ts` / `src/encode-png/mod.ts` /
`src/colorspace/oklab.ts` / `src/dither/blue-noise.ts` respectively.
