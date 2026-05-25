# inksquid

WebGPU-first image color quantizer for the browser. Reduces an RGBA image to a PNG-8 (indexed-color) byte stream, without WASM and without legacy fallbacks.

- **Color space**: OkLab everywhere — k-means, palette selection, dither error all operate in a perceptually uniform space.
- **Palette**: Wu's 1991 algorithm (deterministic) seeds weighted k-means.
- **Importance weighting**: multi-scale Difference-of-Gaussians biases the palette toward detailed regions and softens dither in busy areas.
- **Dither**: `blue-noise` (default, GPU), `floyd-steinberg` (CPU), `scolorq` (CPU + GPU — best perceived quality on natural photos), or `none`.
- **Encoder**: hand-rolled PNG-8 chunks (CRC32 + zlib via `CompressionStream`).
- **Target**: latest Chrome. No WebGL2 / Canvas2D / WASM fallback. On Deno requires `--unstable-webgpu`.

## Install (via JSR)

```ts
import { quantize } from "jsr:@ys319/inksquid";
```

## Usage

```ts
import { quantize } from "jsr:@ys319/inksquid";

const bitmap = await createImageBitmap(file);
const result = await quantize(bitmap, {
  paletteSize: 128,         // 2..256, default 128 — see "Recommended settings" below
  dither: "blue-noise",     // "none" | "blue-noise" | "floyd-steinberg" | "scolorq"
  ditherStrength: 1.0,      // 0..1
  detailWeight: 0.5,        // 0..1, how much to bias toward DoG-detected detail
  kmeansIterations: 15,
  mode: "gpu",              // "gpu" (default), "cpu", "auto"
});

// result.png      — Uint8Array, a complete PNG-8 file
// result.preview  — ImageData, dimensions match input
// result.palette  — Uint8Array (RGB triples)
// result.indices  — Uint8Array, per-pixel palette index (length = w*h)
// result.meta     — { outputBytes, paletteSize, elapsedMs, pipeline, fallbackReason? }
//                   pipeline ∈ "cpu" | "cpu-tiled" | "gpu".
//                   fallbackReason is set when quantize() ran on CPU:
//                   "cpu-forced" (mode: "cpu"), or "no-gpu-adapter" /
//                   "gpu-buffer-overflow" (mode: "auto" fell back).
```

`mode: "gpu"` (default) throws `NoWebGPUError` when no adapter is available, or `GpuBufferOverflowError` when the input exceeds `device.limits.maxBufferSize` (≈ 4 GB / ~250 MP on M-series). `mode: "auto"` catches just those two errors and falls back to CPU; everything else (option-validation errors, unexpected GPU runtime errors) still propagates, so misconfigurations don't silently vanish into a CPU run. `mode: "cpu"` forces the CPU path. Both pipelines run the same algorithms and stay within sub-perceptual divergence (pinned by `src/api/parity.gpu.test.ts`); CPU is just slower on large images.

Above ~2 megapixels the CPU path tiles to keep working-set memory bounded. The tiled path downgrades `dither: "floyd-steinberg"` and `"scolorq"` to `"blue-noise"` (one-shot stderr warning): FS would reset error diffusion at every seam and scolorq's spatial filter doesn't tile cleanly.

Options are validated at the entry point: out-of-range or wrong-typed values (`paletteSize: 0`, `dither: "off"`, `detailWeight: 1.5`, etc.) throw `RangeError` / `TypeError` rather than silently clamping. Use `normalizeOptions(partial)` to apply the same validation ahead of the call.

## Recommended settings

The library ships **no presets** — every input is different. The table below is a starting point from real-photo testing:

| Goal | `paletteSize` | `dither` | Notes |
|---|---:|---|---|
| Size-optimised | **64** | `blue-noise` | Smallest acceptable PNG-8 for photos; visible quantization on large smooth gradients. |
| **Default** | **128** | `blue-noise` | Sweet spot — banding is gone on most content, file size stays compact. |
| High quality | **256** | `blue-noise` | Maximum palette; useful when the input has a wide hue range or subtle gradients. |
| Best perceived quality | **128** | `scolorq` | ~+2 dB PSNR / +0.05 SSIM over `blue-noise` on natural photos. ~5× slower on CPU; GPU runtime is comparable to baseline. |

- `ditherStrength: 1.0` is the default; lower it (e.g. `0.5`) if dither artefacts are visible at small palettes.
- `detailWeight: 0.5` biases the palette toward DoG-detected edges. Set to `0` for uniform weighting (faster, slightly worse on busy images).
- `kmeansIterations: 15` rarely needs tuning — centroids settle by ~10 iterations on most inputs.
- `signal: AbortSignal` cancels a long-running call. Polled at each pipeline-stage boundary, inside the k-means iter loop, and per tile in the tiled path, so abort latency is bounded by "one iter / one tile". A pre-aborted signal short-circuits before any work.

## Advanced exports

`quantize` is the entry point most callers should reach for. The package also exports lower-level pieces:

- `quantizeCpu(input, opts)` / `quantizeGpu(input, opts)` / `quantizeTiled(input, opts)` — pin a specific pipeline. Same `QuantizeOptions` / `QuantizeResult` shape; `meta.pipeline` reflects which one ran.
- `isWebGPUAvailable()` — async probe for UI gating before calling `quantize`.
- `disposeSharedDevice()` — eagerly destroy the process-wide cached `GPUDevice`. Idempotent; the next call that needs the GPU lazily re-acquires. Don't call this while a `quantize` is in flight.
- `normalizeOptions(partial)` — entry-point validation as a pre-flight check.
- `DEFAULT_OPTIONS` — the readonly default option record.
- `NoWebGPUError` / `GpuBufferOverflowError` — the error classes `mode: "auto"` catches when falling back to CPU.

These are part of the public surface and follow the same versioning as `quantize`. Anything not re-exported from `src/mod.ts` is library-internal and may change between patch releases.

Sub-module entry points for callers who only need a single piece:

- `jsr:@ys319/inksquid/png` — `encodePng8` and the byte-level helpers (`concatBytes`, `chunk`).
- `jsr:@ys319/inksquid/oklab` — pure-CPU OkLab ↔ linear-RGB / sRGB converters.
- `jsr:@ys319/inksquid/blue-noise` — standalone blue-noise dither.

## Architecture

See [`docs/architecture.md`](./docs/architecture.md) for the pipeline layout, file/module structure, scolorq mechanism, and CPU/GPU stage mapping. See [`docs/parity-test.md`](./docs/parity-test.md) for the CPU↔GPU parity test thresholds and headroom rationale.

## Development

```sh
deno task test          # full suite (CPU + GPU if an adapter is present)
deno task test:cpu      # CPU-only (no WebGPU permission required)
deno task test:gpu      # *.gpu.test.ts only (auto-skips without an adapter)
deno task check         # type-check the public API
deno task dry-publish   # simulate JSR publish
```

Tests are colocated with their source as `<name>.test.ts` (CPU) / `<name>.gpu.test.ts` (WebGPU).

```sh
deno task demo          # bundle + static server on http://localhost:8000
deno task bundle        # one-shot bundle
deno task serve         # static server (correct MIME types for unbundled .ts)
```

## License

MIT.
