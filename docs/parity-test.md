# CPU/GPU parity test

`src/api/parity.gpu.test.ts` pins the divergence between the CPU and GPU
pipelines to measured-value floors. A WebGPU adapter is required; the
suite auto-skips when one isn't available.

## Floors (calibrated against measured values, then tightened)

| Name | Input / settings | PSNR floor | idx match floor | palette sRGB dist ceiling | Measured |
|---|---|---|---|---|---|
| `gradient/256/p64/iter15/blue-noise` | 256×256 gradient, p64, blue-noise | 44 dB | 95 % | 0.15 | 46.83 / 99.94 % / 0.038 |
| `noise/256/p64/iter15/blue-noise` | 256×256 photoNoise, p64, blue-noise | 60 dB | 99 % | 0.01 | ∞ / 100 % / 0.000 |
| `geometric/256/p16/iter15/none` | 256×256 geometric, p16, none | 60 dB | 99 % | 0.01 | ∞ / 100 % / 0.000 |
| `gradient/128/p8/iter0/none` | 128×128 gradient, p8, iter=0, none (N-B-04 regression guard) | 60 dB | 99 % | 0.01 | ∞ / 100 % / 0.000 |
| `gradient/128/p16/scolorq` | 128×128 gradient, p16, scolorq | 50 dB | 95 % | 0.2 | 64.18 / 100 % / 0.063 |

Plus `PNG size ratio < 1.30` (ratio between the CPU and GPU PNG output
sizes).

## What each floor measures

- **PSNR**: between the CPU preview and the GPU preview, computed on
  sRGB bytes. Bit-exact cases hit `Infinity`; the only thing that
  drives PSNR below it is f32 (GPU) vs Float64 (CPU) accumulator-order
  divergence.
- **idx match rate**: Wu init is shared between pipelines, so slot
  *j* on CPU and slot *j* on GPU are the same starting point. A
  byte-exact comparison of the two index streams measures "did the
  two pipelines pick the same palette entry?".
- **palette sRGB distance**: mean per-slot Euclidean distance in sRGB
  byte space between corresponding palette slots (not CIE ΔE). Both
  pipelines start from the same Wu init, so any drift reflects
  k-means accumulator ordering — exactly the f32-vs-Float64 noise
  this test is built to catch.
- **PNG size ratio**: a meaningfully different index distribution
  (≈ 1.3× or more difference in compressed size) suggests the
  pipelines are picking different palette entries with noticeably
  different frequency, which deserves a separate look.

## How headroom is sized

- **Bit-exact cases (noise / geometric / iter0)**: ~30× headroom on
  the palette distance ceiling (0.01 vs measured 0.000) absorbs f32
  ordering noise on adapters other than the one used for calibration.
- **Gradient blue-noise**: 2.83 dB PSNR + 4.94 pp idx + ~0.11 palette
  dist of headroom. Measurement was deterministic on the calibration
  adapter, but Vulkan / D3D12 backends may show different f32
  ordering, so the floors keep cross-machine slack.
- **Scolorq**: 14 dB PSNR / 5 pp idx / ~3× palette dist of headroom
  (measured 64.18 / 100 % / 0.063 vs floors 50 / 95 % / 0.2).

Every test prints `[parity:<name>] PSNR ..., idx ...%, palette sRGB
dist ...` to stdout on every run, so a regression shows up as a
drifting margin before it crosses a floor.

## N-B-04 regression guard

With `kmeansIterations: 0` and `dither: "none"`, both pipelines skip
the iter loop and run the post-loop assign against the Wu-init
palette. Pre-fix, the GPU produced all-zero indices and the CPU
produced a different all-zero stream (W-A-4 / N-B-04). The
`gradient/128/p8/iter0/none` case locks the fix in at 60 dB / 99 % /
0.01.

## Asymmetry note on `kmeansIterations`

The CPU loop terminates early when `moved < 1e-7`; the GPU loop runs
all iterations unconditionally. This asymmetry is benign: both
pipelines run a final `assignNearestOklab` after the loop so the
returned `indices` always match the final centroids (rather than
lagging by one update step). Together with the shared Wu init, the
remaining divergence is entirely f32-vs-Float64 accumulator ordering.
