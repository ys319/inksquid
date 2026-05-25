// GPU-driven quantization pipeline.
//
// Stages (all WebGPU compute):
//   1. RGBA u8 → OkLab vec4<f32>          (colorspace/oklab.wgsl)
//   2. Extract L channel                  (detail/dog.wgsl: EXTRACT_L)
//   3. Blur L at σ={1,2,4}                (detail/dog.wgsl: BLUR_H, BLUR_V)
//   4. DoG combine                        (detail/dog.wgsl: DOG_COMBINE)
//   5. k-means iterations (assign/accumulate/finalize)
//                                         (palette/kmeans.wgsl)
//   6. Blue-noise dither + assignment     (dither/blue-noise.wgsl)
//
// CPU-only stages: Wu init (small histogram, see HANDOFF Tips), Floyd-Steinberg
// dither (sequential), PNG-8 encode (chunk assembly + DEFLATE via stream),
// preview RGBA (palette splat — readback dominates a hypothetical GPU pass).
//
// Size limit: this single-pass implementation requires the full OkLab buffer
// (n × 16 bytes) to fit within device.limits.maxBufferSize. Larger inputs
// throw; route them through the CPU tiled path instead (quantize() with
// mode "cpu" or "auto" picks tiled automatically above ~2 MP).

import { OKLAB_FORWARD_SHADER } from "../colorspace/oklab.wgsl.ts";
import {
  BLUR_HORIZONTAL_WGSL,
  BLUR_VERTICAL_WGSL,
  DOG_COMBINE_WGSL,
  EXTRACT_L_WGSL,
} from "../detail/dog.wgsl.ts";
import { gaussianKernel1D } from "../detail/dog.ts";
import {
  KMEANS_ACCUMULATE_WGSL,
  KMEANS_ASSIGN_WGSL,
  KMEANS_REDUCE_WGSL,
  KMEANS_WG_SIZE,
} from "../palette/kmeans.wgsl.ts";
import { BLUE_NOISE_WGSL } from "../dither/blue-noise.wgsl.ts";
import { BLUE_NOISE_64_SIZE, getBlueNoise64 } from "../dither/blue-noise-data.ts";
import { getSharedDevice, GpuBufferOverflowError, onSharedDeviceDispose } from "../core/device.ts";
import { oklabPaletteToSrgb, wuQuantizeOklab } from "../palette/wu.ts";
import { floydSteinberg } from "../dither/floyd-steinberg.ts";
import { SCOLORQ_ACCUMULATE_WGSL } from "../palette/scolorq.wgsl.ts";
import { encodePng8 } from "../encode-png/mod.ts";
import {
  normalizeOptions,
  type QuantizeOptions,
  type QuantizeResult,
  type RawImage,
  toRawImage,
} from "../core/types.ts";

// Pack RGBA bytes into u32 in the order R | G<<8 | B<<16 | A<<24. This is
// little-endian byte order: when the GPU reads the u32 as bytes, byte 0 is R.
// The matching WGSL unpack uses `(packed >> 0) & 0xff` for R, `(>> 8) & 0xff`
// for G, etc. Keep both ends in sync — flipping endianness in either place
// produces a fully colour-corrupted output.
function packRgba(data: Uint8ClampedArray): Uint32Array {
  const out = new Uint32Array(data.length / 4);
  for (let i = 0; i < out.length; i++) {
    out[i] = data[i * 4] | (data[i * 4 + 1] << 8) | (data[i * 4 + 2] << 16) |
      (data[i * 4 + 3] << 24);
  }
  return out;
}

// WGSL vec4<f32> is 16-byte aligned, so the GPU centroid buffer pads each
// (L, a, b) triple with a zero. CPU consumers (floydSteinberg, wu palette
// helpers) want a tightly packed (L, a, b) stride-3 array. Single helper
// so the conversion isn't open-coded at every readback site.
function centroidsVec4ToVec3(src: Float32Array, count: number): Float32Array {
  const out = new Float32Array(count * 3);
  for (let j = 0; j < count; j++) {
    out[j * 3] = src[j * 4];
    out[j * 3 + 1] = src[j * 4 + 1];
    out[j * 3 + 2] = src[j * 4 + 2];
  }
  return out;
}

function packBlueNoise(): Uint32Array {
  const m = getBlueNoise64();
  const u32 = new Uint32Array(Math.ceil(m.length / 4));
  for (let i = 0; i < m.length; i++) {
    u32[i >> 2] |= m[i] << ((i & 3) * 8);
  }
  return u32;
}

interface Pipelines {
  oklabForward: GPUComputePipeline;
  extractL: GPUComputePipeline;
  blurH: GPUComputePipeline;
  blurV: GPUComputePipeline;
  dogCombine: GPUComputePipeline;
  kmeansAssign: GPUComputePipeline;
  kmeansAccumulate: GPUComputePipeline;
  kmeansReduce: GPUComputePipeline;
  scolorqAccumulate: GPUComputePipeline;
  blueNoise: GPUComputePipeline;
}

function buildPipelines(device: GPUDevice): Pipelines {
  const mk = (code: string) => {
    const module = device.createShaderModule({ code });
    return device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  };
  return {
    oklabForward: mk(OKLAB_FORWARD_SHADER),
    extractL: mk(EXTRACT_L_WGSL),
    blurH: mk(BLUR_HORIZONTAL_WGSL),
    blurV: mk(BLUR_VERTICAL_WGSL),
    dogCombine: mk(DOG_COMBINE_WGSL),
    kmeansAssign: mk(KMEANS_ASSIGN_WGSL),
    kmeansAccumulate: mk(KMEANS_ACCUMULATE_WGSL),
    kmeansReduce: mk(KMEANS_REDUCE_WGSL),
    scolorqAccumulate: mk(SCOLORQ_ACCUMULATE_WGSL),
    blueNoise: mk(BLUE_NOISE_WGSL),
  };
}

// Process-wide pipeline cache. Pipelines are tied to a specific device
// (compileShaderModule + createComputePipeline both produce device-owned
// resources), so we key by device identity and rebuild on any change.
// The disposal hook below clears this when `disposeSharedDevice` runs or
// `device.lost` resolves, so we don't hold dangling references to a
// destroyed device. Without this cache, every quantize() call spent
// ~5-10 ms recompiling 9 shader modules even though the device was shared.
let _cachedPipelines: { device: GPUDevice; pipelines: Pipelines } | null = null;

onSharedDeviceDispose(() => {
  _cachedPipelines = null;
});

/**
 * Resolve the per-device pipeline set, building once and caching by
 * device identity. Each `quantizeGpu` call holds the returned `pipelines`
 * reference for its entire duration in a local closure, so even if
 * `device.lost` resolves mid-call and the disposal hook nulls
 * `_cachedPipelines`, the in-flight call keeps using the old pipelines
 * (which fail at the GPU queue submit instead). Re-entry into this
 * function after a device-lost event is therefore safely served by a
 * fresh build against the new device on the next `quantize{Gpu}` call.
 */
function getSharedPipelines(device: GPUDevice): Pipelines {
  if (_cachedPipelines && _cachedPipelines.device === device) {
    return _cachedPipelines.pipelines;
  }
  _cachedPipelines = { device, pipelines: buildPipelines(device) };
  return _cachedPipelines.pipelines;
}

function dispatch2D(
  pass: GPUComputePassEncoder,
  width: number,
  height: number,
  wgX = 16,
  wgY = 16,
) {
  pass.dispatchWorkgroups(Math.ceil(width / wgX), Math.ceil(height / wgY));
}

function dispatch1D(pass: GPUComputePassEncoder, n: number, wgX = 64) {
  pass.dispatchWorkgroups(Math.ceil(n / wgX));
}

// Workaround for TS 5.7+ Uint8Array<ArrayBufferLike> tightening when calling
// writeBuffer; the runtime accepts any BufferSource, the assertion just
// re-widens after TS narrowed the input arg.
function writeBuf(
  device: GPUDevice,
  buf: GPUBuffer,
  offset: number,
  data: ArrayBufferView | ArrayBuffer,
) {
  device.queue.writeBuffer(buf, offset, data as BufferSource);
}

// The dst buffer's lifecycle is self-contained: created, mapped, copied
// out, unmapped, and destroyed all within this function. It deliberately
// does *not* go through the caller's `toFree[]` tracker because nothing
// outside this function ever holds a reference to it. The `src` buffer is
// the one the caller owns and tracks.
//
// 4A-02 (4th-pass review): on `mapAsync` rejection (e.g. `device.lost`
// resolved between submit and map) the previous code returned without
// destroying `dst` — a small but real leak. The try/finally below
// guarantees `dst.destroy()` regardless of mapAsync outcome. The
// `mapped` flag avoids calling `unmap()` on an unmapped buffer (which
// itself throws a validation error and would mask the original
// rejection reason).
async function readBuffer(
  device: GPUDevice,
  src: GPUBuffer,
  byteLength: number,
): Promise<ArrayBuffer> {
  const dst = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, 0, dst, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    mapped = true;
    return dst.getMappedRange().slice(0);
  } finally {
    if (mapped) dst.unmap();
    dst.destroy();
  }
}

/**
 * WebGPU quantization pipeline. The fast path on adapters that support
 * the algorithm (≈ everything from Apple M-series, recent NVIDIA, modern
 * Intel iGPU and Vulkan/D3D12 backends). On a 2-megapixel image, expect
 * roughly an order of magnitude speedup over {@link quantizeCpu}.
 *
 * Throws {@link NoWebGPUError} when no adapter is available, or
 * {@link GpuBufferOverflowError} when the input is too large for a
 * single OkLab buffer (`width * height * 16` bytes exceeds
 * `device.limits.maxBufferSize` — ~250 megapixels on M-series).
 * `quantize({ mode: "auto" })` catches both and falls back to CPU
 * transparently.
 */
export async function quantizeGpu(
  input: ImageBitmap | ImageData | RawImage,
  optionsIn: Partial<QuantizeOptions> = {},
): Promise<QuantizeResult> {
  const options = normalizeOptions(optionsIn);
  // A-e: cancel checkpoints before/after each await boundary. Per-pixel
  // GPU dispatch encoding isn't polled — work-per-iter is small (~ms)
  // and polling inside encoder construction would add measurable
  // overhead. Latency is bounded by "one k-means iter / one readback".
  options.signal?.throwIfAborted();
  const di = await getSharedDevice();
  options.signal?.throwIfAborted();
  const { device, limits } = di;
  const t0 = performance.now();

  // Buffers tracked here are destroyed in the trailing `finally` block,
  // including on error paths. Without this, repeated calls leak (the
  // intermediate buffer alone is ~130 MB at 4K with k=256).
  const toFree: GPUBuffer[] = [];
  const trackBuf = (b: GPUBuffer): GPUBuffer => {
    toFree.push(b);
    return b;
  };

  try {
    const img = toRawImage(input);
    const { width, height, data } = img;
    const n = width * height;

    // Pessimistic budget for vec4<f32>: 16 bytes/pixel.
    if (n * 16 > limits.maxBufferSize) {
      throw new GpuBufferOverflowError(n * 16, limits.maxBufferSize);
    }

    const pipelines = getSharedPipelines(device);

    // === Buffers ===
    const rgbaPacked = packRgba(data);
    const rgbaBuf = trackBuf(device.createBuffer({
      size: rgbaPacked.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }));
    writeBuf(device, rgbaBuf, 0, rgbaPacked);

    const oklabBuf = trackBuf(device.createBuffer({
      size: n * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    }));

    // OkLab forward.
    {
      const dims = new Uint32Array([width, height]);
      const dimsBuf = trackBuf(device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }));
      writeBuf(device, dimsBuf, 0, dims);
      const bind = device.createBindGroup({
        layout: pipelines.oklabForward.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: rgbaBuf } },
          { binding: 1, resource: { buffer: oklabBuf } },
          { binding: 2, resource: { buffer: dimsBuf } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipelines.oklabForward);
      pass.setBindGroup(0, bind);
      dispatch2D(pass, width, height);
      pass.end();
      device.queue.submit([enc.finish()]);
    }

    // DoG importance.
    let importanceBuf: GPUBuffer | null = null;
    if (options.detailWeight > 0) {
      const Lbuf = trackBuf(device.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE,
      }));
      // Extract L.
      {
        const dims = new Uint32Array([width, height]);
        const dimsBuf = trackBuf(device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }));
        writeBuf(device, dimsBuf, 0, dims);
        const bind = device.createBindGroup({
          layout: pipelines.extractL.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: oklabBuf } },
            { binding: 1, resource: { buffer: Lbuf } },
            { binding: 2, resource: { buffer: dimsBuf } },
          ],
        });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipelines.extractL);
        pass.setBindGroup(0, bind);
        dispatch2D(pass, width, height);
        pass.end();
        device.queue.submit([enc.finish()]);
      }
      // Blur at sigmas {1, 2, 4}.
      const sigmas = [1, 2, 4];
      const blurredBufs: GPUBuffer[] = [Lbuf];
      for (const sigma of sigmas) {
        const { kernel, radius } = gaussianKernel1D(sigma);
        const kernelBuf = trackBuf(device.createBuffer({
          size: Math.ceil(kernel.byteLength / 16) * 16,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }));
        writeBuf(device, kernelBuf, 0, kernel);
        const dimsBuf = trackBuf(device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }));
        writeBuf(device, dimsBuf, 0, new Uint32Array([width, height, radius, 0]));
        const tempBuf = trackBuf(
          device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE }),
        );
        const outBuf = trackBuf(
          device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE }),
        );
        const bindH = device.createBindGroup({
          layout: pipelines.blurH.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: Lbuf } },
            { binding: 1, resource: { buffer: tempBuf } },
            { binding: 2, resource: { buffer: kernelBuf } },
            { binding: 3, resource: { buffer: dimsBuf } },
          ],
        });
        const bindV = device.createBindGroup({
          layout: pipelines.blurV.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: tempBuf } },
            { binding: 1, resource: { buffer: outBuf } },
            { binding: 2, resource: { buffer: kernelBuf } },
            { binding: 3, resource: { buffer: dimsBuf } },
          ],
        });
        const enc = device.createCommandEncoder();
        const passH = enc.beginComputePass();
        passH.setPipeline(pipelines.blurH);
        passH.setBindGroup(0, bindH);
        dispatch2D(passH, width, height);
        passH.end();
        const passV = enc.beginComputePass();
        passV.setPipeline(pipelines.blurV);
        passV.setBindGroup(0, bindV);
        dispatch2D(passV, width, height);
        passV.end();
        device.queue.submit([enc.finish()]);
        blurredBufs.push(outBuf);
      }
      importanceBuf = trackBuf(device.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }));
      const dimsBuf = trackBuf(device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }));
      const dims = new ArrayBuffer(16);
      new Uint32Array(dims, 0, 2).set([width, height]);
      new Float32Array(dims, 8, 1)[0] = 6;
      writeBuf(device, dimsBuf, 0, dims);
      const bind = device.createBindGroup({
        layout: pipelines.dogCombine.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: blurredBufs[0] } },
          { binding: 1, resource: { buffer: blurredBufs[1] } },
          { binding: 2, resource: { buffer: blurredBufs[2] } },
          { binding: 3, resource: { buffer: blurredBufs[3] } },
          { binding: 4, resource: { buffer: importanceBuf } },
          { binding: 5, resource: { buffer: dimsBuf } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipelines.dogCombine);
      pass.setBindGroup(0, bind);
      dispatch2D(pass, width, height);
      pass.end();
      device.queue.submit([enc.finish()]);
    }

    // Read OkLab back to CPU for Wu init (small histogram, CPU-faster than GPU).
    const oklabArr = new Float32Array(await readBuffer(device, oklabBuf, n * 16));
    // Read importance back (or synthesize ones).
    const importanceArr = importanceBuf
      ? new Float32Array(await readBuffer(device, importanceBuf, n * 4))
      : null;

    const weights = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const alpha = oklabArr[i * 4 + 3];
      if (alpha <= 0) {
        weights[i] = 0;
        continue;
      }
      if (importanceArr) {
        const w = 1 - options.detailWeight + options.detailWeight * (importanceArr[i] * 4 + 0.25);
        weights[i] = alpha * w;
      } else {
        weights[i] = alpha;
      }
    }
    const wu = wuQuantizeOklab({ oklab: oklabArr, weights, paletteSize: options.paletteSize });

    // Upload weights and initial centroids.
    const weightsBuf = trackBuf(device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }));
    writeBuf(device, weightsBuf, 0, weights);
    const k = wu.count;
    const centroidsBuf = trackBuf(device.createBuffer({
      size: k * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }));
    {
      const init = new Float32Array(k * 4);
      for (let j = 0; j < k; j++) {
        init[j * 4] = wu.oklab[j * 3];
        init[j * 4 + 1] = wu.oklab[j * 3 + 1];
        init[j * 4 + 2] = wu.oklab[j * 3 + 2];
      }
      writeBuf(device, centroidsBuf, 0, init);
    }
    const indicesBuf = trackBuf(device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    }));
    // K-means: GPU-only pipeline using two-pass tree reduction (no atomics).
    //
    //   Pass 1 (assign):       per-pixel → nearest centroid → indices[i]
    //   Pass 2 (accumulate):   workgroup-local sum + non-atomic flush to
    //                          intermediate[wg * k + c]
    //   Pass 3 (reduce):       per-centroid scan across workgroups → new centroid
    //
    // Earlier attempts used global atomics (per-pixel f32 CAS, then
    // workgroup-local with global f32 CAS for the flush). Both produced wrong
    // centroids on smooth images where many pixels in a workgroup share the
    // same target centroid. Tree reduction removes the atomic correctness
    // question entirely while keeping all work on GPU.
    const num_workgroups = Math.ceil(n / KMEANS_WG_SIZE);
    const intermediateBuf = trackBuf(device.createBuffer({
      size: num_workgroups * k * 4 * 4,
      usage: GPUBufferUsage.STORAGE,
    }));

    const dimsAssign = trackBuf(device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }));
    writeBuf(device, dimsAssign, 0, new Uint32Array([n, k, 0, 0]));

    const dimsAcc = trackBuf(device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }));
    writeBuf(device, dimsAcc, 0, new Uint32Array([n, k, 0, 0]));

    const dimsReduce = trackBuf(device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }));
    writeBuf(device, dimsReduce, 0, new Uint32Array([k, num_workgroups, 0, 0]));

    const assignBind = device.createBindGroup({
      layout: pipelines.kmeansAssign.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: oklabBuf } },
        { binding: 1, resource: { buffer: centroidsBuf } },
        { binding: 2, resource: { buffer: indicesBuf } },
        { binding: 3, resource: { buffer: dimsAssign } },
      ],
    });
    const accBind = device.createBindGroup({
      layout: pipelines.kmeansAccumulate.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: oklabBuf } },
        { binding: 1, resource: { buffer: weightsBuf } },
        { binding: 2, resource: { buffer: indicesBuf } },
        { binding: 3, resource: { buffer: intermediateBuf } },
        { binding: 4, resource: { buffer: dimsAcc } },
      ],
    });
    const reduceBind = device.createBindGroup({
      layout: pipelines.kmeansReduce.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: intermediateBuf } },
        { binding: 1, resource: { buffer: centroidsBuf } },
        { binding: 2, resource: { buffer: dimsReduce } },
      ],
    });

    // The CPU reference path (src/palette/kmeans.ts) breaks out of this loop
    // once total centroid movement drops below 1e-7. We don't replicate that
    // here because querying centroid drift requires a readback per iteration
    // (the dominant cost of doing so would dwarf the savings), so the GPU
    // always runs all `kmeansIterations`. This is one source of the residual
    // CPU/GPU mismatch tracked by tests/gpu/parity.test.ts on smooth inputs;
    // the post-loop assign below realigns `indicesBuf` with the final
    // centroids so the off-by-one tracked as W-A-4 is closed for the
    // dither="none" path (FS/blue-noise overwrite indices themselves).
    // See STATUS.md "parity" section for the broader picture.
    //
    // CPU-only: dead-cluster re-seeding (Phase 3.1 / 5th-pass). The CPU
    // path re-seeds clusters with sumW=0 onto the worst-residual pixel
    // before the next iter, improving output on inputs where Wu init
    // overpopulates the palette (icons, low-colour graphics). The GPU
    // mirror would need a per-iter readback to identify dead clusters
    // and a tree-reduction over per-pixel residuals to find the worst
    // one — significant WGSL + host orchestration for a niche case
    // that natural photos never trigger (Wu+kmeans on Kodak fixtures
    // never produces a dead cluster at any paletteSize). Asymmetry is
    // documented; parity test thresholds aren't affected because all
    // parity cases use rich-colour inputs.
    if (options.dither === "scolorq") {
      // Phase C: GPU scolorq — soft k-means + geometric annealing,
      // mirroring src/palette/scolorq.ts. Per sweep we dispatch the
      // soft-accumulate shader (replaces both kmeansAssign and
      // kmeansAccumulate) and re-use kmeansReduce unchanged. The
      // host loop here owns the temperature schedule; each sweep
      // re-binds a new Params uniform with the inverse temperature
      // for that sweep. After annealing terminates, one
      // kmeansAssign dispatch produces the final hard indices that
      // feed the PNG-8 encode.
      //
      // Defaults match the CPU path (`src/palette/scolorq.ts`):
      // T0=0.001, Tf=0.00001, 15 sweeps. At T=0.001 the softmax is
      // sharply concentrated on each pixel's 2-3 nearest centroids;
      // distant centroids contribute negligible weight to the
      // per-pixel assignment AND receive negligible cumulative
      // pull from distant pixels' centroid updates — the
      // mode-collapse-toward-majority-colour effect that an earlier
      // T0=0.01 produced is suppressed. See scolorq.ts file header.
      const T0 = 0.001;
      const Tf = 0.00001;
      const totalSweeps = 15;
      const alpha = Math.pow(Tf / T0, 1 / totalSweeps);

      // Params uniform: { Tinv: f32 }, 4 bytes. Pad to 16 for safety
      // — Dawn/wgpu honour scalar uniforms of any size but some
      // bind-group validators are stricter. 16-byte pad costs
      // nothing.
      const paramsBuf = trackBuf(device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }));

      const scolorqAccBind = device.createBindGroup({
        layout: pipelines.scolorqAccumulate.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: oklabBuf } },
          { binding: 1, resource: { buffer: weightsBuf } },
          { binding: 2, resource: { buffer: centroidsBuf } },
          { binding: 3, resource: { buffer: intermediateBuf } },
          { binding: 4, resource: { buffer: dimsAcc } },
          { binding: 5, resource: { buffer: paramsBuf } },
        ],
      });

      let T = T0;
      for (let sweep = 0; sweep < totalSweeps; sweep++) {
        options.signal?.throwIfAborted();
        // Update Params.Tinv for this sweep.
        writeBuf(device, paramsBuf, 0, new Float32Array([1 / T, 0, 0, 0]));
        const enc = device.createCommandEncoder();
        let pass = enc.beginComputePass();
        pass.setPipeline(pipelines.scolorqAccumulate);
        pass.setBindGroup(0, scolorqAccBind);
        dispatch1D(pass, n, KMEANS_WG_SIZE);
        pass.end();
        pass = enc.beginComputePass();
        pass.setPipeline(pipelines.kmeansReduce);
        pass.setBindGroup(0, reduceBind);
        dispatch1D(pass, k);
        pass.end();
        device.queue.submit([enc.finish()]);
        T *= alpha;
      }

      // Final hard assignment for the PNG-8 encode. Mirrors the CPU
      // path's `assignNearestOklab(oklab, centroids, k)` post-anneal
      // step.
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipelines.kmeansAssign);
      pass.setBindGroup(0, assignBind);
      dispatch1D(pass, n, KMEANS_WG_SIZE);
      pass.end();
      device.queue.submit([enc.finish()]);
    } else {
      for (let iter = 0; iter < options.kmeansIterations; iter++) {
        // A-e: poll at iter top. Sync (no await) so we don't add a yield
        // between submits; throwIfAborted is a cheap method call when
        // signal is unset or not yet aborted.
        options.signal?.throwIfAborted();
        const enc = device.createCommandEncoder();
        let pass = enc.beginComputePass();
        pass.setPipeline(pipelines.kmeansAssign);
        pass.setBindGroup(0, assignBind);
        dispatch1D(pass, n, KMEANS_WG_SIZE);
        pass.end();
        pass = enc.beginComputePass();
        pass.setPipeline(pipelines.kmeansAccumulate);
        pass.setBindGroup(0, accBind);
        dispatch1D(pass, n, KMEANS_WG_SIZE);
        pass.end();
        pass = enc.beginComputePass();
        pass.setPipeline(pipelines.kmeansReduce);
        pass.setBindGroup(0, reduceBind);
        dispatch1D(pass, k);
        pass.end();
        device.queue.submit([enc.finish()]);
      }

      // Post-loop assign for `dither: "none"` (N-B-04 / W-A-4): the iter
      // loop's last assign ran against pre-reduce centroids, so `indicesBuf`
      // is one step behind the final palette. blue-noise rewrites
      // `indicesBuf` below, and FS recomputes indices on the CPU, so they
      // don't need this — only the `none` path reads `indicesBuf` directly
      // (including the kmeansIterations=0 case where the loop never ran and
      // indicesBuf is zero-initialised). One extra assign dispatch realigns
      // indices with the final centroids at <1% cost.
      if (options.dither === "none") {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipelines.kmeansAssign);
        pass.setBindGroup(0, assignBind);
        dispatch1D(pass, n, KMEANS_WG_SIZE);
        pass.end();
        device.queue.submit([enc.finish()]);
      }
    }

    // Final assignment with dither.
    if (options.dither === "blue-noise") {
      const noiseU32 = packBlueNoise();
      const noiseBuf = trackBuf(device.createBuffer({
        size: noiseU32.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }));
      writeBuf(device, noiseBuf, 0, noiseU32);
      const importanceForShader = importanceBuf ?? trackBuf(device.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE,
      }));
      const dimsBuf = trackBuf(device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }));
      writeBuf(device, dimsBuf, 0, new Uint32Array([width, height, k, BLUE_NOISE_64_SIZE]));
      // Params layout = { strength: f32 @0, has_importance: u32 @4 } = 8 bytes.
      // Mirror of `struct Params` in src/dither/blue-noise.wgsl.ts — see the
      // comment there for why we keep size=8 instead of padding to 16
      // (Dawn/wgpu/naga currently honour scalar uniform structs of any size).
      const paramsBuf = trackBuf(device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }));
      {
        const buf = new ArrayBuffer(8);
        new Float32Array(buf, 0, 1)[0] = options.ditherStrength;
        new Uint32Array(buf, 4, 1)[0] = importanceBuf ? 1 : 0;
        writeBuf(device, paramsBuf, 0, buf);
      }
      const bind = device.createBindGroup({
        layout: pipelines.blueNoise.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: oklabBuf } },
          { binding: 1, resource: { buffer: centroidsBuf } },
          { binding: 2, resource: { buffer: noiseBuf } },
          { binding: 3, resource: { buffer: importanceForShader } },
          { binding: 4, resource: { buffer: indicesBuf } },
          { binding: 5, resource: { buffer: dimsBuf } },
          { binding: 6, resource: { buffer: paramsBuf } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipelines.blueNoise);
      pass.setBindGroup(0, bind);
      dispatch2D(pass, width, height);
      pass.end();
      device.queue.submit([enc.finish()]);
    }

    // Read centroids back to CPU once and pack to the tight stride-3 layout
    // CPU consumers expect.
    const centroidsArr = new Float32Array(await readBuffer(device, centroidsBuf, k * 16));
    const centroids3 = centroidsVec4ToVec3(centroidsArr, k);
    let indices: Uint8Array;
    if (options.dither === "floyd-steinberg") {
      // FS runs on CPU.
      indices = floydSteinberg({
        width,
        height,
        oklab: oklabArr,
        palette: centroids3,
        paletteCount: k,
        strength: options.ditherStrength,
        importance: importanceArr ?? undefined,
      });
    } else {
      const indicesU32 = new Uint32Array(await readBuffer(device, indicesBuf, n * 4));
      indices = new Uint8Array(n);
      for (let i = 0; i < n; i++) indices[i] = indicesU32[i];
    }

    // Build sRGB palette and alpha table.
    const paletteSrgb = oklabPaletteToSrgb({ oklab: centroids3, count: k });

    let alphaTable: Uint8Array | undefined;
    let needsAlpha = false;
    for (let i = 0; i < n; i++) {
      if (data[i * 4 + 3] !== 255) {
        needsAlpha = true;
        break;
      }
    }
    if (needsAlpha) {
      const sumA = new Float64Array(k);
      const countA = new Uint32Array(k);
      for (let i = 0; i < n; i++) {
        sumA[indices[i]] += data[i * 4 + 3];
        countA[indices[i]]++;
      }
      alphaTable = new Uint8Array(k);
      for (let j = 0; j < k; j++) {
        alphaTable[j] = countA[j] === 0 ? 255 : Math.round(sumA[j] / countA[j]);
      }
    }

    options.signal?.throwIfAborted();
    const png = await encodePng8({
      width,
      height,
      indices,
      palette: { rgb: paletteSrgb, alpha: alphaTable },
      // indices come from k-means assign / FS, both clamped to k=palette
      // size; skip the O(n) re-check on the encode hot path.
      validate: false,
    });

    // Build preview from palette + indices.
    const previewData = new Uint8ClampedArray(n * 4);
    for (let i = 0; i < n; i++) {
      const j = indices[i];
      previewData[i * 4] = paletteSrgb[j * 3];
      previewData[i * 4 + 1] = paletteSrgb[j * 3 + 1];
      previewData[i * 4 + 2] = paletteSrgb[j * 3 + 2];
      previewData[i * 4 + 3] = alphaTable ? alphaTable[j] : 255;
    }
    const preview: ImageData = typeof ImageData !== "undefined"
      ? new ImageData(previewData, width, height)
      : ({ data: previewData, width, height, colorSpace: "srgb" } as ImageData);

    const elapsedMs = performance.now() - t0;
    return {
      png,
      preview,
      palette: paletteSrgb,
      indices,
      meta: { outputBytes: png.byteLength, paletteSize: k, elapsedMs, pipeline: "gpu" },
    };
  } finally {
    // Wait for in-flight GPU work to drain before destroying the backing
    // buffers; the WebGPU spec lets implementations no-op pending commands
    // that reference destroyed resources, but the behavior is impl-defined
    // and waiting here costs nothing for normal exits (queue is already idle
    // by the time we read centroids/indices back via mapAsync).
    await device.queue.onSubmittedWorkDone();
    for (const b of toFree) b.destroy();
  }
}
