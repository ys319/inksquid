/**
 * Resolved (post-validation) option record consumed by every pipeline.
 * Callers usually pass a `Partial<QuantizeOptions>` to `quantize` and the
 * dispatcher fills in defaults via {@link normalizeOptions}.
 */
export interface QuantizeOptions {
  /**
   * Number of palette entries to emit. Integer in `[2, 256]`; default
   * `128`. The PNG-8 cap of 256 is enforced strictly — passing a higher
   * value throws `RangeError`.
   */
  paletteSize: number;
  /**
   * - `"none"`: no dither; nearest-palette assignment in OkLab L2.
   * - `"blue-noise"`: precomputed-mask threshold dither in OkLab.
   *   Cheap, parallel-friendly, GPU-supported. Default.
   * - `"floyd-steinberg"`: classic FS error diffusion in OkLab L2.
   *   Sequential, CPU-only; the tiled CPU path downgrades to
   *   `"blue-noise"` with a one-shot warning.
   * - `"scolorq"`: spatial colour quantization (Puzicha 1998).
   *   Refines the palette via deterministic annealing on soft
   *   assignments and produces dither indices as a by-product of
   *   the same optimisation. CPU-only (tiled & GPU paths
   *   currently downgrade to `"blue-noise"`); higher quality on
   *   natural photos at the cost of ~3-5× runtime.
   */
  dither: "none" | "blue-noise" | "floyd-steinberg" | "scolorq";
  /**
   * Dither perturbation amplitude in `[0, 1]`; default `1.0`. `0`
   * disables dither perturbation (nearest-palette assignment only);
   * lower values can reduce visible noise at small palette sizes.
   */
  ditherStrength: number;
  /**
   * Importance bias from the DoG detail map in `[0, 1]`; default `0.5`.
   * `0` weights every pixel uniformly (faster but slightly worse on busy
   * images); higher values pull centroids toward detected edges.
   */
  detailWeight: number;
  /**
   * Number of weighted k-means refinement iterations to run after Wu
   * init; integer in `[0, 100]`; default `15`. Centroids settle by ~10
   * iterations on most inputs. `0` skips refinement entirely and
   * returns the Wu-init palette.
   */
  kmeansIterations: number;
  /**
   * Optional `AbortSignal` for cancelling a long-running call. When the
   * signal aborts, the pipeline throws `signal.reason` at the next
   * checkpoint (defaulting to a DOMException with `name: "AbortError"`).
   * Checkpoints sit at each pipeline-stage boundary and inside the
   * k-means iter loop / tiled per-tile loop — i.e. the only places where
   * work is divisible. Per-pixel inner loops do *not* poll the signal,
   * so abort latency is bounded by "one iteration / one tile" rather
   * than "instantaneous".
   *
   * A-e (5th-pass review). The field is optional and unvalidated by
   * `normalizeOptions` — it's a runtime context handle, not algorithm
   * config, and passing `undefined` is equivalent to "no cancellation".
   */
  signal?: AbortSignal;
}

/**
 * Default option record used when `quantize` is called with `undefined`
 * or with a `Partial<QuantizeOptions>` that omits some fields. Exposed
 * so callers can show the defaults in a UI or spread them into a
 * partial override (`{ ...DEFAULT_OPTIONS, paletteSize: 64 }`).
 */
export const DEFAULT_OPTIONS: QuantizeOptions = {
  paletteSize: 128,
  dither: "blue-noise",
  ditherStrength: 1.0,
  detailWeight: 0.5,
  kmeansIterations: 15,
};

const VALID_DITHER: ReadonlySet<QuantizeOptions["dither"]> = new Set([
  "none",
  "blue-noise",
  "floyd-steinberg",
  "scolorq",
]);

// `paletteSize` upper bound matches KMEANS_MAX_K in palette/kmeans.wgsl.ts.
// Bumping one without the other will silently corrupt the GPU accumulator.
const MAX_PALETTE_SIZE = 256;
const MAX_KMEANS_ITERATIONS = 100;

/**
 * Merge user-provided options with {@link DEFAULT_OPTIONS} and validate
 * the result. Out-of-range / wrong-typed values throw `RangeError` /
 * `TypeError` rather than silently clamping, so callers learn about
 * mistakes at the call site instead of getting a result that "looks
 * right" with `k = 1` or no dither.
 *
 * Useful as a pre-flight check before calling `quantize`, or for UI
 * code that wants to validate a form submission against the same
 * rules the library applies internally.
 *
 * @throws `TypeError` on wrong-typed fields (e.g. `paletteSize: "16"`).
 * @throws `RangeError` on out-of-range fields (e.g.
 *   `paletteSize: 1000`).
 */
export function normalizeOptions(
  input: Partial<QuantizeOptions> = {},
): QuantizeOptions {
  const merged = { ...DEFAULT_OPTIONS, ...input };

  // NEW-E-004 (2nd-pass review): JS consumers (no TS at the call site) can
  // smuggle strings / booleans / null past the type system. `Number("16")` is
  // `16` and `Number(true)` is `1`, so without an explicit `typeof` check the
  // library would silently accept misuse and the consumer would never learn
  // about it. Each numeric field rejects non-number input with TypeError;
  // range / integerness checks then run on a confirmed-number value.
  const ps = merged.paletteSize;
  if (typeof ps !== "number") {
    throw new TypeError(
      `paletteSize must be a number, got ${typeof ps} (${JSON.stringify(ps)})`,
    );
  }
  if (!Number.isFinite(ps) || ps !== Math.floor(ps)) {
    throw new RangeError(
      `paletteSize must be an integer, got ${ps}`,
    );
  }
  if (ps < 2 || ps > MAX_PALETTE_SIZE) {
    throw new RangeError(
      `paletteSize must be in [2, ${MAX_PALETTE_SIZE}], got ${ps}`,
    );
  }

  if (!VALID_DITHER.has(merged.dither)) {
    throw new TypeError(
      `dither must be one of "none" | "blue-noise" | "floyd-steinberg" | ` +
        `"scolorq", got ${JSON.stringify(merged.dither)}`,
    );
  }

  const ds = merged.ditherStrength;
  if (typeof ds !== "number") {
    throw new TypeError(
      `ditherStrength must be a number, got ${typeof ds} (${JSON.stringify(ds)})`,
    );
  }
  if (!Number.isFinite(ds) || ds < 0 || ds > 1) {
    throw new RangeError(
      `ditherStrength must be in [0, 1], got ${ds}`,
    );
  }

  const dw = merged.detailWeight;
  if (typeof dw !== "number") {
    throw new TypeError(
      `detailWeight must be a number, got ${typeof dw} (${JSON.stringify(dw)})`,
    );
  }
  if (!Number.isFinite(dw) || dw < 0 || dw > 1) {
    throw new RangeError(
      `detailWeight must be in [0, 1], got ${dw}`,
    );
  }

  // kmeansIterations = 0 is intentionally a valid input: it means "use the
  // Wu init palette unchanged and skip refinement". The result `indices`
  // still contains nearest-to-Wu-init assignments (the post-loop assign in
  // kmeansRefine + the GPU dither="none" gate guarantee this), so callers
  // get a well-defined output instead of an all-zero indices stream. Upper
  // bound is `MAX_KMEANS_ITERATIONS` to keep accidental `1e9` typos out of
  // the GPU's iter loop.
  const it = merged.kmeansIterations;
  if (typeof it !== "number") {
    throw new TypeError(
      `kmeansIterations must be a number, got ${typeof it} (${JSON.stringify(it)})`,
    );
  }
  if (!Number.isFinite(it) || it !== Math.floor(it)) {
    throw new RangeError(
      `kmeansIterations must be an integer, got ${it}`,
    );
  }
  if (it < 0 || it > MAX_KMEANS_ITERATIONS) {
    throw new RangeError(
      `kmeansIterations must be in [0, ${MAX_KMEANS_ITERATIONS}], got ${it}`,
    );
  }

  return {
    paletteSize: ps,
    dither: merged.dither,
    ditherStrength: ds,
    detailWeight: dw,
    kmeansIterations: it,
    // signal is a pass-through runtime handle, not algorithm config.
    // Explicitly include the property only when the caller supplied one
    // (avoids attaching `signal: undefined` to the normalised options,
    // which would confuse `"signal" in opts` style checks downstream).
    ...(merged.signal !== undefined ? { signal: merged.signal } : {}),
  };
}

/** Result record returned by every `quantize{,Cpu,Gpu,Tiled}` entry point. */
export interface QuantizeResult {
  /** Complete PNG-8 byte stream (signature + IHDR + PLTE + IDAT + IEND). */
  png: Uint8Array;
  /**
   * Quantised image rendered back to RGBA bytes via the chosen palette.
   * Dimensions match the input.
   */
  preview: ImageData;
  /**
   * Stride-3 RGB palette bytes, length = `paletteSize * 3`. Slot *j* on
   * CPU and slot *j* on GPU correspond to the same Wu-init centroid;
   * downstream drift between the two pipelines is bounded by the parity
   * test (see `docs/parity-test.md`).
   */
  palette: Uint8Array;
  /**
   * Per-pixel palette index in row-major order, length = width * height.
   * Each byte is in [0, palette.length / 3). Useful for palette swapping,
   * re-encoding to other indexed formats, or pinning CPU/GPU parity in
   * tests without going through the rendered preview.
   *
   * The element type is fixed at Uint8Array because paletteSize ≤ 256
   * (PNG-8 limit, matches KMEANS_MAX_K).
   *
   * Invariant across all pipelines (cpu / cpu-tiled / gpu) and all dither
   * modes: `indices[i]` is argmin_j ‖oklab(pixel_i) − centroid_j‖² over the
   * *returned* palette (modulo ditherer-specific offsets for blue-noise /
   * floyd-steinberg). In particular, `kmeansIterations: 0` returns indices
   * computed against the Wu-init palette rather than an all-zero stream.
   */
  indices: Uint8Array;
  /** Per-call metadata. */
  meta: {
    /** Length of `png` in bytes. */
    outputBytes: number;
    /** Number of palette entries actually emitted (equals `palette.length / 3`). */
    paletteSize: number;
    /** Wall-clock time spent inside the quantize call, in milliseconds. */
    elapsedMs: number;
    /**
     * Which pipeline produced this result. Useful for `mode: "auto"`
     * callers who want to know whether they ended up on GPU or fell
     * through to CPU (and from there, whether the tiled path kicked in
     * for large images).
     */
    pipeline: "cpu" | "cpu-tiled" | "gpu";
    /**
     * Only populated by the top-level `quantize(...)` dispatcher when
     * the call ran on a CPU pipeline. Lets callers (telemetry, UI
     * spinners, debug logs) distinguish *why* CPU ran: explicit
     * request, no GPU available, or a runtime GPU constraint forced
     * fallback. The lower-level `quantizeCpu` / `quantizeGpu` /
     * `quantizeTiled` entry points never set this — there is no
     * dispatcher decision to report.
     *
     * Values:
     * - `"no-gpu-adapter"` — `mode: "auto"` and either
     *   `isWebGPUAvailable()` returned false or `quantizeGpu` threw
     *   `NoWebGPUError`.
     * - `"gpu-buffer-overflow"` — `mode: "auto"` and `quantizeGpu`
     *   threw `GpuBufferOverflowError` (input exceeded
     *   `device.limits.maxBufferSize`).
     * - `"cpu-forced"` — caller passed `mode: "cpu"`.
     *
     * Absent when:
     * - `mode: "gpu"` succeeded.
     * - `mode: "auto"` ran on GPU successfully.
     * - One of the direct entry points (`quantizeCpu` /
     *   `quantizeGpu` / `quantizeTiled`) was called directly.
     */
    fallbackReason?: "no-gpu-adapter" | "gpu-buffer-overflow" | "cpu-forced";
  };
}

/**
 * Plain-object RGBA image record used internally by the pipelines.
 * Accepted as input by `quantize{,Cpu,Gpu,Tiled}` alongside
 * `ImageData` / `ImageBitmap`; convert via {@link toRawImage} if you
 * have one of those.
 */
export interface RawImage {
  /** Image width in pixels (positive integer). */
  width: number;
  /** Image height in pixels (positive integer). */
  height: number;
  /**
   * RGBA byte buffer, row-major. `data.length` must equal
   * `width * height * 4`.
   */
  data: Uint8ClampedArray;
}

// 4A-04 (4th-pass review): integrity validation runs once at the pipeline
// entry point so downstream stages can assume `data.length === w*h*4`,
// `w >= 1`, `h >= 1` and all three are integers. Without this, a caller
// passing `{ width: 0, height: 100, data: ... }` reaches Wu / k-means
// with zero-sized buffers and gets either a cryptic indexing error or a
// silent all-zero output — both are worse than an upfront `RangeError`
// pointing at the bad input. Validation is cheap (5 cmps, no scan) so
// it runs unconditionally rather than gated by a `validate: false` knob.
function validateDimensions(width: number, height: number): void {
  if (!Number.isInteger(width) || width < 1) {
    throw new RangeError(
      `toRawImage: width must be a positive integer, got ${width}`,
    );
  }
  if (!Number.isInteger(height) || height < 1) {
    throw new RangeError(
      `toRawImage: height must be a positive integer, got ${height}`,
    );
  }
}

export function toRawImage(input: ImageBitmap | ImageData | RawImage): RawImage {
  if ("data" in input && input.data instanceof Uint8ClampedArray) {
    validateDimensions(input.width, input.height);
    const expected = input.width * input.height * 4;
    if (input.data.length !== expected) {
      throw new RangeError(
        `toRawImage: data.length=${input.data.length} does not match ` +
          `width*height*4=${expected} (width=${input.width}, height=${input.height})`,
      );
    }
    return { width: input.width, height: input.height, data: input.data };
  }
  if (typeof OffscreenCanvas !== "undefined" && "width" in input && "height" in input) {
    validateDimensions(input.width, input.height);
    const canvas = new OffscreenCanvas(input.width, input.height);
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(input as ImageBitmap, 0, 0);
    const id = ctx.getImageData(0, 0, input.width, input.height);
    return { width: id.width, height: id.height, data: id.data };
  }
  throw new Error(
    "Cannot convert input to RawImage: provide ImageData or run where OffscreenCanvas is available",
  );
}
