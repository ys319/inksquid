export interface DeviceInfo {
  device: GPUDevice;
  adapter: GPUAdapter;
  /**
   * Snapshot of the GPU adapter limits the library actually relies on.
   * Currently the only consumer is the single-pass overflow check in
   * `quantize-gpu.ts` (`n * 16 > limits.maxBufferSize` triggers
   * `GpuBufferOverflowError`). Add more fields here only when a concrete
   * consumer needs them — earlier revisions snapshotted three additional
   * limits that nothing read, so they were removed.
   */
  limits: {
    maxBufferSize: number;
  };
}

/**
 * Thrown when no WebGPU adapter is available — typically because the
 * runtime lacks WebGPU (older browser, Deno without `--unstable-webgpu`)
 * or `navigator.gpu.requestAdapter()` returned `null`.
 *
 * `quantize({ mode: "auto" })` catches this and falls back to the CPU
 * pipeline. `mode: "gpu"` propagates it. Use `instanceof NoWebGPUError`
 * in your own error handlers to distinguish it from
 * {@link GpuBufferOverflowError}.
 */
export class NoWebGPUError extends Error {
  /**
   * Construct a new {@link NoWebGPUError}.
   * @param message — optional override; defaults to a generic description.
   */
  constructor(message = "WebGPU is not available in this environment") {
    super(message);
    this.name = "NoWebGPUError";
  }
}

/**
 * Thrown by the GPU pipeline when the input is too large to allocate as
 * a single OkLab buffer on the current adapter (`n × 16` bytes exceeds
 * `device.limits.maxBufferSize` — roughly 250 megapixels on M-series).
 *
 * Exposed as a dedicated class so `quantize({ mode: "auto" })` can fall
 * back to the CPU tiled path on this specific failure without matching
 * error strings, and so callers can distinguish "GPU isn't available"
 * (use {@link NoWebGPUError}) from "GPU can't fit this image" with
 * `instanceof`.
 */
export class GpuBufferOverflowError extends Error {
  /** Bytes the failed allocation requested. */
  readonly required: number;
  /** The adapter's `device.limits.maxBufferSize` at the time of failure. */
  readonly maxBufferSize: number;
  /**
   * Construct a new {@link GpuBufferOverflowError}.
   * @param required — bytes the failed allocation requested.
   * @param maxBufferSize — the adapter's `device.limits.maxBufferSize`.
   */
  constructor(required: number, maxBufferSize: number) {
    super(
      `Image too large for single-pass GPU buffer: ${required} bytes > ` +
        `device.limits.maxBufferSize=${maxBufferSize}. ` +
        `Use mode: "auto" to fall back to the CPU tiled path automatically.`,
    );
    this.name = "GpuBufferOverflowError";
    this.required = required;
    this.maxBufferSize = maxBufferSize;
  }
}

export async function acquireDevice(
  options: { powerPreference?: GPUPowerPreference } = {},
): Promise<DeviceInfo> {
  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new NoWebGPUError("navigator.gpu is not defined");
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: options.powerPreference ?? "high-performance",
  });
  if (!adapter) {
    throw new NoWebGPUError("requestAdapter() returned null");
  }
  const device = await adapter.requestDevice();
  return {
    device,
    adapter,
    limits: {
      maxBufferSize: device.limits.maxBufferSize,
    },
  };
}

/**
 * Maximum wall-clock time (ms) the shared-device acquisition is allowed
 * to take before the pending promise rejects. Without a TTL, a hung
 * adapter (driver wedged mid-init) would leave every concurrent caller
 * waiting on `_pendingAcquire` forever. 10 seconds is a generous ceiling
 * — typical `requestAdapter + requestDevice` resolves in 10-100 ms, so
 * anything past 10 s is a genuine failure, not slow hardware. A-j
 * (5th-pass review).
 */
const ACQUIRE_TTL_MS = 10_000;

/**
 * @internal Acquire-with-retry helper. Exposed (with the `_` prefix
 * indicating @internal) so unit tests can exercise it directly with a
 * stubbed `acquireFn`. Production reads through `getSharedDevice` →
 * `_acquireWithTimeout` → here.
 *
 * Retry policy: `NoWebGPUError` is permanent ("no adapter exists") and
 * propagates without retry — repeating the call won't suddenly conjure
 * a GPU. Anything else (rejected `requestDevice`, transient driver
 * blip) gets one shot at a retry after a 50 ms backoff. The 50 ms is
 * short enough that the cold-start overhead stays imperceptible (a
 * second `requestAdapter + requestDevice` is itself ~10-100 ms) and
 * long enough that a driver hiccup has time to clear. A-a (5th-pass).
 */
export async function _acquireWithRetry(
  acquireFn: (options: { powerPreference?: GPUPowerPreference }) => Promise<DeviceInfo>,
  options: { powerPreference?: GPUPowerPreference } = {},
): Promise<DeviceInfo> {
  try {
    return await acquireFn(options);
  } catch (e) {
    if (e instanceof NoWebGPUError) throw e;
    await new Promise((r) => setTimeout(r, 50));
    return await acquireFn(options);
  }
}

/**
 * @internal Acquire-with-timeout wrapper around `_acquireWithRetry`.
 * Exposed for unit tests so the timeout window can be shortened
 * without making real tests sleep 10 seconds. Production code calls
 * via `getSharedDevice` with the const `ACQUIRE_TTL_MS`.
 *
 * Uses `Promise.race` against a `setTimeout`-driven rejection. The
 * timer is cleared via `.finally` on the winning acquire promise so
 * the event loop doesn't keep the process alive 10 seconds past
 * resolution. A-j (5th-pass).
 */
export function _acquireWithTimeout(
  acquireFn: (options: { powerPreference?: GPUPowerPreference }) => Promise<DeviceInfo>,
  options: { powerPreference?: GPUPowerPreference } = {},
  ttlMs: number = ACQUIRE_TTL_MS,
): Promise<DeviceInfo> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(
      () =>
        reject(
          new Error(
            `getSharedDevice: adapter acquisition exceeded ${ttlMs}ms — possibly ` +
              `hung adapter; the next getSharedDevice() call will attempt a fresh acquire`,
          ),
        ),
      ttlMs,
    );
  });
  return Promise.race([
    _acquireWithRetry(acquireFn, options).finally(() => clearTimeout(timer)),
    timeout,
  ]);
}

/**
 * Async probe that resolves to `true` when `navigator.gpu.requestAdapter()`
 * returns a non-null adapter, and `false` otherwise. Use for UI gating
 * (showing a "GPU available" indicator, hiding the `mode: "gpu"` option,
 * etc.) before calling `quantize`. Never throws; permission denials and
 * missing `navigator.gpu` both surface as `false`.
 */
export async function isWebGPUAvailable(): Promise<boolean> {
  try {
    if (typeof navigator === "undefined" || !navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

// Process-wide cached device. Adapter + device acquisition costs O(10 ms)
// each, and compiling pipelines against a fresh device defeats driver
// caches; sharing across calls makes 2nd-and-later quantize() invocations
// dramatically faster and keeps VRAM usage bounded.
let _cachedDevice: DeviceInfo | null = null;

// In-flight `acquireDevice` promise — used as a single-flight lock so
// concurrent `getSharedDevice()` callers (e.g. `Promise.all([quantize,
// quantize, quantize])` on the first call) all await the same acquisition
// instead of each starting their own. Without this, the cache miss window
// produces orphan devices that never enter `_cachedDevice` and therefore
// never get destroyed by `disposeSharedDevice` — a slow VRAM leak in
// long-running processes. (W-A-5, 3rd-pass review.)
let _pendingAcquire: Promise<DeviceInfo> | null = null;

// Tombstone for the in-flight `_pendingAcquire`. Flipped by
// `disposeSharedDevice` when called while an acquisition is still racing;
// the `.then` handler reads it and destroys the freshly-arrived device
// instead of caching it, so a contract-violating dispose during pending
// acquire doesn't leave a "zombie" device that `_cachedDevice` re-adopts
// after dispose returned. Reset at the start of each new acquisition.
// (4A-01, 4th-pass review.)
let _pendingAcquireDisposed = false;

// Hooks fired right before the cached device is invalidated (either by
// `disposeSharedDevice` or `device.lost`). Used by `quantize-gpu.ts` to
// drop its pipeline cache so the next call doesn't try to reuse pipelines
// tied to a destroyed device. Registered at module load and never removed
// (modules in Deno/JSR live for the entire process), so we don't need a
// dispose API on the hook side.
const _disposalHooks: Array<() => void> = [];

/**
 * Register a callback to fire when the cached device is invalidated
 * (explicit `disposeSharedDevice` or implicit `device.lost`). Used by
 * downstream caches (e.g. compute pipelines in `quantize-gpu.ts`) that
 * become invalid when the device they were built against goes away. The
 * callback runs before the device is destroyed, so it can still touch
 * device-owned resources if needed.
 */
export function onSharedDeviceDispose(fn: () => void): void {
  _disposalHooks.push(fn);
}

function fireDisposalHooks(): void {
  for (const fn of _disposalHooks) {
    try {
      fn();
    } catch {
      // Hooks must not block dispose. A misbehaving hook is a programming
      // bug, not a runtime concern — swallow and continue.
    }
  }
}

/**
 * Get or lazily create the process-wide shared `DeviceInfo`. The cache is
 * automatically dropped if the underlying `device.lost` resolves (driver
 * reset, page lifecycle event, etc.), so the next call re-acquires
 * transparently. Call `disposeSharedDevice` to release VRAM eagerly.
 *
 * Concurrent callers (e.g. `Promise.all([quantize(...), quantize(...)])`
 * on the first call from a cold cache) all share the same in-flight
 * acquisition via the `_pendingAcquire` single-flight lock — only one
 * device is acquired no matter how many callers race the cache miss.
 *
 * **First-caller-wins for `options.powerPreference`**: the first caller's
 * `powerPreference` determines which adapter the device is acquired from
 * for the process lifetime. Later callers passing a different value get
 * the cached device regardless; call `disposeSharedDevice` and re-acquire
 * if you need to switch adapters mid-process.
 *
 * **Disposal during pending acquire**: if `disposeSharedDevice` is called
 * before this promise resolves, the returned promise rejects rather than
 * caching the device — the freshly-acquired device is destroyed on
 * arrival. Concurrent callers waiting on the same in-flight acquisition
 * all see the same rejection. Following the documented "don't dispose
 * while quantize is in flight" contract avoids this rejection entirely;
 * the rejection is the defensive backstop, not an expected code path.
 *
 * **Transient retry**: a single non-`NoWebGPUError` rejection from
 * `acquireDevice` (e.g. flaky `requestDevice`) triggers one retry after
 * a 50 ms backoff. `NoWebGPUError` propagates without retry — no
 * adapter exists, retrying won't help. A-a (5th-pass review).
 *
 * **Acquisition timeout**: if the (retry-capable) acquire doesn't
 * settle within 10 seconds, the pending promise rejects with a clear
 * "hung adapter" error so concurrent callers don't deadlock. After
 * rejection, `_pendingAcquire` is cleared and the next call kicks off
 * a fresh attempt. A-j (5th-pass review).
 */
export function getSharedDevice(
  options: { powerPreference?: GPUPowerPreference } = {},
): Promise<DeviceInfo> {
  if (_cachedDevice) return Promise.resolve(_cachedDevice);
  // Single-flight: if an acquisition is already in flight (started by an
  // earlier concurrent caller), await the same promise instead of kicking
  // off a duplicate adapter+device request. W-A-5 in the 2026-05-23
  // 3rd-pass review explains why the orphan device that the duplicate
  // produced was a real leak.
  if (_pendingAcquire) return _pendingAcquire;
  // Reset tombstone for this fresh acquisition. The previous acquisition's
  // flag is already irrelevant — its promise has settled and its `.finally`
  // ran — but explicit reset here keeps the invariant local to this
  // function instead of relying on `.finally` ordering.
  _pendingAcquireDisposed = false;
  // Routes through `_acquireWithTimeout` → `_acquireWithRetry` →
  // `acquireDevice` so a 10 s hang or a single transient `requestDevice`
  // failure are both handled defensively (A-j + A-a, 5th-pass review).
  // The retry wrapper consumes one transient failure before propagating;
  // the timeout wrapper bounds wall-clock time so concurrent callers
  // waiting on `_pendingAcquire` never deadlock on a wedged adapter.
  _pendingAcquire = _acquireWithTimeout(acquireDevice, options).then((di) => {
    if (_pendingAcquireDisposed) {
      // `disposeSharedDevice` was called while this acquisition was in
      // flight. Honour the dispose: destroy the device we just acquired
      // (it was never cached, so no pipelines bound to it) and reject the
      // promise. Disposal hooks are intentionally not fired here —
      // nothing downstream observed this device, so there is nothing to
      // invalidate.
      di.device.destroy();
      throw new Error(
        "getSharedDevice: shared device disposed before acquisition completed; retry to re-acquire",
      );
    }
    _cachedDevice = di;
    // If the GPU process dies, drop the cache so the next call re-acquires.
    di.device.lost.then(() => {
      if (_cachedDevice === di) {
        _cachedDevice = null;
        fireDisposalHooks();
      }
    });
    return di;
  }).finally(() => {
    // Clear the lock once the promise settles regardless of success /
    // failure. On rejection `_cachedDevice` stays null and the next call
    // re-attempts (so a transient adapter blip doesn't permanently
    // wedge the library); on success the cache is populated above.
    _pendingAcquire = null;
  });
  return _pendingAcquire;
}

/**
 * Destroy the cached device and release its VRAM. Idempotent — calling it
 * twice (or when no device is cached) is a no-op. The next GPU quantize
 * call will re-acquire fresh.
 *
 * **Caller contract**: must not be invoked while a `quantize{,Gpu}` call
 * is in flight. We destroy the device synchronously after firing disposal
 * hooks, which means an in-flight quantize on the same device may see
 * `device.queue.submit` / buffer destroy fail mid-pipeline. The library's
 * own usage sites honour this (tests/gpu/leak.test.ts disposes only
 * between sequential bursts, and `device.lost` resolution flows through
 * the same hook path without ever racing in-flight work). If you need
 * to dispose under concurrent load, await all your `quantize` promises
 * first.
 *
 * **Dispose during pending acquire**: if called while `getSharedDevice`
 * is still acquiring (i.e. the cache is cold and an `acquireDevice`
 * request is in flight), the disposal is honoured: the freshly-acquired
 * device is destroyed on arrival and the pending `getSharedDevice`
 * promise rejects. This is a defensive backstop against the contract
 * violation above for the *acquisition* phase, not a supported pattern
 * — callers waiting on the racing promise will see a rejection they
 * didn't expect. (4A-01, 4th-pass review.)
 */
export function disposeSharedDevice(): void {
  // Tombstone the in-flight acquire (if any). The `.then` handler in
  // `getSharedDevice` reads this flag and destroys the device on arrival
  // instead of caching it.
  if (_pendingAcquire) _pendingAcquireDisposed = true;
  if (!_cachedDevice) return;
  const di = _cachedDevice;
  _cachedDevice = null;
  fireDisposalHooks();
  di.device.destroy();
}
