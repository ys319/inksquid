// Separable Gaussian blur + Difference-of-Gaussians importance map (WGSL).
// Kernel is uploaded as a uniform array; the host picks radii for σ ∈ {1, 2, 4}.
//
// Boundary handling: mirror-without-repeat with period 2*(n-1). The earlier
// single-fold formulation broke once radius >= n (σ=4 → radius=12 → fails
// at width/height < 13 with a single negative cast → u32 UB). The modulo
// form below converges for any radius/dim combination.

const MIRROR_HELPER = /* wgsl */ `
fn mirror_idx(i: i32, n: i32) -> i32 {
  if (n <= 1) { return 0; }
  let period: i32 = 2 * (n - 1);
  var m: i32 = ((i % period) + period) % period;
  if (m >= n) { m = period - m; }
  return m;
}
`;

export const BLUR_HORIZONTAL_WGSL = /* wgsl */ `
// _pad rounds the struct out to 16 bytes so the host can write the uniform
// in one Uint32Array of length 4. WGSL doesn't require this size, but
// keeping uniforms at a 16-byte boundary matches the size we allocate
// host-side and avoids minBindingSize warnings on stricter backends.
struct Dims { width: u32, height: u32, radius: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;
${MIRROR_HELPER}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= dims.width || gid.y >= dims.height) { return; }
  let r = i32(dims.radius);
  var sum: f32 = 0.0;
  for (var t: i32 = -r; t <= r; t = t + 1) {
    let xx = mirror_idx(i32(gid.x) + t, i32(dims.width));
    sum = sum + src[gid.y * dims.width + u32(xx)] * kernel[u32(t + r)];
  }
  dst[gid.y * dims.width + gid.x] = sum;
}
`;

export const BLUR_VERTICAL_WGSL = /* wgsl */ `
// _pad rounds the struct out to 16 bytes so the host can write the uniform
// in one Uint32Array of length 4. WGSL doesn't require this size, but
// keeping uniforms at a 16-byte boundary matches the size we allocate
// host-side and avoids minBindingSize warnings on stricter backends.
struct Dims { width: u32, height: u32, radius: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;
${MIRROR_HELPER}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= dims.width || gid.y >= dims.height) { return; }
  let r = i32(dims.radius);
  var sum: f32 = 0.0;
  for (var t: i32 = -r; t <= r; t = t + 1) {
    let yy = mirror_idx(i32(gid.y) + t, i32(dims.height));
    sum = sum + src[u32(yy) * dims.width + gid.x] * kernel[u32(t + r)];
  }
  dst[gid.y * dims.width + gid.x] = sum;
}
`;

// Sum |b_{k-1} - b_k| across scales, multiply by gain, clamp to [0,1].
export const DOG_COMBINE_WGSL = /* wgsl */ `
// _pad: padding to 16 bytes for host-side write convenience (see BLUR_*_WGSL).
struct Dims { width: u32, height: u32, gain: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read> b0: array<f32>;
@group(0) @binding(1) var<storage, read> b1: array<f32>;
@group(0) @binding(2) var<storage, read> b2: array<f32>;
@group(0) @binding(3) var<storage, read> b3: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
@group(0) @binding(5) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= dims.width || gid.y >= dims.height) { return; }
  let i = gid.y * dims.width + gid.x;
  let d = abs(b0[i] - b1[i]) + abs(b1[i] - b2[i]) + abs(b2[i] - b3[i]);
  dst[i] = clamp(d * dims.gain, 0.0, 1.0);
}
`;

// Extract L from oklab (vec4) into a flat f32 array — needed because the
// blur shaders work on scalar f32 buffers.
export const EXTRACT_L_WGSL = /* wgsl */ `
struct Dims { width: u32, height: u32 };

@group(0) @binding(0) var<storage, read> oklab: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> L: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= dims.width || gid.y >= dims.height) { return; }
  let i = gid.y * dims.width + gid.x;
  L[i] = oklab[i].x;
}
`;
