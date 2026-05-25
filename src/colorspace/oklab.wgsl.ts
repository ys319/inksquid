export const OKLAB_WGSL = /* wgsl */ `
fn srgb_to_linear(c: f32) -> f32 {
  if (c <= 0.04045) { return c / 12.92; }
  return pow((c + 0.055) / 1.055, 2.4);
}

fn linear_to_srgb(c: f32) -> f32 {
  if (c <= 0.0031308) { return c * 12.92; }
  return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

fn srgb_u8_to_oklab(rgba: vec4<u32>) -> vec4<f32> {
  let r_lin = srgb_to_linear(f32(rgba.r) / 255.0);
  let g_lin = srgb_to_linear(f32(rgba.g) / 255.0);
  let b_lin = srgb_to_linear(f32(rgba.b) / 255.0);

  let l_ = 0.4122214708 * r_lin + 0.5363325363 * g_lin + 0.0514459929 * b_lin;
  let m_ = 0.2119034982 * r_lin + 0.6806995451 * g_lin + 0.1073969566 * b_lin;
  let s_ = 0.0883024619 * r_lin + 0.2817188376 * g_lin + 0.6299787005 * b_lin;

  // pow(x, 1/3) rather than a hypothetical cbrt(): for in-gamut sRGB
  // (l_/m_/s_ are non-negative since they come from clamped byte inputs)
  // the result is exact enough that the residual CPU/GPU divergence from
  // CPU's Math.cbrt stays inside the bounds pinned by
  // tests/gpu/parity.test.ts (B-03 deferred in the 2026-05-23 review).
  let l = pow(l_, 1.0 / 3.0);
  let m = pow(m_, 1.0 / 3.0);
  let s = pow(s_, 1.0 / 3.0);

  let L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
  let a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
  let b = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;

  return vec4<f32>(L, a, b, f32(rgba.a) / 255.0);
}

fn oklab_to_srgb_u8(lab: vec4<f32>) -> vec4<u32> {
  let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;

  let l = l_ * l_ * l_;
  let m = m_ * m_ * m_;
  let s = s_ * s_ * s_;

  let r_lin = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  let g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  let b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

  let r = u32(clamp(round(linear_to_srgb(r_lin) * 255.0), 0.0, 255.0));
  let g = u32(clamp(round(linear_to_srgb(g_lin) * 255.0), 0.0, 255.0));
  let b = u32(clamp(round(linear_to_srgb(b_lin) * 255.0), 0.0, 255.0));
  let a = u32(clamp(round(lab.w * 255.0), 0.0, 255.0));
  return vec4<u32>(r, g, b, a);
}
`;

export const OKLAB_FORWARD_SHADER = /* wgsl */ `
${OKLAB_WGSL}

struct Dims { width: u32, height: u32 };

@group(0) @binding(0) var<storage, read> rgba: array<u32>;
@group(0) @binding(1) var<storage, read_write> oklab: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= dims.width || gid.y >= dims.height) { return; }
  let i = gid.y * dims.width + gid.x;
  // rgba[i] is packed as R | G<<8 | B<<16 | A<<24 by packRgba() in
  // src/api/quantize-gpu.ts (little-endian byte order). Keep both ends in
  // sync — a byte-order mismatch shows up as complete colour inversion.
  let packed = rgba[i];
  let p = vec4<u32>(
    (packed >> 0u) & 0xffu,
    (packed >> 8u) & 0xffu,
    (packed >> 16u) & 0xffu,
    (packed >> 24u) & 0xffu,
  );
  oklab[i] = srgb_u8_to_oklab(p);
}
`;
