// Blue-noise threshold dither in OkLab (WGSL).
// Reads the baked 64x64 mask via a R32Uint storage texture or a flat u32 array.
//
// Whole-image single dispatch — no per-tile origin offset because the GPU
// pipeline doesn't tile. The CPU mirror (src/dither/blue-noise.ts) takes
// offsetX/offsetY for the tiled CPU path; here gid.x/gid.y are already
// absolute pixel coordinates so the mask phase is naturally continuous.

export const BLUE_NOISE_WGSL = /* wgsl */ `
struct Dims { width: u32, height: u32, palette_count: u32, noise_size: u32 };
// 8 bytes is below the 16-byte stride wgpu uses for arrays of uniforms,
// but the WebGPU spec permits scalar uniform structs of any size; current
// Dawn/wgpu/naga implementations honour this. If a future backend starts
// warning about minBindingSize < 16, padding to vec4 here + bumping the
// host-side paramsBuf size from 8 to 16 is the smallest fix.
struct Params { strength: f32, has_importance: u32 };

@group(0) @binding(0) var<storage, read> oklab: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> palette: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> noise: array<u32>;  // packed 4 mask values per u32
@group(0) @binding(3) var<storage, read> importance: array<f32>;
@group(0) @binding(4) var<storage, read_write> indices: array<u32>;
@group(0) @binding(5) var<uniform> dims: Dims;
@group(0) @binding(6) var<uniform> params: Params;

// 3C-02 (3rd-pass review): the x/y inputs are u32, and the caller passes
// gid.x + NOISE_OFFSET_*_X (max +37). WGSL u32 add wraps modulo 2^32 by
// spec, so very large gid values (impossible in practice — width is
// capped by maxTextureDimension2D ~ 16384) would wrap. As long as
// noise_size is a power of two, (wrapped_value % noise_size) equals the
// mathematical modulo because 2^32 % noise_size == 0. We rely on this
// equivalence, so noise_size must stay a power of two if anyone resizes
// the baked mask later.
fn sample_noise(x: u32, y: u32) -> f32 {
  let xx = x % dims.noise_size;
  let yy = y % dims.noise_size;
  let idx = yy * dims.noise_size + xx;
  let packed = noise[idx >> 2u];
  let byte = (packed >> ((idx & 3u) * 8u)) & 0xffu;
  return f32(byte) / 255.0 - 0.5;
}

// Coprime-ish offsets (any odd is coprime with 64). Mirror of the
// NOISE_OFFSET_{A,B} / CHROMA_NOISE_SCALE constants in
// src/dither/blue-noise.ts — keep in sync. The L channel samples at
// (0,0) implicitly, so there is no NOISE_OFFSET_L. The parity test
// (tests/gpu/parity.test.ts) catches drift between the two: when a
// case under "dither=blue-noise" produces a non-∞ PSNR mismatch, this
// pair of files is the first place to look.
const NOISE_OFFSET_A_X: u32 = 17u;
const NOISE_OFFSET_A_Y: u32 = 23u;
const NOISE_OFFSET_B_X: u32 = 37u;
const NOISE_OFFSET_B_Y: u32 = 11u;
const CHROMA_NOISE_SCALE: f32 = 0.6;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= dims.width || gid.y >= dims.height) { return; }
  let i = gid.y * dims.width + gid.x;
  let p = oklab[i];
  let tL = sample_noise(gid.x, gid.y);
  let tA = sample_noise(gid.x + NOISE_OFFSET_A_X, gid.y + NOISE_OFFSET_A_Y);
  let tB = sample_noise(gid.x + NOISE_OFFSET_B_X, gid.y + NOISE_OFFSET_B_Y);
  let step = 0.18 / pow(f32(dims.palette_count), 1.0 / 3.0) * params.strength;
  var scale: f32 = 1.0;
  if (params.has_importance != 0u) { scale = 1.0 - importance[i] * 0.5; }
  let offsetL = tL * step * scale;
  let offsetA = tA * step * scale * CHROMA_NOISE_SCALE;
  let offsetB = tB * step * scale * CHROMA_NOISE_SCALE;
  let L = p.x + offsetL;
  let a = p.y + offsetA;
  let b = p.z + offsetB;

  var best: u32 = 0u;
  var best_d: f32 = 1e30;
  for (var j: u32 = 0u; j < dims.palette_count; j = j + 1u) {
    let c = palette[j];
    let d = (L - c.x) * (L - c.x) + (a - c.y) * (a - c.y) + (b - c.z) * (b - c.z);
    if (d < best_d) { best_d = d; best = j; }
  }
  indices[i] = best;
}
`;
