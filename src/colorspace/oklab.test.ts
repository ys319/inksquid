import { assertAlmostEquals } from "jsr:@std/assert@^1.0.0";
import {
  linearRgbToOklab,
  linearToSrgb,
  oklabToLinearRgb,
  srgbToLinear,
  srgbU8ToOklab,
} from "./oklab.ts";

// Known OkLab reference values from Björn Ottosson's post.
// Each row: (linear RGB, OkLab)
const REFERENCE = [
  { rgb: { r: 1, g: 0, b: 0 }, lab: { L: 0.62795, a: 0.22486, b: 0.12585 } },
  { rgb: { r: 0, g: 1, b: 0 }, lab: { L: 0.86644, a: -0.23389, b: 0.17950 } },
  { rgb: { r: 0, g: 0, b: 1 }, lab: { L: 0.45201, a: -0.03246, b: -0.31153 } },
  { rgb: { r: 1, g: 1, b: 1 }, lab: { L: 1.00000, a: 0.0, b: 0.0 } },
  { rgb: { r: 0, g: 0, b: 0 }, lab: { L: 0.0, a: 0.0, b: 0.0 } },
];

Deno.test("linearRgbToOklab matches Ottosson reference values within 1e-3", () => {
  for (const r of REFERENCE) {
    const lab = linearRgbToOklab(r.rgb);
    assertAlmostEquals(lab.L, r.lab.L, 1e-3, `L for ${JSON.stringify(r.rgb)}`);
    assertAlmostEquals(lab.a, r.lab.a, 1e-3, `a for ${JSON.stringify(r.rgb)}`);
    assertAlmostEquals(lab.b, r.lab.b, 1e-3, `b for ${JSON.stringify(r.rgb)}`);
  }
});

Deno.test("oklab linearRgb roundtrip recovers original within 1e-5", () => {
  for (const r of REFERENCE) {
    const lab = linearRgbToOklab(r.rgb);
    const back = oklabToLinearRgb(lab);
    assertAlmostEquals(back.r, r.rgb.r, 1e-5);
    assertAlmostEquals(back.g, r.rgb.g, 1e-5);
    assertAlmostEquals(back.b, r.rgb.b, 1e-5);
  }
});

Deno.test("sRGB transfer curve roundtrip for arbitrary values", () => {
  // Tolerance is 1e-6 instead of 1e-9 because the piecewise curve introduces
  // a small discontinuity in derivative at the 0.04045 / 0.0031308 thresholds,
  // and double-precision rounding amplifies the difference there.
  for (const v of [0, 0.04, 0.04045, 0.05, 0.1, 0.5, 0.9, 1.0]) {
    assertAlmostEquals(srgbToLinear(linearToSrgb(v)), v, 1e-6);
    assertAlmostEquals(linearToSrgb(srgbToLinear(v)), v, 1e-6);
  }
});

Deno.test("sRGB transfer curve boundary: linear/power-law join is continuous", () => {
  // E-012: pin the C0 continuity at the piecewise break-point. The two
  // formulas are deliberately matched to produce the same output at the
  // threshold; if a refactor ever tweaks the constants, this test catches
  // the resulting kink (which would show up as visible banding near
  // mid-shadow tones).
  const SRGB_BREAK = 0.04045;
  const linearAtBreak = srgbToLinear(SRGB_BREAK);
  // Both formulas give ≈ 0.00313 at the break point. Compare against the
  // explicit linear-form value (12.92 division branch) so we pin the
  // numerical match, not just self-consistency.
  assertAlmostEquals(linearAtBreak, SRGB_BREAK / 12.92, 1e-7);
  // Inverse direction.
  const LINEAR_BREAK = 0.0031308;
  const srgbAtBreak = linearToSrgb(LINEAR_BREAK);
  assertAlmostEquals(srgbAtBreak, LINEAR_BREAK * 12.92, 1e-7);
});

Deno.test("srgbU8ToOklab handles the gamut extremes", () => {
  const white = srgbU8ToOklab(255, 255, 255);
  assertAlmostEquals(white.L, 1.0, 1e-3);
  assertAlmostEquals(white.a, 0.0, 1e-3);
  assertAlmostEquals(white.b, 0.0, 1e-3);
  const black = srgbU8ToOklab(0, 0, 0);
  assertAlmostEquals(black.L, 0.0, 1e-9);
  assertAlmostEquals(black.a, 0.0, 1e-9);
  assertAlmostEquals(black.b, 0.0, 1e-9);
});
