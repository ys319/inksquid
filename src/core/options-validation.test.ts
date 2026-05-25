// Pins the boundary contract of normalizeOptions.
//
// We exercise it through quantizeCpu because that's the user-visible surface
// — normalizeOptions itself is exported, but the value of the contract is
// that callers can't accidentally smuggle bad input past the entry point.

import { assertEquals, assertRejects, assertThrows } from "jsr:@std/assert@^1.0.0";
import { normalizeOptions, quantizeCpu } from "../mod.ts";
import { flat as flatHelper } from "../_test-fixtures.ts";

// Thin local wrapper preserving the (w=4, h=4) default at call sites.
function flat(w = 4, h = 4): ImageData {
  return flatHelper(w, h);
}

Deno.test("normalizeOptions: defaults stand alone", () => {
  const o = normalizeOptions();
  assertEquals(o.paletteSize, 128);
  assertEquals(o.dither, "blue-noise");
  assertEquals(o.ditherStrength, 1.0);
  assertEquals(o.detailWeight, 0.5);
  assertEquals(o.kmeansIterations, 15);
});

Deno.test("normalizeOptions: paletteSize rejects 0/1/257/NaN/1.5/-1", () => {
  for (const bad of [0, 1, 257, Number.NaN, 1.5, -1, Number.POSITIVE_INFINITY]) {
    assertThrows(
      () => normalizeOptions({ paletteSize: bad }),
      RangeError,
      undefined,
      `expected throw for paletteSize=${bad}`,
    );
  }
});

Deno.test("normalizeOptions: paletteSize accepts 2..256", () => {
  for (const ok of [2, 16, 128, 255, 256]) {
    const o = normalizeOptions({ paletteSize: ok });
    assertEquals(o.paletteSize, ok);
  }
});

Deno.test("normalizeOptions: dither rejects unknown strings", () => {
  // deno-lint-ignore no-explicit-any
  assertThrows(() => normalizeOptions({ dither: "foo" as any }), TypeError);
  // deno-lint-ignore no-explicit-any
  assertThrows(() => normalizeOptions({ dither: "" as any }), TypeError);
  // deno-lint-ignore no-explicit-any
  assertThrows(() => normalizeOptions({ dither: undefined as any }), TypeError);
});

Deno.test("normalizeOptions: ditherStrength must be in [0,1]", () => {
  for (
    const bad of [-0.01, 1.01, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]
  ) {
    assertThrows(() => normalizeOptions({ ditherStrength: bad }), RangeError);
  }
  for (const ok of [0, 0.5, 1]) {
    assertEquals(normalizeOptions({ ditherStrength: ok }).ditherStrength, ok);
  }
});

Deno.test("normalizeOptions: detailWeight must be in [0,1]", () => {
  for (
    const bad of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]
  ) {
    assertThrows(() => normalizeOptions({ detailWeight: bad }), RangeError);
  }
  for (const ok of [0, 0.25, 1]) {
    assertEquals(normalizeOptions({ detailWeight: ok }).detailWeight, ok);
  }
});

Deno.test("normalizeOptions: kmeansIterations rejects negative/non-integer/>100", () => {
  for (
    const bad of [-1, 1.5, 101, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]
  ) {
    assertThrows(() => normalizeOptions({ kmeansIterations: bad }), RangeError);
  }
  for (const ok of [0, 1, 15, 100]) {
    assertEquals(normalizeOptions({ kmeansIterations: ok }).kmeansIterations, ok);
  }
});

Deno.test("normalizeOptions: numeric fields reject non-number input (NEW-E-004)", () => {
  // JS consumers can pass strings/booleans/null past the TS type. Before this
  // fix, `Number("16") === 16` and `Number(true) === 1` slipped through and
  // either silently quantised at the coerced value (string "16") or threw a
  // confusing RangeError with the coerced numeric value (boolean false → 0,
  // null → 0). After: typeof check throws TypeError with the original input
  // shape in the message so the misuse is obvious at the call site.
  const numericFields = [
    "paletteSize",
    "ditherStrength",
    "detailWeight",
    "kmeansIterations",
  ] as const;
  const badValues: unknown[] = ["16", "0.5", "", true, false, null, {}, []];
  for (const field of numericFields) {
    for (const bad of badValues) {
      assertThrows(
        // deno-lint-ignore no-explicit-any
        () => normalizeOptions({ [field]: bad } as any),
        TypeError,
        field,
        `expected TypeError for ${field}=${JSON.stringify(bad)}`,
      );
    }
  }
});

// 4A-04 (4th-pass review): toRawImage validates dim / data integrity
// before any pipeline stage runs. A caller passing zero-dim or a
// mismatched-length data buffer must hit a `RangeError` at the entry
// point, not a cryptic downstream NaN / OOB indexing failure.
Deno.test("toRawImage: width=0 throws RangeError", async () => {
  await assertRejects(
    () =>
      quantizeCpu({
        width: 0,
        height: 4,
        data: new Uint8ClampedArray(0),
      }),
    RangeError,
    "width",
  );
});

Deno.test("toRawImage: height=0 throws RangeError", async () => {
  await assertRejects(
    () =>
      quantizeCpu({
        width: 4,
        height: 0,
        data: new Uint8ClampedArray(0),
      }),
    RangeError,
    "height",
  );
});

Deno.test("toRawImage: data.length mismatch throws RangeError", async () => {
  await assertRejects(
    () =>
      quantizeCpu({
        width: 4,
        height: 4,
        // 4*4*4 = 64 bytes expected, supply 32.
        data: new Uint8ClampedArray(32),
      }),
    RangeError,
    "data.length",
  );
});

Deno.test("toRawImage: non-integer width throws RangeError", async () => {
  await assertRejects(
    () =>
      quantizeCpu({
        width: 4.5,
        height: 4,
        data: new Uint8ClampedArray(4.5 * 4 * 4),
      }),
    RangeError,
    "width",
  );
});

Deno.test("quantizeCpu surfaces normalizeOptions errors at call time", async () => {
  await assertRejects(
    () => quantizeCpu(flat(), { paletteSize: 0 }),
    RangeError,
    "paletteSize",
  );
  await assertRejects(
    // deno-lint-ignore no-explicit-any
    () => quantizeCpu(flat(), { dither: "foo" as any }),
    TypeError,
    "dither",
  );
  // 3E-010 (3rd-pass review): NEW-E-004's typeof-number gate must also
  // throw through the public API, not just when normalizeOptions is
  // called directly. Otherwise a JS caller could smuggle a string past
  // the type system and observe Number-coerced behaviour at the entry
  // point. One sample is enough — the direct-call coverage above pins
  // the matrix; this test just confirms the wiring.
  await assertRejects(
    // deno-lint-ignore no-explicit-any
    () => quantizeCpu(flat(), { paletteSize: "16" as any }),
    TypeError,
    "paletteSize",
  );
});
