// One-shot script: generate a 64x64 blue-noise mask via void-and-cluster and
// print it as base64 so we can paste it into src/dither/blue-noise-data.ts.
//
//   deno run scripts/generate-blue-noise.ts > blue-noise-64.b64
//
// Compute time is a few seconds; rerunning with the same seed reproduces the
// exact same mask, so the constant is stable.

import { generateBlueNoise } from "../src/dither/void-and-cluster.ts";

const SIZE = parseInt(Deno.args[0] ?? "64", 10);
const SEED = parseInt(Deno.args[1] ?? "1337", 10);

console.error(`Generating ${SIZE}x${SIZE} blue noise (seed=${SEED})...`);
const t0 = performance.now();
const mask = generateBlueNoise(SIZE, SEED);
const ms = (performance.now() - t0).toFixed(1);
console.error(`Done in ${ms}ms.`);

const b64 = btoa(String.fromCharCode(...mask.data));
console.log(b64);

// Quality sanity check: print mean and stddev so the user can verify the
// distribution is uniform.
let sum = 0;
let sumSq = 0;
for (const v of mask.data) {
  sum += v;
  sumSq += v * v;
}
const mean = sum / mask.data.length;
const variance = sumSq / mask.data.length - mean * mean;
console.error(
  `mean=${mean.toFixed(2)} stddev=${
    Math.sqrt(variance).toFixed(2)
  } (uniform expected ~127.5, ~73.9)`,
);
