// Bundles src/mod.ts into a single browser-ready ESM file at
// examples/inksquid.bundle.js so examples/browser.html can be served by any
// plain static server (e.g. `npx serve`) without needing a TS-aware build
// tool.
//
//   deno run -A examples/bundle.ts
//   deno run -A examples/bundle.ts --watch    # rebuild on source changes
//
// Uses esbuild via npm:; Deno fetches it directly from the npm registry, no
// package.json / node_modules needed in this repo.

import * as esbuild from "npm:esbuild@^0.24.0";

const watch = Deno.args.includes("--watch");
const OUT = "examples/inksquid.bundle.js";

const buildOptions: esbuild.BuildOptions = {
  entryPoints: ["src/mod.ts"],
  bundle: true,
  format: "esm",
  target: "es2022",
  outfile: OUT,
  sourcemap: "linked",
  platform: "browser",
  minify: false,
  logLevel: "info",
};

if (watch) {
  const ctx = await esbuild.context(buildOptions);
  await ctx.watch();
  console.log("Watching for changes — Ctrl+C to stop.");
} else {
  const result = await esbuild.build(buildOptions);
  if (result.errors.length) {
    for (const e of result.errors) console.error(e);
    Deno.exit(1);
  }
  const stat = await Deno.stat(OUT);
  console.log(`Wrote ${OUT} (${(stat.size / 1024).toFixed(1)} KB)`);
  esbuild.stop();
}
