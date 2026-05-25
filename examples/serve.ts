// Minimal static server with correct MIME types for the demo.
//
//   deno run --allow-net --allow-read examples/serve.ts
//
// Defaults to port 8000, repo root as the document root.
// Use this instead of `npx serve` when you don't want to install anything —
// `npx serve` mis-detects .ts as MPEG transport stream (video/mp2t), which
// breaks `<script type="module">` loads.

const PORT = parseInt(Deno.env.get("PORT") ?? "8000", 10);
const ROOT = new URL("..", import.meta.url).pathname;

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript",
  ".mjs": "application/javascript",
  ".ts": "application/javascript", // we don't transpile; suitable only for bundled JS named .ts
  ".json": "application/json",
  ".map": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".css": "text/css",
  ".wgsl": "text/plain",
};

function mimeFor(path: string): string {
  const dot = path.lastIndexOf(".");
  if (dot < 0) return "application/octet-stream";
  return MIME[path.slice(dot).toLowerCase()] ?? "application/octet-stream";
}

Deno.serve({ port: PORT }, async (req) => {
  const url = new URL(req.url);
  let path = decodeURIComponent(url.pathname);
  if (path.endsWith("/")) path += "index.html";

  // Prevent directory traversal.
  if (path.includes("..")) return new Response("Forbidden", { status: 403 });

  const fsPath = ROOT + path.replace(/^\//, "");
  try {
    const file = await Deno.open(fsPath, { read: true });
    const stat = await file.stat();
    if (stat.isDirectory) {
      file.close();
      const indexPath = fsPath.replace(/\/?$/, "/index.html");
      const idx = await Deno.open(indexPath, { read: true });
      return new Response(idx.readable, {
        headers: { "content-type": "text/html; charset=utf-8" },
      });
    }
    return new Response(file.readable, {
      headers: { "content-type": mimeFor(fsPath) },
    });
  } catch (e) {
    if (e instanceof Deno.errors.NotFound) return new Response("Not Found", { status: 404 });
    console.error(e);
    return new Response("Server Error", { status: 500 });
  }
});

console.log(`Serving ${ROOT} at http://localhost:${PORT}`);
console.log(`Open http://localhost:${PORT}/examples/browser.html`);
