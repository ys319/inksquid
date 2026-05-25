// PNG IDAT uses zlib-wrapped deflate (RFC 1950 over RFC 1951), so we ask
// CompressionStream for "deflate" — the zlib header + Adler-32 trailer is
// what the PNG spec mandates. Passing "deflate-raw" would produce a
// header-less stream and the resulting PNG would be unreadable.
export async function deflate(data: Uint8Array): Promise<Uint8Array> {
  const cs = new CompressionStream("deflate");
  const writer = cs.writable.getWriter();
  // TS 5.7+ narrows `Uint8Array<ArrayBufferLike>` for `WritableStreamDefaultWriter`
  // to `Uint8Array<ArrayBuffer>`. The runtime accepts any BufferSource, so the
  // cast just re-widens after the input parameter narrowed it. This is the
  // only such cast in the codebase.
  writer.write(data as Uint8Array<ArrayBuffer>);
  writer.close();
  // close() drains the stream; releaseLock() lets `writer` (and its
  // reference to cs.writable) be GC'd promptly. The stream is single-shot
  // so no further writes happen — current behaviour is unchanged, this is
  // just a "no dangling lock" cleanup.
  writer.releaseLock();
  const buf = await new Response(cs.readable).arrayBuffer();
  return new Uint8Array(buf);
}
