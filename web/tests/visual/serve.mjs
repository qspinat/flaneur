// The built site served as a static host would serve it: text compressed (gzip), so that
// timings measured here are those of a real host. Usage: node tests/visual/serve.mjs [port]
import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { extname, join, normalize } from 'node:path';
import { createGzip } from 'node:zlib';

const root = new URL('../../dist/', import.meta.url).pathname;
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.webp': 'image/webp', '.svg': 'image/svg+xml' };
const compressed = new Set(['.html', '.js', '.css', '.json', '.svg']);

createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname));
  let file = join(root, path === '/' ? 'index.html' : path);
  if (!file.startsWith(root)) return res.writeHead(403).end();
  try {
    if (statSync(file).isDirectory()) file = join(file, 'index.html');
  } catch {
    return res.writeHead(404).end();
  }
  const ext = extname(file);
  const gzip = compressed.has(ext) && /gzip/.test(req.headers['accept-encoding'] ?? '');
  res.writeHead(200, { 'Content-Type': types[ext] ?? 'application/octet-stream', ...(gzip ? { 'Content-Encoding': 'gzip' } : {}) });
  const stream = createReadStream(file);
  (gzip ? stream.pipe(createGzip()) : stream).pipe(res);
}).listen(Number(process.argv[2] ?? 4174));
