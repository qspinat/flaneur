// The built site served as a static host would serve it: text compressed (gzip), so that
// timings measured here are those of a real host, under a sub-path as GitHub Pages serves it.
// Usage: [HOST=0.0.0.0] node tests/visual/serve.mjs [port] [sub-path, e.g. /flaneur/]
// Listens on this machine only (127.0.0.1); HOST=0.0.0.0 opens it to the local network, to try
// the site from a phone.
import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { extname, join, normalize } from 'node:path';
import { createGzip } from 'node:zlib';

const root = new URL('../../dist/', import.meta.url).pathname;
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.webp': 'image/webp', '.svg': 'image/svg+xml' };
const compressed = new Set(['.html', '.js', '.css', '.json', '.svg']);

const mount = process.argv[3] ?? '/';

createServer((req, res) => {
  const asked = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (!asked.startsWith(mount)) return res.writeHead(404).end();
  const path = normalize(`/${asked.slice(mount.length)}`);
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
}).listen(Number(process.argv[2] ?? 4174), process.env.HOST || '127.0.0.1', function () {
  const { address, port } = this.address();
  console.log(`http://${address}:${port}${mount}`);
});
