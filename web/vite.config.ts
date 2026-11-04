import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig, type Plugin } from 'vite';

// What the built pages may load: their own files; IGN's map tiles; today's panoramas from the
// hosts Panoramax serves them from, read in the scene being published (a federation: hosts
// come and go). Data that slips past the checks still cannot run script nor load from elsewhere.
const SCENE = resolve(import.meta.dirname, 'public/data/scene.json');
const PANORAMAX = existsSync(SCENE)
  ? [...new Set((JSON.parse(readFileSync(SCENE, 'utf8')).panoramas as { image: string }[]).map((p) => new URL(p.image).origin))]
  : [];
const POLICY = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'", // the map and the 3D view set styles on their elements
  `img-src 'self' data: blob: https://data.geopf.fr ${PANORAMAX.join(' ')}`,
  "connect-src 'self' https://data.geopf.fr", // the map fetches its tiles
  'worker-src blob:', // the map's workers
  "font-src 'self'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'none'",
].join('; ');

/** The policy, in the built pages only: the development server needs its own socket. */
function contentSecurityPolicy(): Plugin {
  return {
    name: 'content-security-policy',
    apply: 'build',
    // right after the charset (which must come first), before anything the policy governs
    transformIndexHtml: (html) => {
      const charset = '<meta charset="UTF-8" />';
      if (!html.includes(charset)) throw new Error(`a page without ${charset}: its security policy would be left out`);
      return html.replace(charset, `$&\n    <meta http-equiv="Content-Security-Policy" content="${POLICY}" />`);
    },
  };
}

// Two pages: the photo walk, and the 3D walk (three.js stays out of the photo walk).
export default defineConfig({
  base: './', // relative: served at qspinat.github.io/flaneur/
  plugins: [contentSecurityPolicy()],
  build: {
    rollupOptions: {
      input: { walk: resolve(import.meta.dirname, 'index.html'), three: resolve(import.meta.dirname, '3d.html') },
    },
  },
});
