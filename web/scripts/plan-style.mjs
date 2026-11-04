// The plan's style, made once from IGN's vector Plan IGN (Géoplateforme, Licence Ouverte 2.0):
// its colours turned to the site's paper (a ramp from --mount to --albumen, style.css, a fifth of
// the original hue kept so that water and gardens still read), and the layers this site never
// needs left out (pictograms and patterns, relief, boundaries, road numbers, the cadastre, and what
// shows only farther out than the plan ever goes). Checked against MapLibre's style rules, then
// written to src/plan-style.json, bundled with the map code (which skips that check when a map
// opens). Made again with `npm run plan-style` (when IGN changes its style or its tiles: the map
// would draw nothing for a renamed layer).
import { readFileSync, writeFileSync } from 'node:fs';
import { Color, validateStyleMin } from '@maplibre/maplibre-gl-style-spec';

const SOURCE = 'https://data.geopf.fr/annexes/ressources/vectorTiles/styles/PLAN.IGN/standard.json';
const OUT = new URL('../src/plan-style.json', import.meta.url);
const HUE_KEPT = 0.2;
// the zooms the plan's tiles are read at: farther out, IGN's tiles weigh 300 KB and more (100 KB
// at most from 15); closer, they hold nothing more the plan draws (only the road markers it leaves
// out), and those of 16 are drawn larger
const LOWEST_ZOOM = 15;
const HIGHEST_ZOOM = 16;
const LEFT_OUT = ['oro_', 'toponyme_oro_', 'limite_', 'toponyme_limite', 'toponyme_routier_borne', 'toponyme_routier_numero', 'parcellaire_', 'toponyme_parcellaire_'];
const SAME_FONT = { 'Source Sans Pro': 'Source Sans Pro Regular' }; // one file under two names: fetched once

// the site's ink and paper (style.css), kept in the style so that a test sees them change
const css = readFileSync(new URL('../src/style.css', import.meta.url), 'utf8');
const token = (name) => {
  const hex = css.match(new RegExp(`--${name}:\\s*(#[0-9a-f]{6})`, 'i'))?.[1];
  if (!hex) throw new Error(`style.css has no --${name} colour: say which colour the plan is faded to`);
  return hex;
};
const TOKENS = { ink: token('mount'), paper: token('albumen') };
const [INK, PAPER] = [TOKENS.ink, TOKENS.paper].map((hex) => Color.parse(hex).rgb.slice(0, 3).map((x) => x * 255));

/** A colour onto the paper: its lightness taken along the ramp from ink to albumen; anything
 *  else (a font, an expression's operator) as it was. */
function faded(value) {
  const c = Color.parse(value)?.rgb;
  if (!c) return value;
  const [r, g, b, a] = c.map((x, i) => (i < 3 ? x * 255 : x));
  const light = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  const out = [0, 1, 2].map((i) => Math.round((INK[i] + (PAPER[i] - INK[i]) * light) * (1 - HUE_KEPT) + [r, g, b][i] * HUE_KEPT));
  return `rgba(${out.join(', ')}, ${+a.toFixed(3)})`;
}

const recolour = (v) =>
  typeof v === 'string' ? faded(v) : Array.isArray(v) ? v.map(recolour) : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, recolour(x)])) : v;

const json = async (url) => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  return response.json();
};
const style = await json(SOURCE);
const sources = Object.entries(style.sources);
if (sources.length !== 1) throw new Error(`IGN's style now has ${sources.length} sources: say which the plan keeps`);
const [[name, { url, ...source }]] = sources;
const { minzoom, maxzoom } = await json(url); // named here: no metadata fetched before a tile

const layers = style.layers.filter(
  (layer) =>
    !LEFT_OUT.some((p) => (layer['source-layer'] ?? '').startsWith(p)) &&
    !(layer.layout?.['icon-image'] && !layer.layout['text-field']) &&
    !layer.paint?.['fill-pattern'] &&
    !layer.paint?.['line-pattern'] &&
    !(layer.maxzoom <= LOWEST_ZOOM),
);
const unpictured = (props) => props && Object.fromEntries(Object.entries(props).filter(([key]) => !key.startsWith('icon-')));
for (const layer of layers) {
  // a name beside a pictogram (a metro station's) kept, without its pictogram, on its point
  if (layer.layout?.['icon-image']) {
    delete layer.layout['text-anchor'];
    delete layer.layout['text-offset'];
  }
  layer.layout = unpictured(layer.layout);
  layer.paint = recolour(unpictured(layer.paint));
  const font = layer.layout?.['text-font'];
  if (Array.isArray(font)) layer.layout['text-font'] = font.map((f) => SAME_FONT[f] ?? f);
}

const made = {
  version: 8,
  name: 'Plan IGN, sur papier',
  metadata: { from: SOURCE, faded: TOKENS },
  glyphs: style.glyphs,
  sources: { [name]: { ...source, minzoom: Math.max(minzoom, LOWEST_ZOOM), maxzoom: Math.min(maxzoom, HIGHEST_ZOOM), attribution: '<a href="https://www.ign.fr/">IGN</a> Plan IGN' } },
  // the paper under the tiles, at once, and where a tile has not come yet
  layers: [{ id: 'paper', type: 'background', paint: { 'background-color': `rgb(${PAPER.map(Math.round).join(', ')})` } }, ...layers],
};
const errors = validateStyleMin(made);
if (errors.length) throw new Error(`the plan's style breaks MapLibre's rules:\n${errors.map((e) => e.message).join('\n')}`);
const text = JSON.stringify(made);
writeFileSync(OUT, text);
console.log(`${style.layers.length} layers, ${layers.length} kept; ${Math.round(text.length / 1024)} KB`);
