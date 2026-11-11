/** The walk's data, as written by `flaneur export`, and what the screen derives from it. */

export interface Link {
  id: string;
  distance_m: number;
}

export type Direction = 'forward' | 'back' | 'left' | 'right';

export interface Today {
  image: string;
  taken: string;
  credit: string;
  url: string;
}

export interface Photo {
  id: string;
  title: string;
  caption: string | null;
  years: [number | null, number | null];
  photographer: string;
  credit: string;
  record_url: string | null;
  lon: number;
  lat: number;
  heading: number | null;
  street: [string, string] | null; // as on its plate: kind, then name
  arrondissement: number | null;
  images: Record<string, string>; // each size by its width in pixels: {"960": …, "1600": …}
  today: Today | null;
  links: Record<Direction, Link | null>;
  eras: string[]; // the same view at other dates, one per date, oldest first
  in_3d: boolean; // shown in the 3D walk (placed in its panorama)
  no_3d: boolean; // no panorama sees it: no 3D walk from it
  flipped?: boolean; // the print turned the right way round: shown mirrored (not in an older export)
  aspect: number | null; // height over width (null in an older export)
}

export interface Walk {
  photos: Photo[];
}

export const DATA = 'data/'; // relative: the site may be served from a sub-path

/** How many device pixels to draw per CSS pixel (maps, the 3D view, the photographs): at most
 *  two, a phone's 2.6 would cost 70% more to draw and fetch, for little to see. Read when
 *  asked: zooming the page changes it. */
export function density(): number {
  return Math.min(2, globalThis.devicePixelRatio || 1); // (1 in the tests: no screen)
}

/** Every size of a photograph, for `srcset`: "…-960.webp 960w, ….webp 1600w". */
export function srcsetOf(images: Record<string, string>): string {
  return Object.entries(images).map(([width, path]) => `${DATA}${path} ${width}w`).join(', ');
}

/** A link out of the site, as the data gives it, if it is an https address (a `javascript:`
 *  one would run on this page); else null. */
export function httpsUrl(href: string | null | undefined): string | null {
  if (!href) return null;
  try {
    return new URL(href).protocol === 'https:' ? href : null;
  } catch {
    return null;
  }
}

/** The size of a photograph to show `pixels` wide: the smallest at least that wide (or within a
 *  tenth of it), else the largest. Chosen here rather than by the browser's srcset, so that
 *  what is fetched ahead is exactly what will be shown. */
export function fitting(images: Record<string, string>, pixels: number): string {
  const widths = Object.keys(images).map(Number).sort((a, b) => a - b);
  return images[widths.find((w) => w >= pixels * 0.9) ?? widths[widths.length - 1]];
}

/** The largest size of a photograph. */
export function largest(images: Record<string, string>): string {
  return fitting(images, Infinity);
}

/** The photo a URL points to: `#/paris_musees:75860`, or null. */
export function photoIdFrom(hash: string): string | null {
  try {
    return decodeURIComponent(hash.replace(/^#\/?/, '')) || null;
  } catch {
    return null; // a malformed link: as if it named none
  }
}

export function hashOf(id: string): string {
  return `#/${encodeURIComponent(id)}`;
}

/** Writes `hash` into the address: as a new entry (`push`), or into this one unless it already
 *  says so. A browser may refuse (Firefox, after very many changes in a few seconds): the photo
 *  must still show, so the refusal is let be. */
export function setAddress(
  hash: string,
  push: boolean,
  history: Pick<History, 'pushState' | 'replaceState'> = window.history,
  location: Pick<Location, 'hash'> = window.location,
): void {
  if (!push && location.hash === hash) return;
  try {
    if (push) history.pushState(null, '', hash);
    else history.replaceState(null, '', hash);
  } catch (error) {
    if (!(error instanceof DOMException && error.name === 'SecurityError')) throw error;
  }
}

/** "1912", "1905–1906", or "" when unknown. */
export function yearsLabel([start, end]: [number | null, number | null]): string {
  if (start == null && end == null) return '';
  if (start == null || end == null || start === end) return String(start ?? end);
  return `${start}–${end}`;
}

/** A photo's date as shown under it: its years, or that they are not known. */
export function whenLabel(years: [number | null, number | null]): string {
  return yearsLabel(years) || 'Date inconnue';
}

/** "40 m", "1,2 km". */
export function distanceLabel(metres: number): string {
  if (metres < 1000) return `${Math.round(metres)} m`;
  return `${(metres / 1000).toFixed(1).replace('.', ',')} km`;
}

/** The arrondissement as on a plate: "5e Arrt", "1er Arrt". */
export function arrondissementLabel(n: number | null): string {
  if (n == null) return '';
  return `${n}${n === 1 ? 'er' : 'e'} Arrt`;
}

/** This photo and the same view at other dates, oldest first. */
export function erasOf(photo: Photo, byId: Map<string, Photo>): Photo[] {
  const all = [photo, ...photo.eras.map((id) => byId.get(id)).filter((p): p is Photo => !!p)];
  return all.sort((a, b) => (a.years[0] ?? 9999) - (b.years[0] ?? 9999));
}

/** Where a walk begins best: the photo with the most ways on (and a view of today). */
export function richest(photos: Photo[]): Photo {
  const ways = (p: Photo) => Object.values(p.links).filter(Boolean).length + (p.today ? 1 : 0);
  return photos.reduce((best, p) => (ways(p) > ways(best) ? p : best));
}

/** The years shown: from the first photographs of Paris to today. */
export const FIRST_YEAR = 1850;
export const LAST_YEAR = new Date().getFullYear();

/** A photo's years as a span (one known year: that year), or null when undated. */
export function span([start, end]: [number | null, number | null]): [number, number] | null {
  if (start == null && end == null) return null;
  return [start ?? end!, end ?? start!];
}

/** Whether a photograph is shown for the years chosen: its years meet them; an undated one
 *  only when every year is chosen (`whole`). */
export function shownIn(years: [number | null, number | null], [min, max]: [number, number], whole: boolean): boolean {
  const known = span(years);
  return known ? known[0] <= max && known[1] >= min : whole;
}

/** Every year some of these photographs were taken, once each, in order (for marks on a bar). */
export function yearsTaken(photos: { years: [number | null, number | null] }[]): number[] {
  const years = new Set<number>();
  for (const p of photos) {
    const known = span(p.years);
    if (known) for (let year = known[0]; year <= known[1]; year++) years.add(year);
  }
  return [...years].sort((a, b) => a - b);
}

/** A place in the 3D walk, as its link says: where, which way, which years. */
export interface Place3d {
  lon: number;
  lat: number;
  heading: number;
  range: [number, number];
}

/** `#2.34712,48.85301,271,1900-1920`. */
export function format3d({ lon, lat, heading, range: [min, max] }: Place3d): string {
  return `#${lon.toFixed(5)},${lat.toFixed(5)},${Math.round(heading)},${min}-${max}`;
}

export function parse3d(hash: string): Place3d | null {
  const found = /^#\/?(-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(\d{4})-(\d{4})$/.exec(hash);
  if (!found) return null;
  const [lon, lat, heading, min, max] = found.slice(1).map(Number);
  if ([lon, lat, heading].some((n) => !Number.isFinite(n)) || min > max) return null;
  return { lon, lat, heading: ((heading % 360) + 360) % 360, range: [min, max] };
}

/** A data file of the site. */
export async function json<T>(name: string): Promise<T> {
  const response = await fetch(`${DATA}${name}`);
  if (!response.ok) throw new Error(`${name}: ${response.status}`);
  return response.json() as Promise<T>;
}

/** What the photo page offers of the 3D walk: a link to it, or a line saying no view of
 *  today is near enough (a photo no panorama sees), or nothing (not placed yet). */
export function offerOf3d(photo: Photo): 'link' | 'far' | null {
  return photo.in_3d ? 'link' : photo.no_3d ? 'far' : null;
}

/** The 3D walk's link opening on a photo, where it was placed: `#photo=<id>,<min>-<max>`. */
export function photo3d(id: string, [min, max]: [number, number]): string {
  return `#photo=${encodeURIComponent(id)},${min}-${max}`;
}

export function parsePhoto3d(hash: string): { id: string; range: [number, number] } | null {
  const found = /^#photo=([^,]+),(\d{4})-(\d{4})$/.exec(hash);
  if (!found) return null;
  const [min, max] = [Number(found[2]), Number(found[3])];
  return min > max ? null : { id: decodeURIComponent(found[1]), range: [min, max] };
}
