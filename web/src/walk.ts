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
  in_3d: boolean; // shown in the 3D walk (its placement confirmed)
}

export interface Walk {
  photos: Photo[];
}

export const DATA = 'data/'; // relative: the site may be served from a sub-path

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

/** The largest size of a photograph. */
export function largest(images: Record<string, string>): string {
  return images[Math.max(...Object.keys(images).map(Number))];
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
