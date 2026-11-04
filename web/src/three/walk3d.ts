import { yearsTaken } from '../walk';

/** The 3D walk's data, as written by `flaneur scene`, and what the screen derives from it.
 * Positions are metres east (x) and north (y) of the scene's origin, z up. */

export type Link = [id: string, bearing: number, distance: number];

export interface Panorama {
  id: string;
  x: number;
  y: number;
  z: number; // the camera, above the origin's ground
  lon: number; // where it is on the map
  lat: number;
  azimuth: number; // compass direction of the panorama's centre
  taken: string;
  credit: string;
  url: string; // the picture in Panoramax's own viewer
  image: string; // the equirectangular picture
  links: Link[];
}

export interface Wall {
  line: [[number, number], [number, number]];
  height: number;
}

export interface Pose {
  x: number;
  y: number;
  z: number;
  heading: number;
  horizon: number; // row of the horizon, as a fraction of the height from the top
  fov: number; // horizontal, degrees
}

export interface OldPhoto {
  id: string;
  title: string;
  years: [number | null, number | null];
  credit: string;
  image: string;
  aspect: number; // height / width
  pose: Pose;
  walls: Wall[]; // the facades it is projected onto, as they stood,
  base: number; // from the street's height there
}

export interface Building {
  footprint: [number, number][];
  base: number;
  height: number;
}

export interface Ground {
  x0: number;
  y0: number;
  step: number;
  columns: number;
  heights: number[]; // row by row, south to north
}

/** A photo of the photo walk, where the 3D walk needs it: in the scene and on the map. */
export interface WalkSpot {
  id: string;
  x: number;
  y: number;
  lon: number;
  lat: number;
  heading: number | null;
  street: [string, string] | null;
  arrondissement: number | null;
}

export interface Scene {
  panoramas: Panorama[];
  walk: WalkSpot[]; // the photo walk's photos: the way back, the plate, the plan
  buildings: Building[];
  cores: Building[]; // the buildings shrunk by 3 m: what hides an old photo behind them
  ground: Ground;
  photos: OldPhoto[];
}

const STEP_DEG = 45;

/** The signed turn from compass direction a to b, in (−180, 180]: positive to the right. */
export function turn(a: number, b: number): number {
  const d = (((b - a) % 360) + 540) % 360 - 180;
  return d === -180 ? 180 : d;
}

/** The way on from a panorama when looking towards `heading`: the link nearest that
 * direction, within 45°. */
export function stepTarget(here: Panorama, heading: number): Link | null {
  const ahead = here.links.filter(([, bearing]) => Math.abs(turn(heading, bearing)) <= STEP_DEG);
  if (!ahead.length) return null;
  return ahead.reduce((a, b) => (Math.abs(turn(heading, a[1])) <= Math.abs(turn(heading, b[1])) ? a : b));
}

function nearest(panoramas: Panorama[], distance: (p: Panorama) => number): Panorama {
  return panoramas.reduce((a, b) => (distance(a) <= distance(b) ? a : b));
}

export function nearestPanorama(scene: Scene, x: number, y: number): Panorama {
  return nearest(scene.panoramas, (p) => (p.x - x) ** 2 + (p.y - y) ** 2);
}

/** Scene metres to three.js axes: east → x, up → y, north → −z. */
export function toThree(x: number, y: number, z: number): [number, number, number] {
  return [x, z, -y];
}

const SHOWN_M = 80;
const FADE_M = 10;
export const MAX_SHOWN = 8;

/** The old photos taken near (x, y), whichever way one looks, with how strongly they show:
 * fully, then fading out over the last 10 m of 80. At most eight, the furthest camera first
 * so that the nearest lies on top. */
export function nearby(photos: OldPhoto[], x: number, y: number): [OldPhoto, number][] {
  return photos
    .map((p) => ({ p, distance: Math.hypot(p.pose.x - x, p.pose.y - y) }))
    .filter(({ distance }) => distance < SHOWN_M)
    .sort((a, b) => a.distance - b.distance)
    .slice(0, MAX_SHOWN)
    .reverse()
    .map(({ p, distance }): [OldPhoto, number] => [p, Math.min(1, (SHOWN_M - distance) / FADE_M)]);
}

/** The years that have photos taken within `metres` of (x, y), for marks on the slider. */
export function yearMarks(photos: OldPhoto[], x: number, y: number, metres = SHOWN_M): number[] {
  return yearsTaken(photos.filter((p) => Math.hypot(p.pose.x - x, p.pose.y - y) <= metres));
}

/** The panorama nearest a point of the map. */
export function nearestTo(scene: Scene, lon: number, lat: number): Panorama {
  const squeeze = Math.cos((lat * Math.PI) / 180); // a degree of longitude is shorter
  return nearest(scene.panoramas, (p) => ((p.lon - lon) * squeeze) ** 2 + (p.lat - lat) ** 2);
}

const RETURN_M = 60;
const METRES_PER_DEGREE_TURNED = 0.2; // facing the other way weighs as 36 m further
const UNKNOWN_TURN_DEG = 90; // a photo whose direction is not known: as if facing aside

/** The photo of the walk nearest where one stands (x, y) and facing most nearly the way one
 * looks; null when none was taken within 60 m. */
export function photoToward(photos: WalkSpot[], x: number, y: number, heading: number): WalkSpot | null {
  let best: WalkSpot | null = null;
  let bestCost = Infinity;
  for (const p of photos) {
    const distance = Math.hypot(p.x - x, p.y - y);
    if (distance > RETURN_M) continue;
    const turned = p.heading == null ? UNKNOWN_TURN_DEG : Math.abs(turn(heading, p.heading));
    const cost = distance + turned * METRES_PER_DEGREE_TURNED;
    if (cost < bestCost) [best, bestCost] = [p, cost];
  }
  return best;
}
