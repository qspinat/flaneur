// The 3D walk's scene (scene.json, see pipeline scene.build3d) and what is worked out of it
// without the viewer: where photos show, the way on, the photo of the walk one faces.
import { shownIn, type Photo } from '../walk';

export type Link = [index: number, bearing: number, metres: number];

/** A panorama of today: levelled from its azimuth (where its picture's centre faces), roll and
 *  pitch (as measured by the pipeline's `flaneur level`). */
export interface Panorama {
  id: string;
  lon: number;
  lat: number;
  azimuth: number;
  roll: number;
  pitch: number;
  taken: string;
  credit: number; // in Scene.credits
  image: string; // on the site (a photo in it), else on Panoramax
  links: Link[]; // the next ones along the street, by index in Scene.panoramas
}

/** An old photo in a panorama: a 3×3 matrix (row by row) from the photo's [0, 1]² to
 *  directions in the panorama picture's own frame (see the pipeline's registration). */
export interface Registration {
  panorama: string;
  m: number[];
  view: { heading: number; pitch: number; fov: number };
}

export interface Shown {
  panorama: string;
  registration: Registration;
  strength: number; // 1 at its home, fading on the neighbours it is carried to
}

export interface OldPhoto {
  id: string;
  title: string;
  years: [number | null, number | null];
  credit: string;
  image: string;
  flipped: boolean;
  shown: Shown[]; // its home first
}

export interface Scene {
  credits: string[];
  panoramas: Panorama[];
  photos: OldPhoto[];
}

/** A photo of the walk, where it was taken (walk.json). */
export type WalkSpot = Pick<Photo, 'id' | 'lon' | 'lat' | 'heading' | 'street' | 'arrondissement'>;

const rad = (degrees: number) => (degrees * Math.PI) / 180;
const STEP_DEG = 45; // a link this far from the way one looks is not ahead
const POINTED_M = 40; // a photo whose home is this near is pointed to when not shown here
const RETURN_M = 60;
const METRES_PER_DEGREE_TURNED = 0.2; // facing the other way weighs as 36 m further
const UNKNOWN_TURN_DEG = 90; // a photo whose direction is not known: as if facing aside

/** From a to b, in degrees within (-180, 180]. */
export function turn(a: number, b: number): number {
  const d = ((((b - a) % 360) + 540) % 360) - 180;
  return d === -180 ? 180 : d;
}

/** Metres east and north from a to b (near enough for a street). */
function offset(a: { lon: number; lat: number }, b: { lon: number; lat: number }): [number, number] {
  const metres = 111_320;
  return [(b.lon - a.lon) * metres * Math.cos(rad(a.lat)), (b.lat - a.lat) * metres];
}

function bearing(a: { lon: number; lat: number }, b: { lon: number; lat: number }): number {
  const [east, north] = offset(a, b);
  return ((Math.atan2(east, north) * 180) / Math.PI + 360) % 360;
}

function metres(a: { lon: number; lat: number }, b: { lon: number; lat: number }): number {
  return Math.hypot(...offset(a, b));
}

/** A camera's right, ahead and up in the world (east, north, up), as rows: facing `azimuth`,
 *  its nose raised `pitch`, then leaning `roll` to its right (degrees): match.camera_axes. */
export function cameraAxes(azimuth: number, pitch: number, roll: number): number[][] {
  const [h, p, r] = [rad(azimuth), rad(pitch), rad(roll)];
  const forward = [Math.sin(h) * Math.cos(p), Math.cos(h) * Math.cos(p), Math.sin(p)];
  const right = [Math.cos(h), -Math.sin(h), 0];
  const down = [
    forward[1] * right[2] - forward[2] * right[1],
    forward[2] * right[0] - forward[0] * right[2],
    forward[0] * right[1] - forward[1] * right[0],
  ];
  return [
    right.map((v, i) => Math.cos(r) * v + Math.sin(r) * down[i]),
    forward,
    right.map((v, i) => Math.sin(r) * v - Math.cos(r) * down[i]),
  ];
}

/** The viewer's correction that levels a panorama (found right in the spike: its tilt the
 *  opposite of the pitch measured, its roll the same). */
export function correction(p: Panorama): { pan: number; tilt: number; roll: number } {
  return { pan: 0, tilt: rad(-p.pitch), roll: rad(p.roll) };
}

/** The viewer's yaw (radians) for a compass bearing (degrees), in a panorama. */
export function yawOf(p: Panorama, bearingDeg: number): number {
  return rad(turn(0, bearingDeg - p.azimuth));
}

/** The compass bearing (degrees) the viewer's yaw (radians) faces, in a panorama. */
export function bearingOf(p: Panorama, yaw: number): number {
  return ((yaw * 180) / Math.PI + p.azimuth + 360) % 360;
}

/** The photos shown on a panorama, in the years chosen, each with how it shows there, the
 *  strongest last (drawn on top). */
export function shownHere(photos: OldPhoto[], panorama: string, range: [number, number], whole: boolean): [OldPhoto, Shown][] {
  const found: [OldPhoto, Shown][] = [];
  for (const photo of photos) {
    const here = photo.shown.find((s) => s.panorama === panorama);
    if (here && shownIn(photo.years, range, whole)) found.push([photo, here]);
  }
  return found.sort((a, b) => a[1].strength - b[1].strength);
}

/** The photos near, in the years chosen, not shown on this panorama: the way to their home. */
const indexed = new WeakMap<Scene, Map<string, Panorama>>();

/** The scene's panoramas by id, made once. */
function byId(scene: Scene): Map<string, Panorama> {
  let found = indexed.get(scene);
  if (!found) indexed.set(scene, (found = new Map(scene.panoramas.map((p) => [p.id, p]))));
  return found;
}

export function pointers(scene: Scene, here: Panorama, range: [number, number], whole: boolean): { photo: OldPhoto; home: Panorama; bearing: number }[] {
  const found = [];
  for (const photo of scene.photos) {
    if (!shownIn(photo.years, range, whole) || photo.shown.some((s) => s.panorama === here.id)) continue;
    const home = byId(scene).get(photo.shown[0].panorama);
    if (home && metres(here, home) <= POINTED_M) found.push({ photo, home, bearing: bearing(here, home) });
  }
  return found;
}

/** The panorama along the link most nearly the way one looks, if one is ahead. */
export function stepTarget(scene: Scene, here: Panorama, heading: number): Panorama | null {
  const ahead = here.links.filter(([, b]) => Math.abs(turn(heading, b)) <= STEP_DEG);
  if (!ahead.length) return null;
  const best = ahead.reduce((a, b) => (Math.abs(turn(heading, a[1])) <= Math.abs(turn(heading, b[1])) ? a : b));
  return scene.panoramas[best[0]];
}

export function nearestTo(scene: Scene, lon: number, lat: number): Panorama {
  const spot = { lon, lat };
  return scene.panoramas.reduce((a, b) => (metres(a, spot) <= metres(b, spot) ? a : b));
}

/** The photo of the walk nearest where one stands and facing most nearly the way one looks;
 *  null when none was taken within 60 m. */
export function toward(walk: WalkSpot[], here: Panorama, heading: number): WalkSpot | null {
  let best: WalkSpot | null = null;
  let bestCost = Infinity;
  for (const p of walk) {
    const distance = metres(here, p);
    if (distance > RETURN_M) continue;
    const turned = p.heading == null ? UNKNOWN_TURN_DEG : Math.abs(turn(heading, p.heading));
    const cost = distance + turned * METRES_PER_DEGREE_TURNED;
    if (cost < bestCost) [best, bestCost] = [p, cost];
  }
  return best;
}
