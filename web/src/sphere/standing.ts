// A photo laid on today's walls from where it was taken: each direction from the panorama goes
// out to the walls around it (or the street, or past the roofs), and that point is seen by the
// camera that took the photo (the pipeline's registration.Standing, composite_standing). The
// shader in photo.ts does the same per pixel; this one answers a click and is tested.

/** The camera where the photographer stood, placed from its panorama (metres east, north,
 *  up), its turn from the world (a rotation vector), focal length (photo pixels), lens shift
 *  (shares of the width, height) and the photo's size (height, width). */
export interface Standing {
  position: number[];
  turn: number[];
  focal: number;
  shift: number[];
  size: [number, number];
}

/** A wall from (x0, y0) to (x1, y1), from the street up to its top: metres from the panorama. */
export type WallRow = [x0: number, y0: number, x1: number, y1: number, top: number];

export const NEAREST_M = 0.5; // a wall nearer than this along a ray is not met (pipeline: Blocks)
export const STREET_REACH_M = 50; // the street met this far at most (facades.STREET_REACH_M)
export const HIDDEN = 1.1; // a point this much farther than a wall in its way is hidden
export const STRETCHED = 6; // a part of the photo laid this many times larger than the photo as a whole: smeared
const STEP_RAD = 1e-4; // how far a direction is moved to see how the photo stretches there
export const ROOFS_PAST_M = 2; // over the roofs, walls are met this far past their ends: the sky
// does not tear between two walls (their corners are joined, not the gaps between blocks)

/** How far along `d` (unit) from `from` it first meets a wall between the street and `top`
 *  (all of the wall above the street: `roofless`), short of `before`; Infinity: none (the walls
 *  come joined at their corners: see the pipeline's scene.walls_shown). */
function toWall(from: number[], d: number[], walls: WallRow[], ground: number, roofless = false, before = Infinity): number {
  let nearest = before;
  for (const [x0, y0, x1, y1, top] of walls) {
    const [ex, ey, ax, ay] = [x1 - x0, y1 - y0, x0 - from[0], y0 - from[1]];
    const across = d[0] * ey - d[1] * ex;
    if (across === 0) continue; // along it
    const t = (ax * ey - ay * ex) / across;
    const s = (ax * d[1] - ay * d[0]) / across;
    const past = roofless ? ROOFS_PAST_M / Math.hypot(ex, ey) : 0;
    const height = from[2] + t * d[2];
    if (t > NEAREST_M && t < nearest && s >= -past && s <= 1 + past && height >= ground && (roofless || height <= top)) nearest = t;
  }
  return nearest;
}

/** Whether along `d` from `from` today's point is the sky: no wall nor street met. */
function isSky(from: number[], d: number[], walls: WallRow[], ground: number): boolean {
  if (toWall(from, d, walls, ground) < Infinity) return false;
  const street = d[2] < 0 ? (ground - from[2]) / d[2] : Infinity;
  return !(street > 0 && street <= STREET_REACH_M);
}

/** How far along `d` (unit, east-north-up) from `from` today's point stands: the nearest wall,
 *  else the street when it points down (within STREET_REACH_M), else (the sky) over the roofs
 *  as far as the wall under it, else Infinity. */
export function reach(from: number[], d: number[], walls: WallRow[], ground: number): number {
  const wall = toWall(from, d, walls, ground);
  if (wall < Infinity) return wall;
  const street = d[2] < 0 ? (ground - from[2]) / d[2] : Infinity;
  if (street > 0 && street <= STREET_REACH_M) return street;
  return d[2] > 0 ? toWall(from, d, walls, ground, true) : Infinity;
}

/** How far along a direction from the panorama today's point stands (see reach). */
export function depthAlong(d: number[], walls: WallRow[], ground: number): number {
  return reach([0, 0, 0], d, walls, ground);
}

/** The rotation of a rotation vector, row by row (Rodrigues). */
function rotation([x, y, z]: number[]): number[][] {
  const angle = Math.hypot(x, y, z);
  if (angle === 0) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const [kx, ky, kz] = [x / angle, y / angle, z / angle];
  const [c, s, v] = [Math.cos(angle), Math.sin(angle), 1 - Math.cos(angle)];
  return [
    [c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s],
    [ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s],
    [kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v],
  ];
}

/** The lens times the turn (K R): from the world to the photo's homogeneous pixels. */
export function lensTurn(s: Standing): number[][] {
  const [h, w] = s.size;
  const [cx, cy] = [(0.5 + s.shift[0]) * w, (0.5 + s.shift[1]) * h];
  const k = [
    [s.focal, 0, cx],
    [0, s.focal, cy],
    [0, 0, 1],
  ];
  const r = rotation(s.turn);
  return k.map((row) => [0, 1, 2].map((j) => row[0] * r[0][j] + row[1] * r[1][j] + row[2] * r[2][j]));
}

const times = (m: number[][], v: number[]) => m.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);

/** Where the point `metres` along `d` from the panorama falls in the photo (pixels; NaN behind
 *  the camera): K R d + (1 / metres) K R (panorama - camera), its depth divided out. */
export function photoPixel(d: number[], metres: number, s: Standing): [number, number] {
  const kr = lensTurn(s);
  const ahead = times(kr, d);
  const step = times(kr, s.position.map((v) => -v));
  const seen = ahead.map((v, i) => v + step[i] / metres);
  return seen[2] > 0 ? [seen[0] / seen[2], seen[1] / seen[2]] : [NaN, NaN];
}

/** How little of the photo a small turn of the direction `d` spans, the way it is most
 *  squeezed: the lesser singular value of the photo's pixels per radian there (as the shader
 *  works it out per pixel of the screen). */
export function squeezed(d: number[], s: Standing, walls: WallRow[], ground: number): number {
  const across = Math.abs(d[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit = (v: number[]) => v.map((x) => x / Math.hypot(...v));
  const e1 = unit(cross(d, across));
  const e2 = cross(d, e1);
  const at = (v: number[]) => photoPixel(v, depthAlong(v, walls, ground), s);
  const here = at(d);
  const [a, b] = [e1, e2].map((e) => {
    const there = at(unit(d.map((x, i) => x + STEP_RAD * e[i])));
    return [(there[0] - here[0]) / STEP_RAD, (there[1] - here[1]) / STEP_RAD];
  });
  const square = a[0] ** 2 + a[1] ** 2 + b[0] ** 2 + b[1] ** 2;
  const det = a[0] * b[1] - a[1] * b[0];
  return Math.sqrt(Math.max(square - Math.sqrt(Math.max(square ** 2 - 4 * det ** 2, 0)), 0) / 2);
}

/** Where a direction from the panorama shows in the photo ([0, 1]², y down), or null: behind
 *  the camera, off the photo, hidden from the photographer by a wall nearer to them, or (the
 *  photo's pixels per radian as laid, `density`, given) where the photo fades out, smeared,
 *  but in the sky (see the shader in photo.ts). */
export function standingAt(d: number[], s: Standing, walls: WallRow[], ground: number, density?: number): [number, number] | null {
  const metres = depthAlong(d, walls, ground);
  const [x, y] = photoPixel(d, metres, s);
  const [u, v] = [x / s.size[1], y / s.size[0]];
  if (!(u >= 0 && u <= 1 && v >= 0 && v <= 1)) return null;
  if (metres < Infinity) {
    const away = d.map((v, i) => v * metres - s.position[i]);
    const length = Math.hypot(...away);
    const toward = away.map((v) => v / length);
    if (toWall(s.position, toward, walls, ground, false, length / HIDDEN) < length / HIDDEN) return null;
  }
  const smeared = () => (squeezed(d, s, walls, ground) * STRETCHED) / density! <= 0.5;
  if (density !== undefined && !isSky([0, 0, 0], d, walls, ground) && smeared()) return null;
  return [u, v];
}
