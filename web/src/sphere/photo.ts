// Old photos drawn inside a panorama of Photo Sphere Viewer, each where its registration puts
// it: a quad in front of the camera whose every pixel's direction is taken back through the
// registration to the photo (exact: the warp is done per pixel, not by the quad's corners).
// A photo whose camera stood off the panorama (see standing.ts) is laid on the walls around it.
import type { Viewer } from '@photo-sphere-viewer/core';
import * as THREE from 'three';
import { HIDDEN, JOINED_M, NEAREST_M, ROOFS_JOINED_M, STREET_REACH_M, lensTurn, reach, standingAt, type Standing, type WallRow } from './standing';
import { cameraAxes, yawOf, type OldPhoto, type Panorama, type Shown } from './walk3d';

const DISTANCE = 5; // inside the viewer's sphere (radius 10)
const KEPT_TEXTURES = 12;
const UNSEEN = 0.05; // a photo fainter than this, faded out, catches no click

const vertexShader = `
varying vec3 here;
void main() {
  here = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const fragmentShader = `
uniform sampler2D photo;
uniform mat3 toPhoto;
uniform float strength;
uniform bool flipped;
varying vec3 here;
void main() {
  vec3 q = toPhoto * here;
  if (q.z <= 0.0) discard;
  vec2 uv = q.xy / q.z;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) discard;
  if (flipped) uv.x = 1.0 - uv.x;
  gl_FragColor = vec4(texture2D(photo, vec2(uv.x, 1.0 - uv.y)).rgb, strength);
}`;

export const MOST_WALLS = 48; // the walls a photo is laid on, the nearest
const STRETCHED = 6; // a part of the photo laid this many times larger than it is: smeared
const FAR_M = 1000; // a ray of the photo meeting nothing, framed as this far

// standing.ts per pixel: the direction from the panorama out to the walls (else the street, else
// over the roofs to the wall under them, else the sky), seen by the photographer's camera
const standingShader = `
uniform sampler2D photo;
uniform mat3 toWorld;
uniform mat3 lensTurn;
uniform vec3 camera;
uniform vec2 size;
uniform float focal;
uniform vec4 ends[${MOST_WALLS}];
uniform float tops[${MOST_WALLS}];
uniform int count;
uniform float ground;
uniform float strength;
uniform bool flipped;
varying vec3 here;
const float NONE = 1e30;

float toWall(vec3 from, vec3 d, bool roofless, float before) {
  float nearest = before;
  for (int i = 0; i < ${MOST_WALLS}; i++) {
    if (i >= count) break;
    vec4 w = ends[i];
    vec2 e = w.zw - w.xy;
    vec2 a = w.xy - from.xy;
    float across = d.x * e.y - d.y * e.x;
    if (across == 0.0) continue;
    float t = (a.x * e.y - a.y * e.x) / across;
    float s = (a.x * d.y - a.y * d.x) / across;
    float past = (roofless ? ${ROOFS_JOINED_M.toFixed(2)} : ${JOINED_M.toFixed(2)}) / length(e);
    float height = from.z + t * d.z;
    if (t > ${NEAREST_M.toFixed(1)} && t < nearest && s >= -past && s <= 1.0 + past && height >= ground && (roofless || height <= tops[i])) nearest = t;
  }
  return nearest;
}

void main() {
  vec3 d = normalize(toWorld * here);
  float metres = toWall(vec3(0.0), d, false, NONE);
  if (metres == NONE && d.z < 0.0 && ground / d.z <= ${STREET_REACH_M.toFixed(1)}) metres = ground / d.z;
  if (metres == NONE && d.z > 0.0) metres = toWall(vec3(0.0), d, true, NONE);
  vec3 seen = lensTurn * d;
  if (metres < NONE) seen -= lensTurn * camera / metres;
  if (seen.z <= 0.0) discard;
  vec2 px = seen.xy / seen.z;
  vec2 uv = px / size;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) discard;
  if (metres < NONE) {
    vec3 away = d * metres - camera;
    float far = length(away) / ${HIDDEN.toFixed(2)};
    if (toWall(camera, normalize(away), false, far) < far) discard;
  }
  // how little of the photo one pixel of the screen spans, the way it is most squeezed, against
  // the photo as a whole: where the photographer saw a wall or the street edge on, the photo
  // would be smeared in streaks; it fades out to today there
  vec2 dx = dFdx(px), dy = dFdy(px);
  float square = dot(dx, dx) + dot(dy, dy);
  float det = dx.x * dy.y - dx.y * dy.x;
  float least = sqrt(max(square - sqrt(max(square * square - 4.0 * det * det, 0.0)), 0.0) / 2.0);
  float whole = focal * length(dFdx(d));
  float kept = smoothstep(0.5, 1.0, least * ${STRETCHED.toFixed(1)} / whole);
  if (kept <= 0.0) discard;
  if (flipped) uv.x = 1.0 - uv.x;
  gl_FragColor = vec4(texture2D(photo, vec2(uv.x, 1.0 - uv.y)).rgb, strength * kept);
}`;

/** A 3×3 matrix given row by row. */
const rows = (values: number[]) => new THREE.Matrix3().fromArray(values).transpose();

/** From the world (east, north, up) to the viewer's frame, standing in panorama `p` (the
 *  viewer levels it by its correction: see walk3d.correction). */
function worldToViewer(viewer: Viewer, p: Panorama): THREE.Matrix3 {
  const at = (yaw: number, pitch: number) => viewer.dataHelper.sphericalCoordsToVector3({ yaw, pitch }).normalize();
  const [east, north, up] = [at(yawOf(p, 90), 0), at(yawOf(p, 0), 0), at(0, Math.PI / 2)];
  return new THREE.Matrix3().set(east.x, north.x, up.x, east.y, north.y, up.y, east.z, north.z, up.z);
}

interface Drawn {
  photo: OldPhoto;
  mesh: THREE.Mesh;
  material: THREE.ShaderMaterial;
  strength: number;
  /** Where a direction of the viewer's frame shows in the photo ([0, 1]²), if it does. */
  at: (direction: THREE.Vector3) => [number, number] | null;
}

export class PhotoLayer {
  private viewer: Viewer;
  private imageOf: (photo: OldPhoto) => string;
  private drawn: Drawn[] = [];
  private materials: THREE.ShaderMaterial[] = []; // kept: their shader is not compiled again
  private standingMaterials: THREE.ShaderMaterial[] = [];
  private textures = new Map<string, THREE.Texture>();
  private loader = new THREE.TextureLoader();
  private fade: number;

  /** `imageOf`: the image of a photo to draw (its size chosen for the screen). */
  constructor(viewer: Viewer, imageOf: (photo: OldPhoto) => string, fade: number) {
    this.viewer = viewer;
    this.imageOf = imageOf;
    this.fade = fade;
  }

  /** Stand in panorama `p`: the photos shown there, each at its strength, the last on top;
   *  one whose camera stood off the panorama laid on its walls, if it has them. */
  stand(p: Panorama, photos: [OldPhoto, Shown][]): void {
    this.clear();
    const a = cameraAxes(p.azimuth, p.pitch, p.roll);
    const fromWorld = worldToViewer(this.viewer, p);
    const toWorld = fromWorld.clone().invert();
    photos.forEach(([photo, shown], order) => {
      const { registration, strength, standing } = shown;
      let geometry: THREE.BufferGeometry | null;
      let uniforms: Record<string, unknown>;
      let at: Drawn['at'];
      if (standing && p.walls?.length && p.ground !== undefined) {
        const walls = nearest(p.walls);
        const ground = p.ground;
        geometry = quad(footprint(standing, walls, ground).map((d) => d.applyMatrix3(fromWorld))) ?? around();
        uniforms = standingUniforms(standing, walls, ground, toWorld);
        at = (direction) => standingAt(direction.clone().applyMatrix3(toWorld).normalize().toArray(), standing, walls, ground);
      } else {
        // from the photo's [0, 1]² to the viewer's frame, and back
        const fromPhoto = fromWorld.clone().multiply(rows(a.flat()).transpose()).multiply(rows(registration.m));
        geometry = quadOf(fromPhoto);
        const toPhoto = fromPhoto.clone().invert();
        uniforms = { toPhoto: { value: toPhoto } };
        at = (direction) => {
          const q = direction.clone().applyMatrix3(toPhoto);
          const [u, v] = [q.x / q.z, q.y / q.z];
          return q.z > 0 && u >= 0 && u <= 1 && v >= 0 && v <= 1 ? [u, v] : null;
        };
      }
      if (!geometry) return;
      const kind = standing && 'lensTurn' in uniforms ? 'standing' : 'flat';
      const kept = kind === 'standing' ? this.standingMaterials : this.materials;
      const material = kept[order] ?? (kept[order] = newMaterial(kind));
      Object.assign(material.uniforms, uniforms, {
        photo: { value: this.texture(photo) },
        strength: { value: strength * this.fade },
        flipped: { value: photo.flipped },
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.renderOrder = 10 + order;
      this.viewer.renderer.addObject(mesh);
      this.drawn.push({ photo, mesh, material, strength, at });
    });
    this.viewer.needsUpdate();
  }

  /** Only these photos shown (the years chosen), the others hidden. */
  only(shown: (photo: OldPhoto) => boolean): void {
    for (const d of this.drawn) d.mesh.visible = shown(d.photo);
    this.viewer.needsUpdate();
  }

  /** How strongly the old photos show against today (0 to 1). */
  setFade(fade: number): void {
    this.fade = fade;
    for (const d of this.drawn) d.material.uniforms.strength.value = d.strength * fade;
    this.viewer.needsUpdate();
  }

  /** The photo shown on top in a direction of the viewer's frame, if any. */
  at(direction: THREE.Vector3): OldPhoto | null {
    for (const d of [...this.drawn].reverse()) {
      if (!d.mesh.visible || d.strength * this.fade < UNSEEN) continue;
      if (d.at(direction)) return d.photo;
    }
    return null;
  }

  /** The photos drawn, shown or not. */
  get photos(): OldPhoto[] {
    return this.drawn.filter((d) => d.mesh.visible).map((d) => d.photo);
  }

  clear(): void {
    for (const d of this.drawn) {
      this.viewer.renderer.removeObject(d.mesh);
      d.mesh.geometry.dispose();
    }
    this.drawn = [];
  }

  /** A photo's texture, kept while used lately (the least lately used let go, never one drawn);
   *  in the colours the viewer draws its panoramas in (it does not manage colour spaces). */
  private texture(photo: OldPhoto): THREE.Texture {
    const image = this.imageOf(photo);
    const found = this.textures.get(image) ?? this.loader.load(image, () => this.viewer.needsUpdate());
    this.textures.delete(image);
    this.textures.set(image, found); // the latest used, last
    const drawn = new Set(this.drawn.map((d) => this.imageOf(d.photo)));
    for (const [old, texture] of this.textures) {
      if (this.textures.size <= KEPT_TEXTURES) break;
      if (old === image || drawn.has(old)) continue;
      texture.dispose();
      this.textures.delete(old);
    }
    return found;
  }
}

function newMaterial(kind: 'flat' | 'standing'): THREE.ShaderMaterial {
  const shown = { photo: { value: null }, strength: { value: 1 }, flipped: { value: false } };
  return new THREE.ShaderMaterial({
    uniforms: kind === 'flat' ? { ...shown, toPhoto: { value: new THREE.Matrix3() } } : shown,
    vertexShader,
    fragmentShader: kind === 'flat' ? fragmentShader : standingShader,
    side: THREE.DoubleSide,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
}

/** The walls a photo is laid on: the MOST_WALLS nearest the panorama. */
function nearest(walls: WallRow[]): WallRow[] {
  const away = ([x0, y0, x1, y1]: WallRow) => {
    const [ex, ey] = [x1 - x0, y1 - y0];
    const s = Math.min(1, Math.max(0, -(x0 * ex + y0 * ey) / (ex * ex + ey * ey || 1)));
    return Math.hypot(x0 + s * ex, y0 + s * ey);
  };
  return walls.length <= MOST_WALLS ? walls : [...walls].sort((a, b) => away(a) - away(b)).slice(0, MOST_WALLS);
}

function standingUniforms(s: Standing, walls: WallRow[], ground: number, toWorld: THREE.Matrix3): Record<string, unknown> {
  const ends = Array.from({ length: MOST_WALLS }, (_, i) => new THREE.Vector4(...(walls[i] ?? [0, 0, 0, 0]).slice(0, 4)));
  const tops = Array.from({ length: MOST_WALLS }, (_, i) => walls[i]?.[4] ?? 0);
  return {
    toWorld: { value: toWorld },
    lensTurn: { value: rows(lensTurn(s).flat()) },
    camera: { value: new THREE.Vector3(...s.position) },
    size: { value: new THREE.Vector2(s.size[1], s.size[0]) },
    focal: { value: s.focal },
    ends: { value: ends },
    tops: { value: tops },
    count: { value: walls.length },
    ground: { value: ground },
  };
}

/** Directions from the panorama (east, north, up) the photo is laid along: where the rays of
 *  its edges, from where it was taken, meet today's walls, street or sky (see reach). */
function footprint(s: Standing, walls: WallRow[], ground: number): THREE.Vector3[] {
  const [h, w] = s.size;
  const kr = rows(lensTurn(s).flat()).invert();
  const found: THREE.Vector3[] = [];
  for (let i = 0; i <= 8; i++) {
    for (const [x, y] of [[(w * i) / 8, 0], [(w * i) / 8, h], [0, (h * i) / 8], [w, (h * i) / 8]]) {
      const ray = new THREE.Vector3(x, y, 1).applyMatrix3(kr).normalize();
      const metres = Math.min(FAR_M, reach(s.position, ray.toArray(), walls, ground));
      found.push(new THREE.Vector3(...s.position).addScaledVector(ray, metres).normalize());
    }
  }
  return found;
}

/** A sphere around the camera: for a photo that wraps around the panorama. */
function around(): THREE.BufferGeometry {
  return new THREE.SphereGeometry(DISTANCE, 64, 32);
}

/** The photo's frame on a plane in front of the camera, across its middle's direction; none
 *  when a corner lies behind it. */
function quadOf(fromPhoto: THREE.Matrix3): THREE.BufferGeometry | null {
  const direction = (u: number, v: number) => new THREE.Vector3(u, v, 1).applyMatrix3(fromPhoto).normalize();
  return quad(
    [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
    ].map(([u, v]) => direction(u, v)),
  );
}

/** A plane in front of the camera holding `directions` (of the viewer's frame), across their
 *  middle; none when one lies behind it. */
function quad(directions: THREE.Vector3[]): THREE.BufferGeometry | null {
  const middle = directions.reduce((sum, d) => sum.add(d.clone().normalize()), new THREE.Vector3()).normalize();
  if (directions.some((d) => d.clone().normalize().dot(middle) <= 0.05)) return null;
  const across = new THREE.Vector3(0, 1, 0).cross(middle).normalize();
  if (across.lengthSq() === 0) across.set(1, 0, 0);
  const up = middle.clone().cross(across).normalize();
  const onPlane = directions.map((d) => d.clone().normalize().multiplyScalar(DISTANCE / d.clone().normalize().dot(middle)));
  const xs = onPlane.map((c) => c.dot(across));
  const ys = onPlane.map((c) => c.dot(up));
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const corner = (x: number, y: number) => middle.clone().multiplyScalar(DISTANCE).addScaledVector(across, x).addScaledVector(up, y);
  const corners = directions.length === 4 ? onPlane : [corner(x0, y0), corner(x1, y0), corner(x1, y1), corner(x0, y1)];
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(corners.flatMap((c) => [c.x, c.y, c.z]), 3));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  return geometry;
}
