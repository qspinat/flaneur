// Old photos drawn inside a panorama of Photo Sphere Viewer, each where its registration puts
// it: a quad in front of the camera whose every pixel's direction is taken back through the
// registration to the photo (exact: the warp is done per pixel, not by the quad's corners).
// A photo whose camera stood off the panorama (see standing.ts) is laid on the walls around it.
import type { Viewer } from '@photo-sphere-viewer/core';
import * as THREE from 'three';
import { HIDDEN, NEAREST_M, ROOFS_PAST_M, STREET_REACH_M, STRETCHED, lensTurn, reach, standingAt, type Standing, type WallRow } from './standing';
import { framing, type Framed } from '../plate';
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

// an Albert-Kahn plate: its frame (the scan's black border and number) and the museum's label
// on it are never drawn (see plate.ts: else the whole photo, nothing left out)
const PLATE = `
uniform vec4 plate;
uniform vec4 label;
bool framedOut(vec2 uv) {
  return uv.x < plate.x || uv.x > plate.z || uv.y < plate.y || uv.y > plate.w
    || (uv.x >= label.x && uv.x <= label.z && uv.y >= label.y && uv.y <= label.w);
}`;

const fragmentShader = `
uniform sampler2D photo;
uniform mat3 toPhoto;
uniform float strength;
uniform bool flipped;
${PLATE}
varying vec3 here;
void main() {
  vec3 q = toPhoto * here;
  if (q.z <= 0.0) discard;
  vec2 uv = q.xy / q.z;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) discard;
  if (flipped) uv.x = 1.0 - uv.x;
  if (framedOut(uv)) discard;
  gl_FragColor = vec4(texture2D(photo, vec2(uv.x, 1.0 - uv.y)).rgb, strength);
}`;

const MOST_WALLS = 48; // the walls a photo is laid on (the pipeline's scene.MOST_WALLS)
const FAR_M = 1000; // a ray of the photo meeting nothing, framed as this far
const FRAMED = 8; // the photo's rays framing where it is laid: this many steps across, each way
const MARGIN = 0.15; // and that frame widened by this share each way: a ray between them may go farther

// standing.ts per pixel: the direction from the panorama out to the walls (else the street, else
// over the roofs to the wall under them, else the sky), seen by the photographer's camera;
// each wall given by an end and its span (ends[i]: x0, y0, x1 - x0, y1 - y0)
const standingShader = `
uniform sampler2D photo;
uniform mat3 toWorld;
uniform mat3 lensTurn;
uniform vec3 fromCamera; // K R (panorama - camera)
uniform vec3 camera;
uniform vec2 size;
uniform float density; // the photo's pixels per radian, laid as a whole, seen from the panorama
uniform vec4 ends[${MOST_WALLS}];
uniform float tops[${MOST_WALLS}];
uniform float spanInverse[${MOST_WALLS}]; // 1 / each wall's length
uniform int count;
uniform float ground;
uniform float strength;
uniform bool flipped;
${PLATE}
varying vec3 here;
const float NONE = 1e30;

// how far along d it meets a wall (see standing.toWall); first: any wall short of before will do
float toWall(vec3 from, vec3 d, bool roofless, float before, bool first) {
  float nearest = before;
  for (int i = 0; i < ${MOST_WALLS}; i++) {
    if (i >= count) break;
    vec4 w = ends[i];
    float across = d.x * w.w - d.y * w.z;
    if (across == 0.0) continue;
    float inv = 1.0 / across;
    vec2 a = w.xy - from.xy;
    float t = (a.x * w.w - a.y * w.z) * inv;
    float s = (a.x * d.y - a.y * d.x) * inv;
    float past = roofless ? ${ROOFS_PAST_M.toFixed(1)} * spanInverse[i] : 0.0;
    float height = from.z + t * d.z;
    if (t > ${NEAREST_M.toFixed(1)} && t < nearest && s >= -past && s <= 1.0 + past && height >= ground && (roofless || height <= tops[i])) {
      nearest = t;
      if (first) break;
    }
  }
  return nearest;
}

// as standing.reachOrSky; sky: met over the roofs (a backdrop at the wall's distance), or not
float reach(vec3 d, out bool sky) {
  sky = false;
  float wall = toWall(vec3(0.0), d, false, NONE, false);
  if (wall < NONE) return wall;
  float street = d.z < 0.0 ? ground / d.z : NONE;
  if (street > 0.0 && street <= ${STREET_REACH_M.toFixed(1)}) return street;
  sky = true;
  return d.z > 0.0 ? toWall(vec3(0.0), d, true, NONE, false) : NONE;
}

void main() {
  vec3 d = normalize(toWorld * here);
  bool sky;
  float metres = reach(d, sky);
  vec3 seen = lensTurn * d;
  if (metres < NONE) seen += fromCamera / metres;
  vec2 px = seen.xy / seen.z;
  // how little of the photo one pixel of the screen spans (as standing.squeezed), the way it is most squeezed, against
  // the photo laid as a whole (registration._stretched): where the photographer saw a wall or the
  // street edge on, the photo would be smeared in streaks; it fades out to today there (worked
  // out before any pixel is left out: the derivatives need all four of a block)
  vec2 dx = dFdx(px), dy = dFdy(px);
  float square = dot(dx, dx) + dot(dy, dy);
  float det = dx.x * dy.y - dx.y * dy.x;
  float least = sqrt(max(square - sqrt(max(square * square - 4.0 * det * det, 0.0)), 0.0) / 2.0);
  // (not the sky: where two walls' backdrops meet, the photo folds there, a line, not a smear)
  float spread = length(dFdx(d));  // before choosing: derivatives need all four of a block
  float kept = sky ? 1.0 : smoothstep(0.5, 1.0, least * ${STRETCHED.toFixed(1)} / (density * spread));
  if (seen.z <= 0.0 || kept <= 0.0) discard;
  vec2 uv = px / size.yx;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) discard;
  if (metres < NONE) {  // hidden from the photographer by a wall nearer to them
    vec3 away = d * metres - camera;
    float far = length(away) / ${HIDDEN.toFixed(2)};
    if (toWall(camera, normalize(away), false, far, true) < far) discard;
  }
  if (flipped) uv.x = 1.0 - uv.x;
  if (framedOut(uv)) discard;
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

type Kind = 'flat' | 'standing';

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
  private materials: Record<Kind, THREE.ShaderMaterial[]> = { flat: [], standing: [] }; // kept: not compiled again
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
      let kind: Kind, geometry: THREE.BufferGeometry | null, uniforms: Record<string, unknown>, at: Drawn['at'];
      if (standing && p.walls?.length && p.ground !== undefined) {
        const { walls, ground } = p as Required<Panorama>;
        const kr = rows(lensTurn(standing).flat());
        kind = 'standing';
        const laid = footprint(standing, kr, walls, ground);
        geometry = quad(laid.map((d) => d.clone().applyMatrix3(fromWorld)), MARGIN) ?? around();
        const across = density(laid, standing);
        uniforms = { ...standingUniforms(standing, kr, walls, ground, toWorld), density: { value: across } };
        at = (direction) => standingAt(direction.clone().applyMatrix3(toWorld).normalize().toArray(), standing, walls, ground, across);
      } else {
        // from the photo's [0, 1]² to the viewer's frame, and back
        const fromPhoto = fromWorld.clone().multiply(rows(a.flat()).transpose()).multiply(rows(registration.m));
        const toPhoto = fromPhoto.clone().invert();
        kind = 'flat';
        geometry = quad([[0, 0], [1, 0], [1, 1], [0, 1]].map(([u, v]) => new THREE.Vector3(u, v, 1).applyMatrix3(fromPhoto)));
        uniforms = { toPhoto: { value: toPhoto } };
        at = (direction) => {
          const q = direction.clone().applyMatrix3(toPhoto);
          const [u, v] = [q.x / q.z, q.y / q.z];
          return q.z > 0 && u >= 0 && u <= 1 && v >= 0 && v <= 1 ? [u, v] : null;
        };
      }
      if (!geometry) return;
      const kept = this.materials[kind];
      const material = kept[order] ?? (kept[order] = newMaterial(kind));
      Object.assign(material.uniforms, uniforms, {
        photo: { value: this.texture(photo) },
        strength: { value: strength * this.fade },
        flipped: { value: photo.flipped },
        ...framedUniforms(photo),
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

/** A photo's plate and label as the shaders take them (see PLATE). */
function framedUniforms(photo: Framed): Record<string, { value: THREE.Vector4 }> {
  const { plate, label } = framing(photo);
  return { plate: { value: new THREE.Vector4(...plate) }, label: { value: new THREE.Vector4(...label) } };
}

function newMaterial(kind: Kind): THREE.ShaderMaterial {
  const shown = { photo: { value: null }, strength: { value: 1 }, flipped: { value: false }, ...framedUniforms({}) };
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

/** What the standing shader is given: the camera (`kr`: its lens times its turn), the walls (at
 *  most MOST_WALLS: the pipeline sends no more), the street, the viewer's frame. */
function standingUniforms(s: Standing, kr: THREE.Matrix3, walls: WallRow[], ground: number, toWorld: THREE.Matrix3): Record<string, unknown> {
  const ends = Array.from({ length: MOST_WALLS }, () => new THREE.Vector4());
  const tops = new Array<number>(MOST_WALLS).fill(0);
  const spanInverse = new Array<number>(MOST_WALLS).fill(0);
  walls.slice(0, MOST_WALLS).forEach(([x0, y0, x1, y1, top], i) => {
    ends[i].set(x0, y0, x1 - x0, y1 - y0);
    tops[i] = top;
    spanInverse[i] = 1 / (Math.hypot(x1 - x0, y1 - y0) || 1);
  });
  const camera = new THREE.Vector3(...s.position);
  return {
    toWorld: { value: toWorld },
    lensTurn: { value: kr },
    fromCamera: { value: camera.clone().applyMatrix3(kr).negate() },
    camera: { value: camera },
    size: { value: new THREE.Vector2(...s.size) },
    ends: { value: ends },
    tops: { value: tops },
    spanInverse: { value: spanInverse },
    count: { value: Math.min(walls.length, MOST_WALLS) },
    ground: { value: ground },
  };
}

/** Directions from the panorama (east, north, up) the photo is laid along: where its rays
 *  (`kr`: its camera's lens times turn), across the whole photo, meet today's walls, street or
 *  sky (see reach) — not its edges alone: a ray inside it may go farther than those around. */
function footprint(s: Standing, kr: THREE.Matrix3, walls: WallRow[], ground: number): THREE.Vector3[] {
  const [h, w] = s.size;
  const back = kr.clone().invert();
  const found: THREE.Vector3[] = [];
  for (let i = 0; i <= FRAMED; i++) {
    for (let j = 0; j <= FRAMED; j++) {
      const ray = new THREE.Vector3((w * i) / FRAMED, (h * j) / FRAMED, 1).applyMatrix3(back).normalize();
      const metres = Math.min(FAR_M, reach(s.position, ray.toArray(), walls, ground));
      found.push(new THREE.Vector3(...s.position).addScaledVector(ray, metres));
    }
  }
  return found;
}

/** The photo's pixels per radian as laid (see footprint), seen from the panorama: the median
 *  over neighbouring rays across its width. */
function density(laid: THREE.Vector3[], s: Standing): number {
  const step = s.size[1] / FRAMED;
  const found: number[] = [];
  for (let i = 0; i < FRAMED; i++) {
    for (let j = 0; j <= FRAMED; j++) {
      const angle = laid[i * (FRAMED + 1) + j].angleTo(laid[(i + 1) * (FRAMED + 1) + j]);
      if (angle > 0) found.push(step / angle);
    }
  }
  found.sort((a, b) => a - b);
  return found[Math.floor(found.length / 2)] ?? s.focal;
}

/** A sphere around the camera: for a photo that wraps around the panorama. */
function around(): THREE.BufferGeometry {
  return new THREE.SphereGeometry(DISTANCE, 64, 32);
}

/** A rectangle on a plane in front of the camera holding `directions` (of the viewer's frame),
 *  across their middle, widened by `margin` (a share of it) each way; none when one lies behind
 *  it. The shader leaves out what is not drawn. */
function quad(directions: THREE.Vector3[], margin = 0): THREE.BufferGeometry | null {
  const unit = directions.map((d) => d.clone().normalize());
  const middle = unit.reduce((sum, d) => sum.add(d), new THREE.Vector3()).normalize();
  if (unit.some((d) => d.dot(middle) <= 0.05)) return null;
  const across = new THREE.Vector3(0, 1, 0).cross(middle);
  if (across.lengthSq() < 1e-9) across.set(1, 0, 0);
  across.normalize();
  const up = middle.clone().cross(across).normalize();
  const onPlane = unit.map((d) => d.multiplyScalar(DISTANCE / d.dot(middle)));
  const xs = onPlane.map((c) => c.dot(across));
  const ys = onPlane.map((c) => c.dot(up));
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const [mx, my] = [(x1 - x0) * margin, (y1 - y0) * margin];
  const corner = (x: number, y: number) => middle.clone().multiplyScalar(DISTANCE).addScaledVector(across, x).addScaledVector(up, y);
  const corners = [corner(x0 - mx, y0 - my), corner(x1 + mx, y0 - my), corner(x1 + mx, y1 + my), corner(x0 - mx, y1 + my)];
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(corners.flatMap((c) => [c.x, c.y, c.z]), 3));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  return geometry;
}
