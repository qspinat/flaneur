// Old photos drawn inside a panorama of Photo Sphere Viewer, each where its registration puts
// it: a quad in front of the camera whose every pixel's direction is taken back through the
// registration to the photo (exact: the warp is done per pixel, not by the quad's corners).
import type { Viewer } from '@photo-sphere-viewer/core';
import * as THREE from 'three';
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
  toPhoto: THREE.Matrix3;
}

export class PhotoLayer {
  private viewer: Viewer;
  private imageOf: (photo: OldPhoto) => string;
  private drawn: Drawn[] = [];
  private materials: THREE.ShaderMaterial[] = []; // kept: their shader is not compiled again
  private textures = new Map<string, THREE.Texture>();
  private loader = new THREE.TextureLoader();
  private fade: number;

  /** `imageOf`: the image of a photo to draw (its size chosen for the screen). */
  constructor(viewer: Viewer, imageOf: (photo: OldPhoto) => string, fade: number) {
    this.viewer = viewer;
    this.imageOf = imageOf;
    this.fade = fade;
  }

  /** Stand in panorama `p`: the photos shown there, each at its strength, the last on top. */
  stand(p: Panorama, photos: [OldPhoto, Shown][]): void {
    this.clear();
    const a = cameraAxes(p.azimuth, p.pitch, p.roll);
    const fromWorld = worldToViewer(this.viewer, p);
    photos.forEach(([photo, { registration, strength }], order) => {
      // from the photo's [0, 1]² to the viewer's frame, and back
      const fromPhoto = fromWorld.clone().multiply(rows(a.flat()).transpose()).multiply(rows(registration.m));
      const geometry = quad(fromPhoto);
      if (!geometry) return;
      const toPhoto = fromPhoto.clone().invert();
      const material = this.materials[order] ?? (this.materials[order] = newMaterial());
      Object.assign(material.uniforms, {
        photo: { value: this.texture(photo) },
        toPhoto: { value: toPhoto },
        strength: { value: strength * this.fade },
        flipped: { value: photo.flipped },
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.renderOrder = 10 + order;
      this.viewer.renderer.addObject(mesh);
      this.drawn.push({ photo, mesh, material, strength, toPhoto });
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
      const q = direction.clone().applyMatrix3(d.toPhoto);
      const [u, v] = [q.x / q.z, q.y / q.z];
      if (q.z > 0 && u >= 0 && u <= 1 && v >= 0 && v <= 1) return d.photo;
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

function newMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { photo: { value: null }, toPhoto: { value: new THREE.Matrix3() }, strength: { value: 1 }, flipped: { value: false } },
    vertexShader,
    fragmentShader,
    side: THREE.DoubleSide,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
}

/** The photo's frame on a plane in front of the camera, across its middle's direction; none
 *  when a corner lies behind it. */
function quad(fromPhoto: THREE.Matrix3): THREE.BufferGeometry | null {
  const direction = (u: number, v: number) => new THREE.Vector3(u, v, 1).applyMatrix3(fromPhoto).normalize();
  const middle = direction(0.5, 0.5);
  const corners = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ].map(([u, v]) => direction(u, v));
  if (corners.some((c) => c.dot(middle) <= 0.05)) return null;
  const onPlane = corners.map((c) => c.multiplyScalar(DISTANCE / c.dot(middle)));
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(onPlane.flatMap((c) => [c.x, c.y, c.z]), 3));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  return geometry;
}
