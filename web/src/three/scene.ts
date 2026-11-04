/** The 3D walk drawn with three.js: stand in a panorama, look around, glide to the next; the
 * old photos around, each on the facades it shows. */
import * as THREE from 'three';
import { DATA, density } from '../walk';
import { EDGE, GONE_DEG, panoramaMaterial, photoMaterial } from './projection';
import { toThree, type OldPhoto, type Panorama, type Scene, type Wall } from './walk3d';

const GLIDE_MS = 600;
const SKY_M = 600;
const LOOK_FOV = 70; // vertical, degrees
const PITCH_LIMIT = 80;
const PHOTO_NEAR = 0.5;
const PHOTO_FAR = 400;

interface Shown {
  photo: OldPhoto;
  material: THREE.ShaderMaterial;
  facades: THREE.Group;
}

/** The camera of an old photo, as projection × view: its heading, its width of view and its
 * horizon (view cameras shifted the lens instead of tilting, so the horizon is rarely
 * mid-height). */
export function photoView(photo: OldPhoto): THREE.Matrix4 {
  const { x, y, z, heading, horizon, fov } = photo.pose;
  const camera = new THREE.Object3D();
  camera.position.set(...toThree(x, y, z));
  camera.rotation.set(0, -THREE.MathUtils.degToRad(heading), 0);
  camera.updateMatrixWorld();
  const half = PHOTO_NEAR * Math.tan(THREE.MathUtils.degToRad(fov) / 2); // half the width
  const height = 2 * half * photo.aspect;
  const projection = new THREE.Matrix4().makePerspective(-half, half, horizon * height, -(1 - horizon) * height, PHOTO_NEAR, PHOTO_FAR);
  return projection.multiply(camera.matrixWorld.clone().invert());
}

function prism(footprint: [number, number][], base: number, height: number): THREE.BufferGeometry {
  const geometry = new THREE.ExtrudeGeometry(new THREE.Shape(footprint.map(([x, y]) => new THREE.Vector2(x, y))), {
    depth: height,
    bevelEnabled: false,
  });
  // extruded along z from the (x, north) plane: stand it up, north to −z
  geometry.rotateX(-Math.PI / 2);
  geometry.translate(0, base, 0);
  return geometry;
}

function ground(scene: Scene): THREE.BufferGeometry {
  const { x0, y0, step, columns, heights } = scene.ground;
  const rows = heights.length / columns;
  const geometry = new THREE.PlaneGeometry((columns - 1) * step, (rows - 1) * step, columns - 1, rows - 1);
  const position = geometry.getAttribute('position');
  for (let i = 0; i < position.count; i++) {
    const column = i % columns;
    const row = rows - 1 - Math.floor(i / columns); // the plane's first vertices are its north-west
    position.setXYZ(i, ...toThree(x0 + column * step, y0 + row * step, heights[row * columns + column]));
  }
  geometry.computeVertexNormals();
  return geometry;
}

function wallMesh(wall: Wall, base: number, material: THREE.Material): THREE.Mesh {
  const [[x0, y0], [x1, y1]] = wall.line;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(Math.hypot(x1 - x0, y1 - y0), wall.height), material);
  mesh.position.set(...toThree((x0 + x1) / 2, (y0 + y1) / 2, base + wall.height / 2));
  mesh.rotation.y = Math.atan2(y1 - y0, x1 - x0); // along the wall, in the (x, −z) plane
  return mesh;
}

/** The facades an old photo is projected onto, as they stood. */
function facadesOf(photo: OldPhoto, material: THREE.Material): THREE.Group {
  const facades = new THREE.Group();
  for (const wall of photo.walls) facades.add(wallMesh(wall, photo.base, material));
  facades.updateMatrixWorld(true);
  return facades;
}

const SHOWING = 0.05; // a photo fainter than this is not picked

/** Whether a photo shows at a point of its facades, seen from `eye`, as the shader draws it:
 * inside its frame (short of its soft edges) and short of the angle where it is gone. */
function drawnAt(shown: Shown, point: THREE.Vector3, eye: THREE.Vector3): boolean {
  const clip = new THREE.Vector4(point.x, point.y, point.z, 1).applyMatrix4(shown.material.uniforms.view.value);
  if (clip.w <= 0) return false;
  const inside = 1 - 2 * EDGE; // the frame is −1 to 1 here, 0 to 1 in the shader
  if (Math.abs(clip.x / clip.w) > inside || Math.abs(clip.y / clip.w) > inside) return false;
  const from = point.clone().sub(shown.material.uniforms.at.value).normalize();
  const seen = point.clone().sub(eye).normalize();
  return from.dot(seen) > Math.cos((GONE_DEG * Math.PI) / 180);
}

export class Scene3D {
  readonly scene: Scene;
  private renderer: THREE.WebGLRenderer;
  private world = new THREE.Scene(); // today
  private cores = new THREE.Scene(); // what hides an old photo: only depth is drawn
  private old = new THREE.Scene(); // the old photos, on their facades
  private camera = new THREE.PerspectiveCamera(LOOK_FOV, 1, 0.1, SKY_M * 2);
  private today = [panoramaMaterial(), panoramaMaterial(THREE.BackSide)]; // buildings and ground, sky
  private loader = new THREE.TextureLoader();
  private textures = new Map<string, Promise<THREE.Texture>>();
  private shown: Shown[] = [];
  private photosAsked = 0;
  private gliding = 0;
  here!: Panorama;
  heading = 0;
  pitch = 0;
  onChange: () => void = () => {};

  constructor(canvas: HTMLCanvasElement, scene: Scene) {
    this.scene = scene;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(density());
    this.loader.setCrossOrigin('anonymous');
    const [solid, sky] = this.today;
    for (const b of scene.buildings) this.world.add(new THREE.Mesh(prism(b.footprint, b.base, b.height), solid));
    this.world.add(new THREE.Mesh(ground(scene), solid), new THREE.Mesh(new THREE.SphereGeometry(SKY_M, 48, 24), sky));
    const depthOnly = new THREE.MeshBasicMaterial({ colorWrite: false });
    for (const c of scene.cores) this.cores.add(new THREE.Mesh(prism(c.footprint, c.base, c.height), depthOnly));
    this.renderer.autoClear = false;
    this.camera.rotation.order = 'YXZ';
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  private resize(): void {
    const canvas = this.renderer.domElement;
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    this.camera.aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
    this.camera.updateProjectionMatrix();
    this.draw();
  }

  private texture(url: string): Promise<THREE.Texture> {
    let found = this.textures.get(url);
    if (!found) {
      found = this.loader.loadAsync(url).then((t) => {
        t.colorSpace = THREE.SRGBColorSpace;
        t.generateMipmaps = false; // a mipmap would smear the panorama's seam
        t.minFilter = THREE.LinearFilter;
        return t;
      });
      found.catch(() => this.textures.delete(url)); // asked again next time
      this.textures.set(url, found);
    }
    return found;
  }

  private setToday(name: string, value: unknown): void {
    for (const m of this.today) m.uniforms[name].value = value;
  }

  private setPanorama(slot: 'A' | 'B', panorama: Panorama, texture: THREE.Texture): void {
    this.setToday(`pano${slot}`, texture);
    this.setToday(`pano${slot}At`, new THREE.Vector3(...toThree(panorama.x, panorama.y, panorama.z)));
    this.setToday(`pano${slot}Azimuth`, THREE.MathUtils.degToRad(panorama.azimuth));
  }

  /** Go to a panorama: a glide when it is near, a jump otherwise. Resolves once there; when
   * its picture cannot be had, the walk stays where it is. */
  async goTo(id: string, heading = this.heading): Promise<void> {
    const next = this.scene.panoramas.find((p) => p.id === id);
    if (!next) return;
    let texture: THREE.Texture;
    try {
      texture = await this.texture(next.image); // the current view stays until it is here
    } catch {
      return;
    }
    const from = this.here;
    this.heading = heading;
    if (from && Math.hypot(next.x - from.x, next.y - from.y) <= 30) {
      this.setPanorama('B', next, texture);
      await this.glide(from, next);
    }
    this.here = next;
    this.setPanorama('A', next, texture);
    this.setToday('toB', 0);
    this.camera.position.set(...toThree(next.x, next.y, next.z));
    this.draw();
    this.onChange();
  }

  private glide(from: Panorama, to: Panorama): Promise<void> {
    const start = performance.now();
    const glide = ++this.gliding;
    const a = new THREE.Vector3(...toThree(from.x, from.y, from.z));
    const b = new THREE.Vector3(...toThree(to.x, to.y, to.z));
    return new Promise<void>((done) => {
      const frame = (now: number) => {
        if (glide !== this.gliding) return done();
        const t = Math.min((now - start) / GLIDE_MS, 1);
        const eased = t * t * (3 - 2 * t);
        this.camera.position.lerpVectors(a, b, eased);
        this.setToday('toB', eased);
        this.draw();
        if (t < 1) requestAnimationFrame(frame);
        else done();
      };
      requestAnimationFrame(frame);
    });
  }

  /** Stand somewhere else than the panorama (to look from an old photo's camera). */
  standAt(x: number, y: number, z: number): void {
    this.camera.position.set(...toThree(x, y, z));
    this.draw();
  }

  look(heading: number, pitch: number): void {
    this.heading = ((heading % 360) + 360) % 360;
    this.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, pitch));
    this.draw();
    this.onChange();
  }

  /** Put these old photos on their facades; a photo whose image cannot be had is left out. */
  async setPhotos(photos: OldPhoto[]): Promise<void> {
    const asked = ++this.photosAsked;
    const settled = await Promise.allSettled(photos.map((p) => this.texture(`${DATA}${p.image}`)));
    if (asked !== this.photosAsked) return; // a newer set was asked for while these loaded
    for (const old of this.shown) {
      this.old.remove(old.facades);
      old.material.dispose();
    }
    this.shown = photos.flatMap((photo, i) => {
      const loaded = settled[i];
      if (loaded.status !== 'fulfilled') return [];
      const { x, y, z } = photo.pose;
      const material = photoMaterial(loaded.value, photoView(photo), new THREE.Vector3(...toThree(x, y, z)));
      const facades = facadesOf(photo, material);
      facades.traverse((o) => (o.renderOrder = i)); // in the order given: the last on top
      this.old.add(facades);
      return [{ photo, material, facades }];
    });
    this.draw();
  }

  /** The old photo drawn under a point of the screen (client pixels), or null: the nearest
   * facade hit, where its photo shows (inside its frame, not seen too far aside) and no
   * building stands in front. */
  photoAt(clientX: number, clientY: number): OldPhoto | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    const ray = new THREE.Raycaster();
    ray.setFromCamera(pointer, this.camera);
    const wall = ray.intersectObjects(this.cores.children, false)[0]?.distance ?? Infinity;
    const showing = this.shown.filter((s) => s.material.uniforms.strength.value > SHOWING);
    for (const hit of ray.intersectObjects(showing.map((s) => s.facades), true)) {
      if (hit.distance > wall) break;
      const shown = showing.find((s) => s.facades === hit.object.parent);
      if (shown && drawnAt(shown, hit.point, this.camera.position)) return shown.photo;
    }
    return null;
  }

  /** The photos shown now, in their order. */
  get photos(): OldPhoto[] {
    return this.shown.map((s) => s.photo);
  }

  /** How strongly each photo shown shows (0 to 1), in their order; today is toned to sepia
   * as much as the strongest. */
  setStrengths(strengths: number[]): void {
    const same = this.shown.every((s, i) => s.material.uniforms.strength.value === (strengths[i] ?? 0));
    if (same && this.shown.length) return; // with none shown, today must lose its sepia
    this.shown.forEach((s, i) => (s.material.uniforms.strength.value = strengths[i] ?? 0));
    this.setToday('sepia', 0.85 * Math.max(0, ...strengths.slice(0, this.shown.length)));
    this.draw();
  }

  /** Where a point of the street, `metres` away towards `bearing`, falls on the screen; null
   * when behind. */
  onScreen(bearing: number, metres: number): { x: number; y: number } | null {
    const r = THREE.MathUtils.degToRad(bearing);
    const point = new THREE.Vector3(
      ...toThree(this.here.x + metres * Math.sin(r), this.here.y + metres * Math.cos(r), this.here.z - 2.3),
    ).project(this.camera);
    if (point.z > 1) return null;
    const canvas = this.renderer.domElement;
    return { x: ((point.x + 1) / 2) * canvas.clientWidth, y: ((1 - point.y) / 2) * canvas.clientHeight };
  }

  draw(): void {
    this.camera.rotation.set(THREE.MathUtils.degToRad(this.pitch), -THREE.MathUtils.degToRad(this.heading), 0);
    // today first; then the old photos, hidden only behind the cores of today's buildings
    this.renderer.clear();
    this.renderer.render(this.world, this.camera);
    this.renderer.clearDepth();
    this.renderer.render(this.cores, this.camera);
    this.renderer.render(this.old, this.camera);
  }
}
