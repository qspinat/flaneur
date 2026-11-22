// A standing photo drawn without holes, as the pipeline draws it (pipeline/src/flaneur/fill.py,
// drawing.py): drawn first apart, its colour weighed by what each pixel is worth (0 where the
// photo has nothing to show: hidden from the photographer, smeared) and where it reaches at all;
// the graphics chip halves both into levels (mipmaps, box averages); one pass over the photo's
// quad reads every level at each pixel and pulls them back from the coarsest: small gaps are
// filled from around them, large ones fade to today, and the photo's edge fades across its
// outline. Its reach is an angle: a gap looks the same filled or faded at any size of screen.
import * as THREE from 'three';

export const FILL_DEG = 2; // gaps up to about this wide are filled; wider ones fade
export const EDGE_DEG = 1; // the photo's edge fades over about this

/** How many levels reach FILL_DEG, for pixels of `pixelDeg` degrees. */
export function levels(pixelDeg: number): number {
  return 1 + Math.ceil(Math.log2(FILL_DEG / pixelDeg));
}

/** The level whose texels are about `degrees` across. */
export function level(degrees: number, pixelDeg: number): number {
  return Math.max(0, Math.round(Math.log2(degrees / pixelDeg)));
}

// on the photo's own quad (only where it may lie), seen as the viewer sees it
const vertexShader = `
void main() {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

// fill.laid, one pixel: every level read here (bilinearly), pulled back from the coarsest
const fragmentShader = `
uniform sampler2D layer; // colour times weight, and weight
uniform sampler2D reach; // where the photo reaches
uniform int depth;
uniform int smoothest; // the level the coverage is read at (two under the coarsest)
uniform int edgiest; // and the photo's reach (about EDGE_DEG)
uniform float strength;
uniform vec2 screen; // the drawing's size, pixels
out highp vec4 laid;
void main() {
  vec2 at = gl_FragCoord.xy / screen;
  float top = float(depth - 1);
  vec4 pulled = textureLod(layer, at, top);
  float reached = textureLod(reach, at, top).r;
  float covered = pulled.a, edge = reached;
  if (pulled.a >= 1.0 && reached >= 1.0) {  // even the coarsest level full: nothing finer is missing
    laid = vec4(textureLod(layer, at, 0.0).rgb, strength);
    return;
  }
  for (int k = depth - 2; k >= 0; k--) {
    vec4 here = textureLod(layer, at, float(k));
    pulled = here + (1.0 - here.a) * pulled;
    if (k >= edgiest) {  // the reach is needed down to its level only
      float there = textureLod(reach, at, float(k)).r;
      reached = there + (1.0 - there) * reached;
      if (k == edgiest) edge = reached;
    }
    if (k == smoothest) covered = pulled.a;
  }
  float shown = smoothstep(0.2, 0.8, covered) * smoothstep(0.5, 1.0, edge);
  if (shown <= 0.0) discard;
  laid = vec4(pulled.rgb / max(pulled.a, 1e-6), shown * strength);
}`;

/** Draws standing photos filled: each drawn first apart, before the viewer's scene (its own
 *  shader writes colour times weight, and where it reaches; one target each), then laid over the
 *  scene through its own quad. */
export class FilledLayer {
  private targets: THREE.WebGLRenderTarget[] = [];
  private material: THREE.ShaderMaterial;
  private size = new THREE.Vector2();
  private clearColour = new THREE.Color();
  private frustum = new THREE.Frustum();
  private viewing = new THREE.Matrix4();
  private bounds = new THREE.Sphere();

  constructor() {
    this.material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms: {
        layer: { value: null },
        reach: { value: null },
        depth: { value: 1 },
        smoothest: { value: 0 },
        edgiest: { value: 0 },
        strength: { value: 1 },
        screen: { value: this.size },
      },
      vertexShader,
      fragmentShader,
      transparent: true,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
    });
  }

  /** Whether `mesh` (a standing photo's quad) may be on the screen at all. */
  inView(mesh: THREE.Mesh, camera: THREE.PerspectiveCamera): boolean {
    const geometry = mesh.geometry;
    if (!geometry.boundingSphere) geometry.computeBoundingSphere();
    this.viewing.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.viewing);
    return this.frustum.intersectsSphere(this.bounds.copy(geometry.boundingSphere!).applyMatrix4(mesh.matrixWorld));
  }

  /** Draw the `n`th standing photo apart (before the scene: no going back to the screen). */
  prepare(renderer: THREE.WebGLRenderer, n: number, mesh: THREE.Mesh, camera: THREE.PerspectiveCamera): void {
    renderer.getDrawingBufferSize(this.size);
    const clearAlpha = renderer.getClearAlpha();
    renderer.getClearColor(this.clearColour);
    renderer.setRenderTarget(this.targetOf(n, this.size.x, this.size.y));
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(mesh, camera);
    renderer.setRenderTarget(null);
    renderer.setClearColor(this.clearColour, clearAlpha);
  }

  /** Lay the `n`th standing photo, drawn apart, over the screen at `strength`. */
  lay(renderer: THREE.WebGLRenderer, n: number, mesh: THREE.Mesh, camera: THREE.PerspectiveCamera, strength: number): void {
    const target = this.targets[n];
    const pixelDeg = camera.fov / this.size.y;
    const depth = levels(pixelDeg);
    const u = this.material.uniforms;
    u.layer.value = target.textures[0];
    u.reach.value = target.textures[1];
    u.depth.value = depth;
    u.smoothest.value = Math.max(0, depth - 2);
    u.edgiest.value = Math.min(level(EDGE_DEG, pixelDeg), depth - 1);
    u.strength.value = strength;
    const [own, autoClear] = [mesh.material, renderer.autoClear];
    mesh.material = this.material; // the pass over the photo's quad only
    renderer.autoClear = false;
    renderer.render(mesh, camera);
    renderer.autoClear = autoClear;
    mesh.material = own;
  }

  dispose(): void {
    for (const target of this.targets) target.dispose();
    this.material.dispose();
  }

  private targetOf(n: number, width: number, height: number): THREE.WebGLRenderTarget {
    const found = this.targets[n];
    if (found && found.width === width && found.height === height) return found;
    found?.dispose();
    const target = new THREE.WebGLRenderTarget(width, height, {
      count: 2,
      generateMipmaps: true,
      minFilter: THREE.LinearMipmapLinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
    });
    target.textures[1].format = THREE.RedFormat; // where the photo reaches: one channel
    this.targets[n] = target;
    return target;
  }
}
