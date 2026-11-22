// A standing photo drawn without holes, as the pipeline draws it (pipeline/src/flaneur/fill.py,
// drawing.py): drawn first apart, its colour weighed by what each pixel is worth (0 where the
// photo has nothing to show: hidden from the photographer, smeared) and where it reaches at all;
// the graphics chip halves both into levels (mipmaps, box averages); one pass reads every level
// at each pixel and pulls them back from the coarsest: small gaps are filled from around them,
// large ones fade to today, and the photo's edge fades across its outline. Its reach is an
// angle: a gap looks the same filled or faded at any size of screen.
import * as THREE from 'three';

export const FILL_DEG = 2; // gaps up to about this wide are filled; wider ones fade
export const EDGE_DEG = 1; // the photo's edge fades over about this

/** 0 at `low`, 1 at `high`, smoothly between (GLSL's smoothstep). */
export function smooth(x: number, low: number, high: number): number {
  const t = Math.min(1, Math.max(0, (x - low) / (high - low)));
  return t * t * (3 - 2 * t);
}

/** How many levels reach FILL_DEG, for pixels of `pixelDeg` degrees. */
export function levels(pixelDeg: number): number {
  return 1 + Math.ceil(Math.log2(FILL_DEG / pixelDeg));
}

/** The level whose texels are about `degrees` across. */
export function level(degrees: number, pixelDeg: number): number {
  return Math.max(0, Math.round(Math.log2(degrees / pixelDeg)));
}

const MOST_LEVELS = 14; // a loop's bound (a screen of 16 384 pixels)

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
  for (int k = ${MOST_LEVELS}; k >= 0; k--) {
    if (k > depth - 2) continue;
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

/** Draws standing photos filled: each drawn first apart (its own shader writes colour times
 *  weight, and where it reaches), then laid over what is on the screen. */
export class FilledLayer {
  private target: THREE.WebGLRenderTarget | null = null;
  private apart = new THREE.Scene();
  private screen = new THREE.Scene();
  private quad: THREE.Mesh;
  private material: THREE.ShaderMaterial;
  private size = new THREE.Vector2();
  private clearColour = new THREE.Color();

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
        screen: { value: new THREE.Vector2() },
      },
      vertexShader,
      fragmentShader,
      transparent: true,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
    });
    this.quad = new THREE.Mesh(undefined, this.material);
    this.screen.add(this.quad);
  }

  /** Lay `mesh` (a standing photo) filled over the screen, at `strength`, seen by `camera`. */
  draw(renderer: THREE.WebGLRenderer, mesh: THREE.Mesh, camera: THREE.PerspectiveCamera, strength: number): void {
    renderer.getDrawingBufferSize(this.size);
    const target = this.targetOf(this.size.x, this.size.y);
    const [clearAlpha, autoClear] = [renderer.getClearAlpha(), renderer.autoClear];
    renderer.getClearColor(this.clearColour);
    const parent = mesh.parent;
    this.apart.add(mesh);
    renderer.setRenderTarget(target);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(this.apart, camera);
    renderer.setRenderTarget(null);
    this.apart.remove(mesh);
    parent?.add(mesh);
    const pixelDeg = camera.fov / this.size.y;
    const depth = levels(pixelDeg);
    Object.assign(this.material.uniforms, {
      layer: { value: target.textures[0] },
      reach: { value: target.textures[1] },
      depth: { value: Math.min(depth, MOST_LEVELS + 1) },
      smoothest: { value: Math.max(0, depth - 2) },
      edgiest: { value: Math.min(level(EDGE_DEG, pixelDeg), depth - 1) },
      strength: { value: strength },
      screen: { value: this.size },
    });
    this.quad.geometry = mesh.geometry; // the pass over the photo's quad only
    this.quad.matrixWorld.copy(mesh.matrixWorld);
    this.quad.matrixAutoUpdate = false;
    renderer.setClearColor(this.clearColour, clearAlpha);
    renderer.autoClear = false;
    renderer.render(this.screen, camera);
    renderer.autoClear = autoClear;
  }

  dispose(): void {
    this.target?.dispose();
    this.material.dispose();
  }

  private targetOf(width: number, height: number): THREE.WebGLRenderTarget {
    if (this.target && this.target.width === width && this.target.height === height) return this.target;
    this.target?.dispose();
    this.target = new THREE.WebGLRenderTarget(width, height, {
      count: 2,
      generateMipmaps: true,
      minFilter: THREE.LinearMipmapLinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
    });
    return this.target;
  }
}
