/** The two materials of the 3D walk. Today's panorama is projected from where it was taken
 * onto today's buildings, the ground and a far sphere. Each old photo is projected from its
 * camera onto the facades it shows, as they stood, see-through outside its frame and as it
 * fades. */
import * as THREE from 'three';

const PI = '3.141592653589793';
/** Soft frame edges, a share of the photo. */
export const EDGE = 0.08;
/** A point of an old photo shows fully while one's line of sight to it is within 12° of the
 * photographer's, and not at all past 35°. */
export const FULL_DEG = 12;
export const GONE_DEG = 35;
const COS_FULL = Math.cos((FULL_DEG * Math.PI) / 180).toFixed(4);
const COS_GONE = Math.cos((GONE_DEG * Math.PI) / 180).toFixed(4);

const vertexShader = `
varying vec3 vWorld;
void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorld = world.xyz;
  gl_Position = projectionMatrix * viewMatrix * world;
}`;

const panoramaShader = `
varying vec3 vWorld;
uniform sampler2D panoA;
uniform sampler2D panoB;
uniform vec3 panoAAt;
uniform vec3 panoBAt;
uniform float panoAAzimuth;
uniform float panoBAzimuth;
uniform float toB;
uniform float sepia;

vec3 panorama(sampler2D picture, vec3 at, float azimuth) {
  vec3 d = normalize(vWorld - at);
  float bearing = atan(d.x, -d.z);
  float u = fract(0.5 + (bearing - azimuth) / (2.0 * ${PI}));
  float v = 0.5 + asin(clamp(d.y, -1.0, 1.0)) / ${PI};
  return texture2D(picture, vec2(u, v)).rgb;
}

void main() {
  vec3 colour = mix(panorama(panoA, panoAAt, panoAAzimuth), panorama(panoB, panoBAt, panoBAzimuth), toB);
  float grey = dot(colour, vec3(0.299, 0.587, 0.114));
  vec3 toned = vec3(grey * 1.07, grey * 0.95, grey * 0.78) * 0.85;
  gl_FragColor = vec4(mix(colour, toned, sepia), 1.0);
}`;

const photoShader = `
varying vec3 vWorld;
uniform sampler2D photo;
uniform mat4 view;
uniform vec3 at;
uniform float strength;

void main() {
  vec4 clip = view * vec4(vWorld, 1.0);
  if (clip.w <= 0.0) discard;
  vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
  vec2 edge = min(uv, 1.0 - uv);
  // seen from where it was taken, a point of the photo is right; the further one's line of
  // sight to it turns from the photographer's, the less: it fades into today's view
  float turned = dot(normalize(vWorld - at), normalize(vWorld - cameraPosition));
  float alpha = smoothstep(0.0, ${EDGE}, min(edge.x, edge.y)) * smoothstep(${COS_GONE}, ${COS_FULL}, turned) * strength;
  if (alpha < 0.01) discard;
  gl_FragColor = vec4(texture2D(photo, uv).rgb, alpha);
}`;

export function panoramaMaterial(side: THREE.Side = THREE.FrontSide): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      panoA: { value: null },
      panoB: { value: null },
      panoAAt: { value: new THREE.Vector3() },
      panoBAt: { value: new THREE.Vector3() },
      panoAAzimuth: { value: 0 },
      panoBAzimuth: { value: 0 },
      toB: { value: 0 },
      sepia: { value: 0 },
    },
    vertexShader,
    fragmentShader: panoramaShader,
    side,
  });
}

/** An old photo seen through its camera (`view`: projection × view matrix, standing `at`). */
export function photoMaterial(photo: THREE.Texture, view: THREE.Matrix4, at: THREE.Vector3): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { photo: { value: photo }, view: { value: view }, at: { value: at }, strength: { value: 0 } },
    vertexShader,
    fragmentShader: photoShader,
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false, // the cores of today's buildings hide it where they stand in front
  });
}
