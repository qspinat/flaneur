import { describe, expect, it } from 'vitest';
import { depthAlong, photoPixel, squeezed, standingAt, type Standing, type WallRow } from './standing';

// From the pipeline (facades.cast, registration.Standing.project) on the same street
const walls: WallRow[] = [
  [-4, 2, -4, 30, 12],
  [4, 2, 4, 30, 14],
  [-6, 30, 6, 30, 10],
];
const ground = -2.5;
const standing: Standing = {
  position: [1.2, -1.0, -0.8],
  turn: [1.5650859288552346, -0.15703238345724274, 0.15703238345724274],
  focal: 700,
  shift: [0.03, -0.05],
  size: [600, 800],
};
const seen: [number[], number, [number, number]][] = [
  [[0, 1, 0], 30, [253.66964590027408, 251.4222962076919]],
  [[-0.28603877677367767, 0.9534625892455922, 0.09534625892455922], 13.984117975602022, [-3.272710295710513, 155.2565450552099]],
  [[0.3651483716701107, 0.9128709291752768, 0.18257418583505536], 10.954451150103322, [458.50440343766104, 97.11485584585537]],
  [[0.04897021068743918, 0.9794042137487836, -0.19588084274975673], 12.762861160413834, [250.79269784295863, 360.72428826705834]], // the street
  [[0, 0.9987523388778446, -0.04993761694389223], 30.03747659175118, [253.66964590027408, 286.25549081827]],
];

describe('a photo laid on the walls from where it was taken', () => {
  it('meets the walls and the street as the pipeline does, and sees them as its camera does', () => {
    for (const [d, metres, px] of seen) {
      expect(depthAlong(d, walls, ground)).toBeCloseTo(metres, 6);
      const [x, y] = photoPixel(d, metres, standing);
      expect(x).toBeCloseTo(px[0], 4);
      expect(y).toBeCloseTo(px[1], 4);
    }
  });

  it('puts the sky over the roofs as far as the wall under it', () => {
    const up = [0.07412493166611012, 0.7412493166611012, 0.667124384994991];

    expect(depthAlong(up, walls, ground)).toBeCloseTo(40.47221268969613, 6);
  });

  it('leaves the sky over no wall infinitely far, seen along its direction alone', () => {
    const behind = [-0.9116846116771036, -0.2279211529192759, 0.3418817293789138]; // no wall there
    const ahead = [-0.9476338693022512, 0.24937733402690823, 0.1995018672215266];
    const [x, y] = photoPixel(ahead, Infinity, standing);

    expect(depthAlong(behind, walls, ground)).toBe(Infinity);
    expect(x).toBeCloseTo(-11773.9724633, 3);
    expect(y).toBeCloseTo(-2217.52770466, 3);
  });

  it('gives the place in the photo, in shares of its width and height', () => {
    const [d, metres, px] = seen[0];
    const uv = standingAt(d, standing, walls, ground);

    expect(depthAlong(d, walls, ground)).toBe(metres);
    expect(uv?.[0]).toBeCloseTo(px[0] / 800, 6);
    expect(uv?.[1]).toBeCloseTo(px[1] / 600, 6);
  });

  it('draws nothing where the photo does not reach', () => {
    expect(standingAt(seen[1][0], standing, walls, ground)).toBeNull(); // left of it
  });

  it('draws nothing the photographer could not see, a wall standing between', () => {
    const between: WallRow = [0.5, 10, 2.0, 10, 5];

    expect(standingAt([0, 1, 0], standing, walls, ground)).not.toBeNull();
    expect(standingAt([0, 1, 0], standing, [...walls, between], ground)).toBeNull();
  });

  it('catches no click where the photo fades out, smeared on a wall the photographer saw edge on', () => {
    const north: Standing = { position: [3.9, -1, 0], turn: [Math.PI / 2, 0, 0], focal: 700, shift: [0, 0], size: [600, 800] };
    const unit = (v: number[]) => v.map((x) => x / Math.hypot(...v));
    const edgeOn = unit([4, 15, 0.5]); // on the right wall, which the camera grazes
    const ahead = unit([0, 30, 0.5]); // on the end wall, seen straight on

    expect(standingAt(edgeOn, north, walls, ground)).not.toBeNull(); // drawn, were it not smeared
    expect(squeezed(edgeOn, north, walls, ground)).toBeLessThan(squeezed(ahead, north, walls, ground) / 6);
    expect(standingAt(edgeOn, north, walls, ground, 700)).toBeNull();
    expect(standingAt(ahead, north, walls, ground, 700)).not.toBeNull();
  });
});
