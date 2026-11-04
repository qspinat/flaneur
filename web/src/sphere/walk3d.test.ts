import { describe, expect, it } from 'vitest';
import {
  bearingOf,
  cameraAxes,
  correction,
  nearestTo,
  pointers,
  shownHere,
  stepTarget,
  toward,
  turn,
  yawOf,
  type OldPhoto,
  type Panorama,
  type Scene,
} from './walk3d';

const REG = { panorama: 'p1', m: [1, 0, 0, 0, 1, 0, 0, 0, 1], view: { heading: 30, pitch: 5, fov: 75 } };

function panorama(id: string, lon: number, links: [number, number, number][] = []): Panorama {
  return { id, lon, lat: 48.852, azimuth: 120, roll: 2, pitch: -1, taken: '2024-05-01', credit: 0, image: `${id}.jpg`, links };
}

function photo(id: string, years: [number, number], shown: [string, number][]): OldPhoto {
  return {
    id,
    title: 'Rue',
    years,
    credit: 'Paris Musées',
    image: `photos/${id}.webp`,
    flipped: false,
    shown: shown.map(([p, strength]) => ({ panorama: p, registration: { ...REG, panorama: p }, strength })),
  };
}

const SCENE: Scene = {
  credits: ['Panoramax'],
  panoramas: [panorama('p1', 2.3466, [[1, 90, 7]]), panorama('p2', 2.3467, [[0, 270, 7]]), panorama('p3', 2.351)],
  photos: [photo('a', [1900, 1900], [['p1', 1], ['p2', 0.4]]), photo('b', [1860, 1860], [['p2', 1]]), photo('c', [1910, 1910], [['p1', 0.3]])],
};

describe('the 3D walk', () => {
  it('levels a panorama as the pipeline does', () => {
    const axes = cameraAxes(198, -8.4, 6.9); // match.camera_axes(198, -8.4, 6.9)
    const expected = [
      [-0.938745, 0.32347, -0.118848],
      [-0.305702, -0.940854, -0.146083],
      [-0.159072, -0.100803, 0.982107],
    ];
    axes.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i][j], 5)));
  });

  it('turns the viewer by the measured tilt, the way the spike found it right', () => {
    expect(correction(SCENE.panoramas[0])).toEqual({ pan: 0, tilt: (1 * Math.PI) / 180, roll: (2 * Math.PI) / 180 });
    expect(yawOf(SCENE.panoramas[0], 150)).toBeCloseTo((30 * Math.PI) / 180, 9);
  });

  it('shows the photos carried onto this panorama, in the years, the strongest on top', () => {
    const here = shownHere(SCENE.photos, 'p1', [1890, 1920], false);

    expect(here.map(([p, s]) => [p.id, s.strength])).toEqual([
      ['c', 0.3],
      ['a', 1],
    ]);
    expect(shownHere(SCENE.photos, 'p1', [1905, 1920], false).map(([p]) => p.id)).toEqual(['c']);
  });

  it('points to the photos near whose copies stopped, toward their home', () => {
    const found = pointers(SCENE, SCENE.panoramas[1], [1850, 1950], true);

    expect(found.map((p) => p.photo.id)).toEqual(['c']); // a and b show here; c only on p1
    expect(found[0].bearing).toBeCloseTo(270, 0); // p1 lies west of p2
  });

  it('steps along the link most nearly ahead, none when none is', () => {
    expect(stepTarget(SCENE, SCENE.panoramas[0], 80)?.id).toBe('p2');
    expect(stepTarget(SCENE, SCENE.panoramas[0], 200)).toBeNull();
  });

  it('starts at the panorama nearest a spot', () => {
    expect(nearestTo(SCENE, 2.3509, 48.852).id).toBe('p3');
  });

  it('finds the photo of the walk nearest where one stands and facing the way one looks', () => {
    const walk = [
      { id: 'w1', lon: 2.3466, lat: 48.852, heading: 90, street: null, arrondissement: 5 },
      { id: 'w2', lon: 2.3466, lat: 48.852, heading: 270, street: null, arrondissement: 5 },
      { id: 'far', lon: 2.36, lat: 48.86, heading: 90, street: null, arrondissement: 5 },
    ];

    expect(toward(walk, SCENE.panoramas[0], 260)?.id).toBe('w2');
    expect(toward(walk, panorama('x', 2.38), 90)).toBeNull();
  });
});

describe('the way one looks', () => {
  it('is a compass bearing turned back from the viewer’s yaw', () => {
    const p = SCENE.panoramas[0]; // azimuth 120
    for (const bearing of [0, 30, 150, 300]) expect(bearingOf(p, yawOf(p, bearing))).toBeCloseTo(bearing, 9);
  });
});

describe('turning', () => {
  it('turns by -180 to 180 degrees, an opposite way by +180', () => {
    expect(turn(0, 180)).toBe(180);
    expect(turn(350, 10)).toBe(20);
    expect(turn(10, 350)).toBe(-20);
  });
});
