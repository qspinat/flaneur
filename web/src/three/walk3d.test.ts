import { describe, expect, it } from 'vitest';
import { FIRST_YEAR, LAST_YEAR, format3d, parse3d } from '../walk';
import {
  inRange,
  nearby,
  nearestPanorama,
  nearestTo,
  photoToward,
  stepTarget,
  toThree,
  turn,
  yearMarks,
  type OldPhoto,
  type Panorama,
  type Scene,
  type WalkSpot,
} from './walk3d';

const pano = (id: string, x: number, y: number, links: Panorama['links'] = []): Panorama => ({
  id, x, y, z: 2.5, lon: 2.347 + x / 73_000, lat: 48.853 + y / 111_200, azimuth: 0, taken: '2025-06-01', credit: 'Panoramax', url: '', image: '', links,
});

describe('the 3D walk', () => {
  it('turns the short way round', () => {
    expect(turn(350, 10)).toBe(20);
    expect(turn(10, 350)).toBe(-20);
  });

  it('steps to the link within 45° of where one looks', () => {
    const here = pano('a', 0, 0, [['north', 30, 5], ['east', 90, 6]]);
    expect(stepTarget(here, 0)?.[0]).toBe('north');
    expect(stepTarget(here, 75)?.[0]).toBe('east');
    expect(stepTarget(here, 200)).toBeNull();
  });

  it('finds the panorama nearest a point', () => {
    const scene = { panoramas: [pano('a', 0, 0), pano('b', 10, 0)] } as Scene;
    expect(nearestPanorama(scene, 7, 1).id).toBe('b');
  });

  it('puts east on x, up on y and north on −z', () => {
    expect(toThree(1, 2, 3)).toEqual([1, 3, -2]);
  });
});

const photo = (id: string, x: number, heading: number): OldPhoto => ({
  id, title: id, years: [1900, 1900], credit: '', image: '', aspect: 1.3,
  pose: { x, y: 0, z: 1.6, heading, horizon: 0.6, fov: 53 }, walls: [], base: 0,
});

describe('the old photos around', () => {
  it('stay whichever way one looks, and fade out between 70 and 80 m', () => {
    const photos = [photo('ahead', 5, 0), photo('behind', 6, 180), photo('fading', 75, 0), photo('gone', 90, 0)];
    expect(nearby(photos, 0, 0).map(([p, s]) => [p.id, s])).toEqual([['fading', 0.5], ['behind', 1], ['ahead', 1]]);
  });

  it('put the nearest camera last, so that its photo lies on top', () => {
    const many = Array.from({ length: 12 }, (_, i) => photo(`p${i}`, i + 1, 0));
    const shown = nearby(many, 0, 0);
    expect(shown).toHaveLength(8);
    expect(shown.at(-1)?.[0].id).toBe('p0');
  });
});

describe('the date range', () => {
  const dated = (years: OldPhoto['years']) => ({ ...photo('p', 0, 0), years });

  it('shows a photo whose span overlaps it', () => {
    expect(inRange(dated([1905, 1910]), [1908, 1930])).toBe(true);
    expect(inRange(dated([1905, 1910]), [1880, 1906])).toBe(true);
    expect(inRange(dated([1905, 1910]), [1911, 1930])).toBe(false);
    expect(inRange(dated([1912, null]), [1912, 1912])).toBe(true);
  });

  it('shows an undated photo only over the whole period', () => {
    expect(inRange(dated([null, null]), [FIRST_YEAR, LAST_YEAR])).toBe(true);
    expect(inRange(dated([null, null]), [1900, LAST_YEAR])).toBe(false);
  });

  it('marks the years that have photos near where one stands', () => {
    const photos = [
      { ...photo('a', 10, 0), years: [1902, 1902] as [number, number] },
      { ...photo('b', 20, 0), years: [1902, 1904] as [number, number] },
      { ...photo('far', 500, 0), years: [1880, 1880] as [number, number] },
    ];
    expect(yearMarks(photos, 0, 0)).toEqual([1902, 1903, 1904]);
  });
});

describe('the link to a place in 3D', () => {
  it('round-trips a point, a heading and a range', () => {
    const state = { lon: 2.34712, lat: 48.85301, heading: 271, range: [1900, 1920] as [number, number] };
    expect(parse3d(format3d(state))).toEqual(state);
  });

  it('reads nothing from a broken link', () => {
    expect(parse3d('#2.3,abc,10,1900-1920')).toBeNull();
    expect(parse3d('#2.3,48.8,10,1920-1900')).toBeNull(); // a range backwards
    expect(parse3d('')).toBeNull();
  });
});

describe('a point of the map', () => {
  it('opens on the panorama nearest it', () => {
    const scene = { panoramas: [pano('a', 0, 0), pano('b', 30, 0), pano('c', 0, 30)] } as Scene;
    expect(nearestTo(scene, 2.347 + 28 / 73_000, 48.853 + 2 / 111_200).id).toBe('b');
  });
});

describe('back to the photos', () => {
  const spot = (id: string, x: number, heading: number | null): WalkSpot => ({
    id, x, y: 0, lon: 0, lat: 0, heading, street: null, arrondissement: null,
  });

  it('goes to the photo nearest where one stands, facing the way one looks', () => {
    const photos = [spot('behind me', 3, 180), spot('my way', 12, 10), spot('far', 400, 0)];

    expect(photoToward(photos, 0, 0, 0)?.id).toBe('my way');
    expect(photoToward(photos, 0, 0, 180)?.id).toBe('behind me');
    expect(photoToward(photos, 1000, 0, 0)).toBeNull(); // none near
  });

  it("prefers a photo facing one's way to one whose direction is not known", () => {
    const photos = [spot('direction unknown', 20, null), spot('facing my way', 22, 0)];

    expect(photoToward(photos, 0, 0, 0)?.id).toBe('facing my way');
  });
});
