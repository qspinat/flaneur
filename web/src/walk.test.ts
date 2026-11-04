import { describe, expect, it } from 'vitest';
import {
  arrondissementLabel,
  distanceLabel,
  erasOf,
  hashOf,
  largest,
  photoIdFrom,
  srcsetOf,
  startOf,
  yearsLabel,
  type Photo,
} from './walk';

function photo(id: string, over: Partial<Photo> = {}): Photo {
  return {
    id,
    title: id,
    caption: null,
    years: [1900, 1900],
    photographer: 'x',
    credit: 'x',
    record_url: null,
    lon: 2.35,
    lat: 48.85,
    heading: 0,
    street: null,
    arrondissement: null,
    images: { '960': 'a-960.webp', '1600': 'a.webp' },
    today: null,
    links: { forward: null, back: null, left: null, right: null },
    eras: [],
    ...over,
  };
}

describe('the URL', () => {
  it('names a photo and reads back', () => {
    expect(photoIdFrom(hashOf('paris_musees:75860'))).toBe('paris_musees:75860');
    expect(photoIdFrom('')).toBeNull();
  });
});

describe('labels', () => {
  it('writes years, distances and arrondissements in French', () => {
    expect(yearsLabel([1912, 1912])).toBe('1912');
    expect(yearsLabel([1905, 1906])).toBe('1905–1906');
    expect(yearsLabel([null, null])).toBe('');
    expect(distanceLabel(42.4)).toBe('42 m');
    expect(distanceLabel(1240)).toBe('1,2 km');
    expect(arrondissementLabel(1)).toBe('1er Arrt');
    expect(arrondissementLabel(5)).toBe('5e Arrt');
  });


});

describe('photograph sizes', () => {
  it('lists every size for srcset and finds the largest', () => {
    const images = { '960': 'photos/a-960.webp', '1600': 'photos/a.webp' };
    expect(srcsetOf(images)).toBe('/data/photos/a-960.webp 960w, /data/photos/a.webp 1600w');
    expect(largest(images)).toBe('photos/a.webp');
  });
});

describe('the walk', () => {
  it('lists the same view in other years oldest first', () => {
    const a = photo('a', { years: [1912, 1912], eras: ['b'] });
    const b = photo('b', { years: [1866, 1868] });
    expect(erasOf(a, new Map([['b', b]])).map((p) => p.id)).toEqual(['b', 'a']);
  });


  it('starts at the photo of the URL, else where there are most ways to go', () => {
    const lone = photo('lone');
    const hub = photo('hub', { links: { forward: { id: 'lone', distance_m: 40 }, back: null, left: null, right: null } });
    const walk = { photos: [lone, hub] };
    expect(startOf(walk, '#/lone').id).toBe('lone');
    expect(startOf(walk, '').id).toBe('hub');
  });
});
