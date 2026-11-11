import { describe, expect, it, vi } from 'vitest';
import {
  arrondissementLabel,
  offerOf3d,
  parsePhoto3d,
  photo3d,
  distanceLabel,
  erasOf,
  fitting,
  hashOf,
  setAddress,
  httpsUrl,
  largest,
  photoIdFrom,
  srcsetOf,
  richest,
  shownIn,
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
    in_3d: false,
    no_3d: false,
    aspect: null,
    ...over,
  };
}

describe('the URL', () => {
  it('names a photo and reads back', () => {
    expect(photoIdFrom(hashOf('paris_musees:75860'))).toBe('paris_musees:75860');
    expect(photoIdFrom('')).toBeNull();
  });

  it('takes a malformed link for none', () => {
    expect(photoIdFrom('#/%E0%A4')).toBeNull();
  });
});

describe('links out of the site', () => {
  it('keeps https addresses only', () => {
    expect(httpsUrl('https://www.parismuseescollections.paris.fr/fr/node/1')).toBe(
      'https://www.parismuseescollections.paris.fr/fr/node/1',
    );
    expect(httpsUrl('javascript:alert(1)')).toBeNull();
    expect(httpsUrl('http://example.org/')).toBeNull();
    expect(httpsUrl('not a url')).toBeNull();
    expect(httpsUrl(null)).toBeNull();
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
    expect(srcsetOf(images)).toBe('data/photos/a-960.webp 960w, data/photos/a.webp 1600w');
    expect(largest(images)).toBe('photos/a.webp');
  });

  it('picks the smallest size about as wide as needed, else the largest', () => {
    const images = { '213': 'photos/a-320.webp', '640': 'photos/a-960.webp', '1067': 'photos/a.webp' };
    expect(fitting(images, 150)).toBe('photos/a-320.webp');
    expect(fitting(images, 700)).toBe('photos/a-960.webp'); // 640 is within a tenth of it
    expect(fitting(images, 1014)).toBe('photos/a.webp');
    expect(fitting(images, 3000)).toBe('photos/a.webp');
  });
});

describe('the walk', () => {
  it('lists the same view in other years oldest first', () => {
    const a = photo('a', { years: [1912, 1912], eras: ['b'] });
    const b = photo('b', { years: [1866, 1868] });
    expect(erasOf(a, new Map([['b', b]])).map((p) => p.id)).toEqual(['b', 'a']);
  });

  it('begins the walk where there are most ways to go', () => {
    const lone = photo('lone');
    const hub = photo('hub', { links: { forward: { id: 'lone', distance_m: 40 }, back: null, left: null, right: null } });
    expect(richest([lone, hub]).id).toBe('hub');
  });
});

describe('the years chosen', () => {
  it('shows a photograph whose years meet them, an undated one only when they are all chosen', () => {
    expect(shownIn([1905, 1906], [1900, 1905], false)).toBe(true);
    expect(shownIn([1866, 1868], [1900, 1950], false)).toBe(false);
    expect(shownIn([1912, null], [1912, 1912], false)).toBe(true);
    expect(shownIn([null, null], [1900, 1950], false)).toBe(false);
    expect(shownIn([null, null], [1850, 1950], true)).toBe(true);
  });
});


describe('the 3D walk, from a photo', () => {
  it('is offered only for a photo placed in it', () => {
    expect(offerOf3d(photo('a', { in_3d: true }))).toBe('link');
    expect(offerOf3d(photo('b'))).toBeNull();
  });

  it('is said not to be for a photo no panorama sees', () => {
    expect(offerOf3d(photo('c', { no_3d: true }))).toBe('far');
  });

  it('is offered when a photo is placed in it, whatever else was answered', () => {
    expect(offerOf3d(photo('d', { in_3d: true, no_3d: true }))).toBe('link');
  });
});


describe('the 3D walk, opened on a photo', () => {
  it('names the photo and the years in its link, and reads them back', () => {
    const link = photo3d('paris_musees:171285', [1855, 1878]);

    expect(parsePhoto3d(link)).toEqual({ id: 'paris_musees:171285', range: [1855, 1878] });
    expect(parsePhoto3d('#2.3466,48.852,40,1850-2026')).toBeNull();
  });
});

describe('setAddress', () => {
  const browser = (hash: string, pushState = vi.fn()) => ({
    history: { pushState, replaceState: vi.fn() },
    location: { hash },
  });

  it('writes a new entry, or this one', () => {
    const b = browser('#/a');
    setAddress('#/b', true, b.history, b.location);
    setAddress('#/c', false, b.history, b.location);
    expect(b.history.pushState).toHaveBeenCalledWith(null, '', '#/b');
    expect(b.history.replaceState).toHaveBeenCalledWith(null, '', '#/c');
  });

  it('leaves the address alone when it already says so', () => {
    const b = browser('#/a');
    setAddress('#/a', false, b.history, b.location);
    setAddress('#/a', true, b.history, b.location);
    expect(b.history.replaceState).not.toHaveBeenCalled();
    expect(b.history.pushState).not.toHaveBeenCalled();
  });

  it('lets a browser that refuses be: the photo still shows', () => {
    const refuse = () => {
      throw new DOMException('The operation is insecure.', 'SecurityError');
    };
    const b = browser('#/a', vi.fn(refuse));
    expect(() => setAddress('#/b', true, b.history, b.location)).not.toThrow();
  });

  it('lets any other error through', () => {
    const b = browser('#/a', vi.fn(() => { throw new TypeError('broken'); }));
    expect(() => setAddress('#/b', true, b.history, b.location)).toThrow(TypeError);
  });
});
