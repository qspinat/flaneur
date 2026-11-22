import { describe, expect, it } from 'vitest';
import { clipPath, drawn, framing } from './plate';

const kahn = { plate: [0.1, 0.02, 0.9, 0.95], label: [0.1, 0.85, 0.3, 0.92] } as const;

describe('an Albert-Kahn plate', () => {
  it('is drawn within its frame, its label left out', () => {
    const photo = { plate: [...kahn.plate], label: [...kahn.label] } as Parameters<typeof framing>[0];
    expect(framing(photo)).toEqual({ plate: kahn.plate, label: kahn.label });
    expect(drawn(photo, 0.5, 0.5)).toBe(true);
    expect(drawn(photo, 0.05, 0.5)).toBe(false); // on the frame
    expect(drawn(photo, 0.2, 0.9)).toBe(false); // on the label
  });

  it('any other photo is drawn whole, nothing left out', () => {
    const { plate, label } = framing({});
    expect(plate).toEqual([0, 0, 1, 1]);
    expect(label[2]).toBeLessThan(0); // an empty box: no pixel falls in it
    expect(drawn({}, 0, 1)).toBe(true);
    expect(drawn({}, 1.01, 0.5)).toBe(false);
  });

  it('is shown in the photo walk cut to its frame, where the image lies in its element', () => {
    const plate = [0.1, 0.02, 0.9, 0.95] as [number, number, number, number];
    // a 1000 × 800 image in a 500 × 800 element: 500 × 400, 200 px above and below it
    expect(clipPath(plate, { width: 500, height: 800 }, { width: 1000, height: 800 })).toBe('inset(208px 50px 220px 50px)');
    expect(clipPath(undefined, { width: 500, height: 800 }, { width: 1000, height: 800 })).toBe('');
    expect(clipPath(plate, { width: 500, height: 800 }, { width: 0, height: 0 })).toBe(''); // not loaded yet
  });
});
