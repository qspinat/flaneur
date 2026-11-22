import { describe, expect, it } from 'vitest';
import { clipPath, framing } from './plate';

describe('an Albert-Kahn plate', () => {
  it('is drawn within its frame, its label left out', () => {
    expect(framing({ plate: [0.1, 0.02, 0.9, 0.95], label: [0.1, 0.85, 0.3, 0.92] })).toEqual({
      plate: [0.1, 0.02, 0.9, 0.95],
      label: [0.1, 0.85, 0.3, 0.92],
    });
  });

  it('any other photo is drawn whole, nothing left out', () => {
    const { plate, label } = framing({});
    expect(plate).toEqual([0, 0, 1, 1]);
    expect(label[2]).toBeLessThan(0); // an empty box: no pixel falls in it
  });

  it('is shown in the photo walk cut to its frame', () => {
    expect(clipPath([0.1, 0.02, 0.9, 0.95])).toBe('inset(2% 10% 5% 10%)');
    expect(clipPath(undefined)).toBe('');
  });
});
