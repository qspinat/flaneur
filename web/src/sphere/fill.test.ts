import { describe, expect, it } from 'vitest';
import { EDGE_DEG, level, levels } from './fill';

const PIXEL_DEG = 90 / 700; // the pipeline's sheets (fill.py's tests): 2° is 16 pixels, 1° 8

describe('the fill, as the pipeline makes it', () => {
  it('reaches the same angle at any size of screen', () => {
    expect(levels(PIXEL_DEG)).toBe(5);
    expect(levels(PIXEL_DEG / 2)).toBe(6); // twice as fine: one level more
    expect(level(EDGE_DEG, PIXEL_DEG)).toBe(3); // texels of 8 pixels: about 1°
  });
});
