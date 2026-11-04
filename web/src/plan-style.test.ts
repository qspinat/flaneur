import { describe, expect, it } from 'vitest';
import planStyle from './plan-style.json';
import css from './style.css?raw';

const token = (name: string) => css.match(new RegExp(`--${name}:\\s*(#[0-9a-f]{6})`, 'i'))?.[1];

describe('the plan', () => {
  it('is faded to the site’s own ink and paper (else: npm run plan-style)', () => {
    expect(planStyle.metadata.faded).toEqual({ ink: token('mount'), paper: token('albumen') });
  });
});
