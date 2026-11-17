"""From `match` to a panorama: the pairs a placement's points give in a panorama's view (a
photo's pixel, the pixel where that panorama matched the same detail), as the placing page
takes pairs — to register a placement there on its own, the way the page does."""

from dataclasses import dataclass

import numpy as np

from flaneur import match, placing, registration

KEPT_DECIMALS = 3  # a point's pixel, as chain.Store.save_points rounds it


@dataclass(frozen=True)
class Paired:
    """Pairs of a placement in a panorama: the photo's pixels (`photo_px`, at the size asked)
    and where the panorama matched the same details in a `view` cut from it (`view_px`, of
    placing.SHOWN pixels square)."""

    panorama: str
    view: registration.ViewOf
    photo_px: np.ndarray
    view_px: np.ndarray


def keys_of(pixels: np.ndarray, observed: match.Observed) -> np.ndarray:
    """The row of `observed.pixels` each pixel is (both rounded to KEPT_DECIMALS, as kept on
    disk); -1 for none."""
    kept = np.round(observed.pixels.astype(float), KEPT_DECIMALS).tolist()
    key_of = {tuple(xy): k for k, xy in enumerate(kept)}
    return np.array(
        [key_of.get(tuple(xy), -1) for xy in np.round(pixels, KEPT_DECIMALS).tolist()], int
    )


def paired(
    points: match.Points, observed: match.Observed, sphere: match.Panorama, size: tuple[int, int]
) -> Paired | None:
    """The points of a placement `sphere` matched itself (its own ray, kept by `match`: never a
    point it is only said to see), in a view of it facing their mean heading, those falling in
    it; the photo's pixels at `size` (height, width). None from fewer than
    registration.MIN_MATCHED: too few to register the photo there."""
    rays = observed.rays
    row_of: dict[int, int] = {}
    for row in np.flatnonzero(rays.sources == sphere.id):
        row_of.setdefault(int(rays.keys[row]), int(row))
    rows = np.array([row_of.get(int(k), -1) for k in keys_of(points.pixels, observed)], int)
    hit = rows >= 0
    if hit.sum() < registration.MIN_MATCHED:
        return None
    ahead = rays.directions[rows[hit]]
    angles = np.arctan2(ahead[:, 0], ahead[:, 1])  # compass: east of north
    heading = float(np.degrees(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())) % 360)
    view = registration.ViewOf(heading)
    view_px = registration.view_pixels(ahead @ sphere.axes.T, view, sphere.axes, placing.SHOWN)
    inside = np.isfinite(view_px).all(axis=1) & (view_px >= 0).all(axis=1)
    inside &= (view_px <= placing.SHOWN).all(axis=1)
    if inside.sum() < registration.MIN_MATCHED:
        return None
    photo_px = points.pixels[hit][inside] * max(size) / match.PHOTO_SIDE
    return Paired(sphere.id, view, photo_px, view_px[inside])
