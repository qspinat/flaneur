"""How far today's details stand from a panorama: its views matched with those of neighbouring
panoramas (see neighbours), each detail found in both placed where the two rays meet (the
panoramas where they were taken, see cli._spheres). They agree with the panoramas by
construction; measured on 2026-11-06: hundreds of details a view, the rays passing within a few
centimetres."""

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import QhullError

from flaneur import match, registration
from flaneur.registration import ViewOf

NEAR_M = (2.0, 15.0)  # a neighbour this far: nearer tells little of depth, farther sees too
# little of the same
SPREAD_DEG = 30.0  # a neighbour is judged on rays at the view's heading and this either side
MOST_NEIGHBOURS = 4
REACH_DEG = 1.5  # a direction takes the depth of the details this near it
MOST_SPREAD = 1.5  # unless they disagree: the farther of them this much farther than the nearer
COARSE = 4  # depth over a view is made at a quarter of its side, then enlarged
COLUMNS = 40  # where the sky begins, found in this many columns of a view
FACING = 3  # views of a panorama whose details a view can show: the nearest its way
MIN_IN_VIEW = 10  # details seen in a view for a depth all over it


@dataclass(frozen=True)
class Depths:
    panorama: str
    directions: np.ndarray  # (n, 3) from the panorama, in the world (east, north, up), unit
    metres: np.ndarray  # (n,) how far along each
    seen_from: np.ndarray  # (n,) the neighbour each was found with
    fixed: np.ndarray  # (n,) whether COLMAP placed that neighbour itself


def neighbours(
    home: match.Panorama, spheres: list[match.Panorama], fixed: set[str], heading: float
) -> list[match.Panorama]:
    """The panoramas NEAR_M from `home`, at most MOST_NEIGHBOURS: those COLMAP placed itself
    (`fixed`) first, then those that see what a view facing `heading` shows from furthest
    aside: the distance between the two panoramas across its rays, at its heading and
    SPREAD_DEG either side, the least of the three (one along a ray tells little of how far
    along it things stand)."""
    bearings = np.radians(heading + np.array([-SPREAD_DEG, 0.0, SPREAD_DEG]))
    rays = np.column_stack([np.sin(bearings), np.cos(bearings)])  # east, north

    def across(p: match.Panorama) -> float:
        east, north = p.x - home.x, p.y - home.y
        return float(np.abs(rays[:, 0] * north - rays[:, 1] * east).min())

    near = [
        p
        for p in spheres
        if p.id != home.id and NEAR_M[0] <= np.hypot(p.x - home.x, p.y - home.y) <= NEAR_M[1]
    ]
    return sorted(near, key=lambda p: (p.id not in fixed, -across(p), p.id))[:MOST_NEIGHBOURS]


def meeting(a: np.ndarray, ray_a: np.ndarray, b: np.ndarray, ray_b: np.ndarray) -> float | None:
    """How far along `ray_a` (from `a`) it meets `ray_b` (from `b`), by match.placed's rules
    (passing within MEET_M, MEET_DEG or more apart, in front of both); None when they do not."""
    rays = [np.asarray(r, float)[None] / np.linalg.norm(r) for r in (ray_a, ray_b)]
    metres = _met(a, b, *rays)[1]
    return float(metres[0]) if len(metres) else None


def _met(
    a: np.ndarray, b: np.ndarray, rays_a: np.ndarray, rays_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Where pairs of rays (one from `a`, one from `b`; a row each) meet by match.placed's
    rules: the points, and how far from `a`; those that do not meet left out."""
    n = len(rays_a)
    points, _, _, _ = match.placed(
        np.tile(np.arange(n), 2),
        np.repeat(["a", "b"], n),
        np.vstack([np.tile(a, (n, 1)), np.tile(b, (n, 1))]),
        np.vstack([rays_a, rays_b]),
        np.full((2 * n, 3), np.nan),  # never cast onto a wall
    )
    return points, np.linalg.norm(points - a, axis=1)


def facing(heading: float) -> list[int]:
    """The FACING views (indices of match.VIEW_HEADINGS) nearest a view facing `heading`:
    those whose details it can show."""
    headings = np.array(match.VIEW_HEADINGS, float)
    off = np.abs((headings - heading + 180) % 360 - 180)
    return np.argsort(off, kind="stable")[:FACING].tolist()


def around(heading: float, made: Callable[[int], "Depths | None"]) -> "Depths | None":
    """The depth around a panorama the way a view faces: its facing views' (`made(k)` for each,
    see facing), joined; none when none has any."""
    parts = [found for k in facing(heading) if (found := made(k)) is not None]
    return joined(parts) if parts else None


def joined(parts: list[Depths]) -> Depths:
    """Depths made apart (views of one panorama, see facing), as one."""
    return Depths(
        parts[0].panorama,
        np.vstack([d.directions for d in parts]),
        np.concatenate([d.metres for d in parts]),
        np.concatenate([d.seen_from for d in parts]),
        np.concatenate([d.fixed for d in parts]),
    )


def triangulated(
    home: match.Panorama,
    other: match.Panorama,
    views: match.Views,
    only: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The details of each view of `home` found in the views of `other` facing the same way or
    a step either side (a near wall turns tens of degrees between two panoramas): their
    directions from `home` and how far along, where the two rays meet by match.placed's
    rules. Only those of `home`'s views in `only` (indices of match.VIEW_HEADINGS), else all."""
    a, b = np.array([home.x, home.y, home.z]), np.array([other.x, other.y, other.z])
    headings = list(match.VIEW_HEADINGS)
    mine, theirs = views.of(home), views.of(other)
    if len(mine) != len(headings) or len(theirs) != len(headings):  # a picture not read
        return np.empty((0, 3)), np.empty(0)
    rays_a, rays_b = [], []
    for k, heading in enumerate(headings):
        if only is not None and k not in only:
            continue
        for step in (-1, 0, 1):
            j = (k + step) % len(headings)
            ia, ib = views.matcher.match(mine[k], theirs[j])
            side_a, side_b = mine[k].size[1], theirs[j].size[1]
            rays_a.append(match.view_directions(mine[k].keypoints[ia], heading, side=side_a))
            rays_b.append(match.view_directions(theirs[j].keypoints[ib], headings[j], side=side_b))
    da, db = np.vstack(rays_a), np.vstack(rays_b)
    if not len(da):
        return np.empty((0, 3)), np.empty(0)
    points, metres = _met(a, b, da, db)
    return (points - a) / metres[:, None], metres


def of_panorama(
    home: match.Panorama,
    near: list[match.Panorama],
    fixed: set[str],
    views: match.Views,
    headings: list[int] | None = None,
) -> Depths | None:
    """The details around `home` with their depth, from the neighbours `near` (in its views
    `headings` only, see facing; else all around); none without any."""
    if not near:
        return None
    found = [(p, *triangulated(home, p, views, headings)) for p in near]
    if not any(len(m) for *_, m in found):
        return None
    return Depths(
        home.id,
        np.vstack([d for _, d, _ in found]),
        np.concatenate([m for *_, m in found]),
        np.concatenate([[p.id] * len(m) for p, _, m in found]),
        np.concatenate([[p.id in fixed] * len(m) for p, _, m in found]).astype(bool),
    )


def at(depths: Depths, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """How far along each direction (world, unit): the median of the details within REACH_DEG
    of it; NaN when there are none, or when they disagree (MOST_SPREAD: a corner, a near
    wall's edge against the street beyond: no depth rather than one in the air between);
    whether those came only from neighbours COLMAP did not place itself; and which neighbours
    they came from (None without a depth)."""
    near = directions @ depths.directions.T >= np.cos(np.radians(REACH_DEG))
    metres = np.full(len(directions), np.nan)
    only_unfixed = np.zeros(len(directions), bool)
    sources: list[list[str] | None] = [None] * len(directions)
    for i, row in enumerate(near):
        found = depths.metres[row]
        if len(found):
            low, high = np.percentile(found, [20, 80])
            metres[i] = np.median(found) if high <= MOST_SPREAD * low else np.nan
            only_unfixed[i] = not depths.fixed[row].any()
            if not np.isnan(metres[i]):
                sources[i] = sorted(set(depths.seen_from[row].tolist()))
    return metres, only_unfixed, sources


def in_view(depths: Depths, view: ViewOf, axes: np.ndarray, side: int) -> np.ndarray:
    """How far along each pixel's ray of a view (side × side): the details seen in it
    interpolated (as inverse depth, which is linear across a flat wall in a picture), the
    nearest beyond them; infinitely far above each column's highest details (the sky). Made
    at a quarter of the side (depth is smooth), the sky's edge at full size. NaN all over when
    fewer than MIN_IN_VIEW details are seen in it: unknown, not sky."""
    px = registration.view_pixels(depths.directions @ axes.T, view, axes, side)
    seen = np.all(np.isfinite(px) & (px >= 0) & (px < side), axis=1)
    px, inverse = px[seen], 1 / depths.metres[seen]
    if len(px) < MIN_IN_VIEW:
        return np.full((side, side), np.nan)
    small = max(side // COARSE, 1)
    at = (np.mgrid[0:small, 0:small][::-1].reshape(2, -1).T + 0.5) * side / small
    try:
        found = LinearNDInterpolator(px, inverse)(at)
    except QhullError:  # all in a line: no triangle to spread them over
        found = np.full(len(at), np.nan)
    beyond = np.isnan(found)
    found[beyond] = NearestNDInterpolator(px, inverse)(at[beyond])
    filled = cv2.resize(found.reshape(small, small), (side, side), interpolation=cv2.INTER_LINEAR)
    # the sky: above the highest details of each column (a few, not one stray)
    width = max(side // COLUMNS, 1)
    bins = np.minimum((px[:, 0] // width).astype(int), COLUMNS - 1)
    top = np.full(COLUMNS, np.nan)
    for b in range(COLUMNS):
        rows = px[bins == b, 1]
        if len(rows) >= 2:
            top[b] = np.percentile(rows, 5)
    known = ~np.isnan(top)
    if known.any():
        centres = (np.arange(COLUMNS) + 0.5) * width
        tops = np.interp(np.arange(side), centres[known], top[known])
        filled[np.arange(side)[:, None] < tops[None, :]] = 0.0
    with np.errstate(divide="ignore"):
        return 1 / filled
