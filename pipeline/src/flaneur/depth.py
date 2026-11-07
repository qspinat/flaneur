"""How far today's details stand from a panorama: its views matched with those of neighbouring
panoramas (see neighbours), each detail found in both placed where the two rays meet (the
panoramas where they were taken, at their heights: see cli._placing_world). They agree with the
panoramas by construction; measured on 2026-11-06: hundreds of details a view, the rays passing
within a few centimetres."""

import numpy as np

from flaneur import match

NEAR_M = (2.0, 15.0)  # a neighbour this far: nearer tells little of depth, farther sees too
# little of the same
SPREAD_DEG = 30.0  # a neighbour is judged on rays at the view's heading and this either side
MOST_NEIGHBOURS = 4
OFF_RAY_DEG = 0.15  # a detail's point this near both its rays, in angle: a pixel or two


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
    (passing within MEET_M, MEET_DEG or more apart, in front of both); None when they do not.
    Clicked by hand, a pixel or two off: not held to OFF_RAY_DEG as matched details are."""
    rays = [np.asarray(r, float)[None] / np.linalg.norm(r) for r in (ray_a, ray_b)]
    metres = _met(a, b, *rays, off_deg=180.0)[1]
    return float(metres[0]) if len(metres) else None


def _met(
    a: np.ndarray,
    b: np.ndarray,
    rays_a: np.ndarray,
    rays_b: np.ndarray,
    off_deg: float = OFF_RAY_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Where pairs of rays (one from `a`, one from `b`; a row each) meet: by match.placed's
    rules, and within `off_deg` of both (the panoramas level at one height, near-horizontal
    rays of a wrong match still pass within its MEET_M); the points, and how far from `a`;
    those that do not meet left out."""
    n = len(rays_a)
    points, pairs, _, _ = match.placed(
        np.tile(np.arange(n), 2),
        np.repeat(["a", "b"], n),
        np.vstack([np.tile(a, (n, 1)), np.tile(b, (n, 1))]),
        np.vstack([rays_a, rays_b]),
        np.full((2 * n, 3), np.nan),  # never cast onto a wall
    )
    off = np.maximum(_off(points - a, rays_a[pairs]), _off(points - b, rays_b[pairs]))
    points = points[off <= off_deg]
    return points, np.linalg.norm(points - a, axis=1)


def _off(seen: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """The angle (degrees) between each direction seen and its ray (unit)."""
    cos = (seen * rays).sum(axis=1) / np.linalg.norm(seen, axis=1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


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
