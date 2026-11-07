"""How far today's details stand from a panorama: its views matched with those of neighbouring
panoramas (see candidates and matched), each pair set right from its own matches (set_right),
each detail found in both placed where the two rays meet (placed); the neighbour giving most
where a view looks is the one shown for second clicks (aimed)."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from flaneur import match
from flaneur.gazetteer import angle

NEAR_M = (2.0, 15.0)  # a neighbour this far: nearer tells little of depth, farther sees too
# little of the same
MOST_CANDIDATES = 12  # the neighbours matched with a panorama, at most
AIM_MARGIN_DEG = 10.0  # a detail counts toward a view this far either side of its width
OFF_RAY_DEG = 0.15  # a detail's point this near both its rays, in angle: a pixel or two
LEAST_MATCHES = 100  # a pair is set right from at least this many matches
WIDEST_DEG = 0.25  # and its matches, once set right, lie this near their epipolar planes
# (median; measured 2026-11-07: 0.08-0.16 for pairs that agree, 0.3 and over for those that
# do not, and a dozen degrees for matches agreeing on nothing)
WIDE_DEG = 4.0  # a matched detail's rays meet this wide at least: narrower, its depth smears
# along them (0.1 degree off, 2 degrees apart: 1.5 m at 30 m), and foliage makes walls pointing
# at the camera (measured 2026-11-07: a sixth fewer walls, agreeing with the panoramas as well;
# 6 degrees, a tenth fewer again, agreed less)
ROBUST_DEG = (1.0, 0.1)  # the scale wrong matches are discounted at, coarse then fine


@dataclass(frozen=True)
class Pair:
    """A neighbour set right against a panorama from their own matches: its rays turned
    (`turn`, 3 × 3: recorded · turn.T) and its camera moved (`at`) so that every match lies on
    its epipolar plane; the baseline keeps its recorded length (the map's scale)."""

    turn: np.ndarray
    at: np.ndarray


def candidates(
    home: match.Panorama, spheres: list[match.Panorama], fixed: set[str]
) -> list[match.Panorama]:
    """The panoramas NEAR_M from `home` to match with it, at most MOST_CANDIDATES: those COLMAP
    placed itself (`fixed`) first, then the nearest."""

    def away(p: match.Panorama) -> float:
        return float(np.hypot(p.x - home.x, p.y - home.y))

    near = [p for p in spheres if p.id != home.id and NEAR_M[0] <= away(p) <= NEAR_M[1]]
    return sorted(near, key=lambda p: (p.id not in fixed, away(p), p.id))[:MOST_CANDIDATES]


def aimed(found: dict[str, np.ndarray], heading: float, fov: float = match.VIEW_FOV) -> list[str]:
    """The neighbours, those whose details (their directions from the panorama, by neighbour)
    are most within a view facing `heading` first (`fov` wide, AIM_MARGIN_DEG more either
    side): the one to show for second clicks sees best what the view shows. (Measured
    2026-11-07 on 18 panoramas, each against a neighbour held out: walls fitted from the 8
    giving most there agreed with it as walls fitted from all of them did, 0.208 against 0.201,
    better in 4 cases and worse in 5 of 33; the walls are fitted from all, whatever the view.)"""

    def within(directions: np.ndarray) -> int:
        bearing = np.degrees(np.arctan2(directions[:, 0], directions[:, 1]))
        return int((np.abs(angle(heading, bearing)) <= fov / 2 + AIM_MARGIN_DEG).sum())

    counted = {n: within(d) for n, d in found.items()}
    return sorted(counted, key=lambda n: (-counted[n], n))


def set_right(a: np.ndarray, b: np.ndarray, rays_a: np.ndarray, rays_b: np.ndarray) -> Pair | None:
    """The neighbour at `b` set right against the panorama at `a` from rays of the same
    details seen from each (unit rows): its orientation (3 angles) and the baseline's
    direction (2) fitted so the pairs of rays lie in planes through both cameras, wrong
    matches discounted; None from too few matches, or matches that do not agree once fitted.
    Measured 2026-11-07: neighbours placed by COLMAP off by 0.05 to 6.8 degrees, mostly in
    tilt; the 0.15 degree test then kept 1 to 52 % of their matches, 35 to 80 % once set
    right."""
    if len(rays_a) < LEAST_MATCHES or np.linalg.norm(b - a) < NEAR_M[0]:  # too near: no depth
        return None
    t0 = (b - a) / np.linalg.norm(b - a)
    u1 = np.cross(t0, [0.0, 0.0, 1.0])
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(t0, u1)

    def unpacked(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = t0 + x[3] * u1 + x[4] * u2
        return Rotation.from_rotvec(x[:3]).as_matrix(), t / np.linalg.norm(t)

    def off(x: np.ndarray) -> np.ndarray:  # each ray's angle off the plane the other makes
        turn, t = unpacked(x)
        seen = rays_b @ turn.T
        na, nb = np.cross(t, seen), np.cross(t, rays_a)
        sines = np.concatenate(
            [
                (rays_a * na).sum(axis=1) / np.linalg.norm(na, axis=1),
                (seen * nb).sum(axis=1) / np.linalg.norm(nb, axis=1),
            ]
        )
        return np.degrees(np.arcsin(np.clip(sines, -1, 1)))

    x = np.zeros(5)
    for scale in ROBUST_DEG:
        x = least_squares(off, x, loss="cauchy", f_scale=scale).x
    if np.median(np.abs(off(x))) > WIDEST_DEG:
        return None
    turn, t = unpacked(x)
    return Pair(turn, a + np.linalg.norm(b - a) * t)


def placed(
    a: np.ndarray, pair: Pair, rays_a: np.ndarray, rays_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The details of a pair set right (rays as matched, the neighbour's as recorded): their
    directions from `a` and how far along, where the two rays meet WIDE_DEG apart at least
    (see _met)."""
    points, metres = _met(a, pair.at, rays_a, rays_b @ pair.turn.T, wide_deg=WIDE_DEG)
    return (points - a) / metres[:, None], metres


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
    wide_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Where pairs of rays (one from `a`, one from `b`; a row each) meet: by match.placed's
    rules, within `off_deg` of both (the panoramas level at one height, near-horizontal rays of
    a wrong match still pass within its MEET_M) and `wide_deg` apart at least; the points, and
    how far from `a`; those that do not meet left out."""
    wide = _off(rays_a, rays_b) >= wide_deg  # (seen alike either way: unit rays)
    rays_a, rays_b = rays_a[wide], rays_b[wide]
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


def matched(
    home: match.Panorama, other: match.Panorama, views: match.Views
) -> tuple[np.ndarray, np.ndarray]:
    """The details of each view of `home` found in the views of `other` facing the same way or
    a step either side (a near wall turns tens of degrees between two panoramas): their rays
    from each (unit rows, as the panoramas are recorded); none if a picture cannot be read."""
    headings = list(match.VIEW_HEADINGS)
    mine, theirs = views.of(home), views.of(other)
    if len(mine) != len(headings) or len(theirs) != len(headings):
        return np.empty((0, 3)), np.empty((0, 3))
    rays_a, rays_b = [], []
    for k, heading in enumerate(headings):
        for step in (-1, 0, 1):
            j = (k + step) % len(headings)
            ia, ib = views.matcher.match(mine[k], theirs[j])
            side_a, side_b = mine[k].size[1], theirs[j].size[1]
            rays_a.append(match.view_directions(mine[k].keypoints[ia], heading, side=side_a))
            rays_b.append(match.view_directions(theirs[j].keypoints[ib], headings[j], side=side_b))
    return np.vstack(rays_a), np.vstack(rays_b)
