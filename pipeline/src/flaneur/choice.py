"""The views of today an old photo is compared with: those covering where its camera may stand,
not every view of the panoramas nearest its caption's place.

The camera stands somewhere around the place its caption gives (measured between place and
camera found: median 13 m, 90 % within 46 m), looking somewhere. A view covers a camera when it
was taken near it (NEAR_M), looking its way (TURN_DEG), and sees what it looks at (the first
wall on its axis) with no wall between. Views are chosen greedily, those covering the most likely
cameras first (Max-k-Cover: the greedy choice is within 63 % of the best), from panoramas COLMAP
placed; one it did not place only fills a hole no placed one covers (its place is borrowed, or
its phone's). Measured offline on 120 views with a known answer, the caption's place drawn off
the truth as measured: 48 views chosen this way cover 99 cameras, the 192 views of the 16
nearest panoramas 90."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from flaneur import match

NEAR_M = 10.0  # a view covers a camera taken this near it
TURN_DEG = 30.0  # looking this near its way
CLEAR_M = 1.0  # and seeing what it looks at: no wall this much before it
SEEN_M = 150.0  # the walls a camera can look at
GRID_M = 5.0  # where a camera may stand, every this
REACH_M = 60.0  # around its place
POOL_M = REACH_M + NEAR_M  # the panoramas a view may be chosen from, around the place
MEDIAN_M, Q90_M = 13.0, 46.0  # place to camera, measured on the photos placed
SIGMA = np.log(Q90_M / MEDIAN_M) / 1.2816  # of the log-normal law with that median and q90
BUDGET = 48  # views compared, at most
EYE_M = 1.6  # a camera's height over the street
HEADINGS = np.asarray(match.VIEW_HEADINGS, float)


@dataclass(frozen=True)
class Chooser:
    """Chooses views (see views) among panoramas, those COLMAP placed (`fixed`) first, with
    today's walls (`blocks`), the street's height (`ground`) and where a camera can stand
    (`free`); each place's choice kept for the run (every matcher compares the same views)."""

    fixed: frozenset[str]
    blocks: object
    ground: Callable[[np.ndarray, np.ndarray], np.ndarray]
    free: Callable[[np.ndarray, np.ndarray], np.ndarray]
    made: dict = field(default_factory=dict, compare=False, repr=False)

    def views(
        self, place: tuple[float, float], pool: Sequence[match.Panorama], budget: int = BUDGET
    ) -> dict[str, list[float]]:
        """The views to compare a photo placed at `place` with, among the panoramas `pool`
        (those within POOL_M of it): each chosen panorama's id and its headings, in the order
        first chosen."""
        key = (round(place[0], 1), round(place[1], 1), tuple(p.id for p in pool), budget)
        if key not in self.made:
            self.made[key] = self._views(place, pool, budget)
        return self.made[key]

    def covering(
        self, views: list[tuple[match.Panorama, float]], cameras: np.ndarray, aims: np.ndarray
    ) -> np.ndarray:
        """Whether each view (a panorama, a heading) covers each camera (x, y, heading) looking
        at its aim (see aims): views × cameras."""
        out = np.zeros((len(views), len(cameras)), bool)
        looks = np.isfinite(aims[:, 0])
        walls = {}  # each panorama's walls around it, cut once
        for v, (p, heading) in enumerate(views):
            near = np.hypot(cameras[:, 0] - p.x, cameras[:, 1] - p.y) <= NEAR_M
            turn = np.abs((cameras[:, 2] - heading + 180) % 360 - 180) <= TURN_DEG
            ok = np.flatnonzero(near & turn & looks)
            if len(ok):
                if p.id not in walls:
                    walls[p.id] = self.blocks.near(p.x, p.y, SEEN_M + REACH_M)
                out[v, ok] = _sees(walls[p.id], p, aims[ok])
        return out

    def aims(self, cameras: np.ndarray) -> np.ndarray:
        """What each camera (x, y, heading) looks at: the first wall on its level axis (NaN:
        none); the walls read once for each spot the cameras stand at."""
        out = np.full((len(cameras), 3), np.nan)
        if not len(cameras):
            return out
        spots, at = np.unique(cameras[:, :2], axis=0, return_inverse=True)
        z = np.asarray(self.ground(spots[:, 0], spots[:, 1]), float) + EYE_M
        centre = spots.mean(axis=0)
        spread = float(np.hypot(*(spots - centre).T).max())
        around = self.blocks.near(centre[0], centre[1], SEEN_M + spread)  # every spot's walls
        h = np.radians(cameras[:, 2])
        ahead = np.column_stack([np.sin(h), np.cos(h), np.zeros_like(h)])
        for k, (x, y) in enumerate(spots):
            rows = np.flatnonzero(at.ravel() == k)
            hit, edge = around.near(x, y, SEEN_M).hit(np.array([x, y, z[k]]), ahead[rows])
            out[rows] = np.where((edge >= 0)[:, None], hit, np.nan)
        return out

    def _views(
        self, place: tuple[float, float], pool: Sequence[match.Panorama], budget: int
    ) -> dict[str, list[float]]:
        cameras, weight = _cameras(place, self.free)
        aims = self.aims(cameras)
        left, chosen, count = weight.astype(np.float32), {}, 0
        for sure in (True, False):  # those COLMAP placed; then, for the holes, the others
            views = [(p, h) for p in pool if (p.id in self.fixed) == sure for h in HEADINGS]
            if not views or count >= budget or not left.any():
                continue
            covers = self.covering(views, cameras, aims).astype(np.float32)
            while count < budget:
                gain = covers @ left
                best = int(np.argmax(gain))
                if gain[best] <= 0:
                    break
                p, heading = views[best]
                chosen.setdefault(p.id, []).append(float(heading))
                count += 1
                left[covers[best] > 0] = 0
        return chosen


def _cameras(
    place: tuple[float, float], free: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Where a camera may stand around `place` (x, y, heading), and how likely each is (the
    log-normal law of the distance, spread over its ring)."""
    span = np.arange(-REACH_M, REACH_M + GRID_M / 2, GRID_M)
    gx, gy = (g.ravel() for g in np.meshgrid(span + place[0], span + place[1]))
    away = np.hypot(gx - place[0], gy - place[1])
    keep = (away <= REACH_M) & np.asarray(free(gx, gy), bool)
    gx, gy, away = gx[keep], gy[keep], np.maximum(away[keep], 0.5)  # as measured
    n = len(HEADINGS)
    cameras = np.column_stack([np.repeat(gx, n), np.repeat(gy, n), np.tile(HEADINGS, len(gx))])
    r = np.repeat(away, n)
    return cameras, np.exp(-(np.log(r / MEDIAN_M) ** 2) / (2 * SIGMA**2)) / r**2


def _sees(walls, p: match.Panorama, aims: np.ndarray) -> np.ndarray:
    """Whether a panorama sees each point (level, no wall of `walls` before it)."""
    origin = np.array([p.x, p.y, p.z])
    d = aims - origin
    far = np.hypot(d[:, 0], d[:, 1])
    d[:, 2] = 0
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    hit, edge = walls.hit(origin, d)
    first = np.where(edge >= 0, np.hypot(hit[:, 0] - p.x, hit[:, 1] - p.y), np.inf)
    return first >= far - CLEAR_M
