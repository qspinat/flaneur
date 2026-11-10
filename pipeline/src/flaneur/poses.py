"""Today's panoramas set right on the map. COLMAP placed the panoramas against each other, and on
the map by their GPS (off by metres between buildings); those it did not place borrow their run's.
So a panorama's walls, right in its own picture, may stand off the buildings on the map: measured
2026-11-07, 4 m for one (bd6b343f), 14° for another, though placed by COLMAP (88e4e410).

A panorama's correction (a shift east and north, a turn about it) is what lays its building
details (see cli._Scenes) on the buildings, fitted twice: against BD TOPO's facades, and against
the outlines of IGN's LiDAR HD surface model (where it stands above the street). Neither survey is
right to better than about a metre here (BD TOPO generalises; LiDAR sees eaves, and takes tree
crowns for buildings: measured 0.9 m apart around one crossroads), so a correction is kept only
when the two agree, and when it is larger than what they disagree by.

Measured, not yet applied (2026-11-08): moving a panorama placed by COLMAP alone breaks the
distances to its neighbours that its pairs' depths come from (depth.set_right), and its walls got
worse (88e4e410: 2.4 m from the facades as placed, 2.8 m moved, 0.5 m had its scene been moved
whole). A correction is the map frame's, to be applied to what goes on the map."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import httpx
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from flaneur.sources import write_atomic, write_text

NEAR_M = 4.0  # details this close to a facade or an outline are the fit's (not interiors, trees)
LEAST_DETAILS = 300  # and at least this many of them
LOSS_M = 0.5  # beyond this off a facade, a detail weighs less and less (soft L1)
AGREE_M = 1.0  # the two surveys' corrections agree when they put the details this close, or
AGREE_SHARE = 0.5  # within this share of the smaller of the two (both far off: closer to each
# other than to the panorama as it stands; the surveys themselves 0.9 m apart round one crossroads)
AGREE_DEG = 1.0  # and turn it alike within this, or AGREE_SHARE of the lesser turn (beyond the
# details, a turn they disagree on puts walls and facades wrong)
LEAST_SHIFT_M = 1.5  # a correction moving them this far at least (or turning LEAST_TURN_DEG):
LEAST_TURN_DEG = 3.0  # beyond the surveys' own disagreement
ABOVE_STREET_M = 3.0  # LiDAR's surface this far above the street: a building (or a tree)
LIDAR = (  # IGN's LiDAR HD surface model (MNS), Lambert-93, as float32 heights
    "https://data.geopf.fr/wms-r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES="
    "&LAYERS=IGNF_LIDAR-HD_MNS_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93&CRS=EPSG:2154"
    "&BBOX={x0},{y0},{x1},{y1}&WIDTH={n}&HEIGHT={n}&FORMAT=image/x-bil;bits=32"
)
CELL_M = 0.5
STEP_M = 0.5  # a sight line sampled this often
FROM_M = 3.0  # from this far off the camera (an awning, a crown right over it: not in the way)
BEFORE_M = 1.5  # up to this short of the detail (its own eaves, balconies)
SIGHT_ROUNDS = 2  # sight lines judged again where the fit puts the details, this many times
HALF_M = 45.0  # the patch fetched around a panorama
STREET_RING_M = 8.0  # the street: LiDAR's lowest tenth within this of the panorama (wide: the
# panorama may be metres off, its ring partly under a roof)
NO_DATA = -1000.0  # a LiDAR height below this: no data there (the service's -99999)
FACADE_ABOVE_M = 2.0  # a panorama's details this far over the street at least (not the street's)
REACH_M = 30.0  # and this close to it
MOST_DETAILS = 20000  # at most this many fitted (drawn at random: a fit's cost grows with them)
PINNED = 0.1  # a shift's weaker direction this much of its stronger at least, else not pinned
# (measured: a lone straight facade 0.01-0.07; a crossroads 0.2-0.7)
# A fit starts where the details' whole shape best meets the walls, as the eye finds it (see
# _searched): every turn and shift within these, on a grid this fine, details this near a wall
# counted; kept only if no other answer (another peak of the counts, putting the details this
# far from where the best does, of the RIVALS highest) counts nearly as many.
SEARCH_TURN_DEG = 20.0
SEARCH_STEP_DEG = 0.5
SEARCH_SHIFT_M = 15.0  # LiDAR patches reach HALF_M: details (REACH_M) shifted this, still in
SEARCH_CELL_M = 0.25
ON_WALL_M = 0.75
RIVAL_M = 3.0
RIVALS = 50
MARGIN = 1.1  # the best counts this many times another answer at least (set on 99, 10/11)


class Correction(NamedTuple):
    """A panorama moved east and north (metres), and turned about itself (degrees,
    counter-clockwise on the plan)."""

    east: float
    north: float
    turn: float


@dataclass(frozen=True)
class Verdict:
    """What was found for a panorama: its correction (None: left as placed), each survey's fit
    (east, north, turn; None: too few details there), the details fitted, and where the panorama
    stood when judged (east, north, heading: judged again if placed otherwise since)."""

    correction: Correction | None
    bdtopo: Correction | None
    lidar: Correction | None
    details: int
    placed: tuple[float, float, float] | None = None


def moved(points: np.ndarray, origin: np.ndarray, c: Correction) -> np.ndarray:
    """Plan points (n × 2) as a panorama at `origin` would see them corrected by `c`."""
    t = np.radians(c.turn)
    q = np.asarray(points, float) - origin
    turned = np.column_stack(
        [np.cos(t) * q[:, 0] - np.sin(t) * q[:, 1], np.sin(t) * q[:, 0] + np.cos(t) * q[:, 1]]
    )
    return origin + turned + [c.east, c.north]


def segment_distance(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Each plan point's distance to the nearest of `segments` (m × 2 × 2; none: infinite)."""
    nearest = np.full(len(points), np.inf)
    for a, b in np.asarray(segments, float).reshape(-1, 2, 2):  # one at a time: no n × m arrays
        ab = b - a
        t = np.clip((points - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
        np.minimum(nearest, np.linalg.norm(points - (a + t[:, None] * ab), axis=1), out=nearest)
    return nearest


def outline_distance(
    heights: np.ndarray, x0: float, y1: float, street: float
) -> Callable[[np.ndarray], np.ndarray]:
    """A plan point's distance to the nearest outline of what stands ABOVE_STREET_M above the
    street in a surface model (`heights`, rows north to south from `y1`, columns east from `x0`,
    CELL_M square; NaN: no data)."""
    built = np.nan_to_num(heights, nan=street) > street + ABOVE_STREET_M
    outline = built & ~ndimage.binary_erosion(built)
    far = ndimage.distance_transform_edt(~outline) * CELL_M

    def distance(points: np.ndarray) -> np.ndarray:  # bilinear: smooth, for the fit to follow
        return ndimage.map_coordinates(far, _cells(points, x0, y1), order=1, mode="nearest")

    return distance


def _cells(points: np.ndarray, x0: float, y1: float) -> np.ndarray:
    """Plan points as (row, column) in a grid whose cells' centres start half a cell in."""
    return np.stack([(y1 - points[:, 1]) / CELL_M - 0.5, (points[:, 0] - x0) / CELL_M - 0.5])


def outline_heights(heights: np.ndarray, x0: float, y1: float, points: np.ndarray) -> np.ndarray:
    """A surface model's heights (see outline_distance for its layout) at plan points (NaN
    outside it)."""
    cells = _cells(points, x0, y1)
    return ndimage.map_coordinates(heights, cells, order=0, mode="constant", cval=np.nan)


def clear_of(
    points: np.ndarray, origin: np.ndarray, heights: np.ndarray, x0: float, y1: float, street: float
) -> np.ndarray:
    """Whether the surface model stands clear between the panorama at `origin` and each plan
    point (from FROM_M off the camera to BEFORE_M short of the point): else what it sees there is
    behind a tree crown or another raised thing, whose outline is not its facade's."""
    away = points - origin
    far = np.linalg.norm(away, axis=1)
    steps = np.arange(FROM_M, max(far.max(initial=0.0) - BEFORE_M, FROM_M), STEP_M)
    if not len(steps):
        return np.ones(len(points), bool)
    unit = away / np.maximum(far, 1e-9)[:, None]
    on = origin + unit[:, None, :] * steps[None, :, None]  # points × steps × 2
    found = outline_heights(heights, x0, y1, on.reshape(-1, 2)).reshape(len(points), len(steps))
    in_way = (found > street + ABOVE_STREET_M) & (steps[None, :] <= (far - BEFORE_M)[:, None])
    return ~in_way.any(axis=1)


def fitted(
    points: np.ndarray,
    origin: np.ndarray,
    distance: Callable[[np.ndarray], np.ndarray],
    start: Correction | None = None,
) -> Correction | None:
    """The correction laying a panorama's details (plan points) on what `distance` measures
    to, from where their whole shape best meets it (from `start` if given: a nudge from there);
    None with fewer than LEAST_DETAILS near it, or two places it meets alike."""
    found = _fit(points, origin, distance, start)
    return found[0] if found else None


def _fit(
    points: np.ndarray,
    origin: np.ndarray,
    distance: Callable[[np.ndarray], np.ndarray],
    start: Correction | None = None,
) -> tuple[Correction, np.ndarray | None] | None:
    """fitted, and the plan direction (unit) its details do not pin a shift along, if one: the
    weaker of the shift's two over the stronger below PINNED (one facade's way; measured, a lone
    straight facade 0.01-0.07, a crossroads 0.2-0.7). Along it the correction does not move."""
    searched = _searched(points, origin, distance) if start is None else None
    if start is None:
        if searched is None:
            return None
        start = searched[0]
    near = points[distance(moved(points, origin, start)) < NEAR_M]
    if len(near) < LEAST_DETAILS:
        return None
    found = least_squares(
        lambda v: distance(moved(near, origin, Correction(*v))),
        np.array(start, float),
        loss="soft_l1",
        f_scale=LOSS_M,
        diff_step=1e-3,
    )
    shift = found.jac[:, :2].T @ found.jac[:, :2]
    strength, directions = np.linalg.eigh(shift)
    free = directions[:, 0] if strength[0] < PINNED * strength[1] else None
    if searched is not None and not _alone(points, origin, *searched[1:], free):
        return None
    correction = Correction(*(float(v) for v in found.x))
    return (_pinned(correction, [free]) if free is not None else correction), free


def _searched(
    points: np.ndarray, origin: np.ndarray, distance: Callable[[np.ndarray], np.ndarray]
) -> tuple[Correction, np.ndarray, np.ndarray, tuple[int, int, int]] | None:
    """Where a panorama's details (plan points, those within REACH_M) best meet what `distance`
    measures to, as the eye finds it: for every turn (SEARCH_TURN_DEG either way) and shift
    (SEARCH_SHIFT_M), how many lie within ON_WALL_M of it, counted at once for all shifts by
    correlating them with the walls (FFT); of answers counting alike, the least move. The best
    as a correction, the turns, their counts (turn × north row × east column, the shift grid's
    middle unmoved) and the best's index; None with no wall about."""
    half = REACH_M + SEARCH_SHIFT_M
    n = int(round(2 * half / SEARCH_CELL_M))
    centres = (np.arange(n) + 0.5) * SEARCH_CELL_M - half
    east, north = np.meshgrid(centres, -centres)  # rows north to south
    grid = origin + np.column_stack([east.ravel(), north.ravel()])
    walls = (distance(grid) < ON_WALL_M).reshape(n, n)
    if not walls.any():
        return None
    near = points[np.linalg.norm(points - origin, axis=1) < REACH_M]
    size = (2 * n, 2 * n)
    on_walls = np.fft.rfft2(walls.astype(float), s=size)
    k = int(round(SEARCH_SHIFT_M / SEARCH_CELL_M))
    turns = np.arange(-SEARCH_TURN_DEG, SEARCH_TURN_DEG + 1e-9, SEARCH_STEP_DEG)
    counts = np.empty((len(turns), 2 * k + 1, 2 * k + 1))
    for t, turn in enumerate(turns):
        q = moved(near, origin, Correction(0.0, 0.0, float(turn))) - origin
        seen = np.zeros((n, n))
        rows = np.floor((half - q[:, 1]) / SEARCH_CELL_M).astype(int)
        np.add.at(seen, (rows, np.floor((q[:, 0] + half) / SEARCH_CELL_M).astype(int)), 1)
        met = np.fft.irfft2(np.conj(np.fft.rfft2(seen, s=size)) * on_walls, s=size)
        counts[t] = np.roll(met, (k, k), axis=(0, 1))[: 2 * k + 1, : 2 * k + 1]
    offsets = (np.arange(2 * k + 1) - k) * SEARCH_CELL_M
    moves = np.abs(turns)[:, None, None] + np.hypot(offsets[:, None], offsets[None, :])[None]
    best = np.unravel_index(np.argmax(np.round(counts) - 1e-3 * moves), counts.shape)
    t, row, column = (int(i) for i in best)
    return (
        Correction(float(offsets[column]), float(-offsets[row]), float(turns[t])),
        turns,
        counts,
        (t, row, column),
    )


def _alone(
    points: np.ndarray,
    origin: np.ndarray,
    turns: np.ndarray,
    counts: np.ndarray,
    best: tuple[int, int, int],
    free: np.ndarray | None,
) -> bool:
    """Whether the best answer of _searched counts MARGIN times as many details on the walls as
    any other: another peak of the counts that puts the details RIVAL_M from where the best does
    (in median; along a `free` direction, where the walls cannot tell, not counted)."""
    k = (counts.shape[1] - 1) // 2
    steps = (np.arange(2 * k + 1) - k) * SEARCH_CELL_M
    peaks = (counts == ndimage.maximum_filter(counts, size=(3, 5, 5))) & (counts > 0)
    ranked = sorted(zip(*np.nonzero(peaks), strict=True), key=lambda i: -counts[i])[:RIVALS]
    sample = points[:: max(1, len(points) // 2000)]

    def where(i: tuple) -> np.ndarray:
        return moved(sample, origin, Correction(steps[i[2]], -steps[i[1]], turns[i[0]]))

    there = where(best)
    for i in ranked:
        apart = where(i) - there
        if free is not None:
            apart = apart - np.outer(apart @ free, free)
        if np.median(np.linalg.norm(apart, axis=1)) >= RIVAL_M:
            return counts[best] >= MARGIN * counts[i]
    return True


def agreed(
    bdtopo: Correction | None,
    lidar: Correction | None,
    points: np.ndarray,
    origin: np.ndarray,
    free: list[np.ndarray] = (),
) -> Correction | None:
    """Both surveys' corrections, averaged, when they put a panorama's details (plan points)
    within AGREE_M of each other, or AGREE_SHARE of how far the lesser moves them (in median:
    two corrections may differ in turn and shift and move the details alike), turn them alike
    (AGREE_DEG), and when it moves them LEAST_SHIFT_M or turns LEAST_TURN_DEG at least; else
    None. Along a `free` direction (unit, on the plan: one survey's details do not pin a shift
    along it), neither is compared nor followed: what it says there is noise."""
    if bdtopo is None or lidar is None:
        return None
    bdtopo, lidar = (_pinned(c, free) for c in (bdtopo, lidar))
    one, other = moved(points, origin, bdtopo), moved(points, origin, lidar)
    lesser = min(np.median(np.linalg.norm(q - points, axis=1)) for q in (one, other))
    if np.median(np.linalg.norm(one - other, axis=1)) > max(AGREE_M, AGREE_SHARE * lesser):
        return None
    if abs(bdtopo.turn - lidar.turn) > max(
        AGREE_DEG, AGREE_SHARE * min(abs(bdtopo.turn), abs(lidar.turn))
    ):
        return None
    mean = Correction(*((a + b) / 2 for a, b in zip(bdtopo, lidar, strict=True)))
    shift = np.median(np.linalg.norm(moved(points, origin, mean) - points, axis=1))
    if shift < LEAST_SHIFT_M and abs(mean.turn) < LEAST_TURN_DEG:
        return None
    return mean


def _pinned(c: Correction, free: list[np.ndarray]) -> Correction:
    """A correction less its shift along the `free` directions (two apart: no shift at all)."""
    shift = np.array([c.east, c.north])
    basis: list[np.ndarray] = []
    for f in free:  # an orthonormal basis of what is free
        rest = f - sum((f @ b) * b for b in basis)
        if np.linalg.norm(rest) > 0.1:
            basis.append(rest / np.linalg.norm(rest))
    for b in basis:
        shift = shift - (shift @ b) * b
    return Correction(float(shift[0]), float(shift[1]), c.turn)


def judged(
    points: np.ndarray,
    origin: np.ndarray,
    bdtopo: Callable[[np.ndarray], np.ndarray],
    lidar: Callable[[np.ndarray], np.ndarray],
    sees: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> Verdict:
    """A panorama's building details (plan points) fitted to each survey (their distances), and
    the correction they agree on, if any. To LiDAR, only the details it `sees` (plan points and
    the panorama: which have a clear sight line, see clear_of), judged where the correction puts
    them: a panorama placed off sees its details inside the buildings, behind their outlines."""
    to_lidar = _fit(points, origin, lidar)
    for _ in range(SIGHT_ROUNDS if sees else 0):
        if to_lidar is None:
            break
        c = to_lidar[0]
        shown = sees(moved(points, origin, c), origin + [c.east, c.north])
        to_lidar = _fit(points[shown], origin, lidar, start=c)
    to_bdtopo = _fit(points, origin, bdtopo)
    fits = [f[0] if f else None for f in (to_bdtopo, to_lidar)]
    free = [f[1] for f in (to_bdtopo, to_lidar) if f and f[1] is not None]
    return Verdict(agreed(*fits, points, origin, free), *fits, len(points))


def judge(
    details: np.ndarray,
    origin: np.ndarray,
    ground: float,
    facades: list,
    folder: Path,
    client: httpx.Client | None,
) -> Verdict:
    """A panorama at `origin` (east, north, height) judged from its building details (n × 3,
    the street at `ground`): against BD TOPO's `facades` (segments on the plan) and LiDAR's
    surface around it (fetched with `client` once, kept in `folder`)."""
    o = np.asarray(origin, float)[:2]
    keep = (details[:, 2] > ground + FACADE_ABOVE_M) & (np.hypot(*(details[:, :2] - o).T) < REACH_M)
    plan = details[keep, :2]
    if len(plan) > MOST_DETAILS:
        plan = plan[np.random.default_rng(0).choice(len(plan), MOST_DETAILS, replace=False)]
    segments = np.asarray(facades, float).reshape(-1, 2, 2)
    heights, x0, y1 = surface(o[0], o[1], folder, client)
    grid = np.linspace(-STREET_RING_M, STREET_RING_M, 13)
    ring = o + np.stack(np.meshgrid(grid, grid), -1).reshape(-1, 2)
    street = float(np.nanpercentile(outline_heights(heights, x0, y1, ring), 10))
    return judged(
        plan,
        o,
        lambda q: segment_distance(q, segments),
        outline_distance(heights, x0, y1, street),
        lambda q, at: clear_of(q, at, heights, x0, y1, street),
    )


def surface(
    x: float, y: float, folder: Path, client: httpx.Client | None
) -> tuple[np.ndarray, float, float]:
    """IGN's LiDAR HD surface model around (x, y): heights, west and north edges (fetched with
    `client` once, kept in `folder`)."""
    x0, y1 = round(x) - HALF_M, round(y) + HALF_M
    n = int(2 * HALF_M / CELL_M)
    path = folder / f"mns_{round(x)}_{round(y)}.npy"
    if not path.exists():
        url = LIDAR.format(x0=x0, y0=y1 - 2 * HALF_M, x1=x0 + 2 * HALF_M, y1=y1, n=n)
        answer = client.get(url, timeout=60)
        answer.raise_for_status()
        kind = answer.headers.get("content-type", "")
        if not kind.startswith("image/") or len(answer.content) != 4 * n * n:
            raise ValueError(
                f"LiDAR HD : réponse inattendue ({kind}, {len(answer.content)} octets)"
            )
        heights = np.frombuffer(answer.content, "<f4").reshape(n, n)
        write_atomic(path, lambda out: np.save(out, heights))
    heights = np.load(path).astype(float)
    heights[heights < NO_DATA] = np.nan
    return heights, x0, y1


def save(path: Path, verdicts: dict[str, Verdict]) -> None:
    write_text(path, json.dumps({i: asdict(v) for i, v in sorted(verdicts.items())}, indent=1))


def load(path: Path) -> dict[str, Verdict]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {
        i: Verdict(
            *(_correction(r[k]) for k in ("correction", "bdtopo", "lidar")),
            r["details"],
            tuple(r["placed"]) if r.get("placed") else None,
        )
        for i, r in rows.items()
    }


def _correction(row: list | None) -> Correction | None:
    return Correction(*row) if row else None


def corrections(verdicts: dict[str, Verdict]) -> dict[str, Correction]:
    """The panoramas to move, and how."""
    return {i: v.correction for i, v in verdicts.items() if v.correction}
