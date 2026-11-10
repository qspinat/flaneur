"""Today's panoramas set right on the map. COLMAP placed the panoramas against each other, and on
the map by their GPS (off by metres between buildings); those it did not place borrow their run's.
So a panorama's walls, right in its own picture, may stand off the buildings on the map: measured
2026-11-07, 4 m for one (bd6b343f), 14° for another, though placed by COLMAP (88e4e410).

A panorama's correction (a shift east and north, a turn about it) is what lays its building
details (see cli._Scenes) on the buildings: found as the eye finds it, the details' whole shape
against BD TOPO's facades (every turn and shift, then refined), and borne out by the outlines of
IGN's LiDAR HD surface model (where it stands above the street): the details within FAR_M of
them, and no further from them than as the panorama stands. Neither survey is right to better
than about a metre here (BD TOPO generalises; LiDAR sees eaves, and takes tree crowns for
buildings: measured 0.9 m apart around one crossroads), so only a correction larger than that
is kept. LiDAR alone does not find (measured 2026-11-10 on 99 panoramas: its outlines, trees
and courtyards among them, meet the details nearly as well in several places, nearly always).

Measured, not yet applied (2026-11-08): moving a panorama placed by COLMAP alone breaks the
distances to its neighbours that its pairs' depths come from (depth.set_right), and its walls got
worse (88e4e410: 2.4 m from the facades as placed, 2.8 m moved, 0.5 m had its scene been moved
whole). A correction is the map frame's, to be applied to what goes on the map."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import httpx
import numpy as np
from scipy import fft, ndimage
from scipy.optimize import least_squares

from flaneur.sources import write_atomic, write_text

NEAR_M = 4.0  # details this close to a facade or an outline are the fit's (not interiors, trees)
LEAST_DETAILS = 300  # and at least this many of them
LOSS_M = 0.5  # beyond this off a facade, a detail weighs less and less (soft L1)
LEAST_SHIFT_M = 1.5  # a correction moving them this far at least (or turning LEAST_TURN_DEG):
LEAST_TURN_DEG = 3.0  # beyond the surveys' own disagreement
FAR_M = 4.0  # a detail's distance to LiDAR, weighed up to this at most (one far off: as far)
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
# far from where the best does) counts nearly as many, and if it lies within what is tried.
SEARCH_TURN_DEG = 20.0
SEARCH_STEP_DEG = 0.5
SEARCH_SHIFT_M = 14.0  # each way: details (REACH_M) shifted this stay within LiDAR's patch
# (HALF_M about a panorama's place rounded to the metre)
SEARCH_CELL_M = 0.25
ON_WALL_M = 0.75
RIVAL_M = 3.0
TURNS = np.arange(-SEARCH_TURN_DEG, SEARCH_TURN_DEG + 1e-9, SEARCH_STEP_DEG)
_K = round(SEARCH_SHIFT_M / SEARCH_CELL_M)
OFFSETS = (np.arange(2 * _K + 1) - _K) * SEARCH_CELL_M  # the shifts tried, either way
MARGIN = 1.02  # the best counts this many times another answer at least: a tie, refused
# (measured on 99 panoramas, 10/11: above it, LiDAR tells right from wrong better than a margin)
# How verdicts are reached: "1", two surveys' fits agreeing (until 2026-11-10); "2", BD TOPO's
# whole shape borne out by LiDAR, with what decides it. A verdict by another is judged again.
# What of the corrections what reads panoramas' poses takes (cli._placed): "all", "turns" (their
# turn only: no shift changes the distances to neighbours its depths come from) or "none".
Applied = Literal["all", "turns", "none"]
APPLIED: Applied = "none"
RULE = "2:" + ",".join(
    str(v)
    for v in (MARGIN, FAR_M, ON_WALL_M, RIVAL_M, NEAR_M, LEAST_SHIFT_M, LEAST_TURN_DEG)
    + (SEARCH_TURN_DEG, SEARCH_STEP_DEG, SEARCH_SHIFT_M, SEARCH_CELL_M, REACH_M, LEAST_DETAILS)
)


class Correction(NamedTuple):
    """A panorama moved east and north (metres), and turned about itself (degrees,
    counter-clockwise on the plan)."""

    east: float
    north: float
    turn: float


@dataclass(frozen=True)
class Verdict:
    """What was found for a panorama: its correction (None: left as placed), BD TOPO's fit
    (east, north, turn; None: too few details there, or two places alike), the details fitted,
    and where the panorama stood when judged (east, north, heading: judged again if placed
    otherwise since)."""

    correction: Correction | None
    bdtopo: Correction | None
    details: int
    placed: tuple[float, float, float] | None = None
    rule: str = RULE


def moved(points: np.ndarray, origin: np.ndarray, c: Correction) -> np.ndarray:
    """Plan points (n × 2) as a panorama at `origin` would see them corrected by `c`."""
    t = np.radians(c.turn)
    q = np.asarray(points, float) - origin
    turned = np.column_stack(
        [np.cos(t) * q[:, 0] - np.sin(t) * q[:, 1], np.sin(t) * q[:, 0] + np.cos(t) * q[:, 1]]
    )
    return origin + turned + [c.east, c.north]


def stands(then: list | tuple | None, now: tuple[float, float, float] | None) -> bool:
    """Whether what was made from a panorama standing `then` (see at) holds with it standing
    `now`: both known, and alike (made before poses were kept, or its panorama no longer at
    hand: made again)."""
    return then is not None and now is not None and tuple(then) == tuple(now)


def carried(
    points: np.ndarray, then: tuple[float, float, float], now: tuple[float, float, float]
) -> np.ndarray:
    """Plan points (n × 2) seen from a panorama that stood `then` (east, north, compass heading)
    and stands `now`: moved and turned with it, about it."""
    (x0, y0, a0), (x1, y1, a1) = then, now
    return moved(points, np.array([x0, y0]), Correction(x1 - x0, y1 - y0, a0 - a1))


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
    outline = built & ~ndimage.binary_erosion(built, border_value=1)  # the patch's edge: none
    far = ndimage.distance_transform_edt(~outline) * CELL_M

    def distance(points: np.ndarray) -> np.ndarray:  # bilinear: smooth, for the fit to follow
        return ndimage.map_coordinates(far, _cells(points, x0, y1), order=1, mode="nearest")

    return distance


def _cells(points: np.ndarray, x0: float, y1: float, cell: float = CELL_M) -> np.ndarray:
    """Plan points as (row, column) in a grid (rows north to south from `y1`, columns east from
    `x0`, `cell` square) whose cells' centres start half a cell in."""
    return np.stack([(y1 - points[:, 1]) / cell - 0.5, (points[:, 0] - x0) / cell - 0.5])


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
    points: np.ndarray, origin: np.ndarray, distance: Callable[[np.ndarray], np.ndarray]
) -> Correction | None:
    """The correction laying a panorama's details (plan points) on what `distance` measures to,
    from where their whole shape best meets it (see _searched), refined there; along a direction
    they do not pin a shift along (see _refined), it does not move. None with fewer than
    LEAST_DETAILS near it, at the edge of what is searched, or when another answer meets it
    nearly as well (see _margin)."""
    searched = _searched(points, origin, distance)
    if searched is None:
        return None
    counts, best = searched
    if any(i in (0, n - 1) for i, n in zip(best, counts.shape, strict=True)):
        return None  # at the edge of what is tried: the answer may lie beyond
    found = _refined(points, origin, distance, _at(best))
    if (
        found is None
        or max(abs(found[0].east), abs(found[0].north)) > SEARCH_SHIFT_M
        or abs(found[0].turn) > SEARCH_TURN_DEG  # refined beyond what is tried: unchecked
        or _margin(points, origin, counts, best, found[1]) < MARGIN
    ):
        return None
    return found[0]


def _refined(
    points: np.ndarray,
    origin: np.ndarray,
    distance: Callable[[np.ndarray], np.ndarray],
    start: Correction,
) -> tuple[Correction, np.ndarray | None] | None:
    """The correction from `start` that lays the details NEAR_M of what `distance` measures to
    on it (soft L1), and the plan direction (unit) they do not pin a shift along, if one: the
    weaker of the shift's two over the stronger below PINNED (one facade's way; measured, a lone
    straight facade 0.01-0.07, a crossroads 0.2-0.7), along which it is not moved."""
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
    east, north, turn = (float(v) for v in found.x)
    if free is not None:
        east, north = np.array([east, north]) - (np.array([east, north]) @ free) * free
    return Correction(float(east), float(north), turn), free


def _searched(
    points: np.ndarray, origin: np.ndarray, distance: Callable[[np.ndarray], np.ndarray]
) -> tuple[np.ndarray, tuple[int, int, int]] | None:
    """Where a panorama's details (plan points, those within REACH_M) best meet what `distance`
    measures to, as the eye finds it: for each of TURNS and OFFSETS (east and north), how many lie
    within ON_WALL_M of it, counted at once for all shifts by correlating them with the walls
    (FFT); of answers counting alike, the least move. Their counts (turn × north row, from the
    north, × east column) and the best's index (see _at); None with no wall about."""
    near = points[np.linalg.norm(points - origin, axis=1) < REACH_M]
    if len(near) < LEAST_DETAILS:
        return None
    half = REACH_M + SEARCH_SHIFT_M
    n = round(2 * half / SEARCH_CELL_M)
    x0, y1 = origin[0] - half, origin[1] + half
    centres = (np.arange(n) + 0.5) * SEARCH_CELL_M
    east, south = np.meshgrid(x0 + centres, y1 - centres)
    walls = (distance(np.column_stack([east.ravel(), south.ravel()])) < ON_WALL_M).reshape(n, n)
    if not walls.any():
        return None
    size = fft.next_fast_len(n + _K)  # no shift tried wraps round onto another
    on_walls = fft.rfft2(walls.astype(np.float32), s=(size, size), workers=-1)
    kept = np.r_[size - _K : size, 0 : _K + 1]  # the shifts tried, from -_K
    counts = np.empty((len(TURNS), len(OFFSETS), len(OFFSETS)))
    for t, turn in enumerate(TURNS):
        row, column = np.floor(
            _cells(moved(near, origin, Correction(0.0, 0.0, float(turn))), x0, y1, SEARCH_CELL_M)
            + 0.5
        ).astype(int)
        seen = np.bincount(row * n + column, minlength=n * n).reshape(n, n).astype(np.float32)
        met = fft.irfft2(
            np.conj(fft.rfft2(seen, s=(size, size), workers=-1)) * on_walls,
            s=(size, size),
            workers=-1,
        )
        counts[t] = np.rint(met[np.ix_(kept, kept)])  # whole details: FFT noise rounded off
    moves = np.abs(TURNS)[:, None, None] + np.hypot(OFFSETS[:, None], OFFSETS[None, :])[None]
    best = np.unravel_index(np.argmax(counts - 1e-3 * moves), counts.shape)

    return counts, tuple(int(i) for i in best)


def _at(i: tuple[int, int, int]) -> Correction:
    """The correction at an index of _searched's counts (turn, north row from the north, east
    column)."""
    return Correction(float(OFFSETS[i[2]]), float(-OFFSETS[i[1]]), float(TURNS[i[0]]))


def _apart(a: np.ndarray, b: np.ndarray, free: np.ndarray | None = None) -> float:
    """How far apart two placings of the same plan points are, in median (along a `free`
    direction, unit: not counted)."""
    d = a - b
    if free is not None:
        d = d - np.outer(d @ free, free)
    return float(np.median(np.linalg.norm(d, axis=1)))


def _margin(
    points: np.ndarray,
    origin: np.ndarray,
    counts: np.ndarray,
    best: tuple[int, int, int],
    free: np.ndarray | None,
) -> float:
    """How many times as many details on the walls the best answer of _searched counts as any
    other (infinite: none): the highest other peak of the counts that puts the details RIVAL_M
    from where the best does (see _apart; along a `free` direction, where the walls cannot tell,
    not counted)."""
    peaks = np.flatnonzero(
        (counts == ndimage.maximum_filter(counts, size=(3, 5, 5))) & (counts > 0)
    )
    t, row, column = np.unravel_index(peaks, counts.shape)
    # no detail (within REACH_M) moves further than this between two answers: those nearer than
    # RIVAL_M this way are the best's own plateau, not other answers
    most = np.hypot(OFFSETS[row] - OFFSETS[best[1]], OFFSETS[column] - OFFSETS[best[2]])
    peaks = peaks[most + np.radians(np.abs(TURNS[t] - TURNS[best[0]])) * REACH_M >= RIVAL_M]
    sample = points[:: max(1, len(points) // 2000)]
    there = moved(sample, origin, _at(best))
    for flat in peaks[np.argsort(-counts.ravel()[peaks], kind="stable")]:
        i = tuple(int(v) for v in np.unravel_index(flat, counts.shape))
        if _apart(moved(sample, origin, _at(i)), there, free) >= RIVAL_M:
            return float(counts[best] / counts[i])
    return float("inf")


def judged(
    points: np.ndarray,
    origin: np.ndarray,
    bdtopo: Callable[[np.ndarray], np.ndarray],
    lidar: Callable[[np.ndarray], np.ndarray],
    sees: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> Verdict:
    """A panorama's building details (plan points) fitted to BD TOPO (its distance, see fitted);
    the correction kept if larger than the surveys' own disagreement and borne out by LiDAR
    (see _borne_out)."""
    found = fitted(points, origin, bdtopo)
    kept = (
        found
        if found
        and (
            _apart(moved(points, origin, found), points) >= LEAST_SHIFT_M
            or abs(found.turn) >= LEAST_TURN_DEG
        )
        and _borne_out(found, points, origin, lidar, sees)
        else None
    )
    return Verdict(kept, found, len(points))


def _borne_out(
    c: Correction,
    points: np.ndarray,
    origin: np.ndarray,
    lidar: Callable[[np.ndarray], np.ndarray],
    sees: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> bool:
    """Whether LiDAR bears a correction out: the details it `sees` (plan points and the
    panorama: which have a clear sight line, see clear_of) from where the correction puts the
    panorama (LEAST_DETAILS at least) lie, in median, nearer than FAR_M to its outlines, and no
    further from them than those it sees from where the panorama stands (if as many). A
    panorama placed off sees its details inside the buildings, behind their outlines."""

    def off(c: Correction) -> float | None:
        q, at = moved(points, origin, c), origin + [c.east, c.north]
        shown = sees(q, at) if sees else np.ones(len(q), bool)
        if shown.sum() < LEAST_DETAILS:
            return None
        return float(np.median(np.minimum(lidar(q[shown]), FAR_M)))

    there = off(c)
    here = off(Correction(0.0, 0.0, 0.0))
    return there is not None and there < FAR_M and (here is None or there <= here)


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
            *(_correction(r[k]) for k in ("correction", "bdtopo")),
            r["details"],
            tuple(r["placed"]) if r.get("placed") else None,
            str(r.get("rule", 1)),
        )
        for i, r in rows.items()
    }


def _correction(row: list | None) -> Correction | None:
    return Correction(*row) if row else None


def still(verdict: Verdict, placed: tuple[float, float, float]) -> bool:
    """Whether a verdict stands for a panorama `placed` so (east, north, heading): judged there,
    by the RULE of now."""
    return stands(verdict.placed, placed) and verdict.rule == RULE


def holding(
    verdicts: dict[str, Verdict], placed: dict[str, tuple[float, float, float]]
) -> dict[str, Verdict]:
    """The verdicts that stand (see still) for the panoramas `placed` so, by id."""
    return {i: v for i, v in verdicts.items() if i in placed and still(v, placed[i])}


def at(x: float, y: float, heading: float) -> tuple[float, float, float]:
    """Where a panorama stands, as a verdict keeps it (Verdict.placed): to the hundredth."""
    return round(x, 2), round(y, 2), round(heading, 2)


def to_apply(
    verdicts: dict[str, Verdict], placed: dict[str, tuple[float, float, float]], applied: Applied
) -> dict[str, tuple[float, float, float]]:
    """The fixes (east, north, change of heading: compass degrees, clockwise) that set panoramas
    right, as register.applied takes them: the corrections by the RULE of now of those still
    `placed` (see at) where they were judged; `applied` (see APPLIED) "turns": the turn only,
    "none": none. A correction turns counter-clockwise on the plan: the heading by its
    opposite."""
    if applied not in ("all", "turns", "none"):
        raise ValueError(f"applied: all, turns or none, not {applied!r}")
    if applied == "none":
        return {}
    shifts = applied == "all"
    return {
        i: (c.east if shifts else 0.0, c.north if shifts else 0.0, -c.turn)
        for i, c in corrections(holding(verdicts, placed)).items()
        if shifts or abs(c.turn) >= LEAST_TURN_DEG  # a turn alone: beyond the surveys' own
    }


def corrections(verdicts: dict[str, Verdict]) -> dict[str, Correction]:
    """The panoramas to move, and how (by the RULE of now)."""
    return {i: v.correction for i, v in verdicts.items() if v.correction and v.rule == RULE}
