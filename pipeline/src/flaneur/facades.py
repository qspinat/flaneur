"""Today's walls around a panorama, for depth: facades are vertical, so each is a segment on the
plan with a top; below them a level street; above and beyond, the sky.

Walls are fitted from the details the panoramas see, in the way of PEARL (Isack & Boykov, 2012):
lines proposed (from pairs of details, and from BD TOPO's facades), then every detail given to
one line or to none by a single cost (how far it lies, whether its neighbours agree, a price per
wall kept), the lines refitted, until it holds. The user may draw walls and remove fitted ones;
those are saved in Lambert-93, for every panorama near them. A ray's depth is where it meets
them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from flaneur import match, poses
from flaneur.blocks import Blocks
from flaneur.registration import ViewOf
from flaneur.segment import BUILT, SKY, UNSEEN
from flaneur.sources import append_row, rows

GROUND = -1  # what a ray met: the street
NOTHING = -2  # nothing known: the sky, or beyond the walls

GROUND_BAND_M = 0.3  # details this close above the street are the street
STREET_REACH_M = 50.0  # a ray meets the street this far at most (beyond: nearly level, unsure)
MOST_POINTS = 12000  # the details a fit is made from, at most (fewer: its draws vary)
SAMPLES = 300  # pairs of details proposing a line
PAIR_M = (1.0, 20.0)  # how far apart a pair's details stand
NEAREST = 48  # a pair's second detail is among the nearest of those far enough (on the plan)
CANDIDATE_BAND_M = 1.0  # how far off BD TOPO's facades may be: where their details are looked for
MIN_WALL_POINTS = 15
INLIER_M = 0.3  # details spread at least this far across a facade: the narrowest band, each
# line's own measured on its details (see _bands; 2026-11-07, pairs set right: 80 % within 0.2 m
# under 10 m away, 0.35 at 10-20, 0.5 at 20-30; a band widening with distance agreed worse)
WIDEST_M = 1.0  # and at most this far (see _bands)
SAME_DEG = 3.0  # lines closer than this (and INLIER_M) are one
LINES_MERGED_M = 2 * INLIER_M  # and fitted lines this close (whatever their bands): one facade
# split in two
OUTLIER_COST = 4.0  # a detail on no wall costs as one two narrowest bands off a wall
SMOOTH = 0.5  # per near neighbour given another line
NEIGHBOURS = 8
NEIGHBOUR_M = 1.5
SWEEPS = 5
WALL_COST = 20.0  # the price of a wall kept, at least (more than OUTLIER_COST: it holds details)
WALL_SHARE = 0.03  # and this much a detail fitted: many details, no slivers along one facade
# (a fixed price keeps sparse facades seen at a grazing angle, but trees with them: 2026-11-07)
ROUNDS = 10
GAP_M = 6.0  # a wall's details further apart than this along it: two walls (a side street)
RUN_SPACINGS = 5  # a detail counts toward a wall with RUN_COUNT others this many of its
RUN_COUNT = 2  # line's usual spacings away (at least RUN_LEAST_M)
RUN_LEAST_M = 0.5
MIN_SPAN_M = 2.0  # a wall's details spread this far along it at least (not a trunk, a post)
EXTENT_M = 1.0  # a wall reaches this far past its outer details
ABOVE_M = 1.0  # and this far above its high ones
SEEN_STEP_M = 0.5  # a wall is weighed against the details seen through it, step by step
SEEN_RATIO = 2.0  # a stretch is seen through when more details lie beyond it, seen across it,
SEEN_LEAST = 5  # than SEEN_RATIO its own plus SEEN_LEAST (over a step and its two neighbours)
BEYOND_M = 1.0  # beyond: this far behind the wall, not in a window's recess
CORNER_M = 2.0  # walls meeting this near an end meet at a corner; farther in, they cross
LONE_M = 6.0  # a wall shorter than this, its ends CORNER_M from any other: a tree, a kiosk
CROSS_DEG = 20.0
OVERLAP_DEG = 20.0  # a drawn or removed wall replaces the stretch of a wall this close in angle
OVERLAP_M = 1.5  # and to its line, alongside it
MIN_PIECE_M = 1.0  # what is left of a wall shorter than this goes too
DRAWN_ABOVE_M = 20.0  # a wall drawn where none was fitted reaches this high over the street
REMOVED_MARGIN_M = 3.0  # a wall removed takes this much more of a wall fitted along it (fitted
# again from other details, its ends move)
NEAR_M = 30.0  # walls drawn serve panoramas this close
TOP_STEP_M = 0.5  # a wall's top read in columns this far apart along it
TOP_DZ_M = 0.25  # each column's rays this far apart in height
TOP_HIGHEST_M = 40.0  # over the street
TOP_COLUMNS = 3  # columns that saw where building meets sky, at least


@dataclass(frozen=True)
class Wall:
    a: tuple[float, float]  # its ends on the plan (Lambert-93)
    b: tuple[float, float]
    top: float | None  # height, as the panoramas' (drawn without one: see with_drawn)
    made: str  # "fitted" or "drawn"
    at: str = field(default="", compare=False)  # drawn: the panorama where it was
    since: int = field(default=0, compare=False)  # drawn: the save it first was in (0: new)


@dataclass(frozen=True)
class Scene:
    """The walls a panorama's depth comes from, around its camera."""

    panorama: str
    origin: tuple[float, float, float]  # its camera (Lambert-93, height)
    ground: float  # the street's height
    walls: tuple[Wall, ...]
    neighbours: tuple[str, ...]  # the panoramas whose details the walls were fitted to
    details: int  # how many
    chosen: tuple[str, ...] | None = None  # the neighbours the user chose (none: by geometry)


Segment = tuple[tuple[float, float], tuple[float, float]]
Line = tuple[np.ndarray, float]  # unit normal on the plan, and offset: normal · p = offset


def fit(
    points: np.ndarray,
    ground: float,
    candidates: list[Segment] = (),
    seed: int = 0,
    origin: np.ndarray | None = None,
) -> list[Wall]:
    """The walls along today's details (n × 3, Lambert-93 and height), those on the street left
    out; BD TOPO's facade segments (`candidates`) proposed beside the lines details suggest,
    kept only where details agree. Seen from a camera at `origin` (the details are what it
    sees): the stretches of walls it sees through are taken off before walls crossing better
    held ones are dropped (see _seen_through)."""
    points = points[np.isfinite(points).all(axis=1)]
    points = points[points[:, 2] > ground + GROUND_BAND_M]
    seen = points  # all of them, before a sample is drawn
    rng = np.random.default_rng(seed)
    if len(points) > MOST_POINTS:
        points = points[rng.choice(len(points), MOST_POINTS, replace=False)]
    if len(points) < MIN_WALL_POINTS:
        return []
    centre = points[:, :2].mean(axis=0)  # Lambert-93's origin is millions of metres away
    xy = points[:, :2] - centre
    tree = cKDTree(xy)
    lines = _proposed(xy, tree, [np.subtract(c, centre) for c in candidates], rng)
    if not lines:
        return []
    labels, lines = _shared(xy, tree, lines)
    found = []  # each wall and how many details hold it
    for i, (normal, offset) in enumerate(lines):
        mine = labels == i
        along = np.array([-normal[1], normal[0]])
        t, heights = xy[mine] @ along, points[mine, 2]
        for part in _stretches(t):
            if np.ptp(t[part]) < MIN_SPAN_M:
                continue
            a, b = (
                tuple((centre + normal * offset + along * x).tolist())
                for x in (t[part].min() - EXTENT_M, t[part].max() + EXTENT_M)
            )
            top = float(np.percentile(heights[part], 95)) + ABOVE_M
            found.append((len(part), Wall(a, b, top, "fitted")))
    found.sort(key=lambda f: f[0], reverse=True)  # the best held first (stable for ties)
    walls = [w for _, w in found]
    if origin is not None:  # each wall's pieces keep its rank
        walls = [piece for w in walls for piece in _seen_through(w, origin, seen, ground)]
    return _uncrossed(_linked(walls))


def topped(
    walls: list[Wall],
    origin: np.ndarray,
    ground: float,
    kind_at: Callable[[np.ndarray], np.ndarray],
) -> list[Wall]:
    """The walls, each top where the camera at `origin` sees building give way to sky above it
    (`kind_at`: what each ray sees, segment.BUILT, SKY...). Read in columns along the
    wall, TOP_STEP_M apart, from those where this wall is the first met: the last height seen
    as building with sky seen right above (a crown over the roof, the view's edge: the column
    says nothing); the median of TOP_COLUMNS or more, else the top as fitted (the details'
    highest). Measured 2026-11-07: 161 walls of 349 raised, 24 lowered, the bench's
    correlation 0.212 to 0.223 (20 cases better, 2 by under 0.005 worse). A taller building
    right behind a wall, seen over it, is taken for the wall (the classes cannot tell them)."""
    origin = np.asarray(origin, float)
    heights = np.arange(ground + 2.0, ground + TOP_HIGHEST_M, TOP_DZ_M)
    # who hides whom: walls as tall as can be (their fitted tops are often too low: measured with
    # them, rays passing over a nearer wall read what stands behind it, 0.223 to 0.203)
    prisms = _prisms(walls, ground, [ground + TOP_HIGHEST_M] * len(walls))
    found = []
    for i, wall in enumerate(walls):
        a, b = np.asarray(wall.a, float), np.asarray(wall.b, float)
        n = max(1, int(np.linalg.norm(b - a) / TOP_STEP_M))
        on = a + np.outer((np.arange(n) + 0.5) / n, b - a)  # each column's foot
        points = np.column_stack([np.repeat(on, len(heights), axis=0), np.tile(heights, n)])
        rays = points - origin
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        kinds = kind_at(rays)
        if not np.any(kinds == SKY):  # no sky to top it at (not segmented: all building)
            found.append(wall)
            continue
        _, met = prisms.hit(origin, rays)  # this wall first: not hidden by a nearer one
        kinds = np.where(met == i, kinds, UNSEEN).reshape(n, len(heights))
        tops = [t for column in kinds if (t := _skyline(column, heights)) is not None]
        found.append(
            replace(wall, top=float(np.median(tops))) if len(tops) >= TOP_COLUMNS else wall
        )
    return found


def _skyline(column: np.ndarray, heights: np.ndarray) -> float | None:
    """Where one column (what is seen, bottom up) goes from building to sky: the last building
    with the next two heights sky (unseen between: where it changes is not known); None if it
    does not."""
    built = np.flatnonzero(column == BUILT)
    if not len(built) or column[built[-1] + 1 : built[-1] + 3].tolist() != [SKY, SKY]:
        return None
    return float(heights[built[-1]] + TOP_DZ_M / 2)


def _linked(walls: list[Wall]) -> list[Wall]:
    """The walls less the short ones standing alone (shorter than LONE_M, neither end within
    CORNER_M of another wall they do not cross: see _cross): facades meet facades, round corners
    and along streets; trees' crowns and posts stand apart, or across a facade."""
    lines = [LineString([w.a, w.b]) for w in walls]

    def alone(i: int) -> bool:
        if lines[i].length >= LONE_M:
            return False
        ends = [Point(walls[i].a), Point(walls[i].b)]
        return not any(
            min(line.distance(e) for e in ends) <= CORNER_M
            for j, line in enumerate(lines)
            if j != i and not _cross(walls[i], walls[j])
        )

    return [w for i, w in enumerate(walls) if not alone(i)]


def _seen_through(wall: Wall, origin: np.ndarray, seen: np.ndarray, ground: float) -> list[Wall]:
    """The wall less its stretches the camera at `origin` sees through: where more of the
    details it sees (`seen`) lie beyond them, across them on the plan, between the street and
    the wall's top, than they hold (measured 2026-11-07: a wall across a street's mouth had 7
    to 11 times its own details seen through it, facades at most 0.9). What is left, in
    pieces no shorter than MIN_PIECE_M."""
    a, b = np.asarray(wall.a, float), np.asarray(wall.b, float)
    span = b - a
    length = float(np.linalg.norm(span))
    along = span / length
    normal, _ = _through(a, b)
    o = origin
    sight = seen[:, :2] - o[:2]
    with np.errstate(divide="ignore", invalid="ignore"):  # parallel to the wall: no crossing
        cross = sight[:, 0] * span[1] - sight[:, 1] * span[0]
        reach = ((a[0] - o[0]) * span[1] - (a[1] - o[1]) * span[0]) / cross  # along the sight
        at = ((a[0] - o[0]) * sight[:, 1] - (a[1] - o[1]) * sight[:, 0]) / cross  # along it
        height = o[2] + reach * (seen[:, 2] - o[2])
    off = np.abs((seen[:, :2] - a) @ normal)
    through = (reach > 0) & (reach < 1) & (at >= 0) & (at <= 1) & (off > BEYOND_M)
    through &= (height >= ground) & (height <= wall.top)
    steps = max(1, int(np.ceil(length / SEEN_STEP_M)))
    place = (seen[:, :2] - a) @ along / length
    mine = (off < INLIER_M) & (place >= 0) & (place <= 1)

    def counted(fraction: np.ndarray) -> np.ndarray:
        found = np.bincount(np.minimum((fraction * steps).astype(int), steps - 1), minlength=steps)
        return np.convolve(found, np.ones(3), "same")  # a step and its two neighbours

    kept = counted(at[through]) <= SEEN_RATIO * counted(place[mine]) + SEEN_LEAST
    if kept.all():
        return [wall]
    runs = np.split(np.arange(steps), np.flatnonzero(np.diff(kept.astype(int))) + 1)
    return [
        replace(
            wall,
            a=tuple((a + span * r[0] / steps).tolist()),
            b=tuple((a + span * (r[-1] + 1) / steps).tolist()),
        )
        for r in runs
        if kept[r[0]] and len(r) * length / steps >= MIN_PIECE_M
    ]


def _uncrossed(walls: list[Wall]) -> list[Wall]:
    """The walls (the best held first) less those crossing one kept: facades meet at corners,
    near their ends, never across one another (a line laid across a dense band of details)."""
    kept: list[Wall] = []
    for wall in walls:
        if not any(_cross(wall, other) for other in kept):
            kept.append(wall)
    return kept


def _cross(one: Wall, other: Wall) -> bool:
    """Whether two walls cross more than CORNER_M from all their ends, CROSS_DEG apart or more."""
    p, q = LineString([one.a, one.b]), LineString([other.a, other.b])
    meet = p.intersection(q)
    if meet.geom_type != "Point":
        return False
    (n, _), (m, _) = (
        _through(np.asarray(w.a, float), np.asarray(w.b, float)) for w in (one, other)
    )
    ends = [Point(x) for x in (one.a, one.b, other.a, other.b)]
    apart = abs(n @ m) <= np.cos(np.radians(CROSS_DEG))
    return apart and min(meet.distance(x) for x in ends) > CORNER_M


def _through(p: np.ndarray, q: np.ndarray) -> Line:
    along = (q - p) / np.linalg.norm(q - p)
    normal = np.array([-along[1], along[0]])
    return normal, float(normal @ p)


def _total_least_squares(xy: np.ndarray) -> Line:
    centre = xy.mean(axis=0)
    normal = np.linalg.svd(xy - centre, full_matrices=False)[2][1]
    return normal, float(normal @ centre)


def _refit(xy: np.ndarray, line: Line, first_band: float) -> tuple[Line, int] | None:
    """The line fitted again to the details near it (within `first_band`, then INLIER_M), and
    how many those are; None with fewer than MIN_WALL_POINTS."""
    for band in (first_band, INLIER_M):
        near = np.abs(xy @ line[0] - line[1]) < band
        count = int(near.sum())
        if count < MIN_WALL_POINTS:
            return None
        line = _total_least_squares(xy[near])
    return line, count


def _proposed(xy: np.ndarray, tree: cKDTree, candidates, rng) -> list[Line]:
    """Lines through pairs of near details and along BD TOPO's facades, each fitted again to
    its details; near duplicates kept once (the one with most details)."""
    found = []
    firsts = rng.integers(len(xy), size=SAMPLES)
    for first, within in zip(firsts, tree.query_ball_point(xy[firsts], PAIR_M[1]), strict=True):
        within = np.asarray(within)
        far = np.linalg.norm(xy[within] - xy[first], axis=1)
        apart = within[far >= PAIR_M[0]][np.argsort(far[far >= PAIR_M[0]])[:NEAREST]]
        if len(apart):
            found.append(_refit(xy, _through(xy[first], xy[rng.choice(apart)]), INLIER_M))
    for a, b in candidates:
        a, b = np.asarray(a, float), np.asarray(b, float)
        if np.linalg.norm(b - a) > 0:
            found.append(_refit(xy, _through(a, b), CANDIDATE_BAND_M))
    kept: list[Line] = []
    for line, _ in sorted((f for f in found if f), key=lambda f: -f[1]):
        if not any(_same(line, other) for other in kept):
            kept.append(line)
    return kept


def _same(one: Line, other: Line, apart: float = INLIER_M) -> bool:
    """Whether two lines are one: within SAME_DEG, their offsets within `apart`."""
    turned = float(one[0] @ other[0])
    if abs(turned) < np.cos(np.radians(SAME_DEG)):
        return False
    return abs(one[1] - np.sign(turned) * other[1]) < apart


def _shared(
    xy: np.ndarray,
    tree: cKDTree,
    lines: list[Line],
) -> tuple[np.ndarray, list[Line]]:
    """Each detail given to one line or to none (label len(lines)), lines worth less than the
    price of a wall dropped (see _worth_keeping), the others fitted again to their details,
    those then nearly one made one (see _merged), each line's band measured on its details
    (see _bands), round after round."""
    distances, around = tree.query(xy, k=NEIGHBOURS + 1)
    close = (distances <= NEIGHBOUR_M) & (around != np.arange(len(xy))[:, None])  # not itself
    who = np.nonzero(close)[0]
    graph = csr_matrix((np.ones(len(who)), (who, around[close])), shape=(len(xy),) * 2)
    graph = ((graph + graph.T) > 0).astype(float)  # neighbours both ways
    labels, banded = None, [(line, INLIER_M) for line in lines]
    for _ in range(ROUNDS):
        costs = _costs(xy, banded)
        before, labels = labels, _labelled(costs, graph)
        kept, given = _worth_keeping(costs, labels)  # each kept line fitted to the details it holds
        fitted = _merged(xy, [given == i for i in kept])
        stable = len(fitted) == len(banded) and np.array_equal(before, labels)
        banded = fitted
        if stable:
            break
    return _labelled(_costs(xy, banded), graph), [line for line, _ in banded]


def _merged(xy: np.ndarray, held: list[np.ndarray]) -> list[tuple[Line, float]]:
    """Lines fitted to the details each holds (`held`: masks), those within SAME_DEG and
    LINES_MERGED_M of a better held one fitted with it as one: squared costs always favour two
    lines over one facade's band of details, which is one wall (measured 2026-11-07: 10 to 30 %
    fewer walls, agreeing with the panoramas as well). Each with its band (see _bands)."""
    groups: list[tuple[Line, np.ndarray]] = []
    for mine in sorted(held, key=lambda m: -int(m.sum())):
        line = _total_least_squares(xy[mine])
        for k, (other, theirs) in enumerate(groups):
            if _same(line, other, LINES_MERGED_M):
                both = theirs | mine
                groups[k] = (_total_least_squares(xy[both]), both)
                break
        else:
            groups.append((line, mine))
    if not groups:
        return []
    return list(zip([line for line, _ in groups], _bands(xy, groups), strict=True))


def _bands(xy: np.ndarray, groups: list[tuple[Line, np.ndarray]]) -> list[float]:
    """How far each line's details lie off it: those on any line (the masks of `groups`)
    within WIDEST_M of it and along its own details' stretch (a facade crossing its line counts
    only where it crosses), twice their robust spread (1.4826 × the median distance, a normal
    law's deviation; twice: a tuned allowance), from INLIER_M to WIDEST_M. Facades are not all
    as thin: trees before one, balconies, one seen at a grazing angle (measured 2026-11-07 on
    18 panoramas: as good against a neighbour held out, 0.209 against 0.201, and a third fewer
    walls)."""
    masks = np.array([mine for _, mine in groups]).T  # details × lines: whose each is
    placed = masks.any(axis=1)
    normals = np.array([n for (n, _), _ in groups])
    offsets = np.array([o for (_, o), _ in groups])
    along = xy @ np.vstack([-normals[:, 1], normals[:, 0]])  # each detail along each line
    lo = np.where(masks, along, np.inf).min(axis=0)  # each line's own stretch
    hi = np.where(masks, along, -np.inf).max(axis=0)
    off = np.abs(xy @ normals.T - offsets)
    near = placed[:, None] & (off < WIDEST_M) & (along >= lo) & (along <= hi)
    with np.errstate(all="ignore"):  # a line with none near: the narrowest band
        spread = np.nanmedian(np.where(near, off, np.nan), axis=0)
    return np.clip(np.nan_to_num(2 * 1.4826 * spread, nan=INLIER_M), INLIER_M, WIDEST_M).tolist()


def _costs(xy: np.ndarray, banded: list[tuple[Line, float]]) -> np.ndarray:
    """Each detail's cost on each line (with its band), as a normal law whose deviation is the
    band would give it (squared bands off, and the band's own width paid for: a wide line does
    not take all), and on none (the last column). The band being twice the spread measured
    (see _bands), a detail stays on a wide line out to about three of its deviations."""
    normals = np.array([n for (n, _), _ in banded]).reshape(-1, 2)
    offsets = np.array([o for (_, o), _ in banded])
    widths = np.array([band for _, band in banded], float)
    off = ((xy @ normals.T - offsets) / widths) ** 2 + 2 * np.log(widths / INLIER_M)
    return np.column_stack([off, np.full(len(xy), OUTLIER_COST)])


def _labelled(costs: np.ndarray, graph: csr_matrix) -> np.ndarray:
    """Each detail's label, its neighbours' (`graph`) heeded: iterated conditional modes, all
    at once, until no label changes."""
    labels = costs.argmin(axis=1)
    count = np.asarray(graph.sum(axis=1))
    for _ in range(SWEEPS):
        agreeing = graph @ np.eye(costs.shape[1])[labels]
        changed = (costs + SMOOTH * (count - agreeing)).argmin(axis=1)
        if np.array_equal(changed, labels):
            break
        labels = changed
    return labels


def _worth_keeping(costs: np.ndarray, labels: np.ndarray) -> tuple[list[int], np.ndarray]:
    """The lines whose details would cost the price of a wall or more elsewhere (WALL_COST, or
    WALL_SHARE of a detail for each detail fitted), the least worth first;
    a line dropped gives its details to the best of those left before the next is weighed (two
    lines sharing one facade: the first goes, the second then holds it alone). And the details'
    labels so given."""
    labels = labels.copy()
    standing = np.ones(costs.shape[1], bool)  # the last: none, always there
    price = max(WALL_COST, WALL_SHARE * len(costs))

    def saving(i: int) -> tuple[float, np.ndarray, np.ndarray]:
        mine = labels == i
        others = standing.copy()
        others[i] = False
        worth = costs[mine][:, others].min(axis=1).sum() - costs[mine, i].sum()
        return float(worth), mine, np.flatnonzero(others)

    for i in sorted(range(len(standing) - 1), key=lambda i: saving(i)[0]):
        worth, mine, others = saving(i)
        if worth < price:
            standing[i] = False
            labels[mine] = others[costs[mine][:, others].argmin(axis=1)]
    return np.flatnonzero(standing[:-1]).tolist(), labels


def _stretches(t: np.ndarray) -> list[np.ndarray]:
    """The details of one line (their places `t` along it) that make a wall, split where they
    leave a gap wider than GAP_M (a side street). Only details with RUN_COUNT others within
    RUN_SPACINGS of the line's usual spacing count (as DBSCAN's core points), with those within
    that reach of one: a lone stray in a gap does not bridge it, nor do strays past its end
    lengthen it, however dense the wall."""
    order = np.argsort(t)
    placed = t[order]
    if len(placed) < 2:
        return []
    reach = max(RUN_LEAST_M, RUN_SPACINGS * float(np.median(np.diff(placed))))
    right = np.searchsorted(placed, placed + reach, "right")
    others = right - np.searchsorted(placed, placed - reach, "left") - 1
    core = placed[others >= RUN_COUNT]
    if not len(core):
        return []
    # and those within reach of one (DBSCAN's border points)
    at = np.clip(np.searchsorted(core, placed), 1, len(core) - 1)
    nearest = np.minimum(np.abs(placed - core[at - 1]), np.abs(placed - core[at]))
    kept = order[nearest <= reach]
    parts = np.split(kept, np.flatnonzero(np.diff(t[kept]) > GAP_M) + 1)
    return [part for part in parts if len(part) >= MIN_WALL_POINTS]


def cast(scene: Scene, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """How far along each world unit direction from the scene's camera: to the nearest wall it
    meets (within its ends, between the street and its top), else to the street when it points
    down (within STREET_REACH_M); NaN meeting nothing. And what it met: a wall's index, GROUND
    or NOTHING. (A wall is met above the street, so before it.)"""
    origin = np.asarray(scene.origin, float)
    d = np.asarray(directions, float)
    points, met = _prisms(scene.walls, scene.ground, [w.top for w in scene.walls]).hit(origin, d)
    metres = np.linalg.norm(points - origin, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        street = (scene.ground - origin[2]) / d[:, 2]
    none = met < 0
    met[none] = NOTHING
    down = none & (d[:, 2] < 0) & (street > 0) & (street <= STREET_REACH_M)
    metres[down], met[down] = street[down], GROUND
    return metres, met


def _prisms(walls, ground: float, tops: list[float]) -> Blocks:
    """The walls as Blocks' prisms, from the street to their `tops`."""
    edges = [(*w.a, *w.b, ground, top) for w, top in zip(walls, tops, strict=True)]
    return Blocks(np.array(edges, float).reshape(-1, 6), np.ones(len(edges), bool))


def in_view(
    scene: Scene,
    view: ViewOf,
    side: int,
    along: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """How far along each pixel's ray of a view (side × side) cut from the scene's panorama, by
    its walls (or `along`: rays to metres); infinitely far meeting nothing (the sky)."""
    rays = match.view_directions(
        match.pixels_of((side, side)), view.heading, view.fov, side, view.pitch
    )
    metres = cast(scene, rays)[0] if along is None else along(rays)
    return np.where(np.isnan(metres), np.inf, metres).reshape(side, side)


def with_drawn(
    fitted: list[Wall], drawn: list[Wall], removed: list[Segment], ground: float
) -> list[Wall]:
    """The walls drawn by hand, and what the drawn and removed ones leave of the fitted. A wall
    drawn without its top takes that of the fitted walls it replaces (fitted again, it follows
    them), else DRAWN_ABOVE_M over the street (`ground`)."""

    def with_top(w: Wall) -> Wall:
        tops = [f.top for f in fitted if _left_of(f, (w.a, w.b), 0.0) != [f]]
        return w if w.top is not None else replace(w, top=max(tops, default=ground + DRAWN_ABOVE_M))

    left = _uncovered(fitted, [(w.a, w.b) for w in drawn])
    return [with_top(w) for w in drawn] + _uncovered(left, removed, REMOVED_MARGIN_M)


def _uncovered(walls: list[Wall], covering: list[Segment], margin: float = 0.0) -> list[Wall]:
    """The walls, less the stretches the `covering` segments lie along (`margin` past their
    ends)."""
    for one in covering:
        walls = [piece for wall in walls for piece in _left_of(wall, one, margin)]
    return walls


def _left_of(wall: Wall, one: Segment, margin: float) -> list[Wall]:
    """What is left of a wall where `one` lies along it (within OVERLAP_DEG, and OVERLAP_M of
    its line midway along the stretch they share): the wall cut between the feet of `one`'s
    ends (`margin` further), pieces under MIN_PIECE_M left out."""
    a, b = np.asarray(wall.a, float), np.asarray(wall.b, float)
    p, q = np.asarray(one, float)
    length = float(np.linalg.norm(b - a))
    if length == 0 or not np.any(q - p):
        return [wall]
    along = (b - a) / length
    normal, offset = _through(p, q)
    if abs(normal @ along) > np.sin(np.radians(OVERLAP_DEG)):
        return [wall]
    low, high = sorted(((p - a) @ along, (q - a) @ along))
    low, high = max(low, 0.0), min(high, length)
    if low >= high or abs(normal @ (a + along * (low + high) / 2) - offset) > OVERLAP_M:
        return [wall]
    low, high = low - margin, high + margin
    return [
        replace(wall, a=tuple((a + along * start).tolist()), b=tuple((a + along * end).tolist()))
        for start, end in ((0.0, low), (high, length))
        if end - start >= MIN_PIECE_M
    ]


def record(
    path: Path,
    panorama: str,
    drawn: list[Wall],
    removed: list[Segment],
    removed_since: list[int] | None = None,
    at: tuple[float, float, float] | None = None,
) -> None:
    """Keep the walls drawn at a panorama and those removed there (a row; the latest holds),
    each with the save it first was in (`since`; `removed_since` for the removals): a wall or a
    removal new, or sent back unchanged. Where the panorama stood (`at`: east, north, heading),
    for them to follow it if it is set right since (see saved_at)."""
    now = 1 + max((s for row in rows(path) for s in _ages(row)), default=0)
    ages = removed_since or [0] * len(removed)
    append_row(
        path,
        {
            "panorama": panorama,
            "drawn": [{"a": w.a, "b": w.b, "top": w.top, "since": w.since or now} for w in drawn],
            "removed": [
                {"a": a, "b": b, "since": s or now} for (a, b), s in zip(removed, ages, strict=True)
            ],
            "panorama_at": None if at is None else [float(v) for v in at],
        },
    )


def _ages(row: dict) -> list[int]:
    return [x["since"] for x in row["drawn"] + row["removed"]]


def _latest(path: Path) -> dict[str, dict]:
    """The latest row of each panorama."""
    latest: dict[str, dict] = {}
    for row in rows(path):
        latest[row["panorama"]] = row
    return latest


def _read(row: dict, now: dict | None = None) -> tuple[list[Wall], list[Segment], list[int]]:
    """A row's walls drawn and its removals, and the save each removal first was in; with where
    the panoramas stand `now` (east, north, heading by id), moved with their panorama since."""
    then, stands = row.get("panorama_at"), (now or {}).get(row["panorama"])

    def end(point) -> tuple[float, float]:
        if not then or stands is None or tuple(then) == tuple(stands):
            return tuple(point)
        return tuple(float(v) for v in poses.carried(np.array([point], float), then, stands)[0])

    drawn = [
        Wall(end(w["a"]), end(w["b"]), w["top"], "drawn", at=row["panorama"], since=w["since"])
        for w in row["drawn"]
    ]
    removed = [(end(r["a"]), end(r["b"])) for r in row["removed"]]
    return drawn, removed, [r["since"] for r in row["removed"]]


def saved_at(
    path: Path, panorama: str, now: dict | None = None
) -> tuple[list[Wall], list[Segment], list[int]]:
    """The walls drawn and removed at a panorama (its latest row), and when each removal was
    (`now`: see _read)."""
    row = _latest(path).get(panorama)
    return _read(row, now) if row else ([], [], [])


def saved_near(
    path: Path, x: float, y: float, radius: float = NEAR_M, now: dict | None = None
) -> tuple[list[Wall], list[Segment]]:
    """The walls drawn and removed (the latest row of each panorama) reaching within `radius`
    of (x, y). A wall drawn at one panorama loses the stretches removed at others after it was
    drawn (a wall drawn after a removal near it stays). (A panorama's row holds the walls drawn
    there, not those drawn at others.)"""
    drawn: list[Wall] = []
    removals: list[tuple[Segment, int, str]] = []
    for panorama, row in _latest(path).items():
        walls, removed, ages = _read(row, now)
        drawn += [w for w in walls if _distance((x, y), (w.a, w.b)) <= radius]
        removals += [
            (r, s, panorama)
            for r, s in zip(removed, ages, strict=True)
            if _distance((x, y), r) <= radius
        ]
    kept = [
        piece
        for wall in drawn
        for piece in _uncovered(
            [wall],
            [r for r, s, at in removals if at != wall.at and s > wall.since],
            REMOVED_MARGIN_M,
        )
    ]
    return kept, [r for r, _, _ in removals]


def _distance(point: tuple[float, float], segment: Segment) -> float:
    return LineString(segment).distance(Point(point))
