"""Today's walls around a panorama, for depth: facades are vertical, so each is a segment on the
plan with a top; below them a level street; above and beyond, the sky.

Walls are fitted from the details the panoramas see, in the way of PEARL (Isack & Boykov, 2012):
lines proposed (from pairs of details, and from BD TOPO's facades), then every detail given to
one line or to none by a single cost (how far it lies, whether its neighbours agree, a price per
wall kept), the lines refitted, until it holds. The user may draw walls and remove fitted ones;
those are saved in Lambert-93, for every panorama near them. A ray's depth is where it meets
them."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from flaneur import match
from flaneur.blocks import Blocks
from flaneur.registration import ViewOf
from flaneur.sources import append_row, rows

GROUND = -1  # what a ray met: the street
NOTHING = -2  # nothing known: the sky, or beyond the walls

GROUND_BAND_M = 0.3  # details this close above the street are the street
STREET_REACH_M = 50.0  # a ray meets the street this far at most (beyond: nearly level, unsure)
MOST_POINTS = 4000
SAMPLES = 300  # pairs of details proposing a line
PAIR_M = (1.0, 20.0)  # how far apart a pair's details stand
NEAREST = 48  # a pair's second detail is among the nearest of those far enough (on the plan)
CANDIDATE_BAND_M = 1.0  # how far off BD TOPO's facades may be: where their details are looked for
MIN_WALL_POINTS = 15
INLIER_M = 0.25
SAME_DEG = 3.0  # lines closer than this (and INLIER_M) are one
OUTLIER_COST = 4.0  # a detail on no wall costs as one 2 × INLIER_M off a wall
SMOOTH = 0.5  # per near neighbour given another line
NEIGHBOURS = 8
NEIGHBOUR_M = 1.5
SWEEPS = 5
WALL_COST = 20.0  # the price of a wall kept (more than OUTLIER_COST: a wall holds details)
ROUNDS = 10
GAP_M = 6.0  # a wall's details further apart than this along it: two walls (a side street)
RUN_SPACINGS = 5  # a detail counts toward a wall with RUN_COUNT others this many of its
RUN_COUNT = 2  # line's usual spacings away (at least RUN_LEAST_M)
RUN_LEAST_M = 0.5
EXTENT_M = 1.0  # a wall reaches this far past its outer details
ABOVE_M = 1.0  # and this far above its high ones
OVERLAP_DEG = 20.0  # a drawn or removed wall replaces the stretch of a wall this close in angle
OVERLAP_M = 1.5  # and to its line, alongside it
MIN_PIECE_M = 1.0  # what is left of a wall shorter than this goes too
REMOVED_MARGIN_M = 3.0  # a wall removed takes this much more of a wall fitted along it (fitted
# again from other details, its ends move)
NEAR_M = 30.0  # walls drawn serve panoramas this close


@dataclass(frozen=True)
class Wall:
    a: tuple[float, float]  # its ends on the plan (Lambert-93)
    b: tuple[float, float]
    top: float  # height, as the panoramas'
    made: str  # "fitted" or "drawn"


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
    points: np.ndarray, ground: float, candidates: list[Segment] = (), seed: int = 0
) -> list[Wall]:
    """The walls along today's details (n × 3, Lambert-93 and height), those on the street left
    out; BD TOPO's facade segments (`candidates`) proposed beside the lines details suggest,
    kept only where details agree."""
    points = points[np.isfinite(points).all(axis=1)]
    points = points[points[:, 2] > ground + GROUND_BAND_M]
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
    walls = []
    for i, (normal, offset) in enumerate(lines):
        mine = labels == i
        along = np.array([-normal[1], normal[0]])
        t, heights = xy[mine] @ along, points[mine, 2]
        for part in _stretches(t):
            a, b = (
                tuple((centre + normal * offset + along * x).tolist())
                for x in (t[part].min() - EXTENT_M, t[part].max() + EXTENT_M)
            )
            walls.append(Wall(a, b, float(np.percentile(heights[part], 95)) + ABOVE_M, "fitted"))
    return walls


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


def _same(one: Line, other: Line) -> bool:
    turned = float(one[0] @ other[0])
    if abs(turned) < np.cos(np.radians(SAME_DEG)):
        return False
    return abs(one[1] - np.sign(turned) * other[1]) < INLIER_M


def _shared(xy: np.ndarray, tree: cKDTree, lines: list[Line]) -> tuple[np.ndarray, list[Line]]:
    """Each detail given to one line or to none (label len(lines)), lines worth less than
    WALL_COST dropped, the others fitted again to their details, round after round."""
    distances, around = tree.query(xy, k=NEIGHBOURS + 1)
    close = (distances <= NEIGHBOUR_M) & (around != np.arange(len(xy))[:, None])  # not itself
    who = np.nonzero(close)[0]
    graph = csr_matrix((np.ones(len(who)), (who, around[close])), shape=(len(xy),) * 2)
    graph = ((graph + graph.T) > 0).astype(float)  # neighbours both ways
    labels = None
    for _ in range(ROUNDS):
        costs = _costs(xy, lines)
        before, labels = labels, _labelled(costs, graph)
        kept = _worth_keeping(costs, labels)
        stable = len(kept) == len(lines) and np.array_equal(before, labels)
        lines = [_total_least_squares(xy[labels == i]) for i in kept]
        if stable:
            break
    return _labelled(_costs(xy, lines), graph), lines


def _costs(xy: np.ndarray, lines: list[Line]) -> np.ndarray:
    """Each detail's cost on each line, and on none (the last column)."""
    normals = np.array([n for n, _ in lines]).reshape(-1, 2)
    offsets = np.array([o for _, o in lines])
    off = ((xy @ normals.T - offsets) / INLIER_M) ** 2
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


def _worth_keeping(costs: np.ndarray, labels: np.ndarray) -> list[int]:
    """The lines whose details would cost WALL_COST or more elsewhere, the least worth first;
    a line dropped gives its details to the best of those left before the next is weighed (two
    lines sharing one facade: the first goes, the second then holds it alone)."""
    labels = labels.copy()
    standing = np.ones(costs.shape[1], bool)  # the last: none, always there

    def saving(i: int) -> tuple[float, np.ndarray, np.ndarray]:
        mine = labels == i
        others = standing.copy()
        others[i] = False
        worth = costs[mine][:, others].min(axis=1).sum() - costs[mine, i].sum()
        return float(worth), mine, np.flatnonzero(others)

    for i in sorted(range(len(standing) - 1), key=lambda i: saving(i)[0]):
        worth, mine, others = saving(i)
        if worth < WALL_COST:
            standing[i] = False
            labels[mine] = others[costs[mine][:, others].argmin(axis=1)]
    return np.flatnonzero(standing[:-1]).tolist()


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
    edges = [(*w.a, *w.b, scene.ground, w.top) for w in scene.walls]
    blocks = Blocks(np.array(edges, float).reshape(-1, 6), np.ones(len(edges), bool))
    points, met = blocks.hit(origin, d)
    metres = np.linalg.norm(points - origin, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        street = (scene.ground - origin[2]) / d[:, 2]
    none = met < 0
    met[none] = NOTHING
    down = none & (d[:, 2] < 0) & (street > 0) & (street <= STREET_REACH_M)
    metres[down], met[down] = street[down], GROUND
    return metres, met


def in_view(scene: Scene, view: ViewOf, side: int) -> np.ndarray:
    """How far along each pixel's ray of a view (side × side) cut from the scene's panorama;
    infinitely far meeting nothing (the sky)."""
    rays = match.view_directions(
        match.pixels_of((side, side)), view.heading, view.fov, side, view.pitch
    )
    metres = cast(scene, rays)[0]
    return np.where(np.isnan(metres), np.inf, metres).reshape(side, side)


def with_drawn(fitted: list[Wall], drawn: list[Wall], removed: list[Segment]) -> list[Wall]:
    """The walls drawn by hand, and what the drawn and removed ones leave of the fitted."""
    left = _uncovered(fitted, [(w.a, w.b) for w in drawn])
    return list(drawn) + _uncovered(left, removed, REMOVED_MARGIN_M)


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


def record(path: Path, panorama: str, drawn: list[Wall], removed: list[Segment]) -> None:
    """Keep the walls drawn at a panorama and those removed there (a row; the latest holds)."""
    row = {"panorama": panorama, "removed": list(removed)}
    append_row(path, {**row, "drawn": [{"a": w.a, "b": w.b, "top": w.top} for w in drawn]})


def saved_near(
    path: Path, x: float, y: float, radius: float = NEAR_M
) -> tuple[list[Wall], list[Segment]]:
    """The walls drawn and removed (the latest row of each panorama) reaching within `radius`
    of (x, y); a wall drawn at one panorama and removed at another afterwards stays removed.
    (A panorama's row holds the walls drawn there, not those drawn at others.)"""
    latest: dict[str, dict] = {}
    for row in rows(path):
        latest.pop(row["panorama"], None)  # in the order saved last
        latest[row["panorama"]] = row
    near = []
    for row in latest.values():
        drawn = [Wall(tuple(w["a"]), tuple(w["b"]), float(w["top"]), "drawn") for w in row["drawn"]]
        removed = [(tuple(a), tuple(b)) for a, b in row["removed"]]
        near.append(
            (
                [w for w in drawn if _distance((x, y), (w.a, w.b)) <= radius],
                [r for r in removed if _distance((x, y), r) <= radius],
            )
        )
    removed = [r for _, theirs in near for r in theirs]
    drawn = [
        wall
        for i, (walls, _) in enumerate(near)
        for wall in _uncovered(walls, [r for _, rs in near[i + 1 :] for r in rs], REMOVED_MARGIN_M)
    ]
    return drawn, removed


def _distance(point: tuple[float, float], segment: Segment) -> float:
    return LineString(segment).distance(Point(point))
