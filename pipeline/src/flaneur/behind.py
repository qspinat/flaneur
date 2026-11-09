"""The depth of the street behind what passes (a car, a person, a bicycle), rebuilt from what
stands around it in the same view: a model gives the passing thing's own depth, of no use for a
point clicked on the facade it hides. A street is buildings standing upright on a ground: a facade
seen above the car goes on straight down to the ground (its distance across the street does not
change along a column of the view); where none is seen above, the facades on either side are
lines on the plan, carried behind it; the ground is the plane fitted on the ground around (a
street slopes, a pavement stands higher), not the flat street under the camera. Each ray takes the
nearest of what it meets.

Measured (2026-11-09, a passing thing's outline laid over truly still pixels, 670 holes in 67
panoramas, a held-out neighbour's view brought through the depth): the walls alone 0.368, this
0.397, the model on the true pixels 0.413; on the ground 0.499 for 0.502."""

import cv2
import numpy as np

from flaneur import facades

ABOVE = 1 / 16  # of the view's side: rows looked at above a hole, for the facade there
RING = 1 / 15  # of the view's side: around a hole, where its facades are taken
GROUND_RING = 2 / 15  # of the view's side: around a hole, where its ground is fitted
LEAST_ABOVE = 5  # facade pixels above a hole's column, for that column to take theirs
LINE_M = 0.25  # a facade point this near a line on the plan lies on it
LEAST_ON_LINE = 30  # points on a line, for it to count
LINES = 2  # facades looked for around a hole (a street corner is two)
OUTSIDE_DEG = 5.0  # a line seen on one side only is carried this far past its points
GROUND_M = 0.1  # a ground point this near a plane lies on it
LEAST_GROUND = 50  # points on the ground plane, for it to count
GROUND_REACH_M = 30.0  # ground points farther than this are not fitted on
STEEPEST = 0.15  # a fitted ground steeper than this is no street
TRIES = 200  # draws, for a line or a plane (seeded: the same answer each time)
MOST_POINTS = 4000  # points drawn from, at most (more: the same fit, slower)


def filled(
    hole: np.ndarray,
    known: np.ndarray,
    built: np.ndarray,
    ground: np.ndarray,
    rays: np.ndarray,
    origin: np.ndarray,
    street: float,
    side: int | None = None,
) -> np.ndarray:
    """How far along each ray of a view (or of a crop of one around the hole, h × w) the still
    street stands behind `hole`: from the distances `known` around it (NaN where none), the pixels
    seen as building (`built`) and as ground (`ground`), their world `rays` (h·w × 3) from
    `origin`, and the street's height under the camera; `side`: the whole view's, which sizes the
    rings looked at (by default the hole's). NaN outside the hole, and where nothing is met."""
    shape = hole.shape
    side = side or shape[0]
    rng = np.random.default_rng(0)
    on_ground = _ground_plane(hole, known, ground, rays, origin, rng, side)
    to_ground = _to_ground(rays, origin, street, on_ground)
    horiz = np.hypot(rays[:, 0], rays[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        above = _facade_above(hole, known, built, horiz.reshape(shape), side).ravel() / horiz
    from_above = _nearest(above, to_ground)
    lines = _plan_lines(hole, known, built, rays, origin, street, rng, side)
    found = np.where(np.isfinite(above), from_above, _nearest(*lines, to_ground))
    return np.where(hole.ravel(), found, np.nan).reshape(shape)


def _nearest(*lengths: np.ndarray) -> np.ndarray:
    """Per ray, the nearest positive length among several (NaN: none)."""
    stack = np.vstack([np.where(np.isfinite(t) & (t > 0), t, np.inf) for t in lengths])
    best = stack.min(axis=0)
    return np.where(np.isfinite(best), best, np.nan)


def _ring(hole: np.ndarray, share: float, side: int) -> np.ndarray:
    width = max(3, round(side * share)) | 1
    grown = cv2.dilate(hole.astype(np.uint8), np.ones((width, width), np.uint8)) > 0
    return grown & ~hole


def _to_ground(rays, origin, street, plane) -> np.ndarray:
    """How far along each ray to the ground: the plane z = a·x + b·y + c, or the flat street;
    NaN beyond the street's reach (nearly level rays: unsure)."""
    a, b, c = plane if plane is not None else (0.0, 0.0, street)
    den = rays[:, 2] - a * rays[:, 0] - b * rays[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (a * origin[0] + b * origin[1] + c - origin[2]) / den
    return np.where((t > 0) & (t <= facades.STREET_REACH_M), t, np.nan)


def _around(hole, share, seen, known, rays, origin, rng, side) -> np.ndarray:
    """The world points of the pixels around the hole (a ring `share` of the view's `side` wide)
    that show `seen` and whose distance is known; MOST_POINTS of them at most."""
    near = np.flatnonzero(
        _ring(hole, share, side).ravel() & seen.ravel() & np.isfinite(known.ravel())
    )
    if len(near) > MOST_POINTS:
        near = rng.choice(near, MOST_POINTS, replace=False)
    return origin + rays[near] * known.ravel()[near, None]


def _ground_plane(
    hole, known, ground, rays, origin, rng, side
) -> tuple[float, float, float] | None:
    """The plane z = a·x + b·y + c of the ground seen around the hole (RANSAC), or None."""
    points = _around(hole, GROUND_RING, ground, known, rays, origin, rng, side)
    points = points[np.hypot(*(points[:, :2] - origin[:2]).T) < GROUND_REACH_M]
    if len(points) < LEAST_GROUND:
        return None
    design = np.column_stack([points[:, :2], np.ones(len(points))])
    best = None
    for _ in range(TRIES):
        pick = rng.choice(len(points), 3, replace=False)
        try:
            coef = np.linalg.solve(design[pick], points[pick, 2])
        except np.linalg.LinAlgError:
            continue
        on = np.abs(design @ coef - points[:, 2]) < GROUND_M
        if best is None or on.sum() > best.sum():
            best = on
    if best is None or best.sum() < LEAST_GROUND:
        return None
    a, b, c = np.linalg.lstsq(design[best], points[best, 2], rcond=None)[0]
    return None if np.hypot(a, b) > STEEPEST else (float(a), float(b), float(c))


def _facade_above(hole, known, built, horiz, side) -> np.ndarray:
    """Per pixel of the hole, the horizontal distance of the facade seen just above it in its
    column (a facade is upright: the same all the way down); NaN where none is seen."""
    across = known * horiz  # horizontal distance of each pixel known
    usable = built & np.isfinite(across)
    rows = max(1, round(side * ABOVE))
    found = np.full(hole.shape, np.nan)
    for column in np.flatnonzero(hole.any(axis=0)):
        top = int(np.argmax(hole[:, column]))
        span = slice(max(0, top - rows), top)
        values = across[span, column][usable[span, column]]
        if len(values) >= LEAST_ABOVE:
            found[hole[:, column], column] = np.median(values)
    return found


def _plan_lines(hole, known, built, rays, origin, street, rng, side) -> list[np.ndarray]:
    """How far along each ray to the facades seen around the hole, as lines on the plan (one or
    two, RANSAC), between the street and their top; each carried behind the hole if its points
    stand on both sides, else only a little past them."""
    points = _around(hole, RING, built, known, rays, origin, rng, side)
    if len(points) < LEAST_ON_LINE:
        return []
    top = np.percentile(points[:, 2], 95)
    toward = rays[hole.ravel(), :2].sum(axis=0)  # the hole's direction: bearings from it, so
    turn = np.arctan2(toward[1], toward[0])  # that west's jump from +π to -π never falls inside

    def bearing(xy: np.ndarray) -> np.ndarray:
        return (np.arctan2(xy[:, 1], xy[:, 0]) - turn + np.pi) % (2 * np.pi) - np.pi

    rays_bearing = bearing(rays[:, :2])
    lengths = []
    for normal, offset, kept in _lines(points[:, :2], rng):
        seen = bearing(kept - origin[:2])
        low, high = seen.min(), seen.max()
        across = rays[:, :2] @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (offset - origin[:2] @ normal) / across
        z = origin[2] + t * rays[:, 2]
        ok = (t > 0) & (z >= street - 0.2) & (z <= top + 0.5)
        if not low <= 0 <= high:  # seen on one side of the hole only: carried a little past
            margin = np.radians(OUTSIDE_DEG)
            ok &= (rays_bearing >= low - margin) & (rays_bearing <= high + margin)
        lengths.append(np.where(ok, t, np.nan))
    return lengths


def _lines(points: np.ndarray, rng) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Up to LINES lines through plan points (normal, offset, the points on it), each refitted on
    its own points."""
    found, left = [], points
    for _ in range(LINES):
        if len(left) < LEAST_ON_LINE:
            break
        best = None
        for _ in range(TRIES):
            a, b = left[rng.choice(len(left), 2, replace=False)]
            if np.hypot(*(b - a)) < 0.5:
                continue
            normal, offset = facades._through(a, b)
            on = np.abs(left @ normal - offset) < LINE_M
            if best is None or on.sum() > best.sum():
                best = on
        if best is None or best.sum() < LEAST_ON_LINE:
            break
        kept = left[best]
        normal, offset = facades._total_least_squares(kept)
        found.append((normal, offset, kept))
        left = left[~best]
    return found
