"""Line an old photograph up with the street: the direction its camera faced, where its
horizon lies and how wide it saw, so that the 3D walk can draw it on the buildings.

Matching the photo against today's panorama was tried first (SIFT, rue du Petit-Pont): 2 fits
out of 29, both wrong. Most facades in these photos are gone, so the geometry of the street
is used instead."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

from flaneur import walls
from flaneur.export import WalkPhoto, image_name
from flaneur.gazetteer import angle, to_metres

# Lining up by geometry: where the street's lines run to (the vanishing point) gives the
# direction the camera faced, against the direction of the facades around it; the row of
# that point gives the horizon (view cameras kept verticals upright by shifting the lens, so
# the horizon is rarely mid-height).
SEGMENT_MIN = 0.04  # of the picture's diagonal
UPRIGHT_DEG = 20  # segments this close to vertical are uprights, not street lines
LEVEL_DEG = 3  # nor these, close to level: facades seen square on
AGREE_DEG = 1.5
VP_TRIES = 2000
VP_RANGE = 3.0  # the point may lie outside the picture, up to 3 widths away
STREET_DEG = 40
SAME_STREET_DEG = 5
TURN_LIMIT_DEG = 60


# Width of view when nothing better is known: Atget's 18 × 24 cm plates behind a lens of
# about 180 mm (to be calibrated on the photos the user checks).
FOV_UPRIGHT = 53
FOV_LANDSCAPE = 67
CAMERA_HEIGHT_M = 1.6  # a view camera on its tripod


def vanishing_point(grey: np.ndarray) -> tuple[float, float] | None:
    """Where most of a grey picture's slanting lines meet, as fractions of its width and
    height (RANSAC over pairs of segments, each scored by the length of segments pointing at
    it)."""
    h, w = grey.shape
    found = cv2.createLineSegmentDetector().detect(grey)[0]
    if found is None:
        return None
    segments = found.reshape(-1, 4)
    starts, ends = segments[:, :2], segments[:, 2:]
    length = np.hypot(*(ends - starts).T)
    slope = np.degrees(np.arctan2(*(ends - starts).T[::-1])) % 180
    keep = (
        (length > SEGMENT_MIN * np.hypot(w, h))
        & (np.abs(slope - 90) > UPRIGHT_DEG)
        & (np.minimum(slope, 180 - slope) > LEVEL_DEG)
    )
    starts, ends, length = starts[keep], ends[keep], length[keep]
    if len(length) < 4:
        return None
    lines = np.cross(np.c_[starts, np.ones(len(starts))], np.c_[ends, np.ones(len(ends))])
    middles, directions = (starts + ends) / 2, (ends - starts) / length[:, None]
    rng = np.random.default_rng(0)
    best, best_score = None, 0.0
    for _ in range(VP_TRIES):
        i, j = rng.choice(len(lines), 2, replace=False)
        point = np.cross(lines[i], lines[j])
        if abs(point[2]) < 1e-9:
            continue
        x, y = point[:2] / point[2]
        if not (-VP_RANGE * w < x < (1 + VP_RANGE) * w and -h < y < 2 * h):
            continue
        towards = np.array([x, y]) - middles
        towards /= np.linalg.norm(towards, axis=1)[:, None] + 1e-9
        off = np.degrees(np.arccos(np.clip(np.abs((towards * directions).sum(1)), 0, 1)))
        score = length[off < AGREE_DEG].sum()
        if score > best_score:
            best, best_score = (x / w, y / h), score
    return best


def street_direction(bearings: list[tuple[float, float]], guess: float) -> float | None:
    """Among facade directions around the camera, (bearing, metres of facade), the one with
    the most facade within 40° of the proposed heading; None when there is none."""
    near = [(b, length) for b, length in bearings if abs(angle(guess, b)) <= STREET_DEG]
    if not near:
        return None
    # the facade length running within 5° of each direction: a street's facade comes in pieces
    support = [sum(m for c, m in near if abs(angle(b, c)) <= SAME_STREET_DEG) for b, _ in near]
    best = near[int(np.argmax(support))][0]
    group = [(c, m) for c, m in near if abs(angle(best, c)) <= SAME_STREET_DEG]
    turns = sum(angle(best, c) * m for c, m in group) / sum(m for _, m in group)
    return (best + turns) % 360


def orient(
    guess: float, vanishing: tuple[float, float] | None, street: float | None, fov: float
) -> tuple[float, float] | None:
    """Heading and horizon (fraction of the height from the top): the street's direction,
    turned by the vanishing point's offset from the centre. None when either is unknown, or
    when the result turns more than 60° away from the proposed heading."""
    if vanishing is None or street is None:
        return None
    f = 0.5 / np.tan(np.radians(fov) / 2)  # in photo widths
    heading = (street - np.degrees(np.arctan((vanishing[0] - 0.5) / f))) % 360
    if abs(angle(guess, heading)) > TURN_LIMIT_DEG:
        return None
    return float(heading), vanishing[1]


@dataclass(frozen=True)
class Pose:
    x: float  # Lambert-93 metres
    y: float
    z: float  # above the street
    heading: float
    horizon: float  # row of the horizon, as a fraction of the height from the top
    fov: float  # horizontal, degrees


def _lines_of(path: Path) -> tuple[tuple[float, float] | None, tuple[int, int]]:
    """The photo's vanishing point and its size."""
    with Image.open(path) as image:
        return vanishing_point(np.asarray(image.convert("L"))), image.size


def propose(
    photos: list[WalkPhoto],
    data_dir: Path,
    today: list[Polygon],
    old: list[Polygon],
) -> list[dict]:
    """Each photo's pose, from its placement and its street's lines, and the facades it is
    projected onto (see walls.facades). Status "lines" when the street's lines turned the
    heading, "placement" when the proposed heading was kept."""
    photos = [
        me for me in photos if me.heading is not None and (data_dir / image_name(me.id)).exists()
    ]
    with ProcessPoolExecutor() as pool:  # finding the lines takes most of the time
        lines = list(pool.map(_lines_of, [data_dir / image_name(me.id) for me in photos]))
    every_edge = walls.edges(today) + walls.edges(old)
    proposals = []
    for me, (point, (width, height)) in zip(photos, lines, strict=True):
        here = to_metres(Point(me.lon, me.lat))
        fov = FOV_UPRIGHT if height > width else FOV_LANDSCAPE
        street = street_direction(walls.facade_bearings(every_edge, here), me.heading)
        # a photo facing a facade keeps its heading but still has its street, for its facades
        turned = orient(me.heading, point, street, fov)
        heading, horizon = turned or (me.heading, 0.5)
        pose = Pose(
            round(here.x, 2),
            round(here.y, 2),
            CAMERA_HEIGHT_M,
            round(heading, 2),
            round(horizon, 3),
            fov,
        )
        proposals.append(
            {
                "photo_id": me.id,
                "status": "lines" if turned else "placement",
                "pose": asdict(pose),
                "walls": [asdict(w) for w in walls.facades(here, heading, street, fov, every_edge)],
            }
        )
    return proposals


def matched(photo_id: str, m) -> dict:
    """The proposal for a photo whose camera was found against today's panoramas (see
    flaneur.match.Situated): that camera, and the facades its points lie on."""
    pose = Pose(m.x, m.y, m.above_street, m.heading, m.horizon, m.fov)
    return {"photo_id": photo_id, "status": "matched", "pose": asdict(pose), "walls": m.walls}
