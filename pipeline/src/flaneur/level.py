"""How a panorama's camera leaned when it took its picture.

Panoramax's 360° pictures are taken from poles, helmets and cars, and are not quite level: by
5° (median) around the pilot, up to 13°. Read as level, every direction found in them is off
by up to that much, and points seen from two of them do not meet.

A level view cut from a level picture shows the walls' vertical edges upright. Cut from a
picture that leans, they lean too, by how much the picture leans about the direction the view
faces: in a view facing h (from the picture's centre), by about roll·cos h + pitch·sin h,
the other way. So the edges' lean in views all around gives the roll and pitch (a view
leaning otherwise, on a slanting facade or a ramp, set aside), which are undone, and what is
left measured again from the views then cut. A picture whose views do not give one lean
surely enough leans as the pictures taken near it in the same run, if they do (borrowed)."""

import hashlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from flaneur import match, register
from flaneur.panoramax import Picture
from flaneur.sources import write_json

HEADINGS = np.arange(0, 360, 30)  # views all around (degrees from the picture's centre)
FOV, SIDE = 60.0, 600  # finer than the picture: edges read to a tenth of a degree more
UPRIGHT_DEG = 15  # segments this near upright are taken for walls' vertical edges
SHORTEST_PX = 40  # and this long at least
MIDDLE = 0.2  # near the view's middle (this share of its width each side): there they stand
# upright in a level view even while it still looks up or down
MIN_EDGES = 3  # in a view, for its lean to count
MIN_VIEWS = 5  # views with a lean, for the picture's (and left once some are set aside)
ROUNDS = 2  # the second corrects what the first left: 0.2° (median of 160 pictures)
APART = 3  # a view leaning this many times further from the fit than most is set aside
NEAR_DEG = 0.5  # (or at least this far)
MOST_UNSURE_DEG = 0.7  # a lean this unsure is not kept: on 160 pictures, all those whose
# edges were left more than 1° from upright were unsure by 0.8° or more, nine in ten others
# by less than 0.4°
UNSETTLED_DEG = 1.0  # the second round still correcting more: not kept either
BORROW_M, BORROW_RANKS = 10, 3  # a picture not measured leans as those of its run this near
# and this few pictures before or after it (two at least): off by 0.4° (median), against 1.5°
# taken as level; as measured, not turned to its own heading (that did no better)
MOST_DEG = 20.0  # leaning more is not to be trusted either (the most seen: 13°)

_segments = cv2.createLineSegmentDetector()


def tilt(picture: np.ndarray) -> tuple[float, float] | None:
    """How the camera of a 360° picture (BGR) leaned: its roll and pitch (degrees, see
    match.camera_axes), or None where it shows too few walls, or they do not agree on one
    lean surely enough, or on one more than any camera's."""
    roll = pitch = 0.0
    unsure = np.inf
    for _ in range(ROUNDS):
        axes = match.camera_axes(0.0, pitch, roll)
        found = [
            (heading, lean)
            for heading in HEADINGS
            if (lean := _lean(match.perspective(picture, axes, heading, FOV, SIDE, 0.0)))
            is not None
        ]
        if len(found) < MIN_VIEWS:
            return None
        headings, leans = np.array(found).T
        across = np.column_stack([np.cos(np.radians(headings)), np.sin(np.radians(headings))])
        if (fit := _fit(across, leans)) is None:
            return None
        (still_rolled, still_raised), unsure = fit
        roll, pitch = roll - still_rolled, pitch - still_raised
    settled = np.hypot(still_rolled, still_raised) <= UNSETTLED_DEG
    if not (settled and unsure <= MOST_UNSURE_DEG and np.hypot(roll, pitch) <= MOST_DEG):
        return None
    return float(roll), float(pitch)


def _fit(across: np.ndarray, leans: np.ndarray) -> tuple[np.ndarray, float] | None:
    """The roll and pitch whose leans (`across` @ them) the views' `leans` follow, views far
    from the others set aside (None: too few left); and how unsure they are (degrees, one
    standard deviation the way they are least sure: from how far the views kept lean from
    them, and how well their headings, all around or bunched, pin them down)."""
    found, *_ = np.linalg.lstsq(across, leans, rcond=None)
    miss = np.abs(across @ found - leans)
    kept = miss <= max(APART * np.median(miss), NEAR_DEG)
    if kept.sum() < MIN_VIEWS:
        return None
    across, leans = across[kept], leans[kept]
    found, *_ = np.linalg.lstsq(across, leans, rcond=None)
    miss = across @ found - leans
    variance = (miss @ miss) / (len(miss) - len(found))
    unsure = np.sqrt(variance / np.linalg.eigvalsh(across.T @ across).min())
    return found, float(unsure)


def _lean(view: np.ndarray) -> float | None:
    """How far the vertical edges near the middle of a view lean (degrees, their tops to the
    right: positive; the median by length), or None if it shows too few."""
    grey = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    found = _segments.detect(grey)[0]
    if found is None:
        return None
    x0, y0, x1, y1 = found.reshape(-1, 4).T
    across, down = x1 - x0, y1 - y0
    length = np.hypot(across, down)
    lean = np.degrees(np.arctan2(np.where(down < 0, across, -across), np.abs(down)))
    middle = np.abs((x0 + x1) / 2 - grey.shape[1] / 2) < MIDDLE * grey.shape[1]
    edges = (np.abs(lean) < UPRIGHT_DEG) & (length >= SHORTEST_PX) & middle
    if edges.sum() < MIN_EDGES:
        return None
    lean, length = lean[edges], length[edges]
    order = np.argsort(lean)
    weight = np.cumsum(length[order])
    return float(lean[order][np.searchsorted(weight, weight[-1] / 2)])


def setup() -> str:
    """How tilts are now measured: those measured otherwise are measured again."""
    return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:8]


def kept(path: Path) -> dict[str, list[float] | None]:
    """The tilts kept in `path` (see measure), by panorama id."""
    return register.load(path).get("tilts", {})


def measure(pictures: dict[str, Path], out: Path) -> dict[str, list[float] | None]:
    """The roll and pitch of each panorama's picture (by id; None: not measured, see tilt),
    kept in `out` as they come; those already there, measured the same way, are not
    measured again."""
    now = setup()
    done = kept(out) if register.load(out).get("setup") == now else {}
    todo = [(i, path) for i, path in pictures.items() if i not in done]
    if todo:
        with ProcessPoolExecutor() as pool:
            for n, (i, found) in enumerate(pool.map(_one, todo, chunksize=8), 1):
                done[i] = found
                if n % 100 == 0:
                    write_json(out, {"setup": now, "tilts": done})
        write_json(out, {"setup": now, "tilts": done})
    return {i: done[i] for i in pictures}


def _one(job: tuple[str, Path]) -> tuple[str, list[float] | None]:
    picture_id, path = job
    picture = cv2.imread(str(path))
    found = None if picture is None else tilt(picture)
    return picture_id, None if found is None else [round(found[0], 2), round(found[1], 2)]


def summary(tilts: dict[str, list[float] | None]) -> dict[str, float]:
    leaning = [float(np.hypot(*t)) for t in tilts.values() if t is not None]
    return {
        "panoramas": len(tilts),
        "measured": len(leaning),
        "median_deg": register.median(leaning),
        "p90_deg": float(np.percentile(leaning, 90)) if leaning else 0.0,
    }


def borrowed(
    tilts: dict[str, list[float] | None], pictures: list[Picture]
) -> dict[str, list[float] | None]:
    """The tilts (see measure), each picture not measured leaning as the pictures measured in
    its run near it (see BORROW_M; the median, of two at least), where there are such."""
    runs = defaultdict(list)
    for p in pictures:
        if p.sequence and p.rank is not None and tilts.get(p.id) is not None:
            runs[p.sequence].append((p.point.x, p.point.y, p.rank, *tilts[p.id]))
    measured = {run: np.array(rows) for run, rows in runs.items()}
    found = dict(tilts)
    for p in pictures:
        if tilts.get(p.id) is not None or p.rank is None or p.sequence not in measured:
            continue
        run = measured[p.sequence]
        near = (np.hypot(run[:, 0] - p.point.x, run[:, 1] - p.point.y) <= BORROW_M) & (
            np.abs(run[:, 2] - p.rank) <= BORROW_RANKS
        )
        if near.sum() >= 2:
            found[p.id] = np.median(run[near, 3:], axis=0).tolist()
    return found


def current(path: Path) -> bool:
    """Whether the tilts kept in `path` were measured as tilts are now (or none are kept)."""
    return not path.exists() or register.load(path).get("setup") == setup()
