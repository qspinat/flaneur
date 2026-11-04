"""Line today's panoramas up with today's buildings, by their roofs and corners.

A panorama's recorded position and heading come from the phone or camera that took it: off by
several metres and up to 15-20° (seen on rue du Petit-Pont: 7 m west, 7.5 m north, 15°). The
3D walk projects each panorama onto BD TOPO's buildings from where it was taken, and old
photos are placed against them, so both must agree. The buildings' roof line and corners are
easy to compute for any position, and a roof against the sky or a building's corner is a
sharp edge in the picture: the shift and the turn that put the ones on the others are the
panorama's correction. (Finding the sky itself was tried first: white clouds broke it.)

No longer used to place panoramas (cli._placed). Checked against landmarks' tops (LiDAR), its
correction was the wrong one wherever it and COLMAP's differed, by 10 to 35°; its corrections
lie 8 m from COLMAP's (median), the phone's own position 0.9 m. BD TOPO's prisms stop at the
gutter, and the roofs rise 2.9 m above it (median, LiDAR, around Saint-Séverin): the roof line
seen from the street is not the one it was fitted to."""

import itertools
import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import httpx
import numpy as np
from shapely.geometry import Point

from flaneur.blocks import Blocks
from flaneur.gazetteer import to_degrees
from flaneur.panoramax import Picture
from flaneur.pilot import DOWNLOADS_AT_ONCE, cached_image
from flaneur.sources import write_json

COLUMNS, ROWS = 720, 360  # the panorama is read at half a degree per pixel
RANGE_M = 150  # buildings further away barely rise above the horizon
# Search: a coarse grid, then a fine one around its best (metres, degrees; turns in
# multiples of half a degree).
COARSE = (12, 2, 24, 3)  # ± shift, step, ± turn, step
COARSE_SPREAD_DEG = 2  # edges widened this much for the coarse search
CANDIDATES = 8  # the coarse search's best, each searched finely
SPREADS = np.array([0, 1, 2, 4, 8, 16, 32, 64])  # columns either way, for the coarse corners
FINE = (2, 0.5, 3, 0.5)
BETTER = 0.05  # a correction is kept when the buildings fall on edges this much stronger,
MIN_FIT = 0.15  # and at least this strong (a picture with no edges improves on nothing)
CORNERS_M = 60  # corners further away are too thin to tell
CORNER_FROM_M = 3  # a corner is read from this high above its base (cars hide the bottom)
CORNER_SAMPLES = 8  # rows read along each corner
CORNERS_FULL = 10  # corners agreeing for the corners' fit to be full


@dataclass(frozen=True)
class Edges:
    """How sharply a panorama changes (0 to 1, at half a degree per pixel): from one row to
    the next (a roof against the sky) and from one column to the next (a building's corner,
    the end of a facade where a street opens). A cloud's edge is a soft one."""

    across_rows: np.ndarray
    across_columns: np.ndarray
    # for the coarse search: the edges across columns spread by each of SPREADS columns
    # either way, and the grid step (metres, degrees) they must not be stepped over by
    spread: np.ndarray | None = None
    step: tuple[float, float] = (0.0, 0.0)

    def coarse(self, step_m: float, step_deg: float) -> "Edges":
        """What the coarse search reads: roofs spread COARSE_SPREAD_DEG either way, and each
        corner read as widely as it can move within one step of the grid (a near corner moves
        fast as the camera moves, a far one slowly)."""
        n = 2 * round(COARSE_SPREAD_DEG * COLUMNS / 360) + 1
        return Edges(
            cv2.dilate(self.across_rows, np.ones((n, 1), np.uint8)),
            self.across_columns,
            np.stack(
                [
                    cv2.dilate(self.across_columns, np.ones((1, 2 * k + 1), np.uint8))
                    for k in SPREADS
                ]
            ),
            (step_m, step_deg),
        )


def picture_edges(equirectangular: np.ndarray) -> Edges:
    small = cv2.resize(equirectangular, (COLUMNS, ROWS), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small.astype(np.float32), (3, 3), 0)

    def across(dx: int, dy: int, slack: tuple[int, int]) -> np.ndarray:
        # every colour counts: a blue sky over pale stone differs more in colour than in grey
        edges = np.abs(cv2.Sobel(small, cv2.CV_32F, dx, dy, ksize=3)).sum(axis=2)
        edges = cv2.dilate(edges, np.ones(slack, np.uint8))  # half a degree either way
        scale = float(np.percentile(edges[: ROWS // 2], 98)) or 1.0
        return np.minimum(edges / scale, 1.0)

    return Edges(across(0, 1, (3, 1)), across(1, 0, (1, 3)))


@dataclass(frozen=True)
class Correction:
    """What to add to a panorama's recorded position (metres) and heading (degrees), and how
    strong the picture's edges were where the buildings' roofs and corners fall, before and
    after (0 to 1)."""

    east: float
    north: float
    turn: float
    fit_before: float
    fit_after: float

    @property
    def better(self) -> bool:
        return self.fit_after >= max(self.fit_before + BETTER, MIN_FIT)


def register(
    edges: Edges,
    azimuth: float,
    x: float,
    y: float,
    camera_z: Callable[[float, float], float],
    blocks: Blocks,
) -> Correction:
    """The shift and turn that put the buildings' roofs and corners on the panorama's
    strongest edges (`edges` from picture_edges; the image's middle column looks towards
    `azimuth`), the camera standing at `camera_z(x, y)` wherever it is moved. The fit is the
    mean of the roofs' and the corners': where trees hide the roofs, corners still tell."""
    blocks = blocks.near(x, y, RANGE_M + COARSE[0])
    columns = np.arange(COLUMNS)
    bearings = np.radians(azimuth + (columns + 0.5) * 360 / COLUMNS - 180)

    def row_of(elevation: np.ndarray) -> np.ndarray:
        return np.clip(np.round((0.5 - elevation / np.pi) * ROWS), 0, ROWS - 1).astype(int)

    def fits(east: float, north: float, turns: np.ndarray, edges: Edges = edges) -> np.ndarray:
        """The fit for each turn: the roof line and the corners are found once around the
        horizon, and a turn is a shift along it."""
        here = (x + east, y + north, camera_z(x + east, y + north))
        shift = np.round(turns * COLUMNS / 360).astype(int)[:, None]
        roofs = blocks.skyline(*here, bearings)
        rows = np.where(np.isfinite(roofs), row_of(roofs), -1)  # -1: no building that way
        seen_rows = rows[(columns[None] + shift) % COLUMNS]
        on_roofs = np.where(
            seen_rows >= 0, edges.across_rows[np.maximum(seen_rows, 0), columns[None]], 0.0
        ).mean(axis=1)
        corners = blocks.corners(*here, metres=CORNERS_M)
        if not len(corners):
            return on_roofs / 2
        bearing, distance, base, top = corners.T
        heights = (
            base[:, None]
            + CORNER_FROM_M
            + np.linspace(0, 1, CORNER_SAMPLES)[None]
            * (np.maximum(top - base - CORNER_FROM_M, 0)[:, None])
        )
        corner_rows = row_of(np.arctan2(heights - here[2], distance[:, None]))
        offset = (np.degrees(bearing) - azimuth + 180) * COLUMNS / 360 - 0.5
        corner_columns = np.round(offset[None] - shift).astype(int) % COLUMNS
        if edges.spread is None:
            on_corners = edges.across_columns[corner_rows[None], corner_columns[:, :, None]]
        else:  # half a step either way, in metres and in turn
            step_m, step_deg = edges.step
            moves = np.degrees(np.arctan2(step_m * 0.75, distance)) + step_deg / 2
            level = np.searchsorted(SPREADS, moves * COLUMNS / 360).clip(0, len(SPREADS) - 1)
            on_corners = edges.spread[
                level[None, :, None], corner_rows[None], corner_columns[:, :, None]
            ]
        # each corner counts for (on an edge) or against (on nothing): more corners agreeing
        # is better, and corners where the picture shows none cost something
        votes = (2 * on_corners.mean(axis=2) - 1).sum(axis=1)
        return (on_roofs + np.clip(votes / CORNERS_FULL, -1, 1)) / 2

    def search(east: float, north: float, turn: float, grid: tuple[float, ...], edges: Edges):
        shift, shift_step, half_turn, turn_step = grid
        shifts = np.arange(-shift, shift + shift_step / 2, shift_step)
        turns = turn + np.arange(-half_turn, half_turn + turn_step / 2, turn_step)
        return sorted(
            (
                (float(fit), east + e, north + n, float(t))
                for e, n in itertools.product(shifts, shifts)
                for fit, t in zip(fits(east + e, north + n, turns, edges), turns, strict=True)
            ),
            reverse=True,
        )

    def distinct(found: list[tuple[float, float, float, float]]):
        """The best candidates, at least one coarse step apart."""
        kept: list[tuple[float, float, float, float]] = []
        for c in found:
            if all(
                max(abs(c[1] - k[1]), abs(c[2] - k[2])) > COARSE[1] or abs(c[3] - k[3]) > COARSE[3]
                for k in kept
            ):
                kept.append(c)
            if len(kept) == CANDIDATES:
                break
        return kept

    before = float(fits(0.0, 0.0, np.zeros(1))[0])
    # the widened coarse fit ranks candidates only roughly: each of the best is searched
    # finely, on the sharp edges
    coarse = distinct(search(0.0, 0.0, 0.0, COARSE, edges.coarse(COARSE[1], COARSE[3])))
    best, east, north, turn = max(search(*c[1:], FINE, edges)[0] for c in coarse)
    return Correction(
        round(float(east), 2),
        round(float(north), 2),
        round(turn, 2),
        round(before, 4),
        round(best, 4),
    )


_WORLD: tuple[Blocks, Callable[[float, float], float]] | None = None  # sent once per worker


def _keep(blocks: Blocks, camera_z: Callable[[float, float], float]) -> None:
    global _WORLD
    _WORLD = blocks, camera_z


FAILED = "failed"  # the picture could not be read: not recorded, tried again next time


def _one(job: tuple[str, Path, float, float, float]) -> tuple[str, dict | str]:
    picture_id, path, azimuth, x, y = job
    # decoded at half its 2048 px: it is read at 720 columns
    equirectangular = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
    if equirectangular is None or _WORLD is None or not is_sphere(*equirectangular.shape[:2]):
        return picture_id, FAILED
    blocks, camera_z = _WORLD
    found = register(picture_edges(equirectangular), azimuth, x, y, camera_z, blocks)
    return picture_id, {**asdict(found), "recorded": recorded(x, y, azimuth)}


def load(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text()) if path.exists() else {}


def correct(
    pictures: list[Picture],
    camera_z: Callable[[float, float], float],
    blocks: Blocks,
    client: httpx.Client,
    cache_dir: Path,
    out: Path,
) -> dict[str, dict]:
    """Line each panorama up (its camera at `camera_z(x, y)`, a picklable function, in the
    frame of `blocks`); results are kept in `out` as they come, and a panorama already there
    for the same recorded position and heading is not lined up again."""
    done = load(out)
    todo = [p for p in pictures if not still(done.get(p.id), p)]

    def fetch(p: Picture) -> tuple[str, Path, float, float, float] | None:
        path = cache_dir / f"{p.id}.jpg"
        try:
            cached_image(client, p.sd_url, path)
        except (httpx.HTTPError, OSError):
            return None  # asked again next time
        return p.id, path, p.azimuth or 0.0, p.point.x, p.point.y

    with ThreadPoolExecutor(DOWNLOADS_AT_ONCE) as downloads:
        jobs = [job for job in downloads.map(fetch, todo) if job]
    with ProcessPoolExecutor(initializer=_keep, initargs=(blocks, camera_z)) as pool:
        for n, (picture_id, found) in enumerate(pool.map(_one, jobs, chunksize=4), 1):
            if found != FAILED:
                done[picture_id] = found
            if n % 50 == 0:
                write_json(out, done)
    write_json(out, done)
    return done


def recorded(x: float, y: float, azimuth: float | None) -> list[float]:
    """A panorama's recorded position and heading, as kept beside a correction found for it
    (see still)."""
    return [round(x, 2), round(y, 2), round(azimuth or 0.0, 2)]


def is_sphere(height: int, width: int) -> bool:
    """Whether a picture is a whole 360° sphere (2:1); a few are cropped at the top or bottom,
    and lining them up as spheres would mean nothing."""
    return width == 2 * height


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def still(entry: dict | None, p: Picture) -> bool:
    """Whether a stored result was found for this panorama as it is recorded now."""
    if not entry:
        return False
    x, y, azimuth = entry["recorded"]
    return (
        abs(x - p.point.x) < 0.05
        and abs(y - p.point.y) < 0.05
        and abs(azimuth - (p.azimuth or 0.0)) < 0.05
    )


def kept(corrections: dict[str, dict], pictures: list[Picture]) -> dict[str, Correction]:
    """The corrections worth applying to these panoramas: found for them as they are
    recorded now, with a skyline that meets the buildings better."""
    fields = Correction.__dataclass_fields__
    found = {
        p.id: Correction(**{k: corrections[p.id][k] for k in fields})
        for p in pictures
        if still(corrections.get(p.id), p)
    }
    return {i: c for i, c in found.items() if c.better}


def applied(pictures: list[Picture], fixes: dict[str, tuple[float, float, float]]) -> list[Picture]:
    """The panoramas, each moved and turned by its fix (east, north, turn), if it has one."""
    moved = []
    for p in pictures:
        fix = fixes.get(p.id)
        if fix is None:
            moved.append(p)
            continue
        east, north, turn = fix
        there = to_degrees(Point(p.point.x + east, p.point.y + north))
        moved.append(
            replace(p, lon=there.x, lat=there.y, azimuth=((p.azimuth or 0.0) + turn) % 360)
        )
    return moved


def summary(corrections: dict[str, dict], pictures: list[Picture]) -> dict[str, float]:
    fixes = list(kept(corrections, pictures).values())
    current = [corrections[p.id] for p in pictures if still(corrections.get(p.id), p)]
    return {
        "panoramas": len(current),
        "kept": len(fixes),
        "median_shift_m": median([np.hypot(c.east, c.north) for c in fixes]),
        "median_turn_deg": median([abs(c.turn) for c in fixes]),
        "fit_before": median([c.fit_before for c in fixes]),
        "fit_after": median([c.fit_after for c in fixes]),
    }
