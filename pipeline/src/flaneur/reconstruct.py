"""Place today's panoramas with each other, by Structure-from-Motion (COLMAP), and tie them to
the map.

A panorama's recorded position and heading come from a phone: off by metres and by up to 35°
in these streets, much the same from one picture of a capture run to the next (it drifts
along the run, by up to tens of degrees). COLMAP matches the windows, doors and cornices the
panoramas share and solves all their positions and the facades' points
together (pycolmap's panorama pipeline: each 360° picture cut into a rig of views). Its
reconstruction has its own frame: it is levelled by the panoramas' own "up" (they are levelled
pictures), then fitted to the recorded positions as a whole (one scale, turn and shift, robust
to the few far off). Tested on rue du Petit-Pont: 298 of 305 panoramas placed, two sessions
turned 11° and 34° brought onto today's buildings.

The area is done in overlapping tiles of a few hundred panoramas, each the size of that test.
COLMAP's own poses are kept for each tile (hours to compute); the fit to the map is redone
from them on every run (seconds), so that its rules can be tuned on what COLMAP found."""

import hashlib
import json
import logging
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from flaneur import register
from flaneur.gazetteer import angle
from flaneur.panoramax import Picture
from flaneur.sources import write_json

TILE_M = 250  # tiles on a fixed grid of Lambert-93 metres: a new panorama changes one tile
OVERLAP_M = 50  # each tile is reconstructed with this margin, for the panoramas at its edge
MARGIN_SHARE = 0.2  # and at most this share of its side: a split tile's margin shrinks too
MOST_WITH_MARGIN = 750  # and narrower still where it reaches into a crowd
MOST = 500  # panoramas in a tile's core: a crowded tile is split (COLMAP was tested at 300)
SMALLEST_M = 60
INLIER_M = 6  # GPS this close to the fitted reconstruction counts in the fit
FAR_OFF_M = 30  # a panorama this far from its GPS after the fit: placed wrong, not used
MIN_AGREEING = 0.5  # share of a tile's panoramas agreeing with its fit, for the tile to count
# A panorama COLMAP did not place moves as its run's nearest placed ones: this many pictures
# away at most, and recorded this near it. Measured on those placed, each left out (90%
# within): a lender within 10 m, up to 5 pictures away, 0.5 to 1.4 m and 1 to 4°; 10 to 15 m
# away, 4.8 m and 27° (as recorded: 5.6 m and 21°). A worse lender costs more than it gains:
# borrowed 6 to 8 pictures away, one run no longer met another left as recorded, and four
# photos placed through both were lost.
BORROW_RANKS = 5
BORROW_M = 10.0
FIT_TRIES = 2000
OPTIONS = "spatial global perspective-overlapping"  # COLMAP's, part of a kept result's name
POSES = ("centres", "forwards", "ups")


@dataclass(frozen=True)
class Tile:
    core: np.ndarray  # the panoramas it places (indices)
    members: np.ndarray  # those it is reconstructed with: the core and a margin


def tiles(
    points: np.ndarray,
    side: float = TILE_M,
    overlap: float = OVERLAP_M,
    most: int = MOST,
    smallest: float = SMALLEST_M,
) -> list[Tile]:
    """Square tiles over the points (metres), on a grid of multiples of `side`, each panorama
    in one tile's core; a tile with more than `most` panoramas in its core is split in four
    (down to `smallest`)."""
    found: list[Tile] = []

    def cover(corner: np.ndarray, size: float, inside: np.ndarray) -> None:
        if not len(inside):
            return
        half = size / 2
        if len(inside) > most and half >= smallest:
            quarter = np.floor((points[inside] - corner) / half).clip(0, 1).astype(int)
            for key in ((0, 0), (0, 1), (1, 0), (1, 1)):
                cover(corner + np.array(key) * half, half, inside[(quarter == key).all(axis=1)])
            return
        margin = min(overlap, MARGIN_SHARE * size)
        while True:  # a margin reaching into a crowd is narrowed
            near = ((points >= corner - margin) & (points < corner + size + margin)).all(axis=1)
            if near.sum() <= MOST_WITH_MARGIN or margin < 1:
                break
            margin /= 2
        found.append(Tile(inside, np.flatnonzero(near)))

    cell = np.floor(points / side).astype(int)
    for key in np.unique(cell, axis=0):
        cover(key * side, side, np.flatnonzero((cell == key).all(axis=1)))
    return found


def axes(cam_from_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The world directions a COLMAP panorama faces (the picture's centre) and its up: its
    camera has x to the right, y down and z towards the centre of the picture."""
    world_from_cam = cam_from_world.T
    return world_from_cam @ [0, 0, 1.0], -(world_from_cam @ [0, 1.0, 0])


@dataclass(frozen=True)
class Fit:
    """A reconstruction's frame to the map's: levelled, then scaled, turned and shifted."""

    level: np.ndarray  # 3 × 3: the model's axes to (any, any, up)
    scale: float
    turn: np.ndarray  # 2 × 2
    shift: np.ndarray  # 2
    agreeing: np.ndarray  # per panorama: within INLIER_M of its GPS
    far_off: np.ndarray  # per panorama: too far from its GPS after the fit

    def _flat(self, v: np.ndarray) -> np.ndarray:
        return (v @ self.level.T)[:, :2] @ self.turn.T

    def place(self, model: np.ndarray) -> np.ndarray:
        """Model points to map metres (east, north)."""
        return self.scale * self._flat(model) + self.shift

    def heading(self, forward: np.ndarray) -> np.ndarray:
        """Model directions to compass bearings (degrees)."""
        flat = self._flat(forward)
        return np.degrees(np.arctan2(flat[:, 0], flat[:, 1])) % 360


def _similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """The scale, rotation and shift taking points `a` onto `b` best (Umeyama, no mirror)."""
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    u, s, vt = np.linalg.svd((b - mb).T @ (a - ma))
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ d @ vt
    scale = float((s * np.diag(d)).sum() / max(((a - ma) ** 2).sum(), 1e-12))
    return scale, rotation, mb - scale * rotation @ ma


def fit_to_gps(model: np.ndarray, up: np.ndarray, gps: np.ndarray, seed: int = 0) -> Fit:
    """Level the reconstruction by the panoramas' mean up, then fit it to their recorded
    positions: RANSAC over pairs, then least squares on those within INLIER_M."""
    z = up.mean(axis=0)
    z /= np.linalg.norm(z)
    reference = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(reference, z)
    x /= np.linalg.norm(x)
    level = np.stack([x, np.cross(z, x), z])
    flat = (model @ level.T)[:, :2]

    def gap(scale: float, turn: np.ndarray, shift: np.ndarray) -> np.ndarray:
        return np.linalg.norm(scale * flat @ turn.T + shift - gps, axis=1)

    rng = np.random.default_rng(seed)
    best, most = np.zeros(len(flat), bool), -1
    for _ in range(FIT_TRIES):
        pair = rng.choice(len(flat), 2, replace=False)
        scale, turn, shift = _similarity(flat[pair], gps[pair])
        if not np.isfinite(scale) or scale < 1e-9:  # the two at one spot: no scale to tell
            continue
        inside = gap(scale, turn, shift) < INLIER_M
        if inside.sum() > most:
            best, most = inside, int(inside.sum())
    if most < 3:  # nothing agrees: a fit that fails every panorama
        nothing = np.zeros(len(flat), bool)
        return Fit(level, 1.0, np.eye(2), np.zeros(2), nothing, ~nothing)
    scale, turn, shift = _similarity(flat[best], gps[best])
    after = gap(scale, turn, shift)
    return Fit(level, scale, turn, shift, after < INLIER_M, after > FAR_OFF_M)


def usable(pictures: list[Picture], cache_dir: Path) -> list[Picture]:
    """The panoramas whose picture is downloaded and is a whole sphere."""
    found = []
    for p in pictures:
        try:
            with Image.open(cache_dir / f"{p.id}.jpg") as image:
                if register.is_sphere(image.height, image.width):
                    found.append(p)
        except OSError:  # not downloaded, or not a picture: left out
            continue
    return found


@dataclass(frozen=True)
class Poses:
    """COLMAP's placement of a tile's panoramas, in its own frame."""

    ids: list[str]
    centres: np.ndarray
    forwards: np.ndarray
    ups: np.ndarray


NO_POSES = Poses([], np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)))


def reconstruct_tile(pictures: list[Picture], cache_dir: Path, work: Path) -> Poses:
    """COLMAP's largest reconstruction of these panoramas (none when nothing was placed)."""
    # imported here only: pycolmap's OpenMP and torch's (match.Matcher) abort a process that
    # loads both, without a word
    if "torch" in sys.modules:
        raise RuntimeError("torch est chargé : pycolmap ne peut pas l'être avec lui")
    import pycolmap
    from pycolmap.panorama import Mapper, Matcher, PanoramaReconstructionOptions

    if len(pictures) < 3:
        return NO_POSES
    panos = work / "panos"
    shutil.rmtree(work, ignore_errors=True)
    panos.mkdir(parents=True)
    for p in pictures:
        (panos / f"{p.id}.jpg").symlink_to((cache_dir / f"{p.id}.jpg").resolve())
    models = pycolmap.panorama.reconstruct(
        panos,
        work / "out",
        PanoramaReconstructionOptions(
            matcher=Matcher.SPATIAL, mapper=Mapper.GLOBAL, use_gpu=False, show_progress=False
        ),
    )
    if not models:
        return NO_POSES
    # the model placing the most panoramas (each is one image there, as a 360° camera)
    rec = max(
        (pycolmap.Reconstruction(m) for m in (work / "out" / "sparse_equirectangular").glob("*")),
        key=lambda r: r.num_reg_images(),
    )
    placed = [
        (Path(i.name).stem, i.projection_center(), *axes(i.cam_from_world().rotation.matrix()))
        for i in rec.images.values()
        if i.has_pose
    ]
    if not placed:
        return NO_POSES
    ids, centres, forwards, ups = zip(*placed, strict=True)
    return Poses(list(ids), np.array(centres), np.array(forwards), np.array(ups))


def _cached_poses(members: list[Picture], cache_dir: Path, work_dir: Path) -> Poses:
    """A tile's COLMAP poses, computed once for its exact set of panoramas."""
    named = " ".join([OPTIONS, *sorted(p.id for p in members)])
    key = hashlib.sha1(named.encode()).hexdigest()[:16]
    done = work_dir / f"poses_{key}.json"
    if done.exists():
        kept = json.loads(done.read_text())
        return Poses(kept["ids"], *(np.array(kept[k], float).reshape(-1, 3) for k in POSES))
    logging.info("COLMAP: %d panoramas", len(members))
    poses = reconstruct_tile(members, cache_dir, work_dir / key)
    shutil.rmtree(work_dir / key, ignore_errors=True)  # gigabytes of views
    if not poses.ids:  # nothing placed: tried again next time, not kept as an answer
        return poses
    write_json(
        done, {"ids": poses.ids, **{k: np.round(getattr(poses, k), 6).tolist() for k in POSES}}
    )
    return poses


def _fix(p: Picture, x: float, y: float, heading: float) -> dict:
    turn = angle(p.azimuth or 0.0, heading)
    return {
        "fix": [round(x - p.point.x, 2), round(y - p.point.y, 2), round(turn, 2)],
        "recorded": register.recorded(p.point.x, p.point.y, p.azimuth),
    }


def run(
    pictures: list[Picture], cache_dir: Path, work_dir: Path, out: Path
) -> tuple[dict[str, dict], int]:
    """Each panorama's fix (east, north, turn) where COLMAP placed it, tile by tile, and how
    many panoramas could be tried. A panorama is placed by the tile whose core holds it, else
    by a neighbour that has it in its margin. The seams are logged: how far apart two tiles
    place the panoramas they share."""
    pictures = usable(pictures, cache_dir)
    by_id = {p.id: p for p in pictures}
    points = np.array([(p.point.x, p.point.y) for p in pictures])
    placed: dict[str, list[tuple[float, float]]] = {}
    fixes: dict[str, dict] = {}
    spare: dict[str, dict] = {}  # placed in a tile's margin only
    for tile in tiles(points):
        poses = _cached_poses([pictures[i] for i in tile.members], cache_dir, work_dir)
        if len(poses.ids) < 3:
            continue
        gps = np.array([(by_id[i].point.x, by_id[i].point.y) for i in poses.ids])
        fit = fit_to_gps(poses.centres, poses.ups, gps)
        if fit.agreeing.mean() < MIN_AGREEING:
            logging.info("a tile left out: %.0f%% agree with its fit", 100 * fit.agreeing.mean())
            continue
        core = {pictures[i].id for i in tile.core}
        where = zip(poses.ids, fit.place(poses.centres), fit.heading(poses.forwards), strict=True)
        for (i, (x, y), heading), far in zip(where, fit.far_off, strict=True):
            if far:
                continue
            placed.setdefault(i, []).append((x, y))
            (fixes if i in core else spare).setdefault(i, _fix(by_id[i], x, y, heading))
    seams = [np.hypot(*np.subtract(*twice[:2])) for twice in placed.values() if len(twice) > 1]
    if seams:
        logging.info(
            "seams: %d panoramas placed by two tiles, %.1f m apart (median), %.1f m (90%%)",
            len(seams),
            np.median(seams),
            np.percentile(seams, 90),
        )
    fixes = spare | fixes
    write_json(out, fixes)
    return fixes, len(pictures)


def borrowed(
    fixes: dict[str, tuple[float, float, float]], pictures: list[Picture]
) -> dict[str, tuple[float, float, float]]:
    """The fixes (east, north, turn) of the panoramas placed, and of each panorama not placed,
    its run's nearest placed panoramas' (the median of those as near, within BORROW_RANKS and
    BORROW_M): the phone's error is much the same from one picture of a run to the next."""
    at = defaultdict(
        list
    )  # (run, rank): where the panoramas placed there were recorded, and their fixes
    for p in pictures:
        if p.id in fixes and p.sequence and p.rank is not None:
            at[p.sequence, p.rank].append((p.point, fixes[p.id]))
    found = dict(fixes)
    for p in pictures:
        if p.id in fixes or p.rank is None:
            continue
        for gap in range(1, BORROW_RANKS + 1):
            near = at.get((p.sequence, p.rank - gap), []) + at.get((p.sequence, p.rank + gap), [])
            if as_near := [fix for point, fix in near if point.distance(p.point) <= BORROW_M]:
                found[p.id] = _median_fix(np.array(as_near))
                break
    return found


def _median_fix(fixes: np.ndarray) -> tuple[float, float, float]:
    """The median of fixes (east, north, turn), the turns taken around the first's."""
    turns = fixes[0, 2] + (fixes[:, 2] - fixes[0, 2] + 180) % 360 - 180
    east, north = np.median(fixes[:, :2], axis=0)
    return float(east), float(north), float((np.median(turns) + 180) % 360 - 180)


def summary(fixes: dict[str, dict]) -> dict[str, float]:
    return {
        "placed": len(fixes),
        "median_shift_m": register.median([float(np.hypot(*f["fix"][:2])) for f in fixes.values()]),
        "median_turn_deg": register.median([abs(f["fix"][2]) for f in fixes.values()]),
    }
