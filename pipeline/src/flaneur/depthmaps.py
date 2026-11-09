"""How far each pixel of a panorama's views stands, as a model trained on single pictures sees it
(MoGe-2), for the points clicked where today's walls say nothing; made live, the first time a
panorama needs it (0.15 s a view), kept in memory, never on disk.

Its own metres are no use here: each view is scaled to the walls it sees. Measured on the walls'
test (a held-out neighbour's view brought through the depth, 81 cases, on building pixels): the
walls alone 0.387; the walls, else MoGe-2 on what stands (buildings, ground, trees, street
furniture), 0.441, 80 cases better and none worse (Depth Anything 3: 0.436; Depth Pro: 0.425).
On what passes (a car, a person) its depth is the thing's, not the street's: the street behind is
rebuilt from what stands around it (see behind). MoGe-2 runs in half precision at its lowest
resolution: as good (0.441), ten times faster than at its finest in single precision. Its view
kept at full size: kept at a quarter, one pixel in three along a facade's edge was off by 10 %.

MoGe-2 is Microsoft's (MIT), its weights (Ruicheng/moge-2-vitl-normal) fetched once, kept outside
the repository; loaded in safe mode (weights only)."""

import functools
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable

import cv2
import numpy as np

from flaneur import behind, facades, match, segment

MODEL = "Ruicheng/moge-2-vitl-normal"
LEVEL = 0  # MoGe-2's resolution level: its lowest, as good here as its finest
NEAR_M = 30.0  # a wall this near scales a view (farther, its own depth is less sure)
FAR_M = 150.0  # scaled, farther than this: the sky (models put it hundreds of metres off)
SCALED_ON = 40  # cells across each view where the walls are cast, for its scale
LEAST_CELLS = 25  # of those on walls, for a view to have a scale of its own
LEAST_VIEWS = 3  # views with a scale, for those without to take theirs (the median)
KEPT = 4  # panoramas' maps kept in memory (20 MB each)
GROWN = 2  # pixels a passing thing's outline is grown by (a model blurs depth at its edges)


def scales(raw: np.ndarray, walls: np.ndarray) -> np.ndarray | None:
    """Each view's scale (views × side × side maps: the model's distances `raw`, the walls'
    `walls`, NaN where none within NEAR_M): the median of walls over model on the cells a wall
    meets; a view with too few takes the others' median; None when too few views have one."""
    found = np.full(len(raw), np.nan)
    for v, (mine, theirs) in enumerate(zip(raw, walls, strict=True)):
        on = np.isfinite(theirs) & np.isfinite(mine) & (mine > 0)
        if on.sum() >= LEAST_CELLS:
            found[v] = np.median(theirs[on] / mine[on])
    if np.isfinite(found).sum() < LEAST_VIEWS:
        return None
    found[~np.isfinite(found)] = np.median(found[np.isfinite(found)])
    return found


def filled(maps: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """How far along each world ray (unit rows, from the panorama), by the views' maps (see
    grid_rays); NaN for a ray in no view."""
    return segment.at_rays(maps, rays, np.nan)


@functools.cache
def grid_rays(side: int) -> np.ndarray:
    """The world directions of each cell of the views' maps (views × side × side × 3)."""
    pixels = match.pixels_of((side, side))
    return np.stack(
        [match.view_directions(pixels, h, side=side) for h in match.VIEW_HEADINGS]
    ).reshape(len(match.VIEW_HEADINGS), side, side, 3)


class Filler:
    """How far along rays from a scene's panorama, by a model's maps of its views scaled to the
    scene's walls (see scales), taken only where the views' `parts` (see segment.Part, cut as the
    maps are; none: the model everywhere) show what stands; behind what passes, the street rebuilt
    around it (see behind), written into a view's map the first time a ray falls there. Scaled the
    first time it is asked (most clicks meet a wall); NaN everywhere when too few walls are seen
    to scale it."""

    def __init__(self, scene: facades.Scene, maps: np.ndarray, parts: np.ndarray | None = None):
        if parts is not None and parts.shape != maps.shape:
            raise ValueError("les classes et la profondeur ne sont pas coupées pareil")
        self.scene, self.maps, self.parts = scene, maps, parts
        self.lock = threading.Lock()
        self.scaled: np.ndarray | None = None
        self.passing: np.ndarray | None = None  # per pixel: its view + 1 where something passes
        self.left: set[int] = set()  # the views whose passing things have no street behind yet
        if parts is not None:
            moving = parts == segment.Part.MOVING
            views = np.arange(1, len(parts) + 1, dtype=np.int16)[:, None, None]
            self.passing = np.where(moving, views, np.int16(0))
            self.left = {int(v) for v in np.flatnonzero(moving.any(axis=(1, 2)))}

    def __call__(self, rays: np.ndarray) -> np.ndarray:
        with self.lock:
            if self.scaled is None:
                self.scaled, self.maps = self._scaled(), None  # the model's own: read once
        if not len(self.scaled):
            return np.full(len(rays), np.nan)
        if self.left:
            met = segment.at_rays(self.passing, rays, 0)
            for view in {int(v) - 1 for v in np.unique(met[met > 0])} & self.left:
                street = self._street_behind(view)  # outside the lock: the same each time
                with self.lock:
                    if view in self.left:
                        rebuilt = np.isfinite(street)
                        self.scaled[view][rebuilt] = street[rebuilt]
                        self.left.discard(view)
        return filled(self.scaled, rays)

    def _scaled(self) -> np.ndarray:
        side = self.maps.shape[1]
        step = max(1, side // SCALED_ON)
        cells = slice(step // 2, None, step)
        coarse = self.maps[:, cells, cells]
        grid = grid_rays(coarse.shape[1])
        metres, met = facades.cast(self.scene, grid.reshape(-1, 3))
        walls = np.where((met >= 0) & (metres < NEAR_M), metres, np.nan)
        scale = scales(coarse, walls.reshape(coarse.shape))
        if scale is None:
            return np.empty(0, np.float32)
        scaled = self.maps * scale.astype(np.float32)[:, None, None]
        scaled[scaled > FAR_M] = np.nan
        if self.parts is not None:
            scaled[~np.isin(self.parts, segment.STILL)] = np.nan
        return scaled

    def _street_behind(self, view: int) -> np.ndarray:
        """The street behind each thing passing in a view, one outline at a time, each rebuilt
        from what stands around it (side × side, NaN elsewhere)."""
        side = self.parts.shape[1]
        parts = self.parts[view]
        grown = np.ones((2 * GROWN + 1, 2 * GROWN + 1), np.uint8)
        moving = cv2.dilate((parts == segment.Part.MOVING).astype(np.uint8), grown)
        count, outlines, stats, _ = cv2.connectedComponentsWithStats(moving)
        rays = match.view_directions(
            match.pixels_of((side, side)), match.VIEW_HEADINGS[view], side=side
        ).reshape(side, side, 3)
        built, ground = parts == segment.Part.BUILT, parts == segment.Part.GROUND
        origin = np.array(self.scene.origin)
        margin = round(side * max(behind.RING, behind.GROUND_RING, behind.ABOVE))
        street = np.full((side, side), np.nan, np.float32)
        for outline, (x, y, w, h, _) in enumerate(stats[1:count], 1):
            crop = (
                slice(max(0, y - margin), min(side, y + h + margin)),
                slice(max(0, x - margin), min(side, x + w + margin)),
            )
            hole = outlines[crop] == outline
            around = rays[crop].reshape(-1, 3)
            metres, met = facades.cast(self.scene, around)  # the walls first, else the model
            known = np.where(met >= 0, metres, self.scaled[view][crop].ravel())
            known = np.where(moving[crop].ravel() > 0, np.nan, known).reshape(hole.shape)
            found = behind.filled(
                hole,
                known,
                built[crop],
                ground[crop],
                around,
                origin,
                self.scene.ground,
                side,
            )
            street[crop][hole] = found[hole]
        street[street > FAR_M] = np.nan  # the sky, as for the model
        return street


def fills(
    depth: Callable[[str], np.ndarray | None], parts: Callable[[str], np.ndarray | None]
) -> Callable[[facades.Scene], Filler | None]:
    """A scene's filler (see Filler), from its panorama's maps by the model (`depth`) and what
    its views show (`parts`); None without maps."""

    def fill(scene: facades.Scene) -> Filler | None:
        maps = depth(scene.panorama)
        return None if maps is None else Filler(scene, maps, parts(scene.panorama))

    return fill


def _along(side: int) -> np.ndarray:
    """How much longer each pixel's ray is than its depth along the view's axis."""
    f = match.focal(match.VIEW_FOV, side)
    c = np.arange(side) + 0.5 - side / 2
    return np.sqrt(f**2 + c[None, :] ** 2 + c[:, None] ** 2) / f


class Depthmaps:
    """The distance along each pixel of a panorama's views (views × side × side), made live by
    the model (`estimate`: a view, BGR, to its depth along the axis; MoGe-2 by default, loaded the
    first time, `lock` held while it runs) and kept in memory for the last few panoramas. A model
    that fails (not fetched, memory short) is said once and not asked again: the walls alone."""

    def __init__(
        self,
        lock: threading.Lock,
        estimate: Callable[[np.ndarray], np.ndarray] | None = None,
        side: int = match.VIEW_SIDE,
    ):
        self.lock, self.side = lock, side
        self.estimate = estimate or self._moge
        self.loaded = None
        self.broken: Exception | None = None
        self.kept: OrderedDict[str, np.ndarray] = OrderedDict()  # panorama id: maps

    def maps(self, panorama: match.Panorama) -> np.ndarray | None:
        """The panorama's maps (metres along each ray, as the model says), made the first time;
        None if its picture cannot be read, or the model failed."""
        if self.broken is not None:
            return None
        found = self._kept(panorama.id)
        if found is not None:
            return found
        picture = cv2.imread(str(panorama.path))
        if picture is None:
            return None
        views = [
            match.perspective(picture, panorama.axes, h, side=self.side)
            for h in match.VIEW_HEADINGS
        ]
        along = _along(self.side)
        with self.lock:  # the model only: one thing at a time on the graphics chip
            found = self._kept(panorama.id)  # made meanwhile, by the request that held the lock
            if found is None and self.broken is None:
                try:
                    found = np.array([self.estimate(v) * along for v in views], np.float32)
                except Exception as error:  # the walls alone, said once
                    self.broken = error
                    print(f"profondeur du modèle impossible : {error}", file=sys.stderr)
                    return None
                self.kept[panorama.id] = found
                while len(self.kept) > KEPT:
                    self.kept.popitem(last=False)
        return found

    def _kept(self, panorama: str) -> np.ndarray | None:
        found = self.kept.get(panorama)
        if found is not None:
            self.kept.move_to_end(panorama)
        return found

    def _moge(self, view: np.ndarray) -> np.ndarray:
        """MoGe-2's depth along the axis of a view (BGR, square), at its size."""
        if self.loaded is None:
            if "pycolmap" in sys.modules:  # their OpenMP runtimes abort one process (see match)
                raise RuntimeError("MoGe-2 et pycolmap ne se chargent pas ensemble")
            import torch
            from moge.model.v2 import MoGeModel

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            half = torch.float16 if device == "mps" else torch.float32
            model = MoGeModel.from_pretrained(MODEL).to(device).eval().to(half)
            self.loaded = (model, device, half)
        import torch

        model, device, half = self.loaded
        image = torch.from_numpy(segment.rgb(view)).permute(2, 0, 1).to(device, half) / 255
        with torch.inference_mode():
            found = model.infer(
                image,
                fov_x=match.VIEW_FOV,
                apply_mask=False,
                use_fp16=False,
                resolution_level=LEVEL,
            )
        return found["depth"].float().cpu().numpy()
