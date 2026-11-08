"""How far each pixel of a panorama's views stands, as a model trained on single pictures sees it
(Depth Pro), for the points clicked where today's walls say nothing. Its own metres are no use
here (measured 2026-11-08: off by about half, from view to view); each view is scaled to the walls
it sees. On the walls' test (a held-out neighbour's view brought through the depth, 27 cases), the
walls with Depth Pro's depth where they meet no wall: 0.233 → 0.273, 24 better and 2 worse (Depth
Anything V2: 0.256; the street plane kept: 0.259). Made ahead, like the segmentation (flaneur
profondeur): the model takes more than 5 GB, never in the page.

Depth Pro is Apple's (apple/DepthPro-hf, through transformers), its weights under Apple's research
licence, for non-commercial use as Flâneur is; fetched once, kept outside the repository."""

import functools
import threading
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from flaneur import facades, match, segment
from flaneur.sources import write_atomic

MODEL = "apple/DepthPro-hf"
SIDE = 160  # each view's depth kept this many cells across (its view is cut at match.VIEW_SIDE)
NEAR_M = 30.0  # a wall this near scales a view (farther, its own depth is less sure)
SCALED_ON = 40  # cells across each view where the walls are cast, for its scale
LEAST_CELLS = 25  # of those on walls, for a view to have a scale of its own
LEAST_VIEWS = 3  # views with a scale, for those without to take theirs (the median)
KEPT = 16  # panoramas' maps kept in memory


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
def grid_rays(side: int = SIDE) -> np.ndarray:
    """The world directions of each cell of the views' maps (views × side × side × 3)."""
    pixels = match.pixels_of((side, side))
    return np.stack(
        [match.view_directions(pixels, h, side=side) for h in match.VIEW_HEADINGS]
    ).reshape(len(match.VIEW_HEADINGS), side, side, 3)


class Filler:
    """How far along rays from a scene's panorama, by a model's maps of its views scaled to the
    scene's walls (see scales); scaled the first time it is asked (most clicks meet a wall), and
    NaN everywhere when too few walls are seen to scale it."""

    def __init__(self, scene: facades.Scene, maps: np.ndarray):
        self.scene, self.maps = scene, maps
        self.lock = threading.Lock()
        self.scaled: np.ndarray | None = None

    def __call__(self, rays: np.ndarray) -> np.ndarray:
        with self.lock:
            if self.scaled is None:
                self.scaled = self._scaled()
        if not len(self.scaled):
            return np.full(len(rays), np.nan)
        return filled(self.scaled, rays)

    def _scaled(self) -> np.ndarray:
        step = max(1, self.maps.shape[1] // SCALED_ON)
        cells = slice(step // 2, None, step)
        coarse = self.maps[:, cells, cells]
        grid = grid_rays(self.maps.shape[1])[:, cells, cells]
        metres, met = facades.cast(self.scene, grid.reshape(-1, 3))
        walls = np.where((met >= 0) & (metres < NEAR_M), metres, np.nan)
        scale = scales(coarse, walls.reshape(coarse.shape))
        if scale is None:
            return np.empty(0, np.float32)
        return (self.maps * scale[:, None, None]).astype(np.float32)


def _along(side: int) -> np.ndarray:
    """How much longer each pixel's ray is than its depth along the view's axis."""
    f = match.focal(match.VIEW_FOV, side)
    c = np.arange(side) + 0.5 - side / 2
    return np.sqrt(f**2 + c[None, :] ** 2 + c[:, None] ** 2) / f


class Depthmaps:
    """The distance along each cell of a panorama's views (see grid_rays), made once by the
    model (`estimate`: a view, BGR, to its depth along the axis; Depth Pro by default) and kept
    in `folder`, named as the panorama is levelled (its views change with it). Without `make`,
    only those kept are read."""

    def __init__(
        self,
        folder: Path,
        lock: threading.Lock,
        name: Callable[[match.Panorama], str],
        make: bool = True,
        estimate: Callable[[np.ndarray], np.ndarray] | None = None,
        side: int = SIDE,
    ):
        self.folder, self.lock, self.name, self.make, self.side = folder, lock, name, make, side
        self.estimate = estimate or self._depth_pro
        self.loaded = None
        self.maps = functools.lru_cache(maxsize=KEPT)(self._maps)

    def _maps(self, panorama: match.Panorama) -> np.ndarray | None:
        """The panorama's maps (views × side × side, metres as the model says), made if they
        were not and `make`; None otherwise, or if its picture cannot be read."""
        path = self._path(panorama)
        if not self.kept(panorama):
            if not self.make:
                return None
            picture = cv2.imread(str(panorama.path))
            if picture is None:
                return None
            with self.lock:
                if not self.kept(panorama):  # made meanwhile, by the request that held the lock
                    self._write(path, picture, panorama)
        with np.load(path) as kept:
            return kept["metres"].astype(np.float32)

    def kept(self, panorama: match.Panorama) -> bool:
        """Whether the panorama's maps are kept, made by the model."""
        return segment.by_model(self._path(panorama), MODEL)

    def _path(self, panorama: match.Panorama) -> Path:
        return self.folder / f"{self.name(panorama)}.npz"

    def _write(self, path: Path, picture: np.ndarray, panorama: match.Panorama) -> None:
        maps = []
        for heading in match.VIEW_HEADINGS:
            view = match.perspective(picture, panorama.axes, heading)
            ray = self.estimate(view) * _along(view.shape[0])
            maps.append(cv2.resize(ray, (self.side, self.side), interpolation=cv2.INTER_AREA))
        if self.loaded is not None:
            segment.let_go(self.loaded[2])  # once a panorama: not view by view
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(
            path,
            lambda out: np.savez_compressed(out, metres=np.array(maps, np.float16), model=MODEL),
        )

    def _depth_pro(self, view: np.ndarray) -> np.ndarray:
        """Depth Pro's depth along the axis of a view (BGR, square), at its size."""
        import torch

        if self.loaded is None:
            self.loaded = segment.loaded(MODEL, "AutoModelForDepthEstimation")
        model, processor, device, half = self.loaded
        inputs = processor(images=view[:, :, ::-1], return_tensors="pt").to(device)
        inputs = {k: v.to(half) if v.is_floating_point() else v for k, v in inputs.items()}
        with torch.no_grad():
            found = model(**inputs)
        side = view.shape[0]
        depth = processor.post_process_depth_estimation(found, target_sizes=[(side, side)])
        return depth[0]["predicted_depth"].float().cpu().numpy()
