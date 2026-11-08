"""What today's panoramas show where: the views cut from each (see match.Views) segmented by a
model trained on street pictures (Mask2Former on Mapillary Vistas: building, wall, vegetation,
sky, road, car...), so that details matched on trees, cars or people make no walls. Each
pixel's class is kept (Segmenter.classes); a ray is read in the view that faces it most
(at_rays).

The model's code is MIT (facebookresearch/Mask2Former); its weights, trained on Mapillary
Vistas, are for non-commercial use, as Flâneur is. They are fetched once, kept outside the
repository."""

import functools
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from flaneur import match
from flaneur.sources import write_atomic

MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
STRUCTURE = ("Building", "Wall", "Bridge", "Tunnel")  # what a wall may stand on
UNSEEN, BUILT, SKY, OTHER = 0, 1, 2, 3  # what a ray sees in a panorama's views (see seen)
KEPT = 16  # panoramas' kinds kept in memory (a scene reads its own once per neighbour)
_AXES = np.array([match._view_axes(h, match.VIEW_PITCH) for h in match.VIEW_HEADINGS])


def at_rays(maps: np.ndarray, rays: np.ndarray, outside=0) -> np.ndarray:
    """What per-view maps (one square map per VIEW_HEADINGS) hold where each world ray (unit
    rows, as the panorama is recorded) falls, in the view that faces it most; `outside` for a
    ray that falls in none."""
    side = maps.shape[1]
    local = np.einsum("nc,vrc->nvr", rays, _AXES)  # each ray in each view: right, down, ahead
    best = np.argmax(local[:, :, 2], axis=1)
    seen = local[np.arange(len(rays)), best]
    f = match.focal(match.VIEW_FOV, side)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = np.rint(seen[:, :2] / seen[:, 2:] * f + side / 2 - 0.5)
    inside = (seen[:, 2] > 0) & np.all((px >= 0) & (px < side), axis=1)
    found = np.full(len(rays), outside, dtype=maps.dtype)
    x, y = px[inside].astype(int).T
    found[inside] = maps[best[inside], y, x]
    return found


def seen(kinds: np.ndarray | None, rays: np.ndarray) -> np.ndarray:
    """What each ray sees (`kinds`: per view, BUILT, SKY or OTHER; see Segmenter.kinds), UNSEEN
    outside every view. Without kinds, BUILT for every ray: as before segmentation (every
    detail kept, no sky to top a wall at)."""
    if kinds is None:
        return np.full(len(rays), BUILT, np.int8)
    return at_rays(kinds, rays, UNSEEN)


class _NotRead(Exception):
    """A panorama whose picture could not be read: nothing kept, asked again next time."""


def by_model(path: Path, model: str = MODEL) -> bool:
    """Whether what is kept at `path` was made by `model` (another's is made again)."""
    if not path.exists():
        return False
    with np.load(path) as kept:
        return str(kept["model"]) == model


def loaded(name: str, kind: str) -> tuple:
    """A model of transformers (`kind`: its class's name) and its processor, on the graphics
    chip if there is one, in half precision there (half the memory), and the device and number
    type: (model, processor, device, half). Torch and transformers are imported only here: they
    take seconds, and most commands never use a model."""
    if "pycolmap" in sys.modules:  # their OpenMP runtimes abort one process (see match)
        raise RuntimeError(f"{name} et pycolmap ne se chargent pas ensemble")
    import torch
    import transformers

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    half = torch.float16 if device == "mps" else torch.float32
    processor = transformers.AutoImageProcessor.from_pretrained(name)
    model = getattr(transformers, kind).from_pretrained(name, dtype=half)
    return model.to(device).eval(), processor, device, half


def rgb(view: np.ndarray) -> np.ndarray:
    """A view as models take it: RGB, contiguous (transformers' processors refuse an OpenCV picture
    reversed in place, its strides negative, once torchvision is installed)."""
    return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)


def let_go(device: str) -> None:
    """The graphics chip's cache let go: memory is short."""
    if device == "mps":
        import torch

        torch.mps.empty_cache()


class NotSegmented(Exception):
    """A panorama not segmented yet, by a segmenter that does not make them."""


class Segmenter:
    """Each pixel's class in a panorama's views, made with the model on the graphics chip if
    there is one (loaded the first time, `lock` held while it runs) and kept in `folder`,
    named as the panorama is levelled (its views change with it), for the model. Without
    `make`, only those kept are read (NotSegmented otherwise): the model takes gigabytes."""

    def __init__(
        self,
        folder: Path,
        lock: threading.Lock,
        name: Callable[[match.Panorama], str],
        make: bool = True,
    ):
        self.folder, self.lock, self.name, self.make = folder, lock, name, make
        self.loaded = None
        self._kept = functools.lru_cache(maxsize=KEPT)(self._kinds)

    def kinds(self, panorama: match.Panorama) -> np.ndarray | None:
        """What each pixel of the panorama's views shows: BUILT (a building, a wall, a bridge, a
        tunnel), SKY or OTHER; None if its picture cannot be read (asked again next time)."""
        try:
            return self._kept(panorama)
        except _NotRead:
            return None

    def _kinds(self, panorama: match.Panorama) -> np.ndarray:
        classes, labels = self.classes(panorama)
        names = np.array(labels)
        structure = np.isin(names, STRUCTURE)
        if not structure.any():  # a model naming them otherwise: every detail would go
            raise ValueError(f"aucune des classes {STRUCTURE} dans le modèle {MODEL}")
        kinds = np.full(len(names), OTHER, np.int8)
        kinds[structure] = BUILT
        kinds[names == "Sky"] = SKY
        return kinds[classes]

    def kept(self, panorama: match.Panorama) -> bool:
        """Whether the panorama's classes are kept, made by the model."""
        return by_model(self._path(panorama))

    def made(self, panorama: match.Panorama) -> Path | None:
        """Where the panorama's classes are kept, made if they were not (or by another model);
        None if its picture cannot be read."""
        path = self._path(panorama)
        if not by_model(path):
            if not self.make:
                raise NotSegmented(panorama.id)
            picture = cv2.imread(str(panorama.path))
            if picture is None:
                return None
            views = [match.perspective(picture, panorama.axes, h) for h in match.VIEW_HEADINGS]
            with self.lock:
                if not by_model(path):  # made meanwhile, by the request that held the lock
                    self._write(path, views)
        return path

    def _path(self, panorama: match.Panorama) -> Path:
        return self.folder / f"{self.name(panorama)}.npz"

    def classes(self, panorama: match.Panorama) -> tuple[np.ndarray, list[str]]:
        """The class of each pixel of the panorama's views (views × side × side), and the
        classes' names; _NotRead if its picture cannot be read."""
        path = self.made(panorama)
        if path is None:
            raise _NotRead
        with np.load(path) as kept:
            return kept["classes"], list(kept["labels"])

    def _write(self, path: Path, views: list[np.ndarray]) -> None:
        model, processor, device, half, labels = self._model()
        classes = np.array([self._segmented(v, model, processor, device, half) for v in views])
        let_go(device)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(
            path,
            lambda out: np.savez_compressed(
                out, classes=classes.astype(np.uint8), labels=np.array(labels), model=MODEL
            ),
        )

    def _model(self) -> tuple:
        """The model, its processor, the device and number type, and its classes' names (loaded
        once; torch and transformers imported only here: they take seconds, and most commands
        never segment)."""
        if self.loaded is None:
            model, processor, device, half = loaded(MODEL, "Mask2FormerForUniversalSegmentation")
            names = model.config.id2label
            labels = [names[i] for i in range(len(names))]
            self.loaded = (model, processor, device, half, labels)
        return self.loaded

    @staticmethod
    def _segmented(view: np.ndarray, model, processor, device: str, half) -> np.ndarray:
        """Each pixel's class in a view (BGR, square)."""
        import torch

        side = view.shape[0]
        inputs = processor(
            images=rgb(view),
            size={"shortest_edge": side, "longest_edge": side},  # as cut: not 2048 (memory)
            return_tensors="pt",
        ).to(device, half)
        with torch.no_grad():
            found = model(**inputs)
        classes = processor.post_process_semantic_segmentation(found, target_sizes=[(side, side)])
        return classes[0].cpu().numpy()
