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
KEPT = 16  # panoramas' masks kept in memory (a scene reads its own once per neighbour)
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


def on_structure(masks: np.ndarray | None, rays: np.ndarray) -> np.ndarray:
    """Whether each ray falls on a building (`masks`: per view, True on structure; see
    at_rays); every ray, without masks."""
    return np.ones(len(rays), bool) if masks is None else at_rays(masks, rays, False)


class _NotRead(Exception):
    """A panorama whose picture could not be read: nothing kept, asked again next time."""


class Segmenter:
    """Each pixel's class in a panorama's views, made with the model on the graphics chip if
    there is one (loaded the first time, `lock` held while it runs) and kept in `folder`,
    named as the panorama is levelled (its views change with it), for the model."""

    def __init__(self, folder: Path, lock: threading.Lock, name: Callable[[match.Panorama], str]):
        self.folder, self.lock, self.name = folder, lock, name
        self.loaded = None
        self._kept = functools.lru_cache(maxsize=KEPT)(self._masks)

    def masks(self, panorama: match.Panorama) -> np.ndarray | None:
        """Where the panorama's views show a building, a wall, a bridge or a tunnel; None if
        its picture cannot be read (asked again next time)."""
        try:
            return self._kept(panorama)
        except _NotRead:
            return None

    def _masks(self, panorama: match.Panorama) -> np.ndarray:
        classes, labels = self.classes(panorama)
        structure = np.flatnonzero(np.isin(labels, STRUCTURE))
        if not len(structure):  # a model naming them otherwise: every detail would go
            raise ValueError(f"aucune des classes {STRUCTURE} dans le modèle {MODEL}")
        return np.isin(classes, structure)

    def classes(self, panorama: match.Panorama) -> tuple[np.ndarray, list[str]]:
        """The class of each pixel of the panorama's views (views × side × side), and the
        classes' names; _NotRead if its picture cannot be read."""
        path = self.folder / f"{self.name(panorama)}.npz"
        if not path.exists():
            picture = cv2.imread(str(panorama.path))
            if picture is None:
                raise _NotRead
            views = [match.perspective(picture, panorama.axes, h) for h in match.VIEW_HEADINGS]
            with self.lock:
                if not path.exists():  # made meanwhile, by the request that held the lock
                    self._write(path, views)
        with np.load(path) as kept:
            if str(kept["model"]) == MODEL:
                return kept["classes"], list(kept["labels"])
        path.unlink(missing_ok=True)  # made by another model: made again
        return self.classes(panorama)

    def _write(self, path: Path, views: list[np.ndarray]) -> None:
        model, processor, device, half, labels = self._model()
        classes = np.array([self._segmented(v, model, processor, device, half) for v in views])
        if device == "mps":
            import torch

            torch.mps.empty_cache()  # memory is short: the chip's cache let go
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
            if "pycolmap" in sys.modules:  # their OpenMP runtimes abort one process (see match)
                raise RuntimeError("segmentation et pycolmap ne se chargent pas ensemble")
            import torch
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            half = torch.float16 if device == "mps" else torch.float32  # half the memory
            processor = AutoImageProcessor.from_pretrained(MODEL)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL, dtype=half)
            names = model.config.id2label
            labels = [names[i] for i in range(len(names))]
            self.loaded = (model.to(device).eval(), processor, device, half, labels)
        return self.loaded

    @staticmethod
    def _segmented(view: np.ndarray, model, processor, device: str, half) -> np.ndarray:
        """Each pixel's class in a view (BGR, square)."""
        import torch

        side = view.shape[0]
        inputs = processor(
            images=view[:, :, ::-1],
            size={"shortest_edge": side, "longest_edge": side},  # as cut: not 2048 (memory)
            return_tensors="pt",
        ).to(device, half)
        with torch.no_grad():
            found = model(**inputs)
        classes = processor.post_process_semantic_segmentation(found, target_sizes=[(side, side)])
        return classes[0].cpu().numpy()
