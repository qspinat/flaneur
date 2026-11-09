"""LoMa-G (DaD keypoints, DeDoDe-G descriptors, the LoMa-G matcher; MIT licence), behind
match.Matcher's interface. Measured against DISK + LightGlue: far ahead on LTLL (old and new
pictures of the same places), and more photos rightly placed by the user's verdicts; matching
tries it first, DISK after (see cli.match_photos).

Its weights are pickles: each is known by its file name and SHA-256, checked before loading,
and loaded as tensors only (`weights_only`); LoMa's own loader (`torch.hub`) would run any code
they held. Torch is imported here: never where pycolmap is (see match.Matcher)."""

import sys

if "pycolmap" in sys.modules:  # it would abort the process, without a word
    raise ImportError("pycolmap est chargé : torch ne peut pas l'être avec lui")

import contextlib  # noqa: E402 (after the guard: torch must not load beside pycolmap)
import hashlib  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from flaneur import match  # noqa: E402
from flaneur.match import KEYPOINTS, Features  # noqa: E402

WEIGHTS = {  # file name → SHA-256, of every file LoMa-G loads
    "loma_G.pth": "9d925dc9861fb8e52489a1d2d2354780d5d38e34cbe0c6b10830d0ebe6055941",
    "dad.pth": "7d827844649faf5213ba97e7adb43e16b936dd3d11ab2d0fc0d602c67fae781f",
    "dinov2_vitl14_pretrain.pth": (
        "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"
    ),
}
DESCRIBED_SIDE = 784  # DeDoDe reads the picture as a square of this side (LoMa's own reading)


def pinned(url: str, folder: Path, map_location=None) -> dict:
    """The weights at `url`, kept in `folder`: downloaded if missing, then loaded only if their
    SHA-256 is the one pinned."""
    name = url.rsplit("/", 1)[-1]
    if name not in WEIGHTS:
        raise RuntimeError(f"poids non épinglés, refusés : {url}")
    path = folder / name
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(path))
    with path.open("rb") as weights:
        digest = hashlib.file_digest(weights, "sha256").hexdigest()
    if digest != WEIGHTS[name]:
        raise RuntimeError(f"{path} : SHA-256 inattendu, poids refusés")
    return torch.load(path, map_location=map_location, weights_only=True)


@contextlib.contextmanager
def loading_pinned(folder: Path) -> Iterator[None]:
    """LoMa's weights loaded through `pinned` (from `folder`) while inside: torch.hub's
    loader, which LoMa calls for each of them, swapped for it."""
    before = torch.hub.load_state_dict_from_url
    torch.hub.load_state_dict_from_url = lambda url, *args, **kwargs: pinned(
        url, folder, kwargs.get("map_location")
    )
    try:
        yield
    finally:
        torch.hub.load_state_dict_from_url = before


def _tensor(rgb: Image.Image, size: tuple[int, int], device) -> torch.Tensor:
    return (
        torch.from_numpy(np.array(rgb.resize(size)) / 255.0)
        .permute(2, 0, 1)
        .float()[None]
        .to(device)
    )


def picture(bgr: np.ndarray) -> Image.Image:
    """A picture (OpenCV's) as LoMa reads it: RGB, through PIL."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def detector_input(rgb: Image.Image, detector, device) -> torch.Tensor:
    """The picture as DaD reads it from a file (DaD.load_image), without the file."""
    w, h = rgb.size
    if detector.keep_aspect_ratio:
        scale = detector.resize / max(w, h)
        size = (int((scale * w) // 8 * 8), int((scale * h) // 8 * 8))
    else:
        size = (detector.resize, detector.resize)
    return _tensor(rgb, size, device)


def descriptor_input(rgb: Image.Image, device) -> torch.Tensor:
    """The picture as DeDoDe reads it from a file (read_image), without the file."""
    return _tensor(rgb, (DESCRIBED_SIDE, DESCRIBED_SIDE), device)


class LoMaGMatcher(match.Matcher):
    """LoMa-G's keypoints and pairs, as match.Matcher's (pixels, strongest first)."""

    def __init__(self, folder: Path):
        from loma import LoMa, LoMaG

        with loading_pinned(folder / "torch" / "checkpoints"):
            self.model = LoMa(LoMaG())
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.precision = "lomag"  # names the photos' pairs it makes (chain.Store.pairs)

    @torch.inference_mode()
    def features(self, bgr: np.ndarray) -> Features:
        h, w = bgr.shape[:2]
        detector, descriptor = self.model._detector, self.model._descriptor
        rgb = picture(bgr)
        found = detector.detect(detector_input(rgb, detector, self.device), num_keypoints=KEYPOINTS)
        keypoints = found["keypoints"]
        described = descriptor.describe_keypoints(descriptor_input(rgb, self.device), keypoints)
        pixels = detector.to_pixel_coords(keypoints, h, w)[0].float().cpu().numpy()
        descriptors = described["descriptions"][0].float().cpu().numpy()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        return Features(pixels.astype(np.float32), descriptors.astype(np.float32), (h, w))

    @torch.inference_mode()
    def scored_pairs(self, a: Features, b: Features) -> tuple[np.ndarray, np.ndarray]:
        from loma.loma import filter_matches

        if not len(a.keypoints) or not len(b.keypoints):
            return np.empty((0, 2), np.int64), np.empty(0, np.float32)

        def normalized(f: Features) -> torch.Tensor:
            h, w = f.size
            k = torch.from_numpy(f.keypoints).float()
            return torch.stack([2 * k[:, 0] / w - 1, 2 * k[:, 1] / h - 1], -1)[None].to(self.device)

        scores = self.model(
            normalized(a),
            normalized(b),
            torch.from_numpy(a.descriptors)[None].to(self.device),
            torch.from_numpy(b.descriptors)[None].to(self.device),
        )["scores"]
        to, _, sure, _ = filter_matches(scores.float(), self.model.cfg.filter_threshold)
        to, sure = to[0].cpu().numpy(), sure[0].cpu().numpy()
        paired = np.nonzero(to > -1)[0]
        return np.column_stack([paired, to[paired]]), sure[paired].astype(np.float32)


def released() -> None:
    """The graphics chip's memory handed back once a matcher is dropped."""
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
