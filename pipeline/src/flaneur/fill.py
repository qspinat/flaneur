"""A photo laid in a view without holes: each pixel of the layer weighs what it is worth (0
where the photo has nothing to show there), small gaps are filled from around them and large
ones fade to the view, and the photo's edge fades softly.

Done the way a graphics chip does it, so that the 3D walk (web/src/sphere/fill.ts) draws the
same: the layer's levels halve by box averages (mipmaps), each read at a pixel as a texture is
(bilinear), and pulled back from the coarsest one pixel at a time (push-pull). Its reach is an
angle, not pixels: a gap looks the same filled or faded at any size of view."""

import cv2
import numpy as np

FILL_DEG = 2.0  # gaps up to about this wide are filled; wider ones fade
EDGE_DEG = 1.0  # the photo's edge fades over about this, inside it


def smooth(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """0 at `low`, 1 at `high`, smoothly between (GLSL's smoothstep; `low` > `high`: falling)."""
    t = np.clip((np.asarray(x, np.float32) - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def levels(pixel_deg: float) -> int:
    """How many levels of the pyramid reach FILL_DEG, for pixels of `pixel_deg` degrees."""
    return 1 + int(np.ceil(np.log2(FILL_DEG / pixel_deg)))


def level(degrees: float, pixel_deg: float) -> int:
    """The level whose texels are about `degrees` across."""
    return max(0, round(np.log2(degrees / pixel_deg)))


def laid(
    colour: np.ndarray, weight: np.ndarray, reached: np.ndarray, pixel_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """The layer filled: its colour, and how much of it shows (0 to 1). `colour` (h × w × 3)
    weighed by `weight` (h × w, 0 to 1); `reached`: where the photo is at all (its edge)."""
    depth = levels(pixel_deg)
    w = np.clip(weight, 0, 1).astype(np.float32)[..., None]
    layer = np.concatenate(
        [colour.astype(np.float32) * w, w, reached.astype(np.float32)[..., None]], axis=-1
    )
    read = _levels_read(layer, depth)
    # pulled back: each level where it covers, the coarser ones where it does not (the photo's
    # reach too: a notch narrower than a level is closed by the coarser ones); the coverage kept
    # as smooth as the fill's reach (two levels up), so a wide gap fades without a step, and the
    # reach read at about EDGE_DEG, so the photo's edge fades over that
    pulled = read[-1][..., 3:]
    covered, edge = pulled[..., 0], pulled[..., 1]
    colour_pulled = read[-1][..., :3]
    smoothest, edgiest = max(0, depth - 2), min(level(EDGE_DEG, pixel_deg), depth - 1)
    for k in range(depth - 2, -1, -1):
        rest = 1 - read[k][..., 3:]
        colour_pulled = read[k][..., :3] + rest[..., :1] * colour_pulled
        pulled = read[k][..., 3:] + rest * pulled
        if k == smoothest:
            covered = pulled[..., 0].copy()
        if k == edgiest:
            edge = pulled[..., 1].copy()
    rgb = colour_pulled / np.maximum(pulled[..., :1], 1e-6)
    return rgb, smooth(covered, 0.2, 0.8) * smooth(edge, 0.5, 1.0)


def _levels_read(layer: np.ndarray, depth: int) -> list[np.ndarray]:
    """The layer's levels (each half the last, by box averages), each read at every pixel of the
    layer, bilinearly."""
    h, w = layer.shape[:2]
    found, current = [layer], layer
    for _ in range(depth - 1):
        size = (max(1, current.shape[1] // 2), max(1, current.shape[0] // 2))
        current = _resized(current, size, cv2.INTER_AREA)
        found.append(_resized(current, (w, h), cv2.INTER_LINEAR))
    return found


def _resized(layer: np.ndarray, size: tuple[int, int], how: int) -> np.ndarray:
    """cv2.resize on any number of channels (it takes at most 4 at a time)."""
    parts = [
        cv2.resize(layer[..., i : i + 4], size, interpolation=how)
        for i in range(0, layer.shape[2], 4)
    ]
    return np.concatenate([p if p.ndim == 3 else p[..., None] for p in parts], axis=-1)
