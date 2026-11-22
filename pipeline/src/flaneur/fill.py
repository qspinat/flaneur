"""A photo laid in a view without holes: each pixel of the layer weighs what it is worth (0
where the photo has nothing to show there), small gaps are filled from around them and large
ones fade to the view (push-pull: a premultiplied pyramid, pulled back), and the photo's frame
ends softly. The 3D walk does the same (web/src/sphere/fill.ts): one rule for both.

Its reach is an angle, not pixels: a gap looks the same filled or faded at any size of view."""

import cv2
import numpy as np

FILL_DEG = 2.0  # gaps up to about this wide are filled; wider ones fade
EDGE_DEG = 1.0  # the frame's edge fades over this, inside the frame


def levels(pixel_deg: float) -> int:
    """How many levels of the pyramid reach FILL_DEG, for pixels of `pixel_deg` degrees."""
    return 1 + int(np.ceil(np.log2(FILL_DEG / pixel_deg)))


def push_pull(colour: np.ndarray, weight: np.ndarray, depth: int) -> tuple[np.ndarray, np.ndarray]:
    """The layer filled: its colour and its coverage (0 to 1), from `colour` (h × w × 3) weighed
    by `weight` (h × w, 0 to 1), through `depth` levels. The coverage is smoothed over the fill's
    own reach: a large gap fades across its edge, not from a step at it."""
    weight = np.clip(weight, 0, 1).astype(np.float32)
    pyramid = [(colour.astype(np.float32) * weight[..., None], weight)]
    while len(pyramid) < depth and min(pyramid[-1][1].shape) > 1:
        c, a = pyramid[-1]
        pyramid.append((cv2.pyrDown(c), cv2.pyrDown(a)))
    c, a = pyramid[-1]
    for fine_c, fine_a in reversed(pyramid[:-1]):
        size = (fine_a.shape[1], fine_a.shape[0])
        rest = (1 - fine_a)[..., None]
        c = fine_c + rest * cv2.resize(c, size, interpolation=cv2.INTER_LINEAR)
        a = fine_a + rest[..., 0] * cv2.resize(a, size, interpolation=cv2.INTER_LINEAR)
    a = np.clip(a, 0, 1)
    rgb = c / np.maximum(a, 1e-6)[..., None]
    smooth = cv2.GaussianBlur(a, (0, 0), 2.0 ** max(0, depth - 2))
    return rgb, np.clip(smooth, 0, 1)


def alpha(coverage: np.ndarray) -> np.ndarray:
    """How much of the layer shows, from its coverage: none under 0.2, all over 0.8."""
    t = np.clip((np.asarray(coverage, float) - 0.2) / 0.6, 0, 1)
    return t * t * (3 - 2 * t)


def frame(reached: np.ndarray, reach_px: int) -> np.ndarray:
    """The photo's frame in a view: where it reaches (`reached`), its notches narrower than
    `reach_px` closed and every gap inside it kept (filled or faded like any other); outside, what
    joins the view's edge."""
    side = max(1, reach_px) | 1
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    closed = cv2.morphologyEx(reached.astype(np.uint8), cv2.MORPH_CLOSE, disc).astype(bool)
    _, labels = cv2.connectedComponents((~closed).astype(np.uint8), connectivity=4)
    edges = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    outside = np.isin(labels, edges[edges > 0])
    return ~outside


def edge(inside: np.ndarray, edge_px: float) -> np.ndarray:
    """1 well inside the frame, falling smoothly to 0 at its edge over `edge_px`."""
    near = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)
    t = np.clip(near / max(edge_px, 1e-6), 0, 1)
    return np.where(inside, t * t * (3 - 2 * t), 0.0)
