"""Does a camera found for an old photo see the buildings then standing where the photo shows
edges? Their roof line, against the sky, and their corners, upright, are sharp edges in a
photo of 1900 as today; BD TOPO dates most buildings, so the outlines are those of the
buildings standing at the photo's date. Matched points can agree by chance on a wrong camera;
outlines checked over the whole photo rarely do."""

from dataclasses import dataclass, replace

import cv2
import numpy as np

from flaneur.blocks import Blocks
from flaneur.match import View, focal, project
from flaneur.register import CORNER_FROM_M, CORNER_SAMPLES, CORNERS_M

NEAR_PX = 2.5  # an outline this close to an edge of the photo lies on it
COLUMN_STEP = 2  # the roof line is read every this many columns
UPRIGHT = 2.0  # an edge is upright where it changes this many times more across than down
# the same camera turned, and moved: how well outlines fit a camera that is not the right one
TURNS_DEG = (-12.0, -6.0, 6.0, 12.0)
MOVES_M = (-4.0, 4.0)


@dataclass(frozen=True)
class Edges:
    """How far each pixel of a photo is from an edge, and from an upright edge."""

    away: np.ndarray
    away_upright: np.ndarray


def _canny(photo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A grey `photo`'s edges (thresholds from its own contrast), and its gradients."""
    soft = cv2.GaussianBlur(photo, (0, 0), 1.2)
    gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1)
    low, high = np.percentile(np.hypot(gx, gy), (70, 90))
    return cv2.Canny(soft, max(low, 1.0), max(high, 2.0), L2gradient=True) > 0, gx, gy


def edge_mask(photo: np.ndarray) -> np.ndarray:
    """The edges of a grey `photo`, as edges_of finds them (for drawing them)."""
    return _canny(photo)[0]


def edges_of(photo: np.ndarray) -> Edges:
    """The edges of a grey `photo`, found once for every camera tried on it."""
    edges, gx, gy = _canny(photo)
    upright = edges & (np.abs(gx) > UPRIGHT * np.abs(gy))

    def away(mask: np.ndarray) -> np.ndarray:
        return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)

    return Edges(away(edges), away(upright))


def _on(away: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """For pixels (column, row) inside the photo, whether each lies on an edge."""
    c, r = np.round(pixels).astype(int).T
    return away[r, c] <= NEAR_PX


def _inside(pixels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    finite = np.isfinite(pixels).all(axis=1)
    c, r = np.where(finite[:, None], pixels, -1).T
    return finite & (c >= 0) & (c <= w - 1) & (r >= 0) & (r <= h - 1)


def _samples(
    view: View, blocks: Blocks, size: tuple[int, int], corners: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Where `view` sees the roof line (pixels, one per COLUMN_STEP columns) and the corners
    (CORNER_SAMPLES pixels up each; `corners` as Blocks.corners gives them from where it
    stands) of `blocks`, inside a photo of `size`."""
    h, w = size
    f = focal(view.fov, w)
    columns = np.arange(0, w, COLUMN_STEP, dtype=float)
    across = (columns - (w / 2 - 0.5)) / f
    bearings = np.radians(view.heading) + np.arctan(across)
    elevation = blocks.skyline(view.x, view.y, view.z, bearings)
    rise = np.tan(np.where(np.isfinite(elevation), elevation, np.nan))  # NaN: sky, left out
    rows = view.horizon * h - 0.5 - f * rise * np.sqrt(1 + across**2)
    roof = np.column_stack([columns, rows])
    roof = roof[_inside(roof, size)]

    tall = corners[corners[:, 3] - corners[:, 2] > CORNER_FROM_M]
    if not len(tall):
        return roof, np.empty((0, 2))
    bearing, distance, base, top = tall.T
    up = np.linspace(0, 1, CORNER_SAMPLES)[None]
    heights = base[:, None] + CORNER_FROM_M + up * (top - base - CORNER_FROM_M)[:, None]
    points = np.stack(
        [
            np.broadcast_to((view.x + distance * np.sin(bearing))[:, None], heights.shape),
            np.broadcast_to((view.y + distance * np.cos(bearing))[:, None], heights.shape),
            heights,
        ],
        axis=-1,
    ).reshape(-1, 3)
    upright = project(view, points, size)
    return roof, upright[_inside(upright, size)]


def fit(photo: Edges, view: View, blocks: Blocks, corners: np.ndarray | None = None) -> float:
    """The share of the outlines of `blocks` seen by `view` that lie on edges of the `photo`
    (0 when none is seen): the roof line on any edge, the corners on upright ones (`corners`:
    those seen from where it stands, when already known)."""
    if corners is None:
        corners = blocks.corners(view.x, view.y, view.z, CORNERS_M)
    roof, upright = _samples(view, blocks, photo.away.shape, corners)
    if not len(roof) + len(upright):
        return 0.0
    on = _on(photo.away, roof).sum() + _on(photo.away_upright, upright).sum()
    return float(on / (len(roof) + len(upright)))


def contrast(photo: np.ndarray, view: View, blocks: Blocks) -> float:
    """How much better the outlines fit `view` on the grey `photo` than the same camera turned
    a little or moved a few metres (the median of those): near 0 when the photo shows nothing
    of them there."""
    edges = edges_of(photo)
    here = blocks.corners(view.x, view.y, view.z, CORNERS_M)  # the same, turned
    turned = [fit(edges, replace(view, heading=view.heading + t), blocks, here) for t in TURNS_DEG]
    moved = [
        fit(edges, replace(view, x=view.x + dx, y=view.y + dy), blocks)
        for dx in MOVES_M
        for dy in MOVES_M
    ]
    return fit(edges, view, blocks, here) - float(np.median(turned + moved))
