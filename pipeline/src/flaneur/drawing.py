"""A photo laid in a view by its standing camera (see registration.Standing), as the 3D walk
draws it: each pixel of the view taken to today's point along its ray (facades.background:
walls, street, over the roofs, the sky), seen by the photographer's camera; nothing cut hard —
a point hidden from the photographer, or where the photo would be smeared (a wall seen edge on),
weighs less, and the layer is filled (fill: small gaps interpolated, large ones fading to the
view, the frame's edge soft). An Albert-Kahn plate: only its plate, never its label."""

import cv2
import numpy as np

from flaneur import facades, fill, match
from flaneur.registration import HIDDEN, Standing, ViewOf

FULL, GONE = 6.0, 12.0  # the photo laid this many times larger than as a whole: kept, gone
Box = tuple[float, float, float, float]  # x0, y0, x1, y1: shares of the photo (see plates)


def composite_standing(
    view_bgr: np.ndarray,
    photo_bgr: np.ndarray,
    s: Standing,
    scene: facades.Scene,
    view: ViewOf,
    metres: np.ndarray | None = None,
    plate: Box | None = None,
    label: Box | None = None,
) -> np.ndarray:
    """The view (square, cut from the scene's panorama) with the photo laid in; `metres` along
    each pixel's ray (side × side), else facades.background; `plate`, `label`: what of the photo
    is shown, and what never is."""
    side = view_bgr.shape[0]
    if view_bgr.shape[1] != side:
        raise ValueError("une vue est carrée")
    rays = match.view_directions(
        match.pixels_of((side, side)), view.heading, view.fov, side, view.pitch
    )
    far = (facades.background(scene, rays) if metres is None else np.asarray(metres, float)).ravel()
    at, reached = _seen(s, scene.origin, rays, far, plate)
    weight = (reached & ~_hidden(s, scene, rays, far, reached) & ~_on(at, s.size, label)).astype(
        np.float32
    ).reshape(side, side) * stretch_kept(at.reshape(side, side, 2), reached.reshape(side, side))
    maps = np.where(reached[:, None], at - 0.5, -1.0).astype(np.float32).reshape(side, side, 2)
    colour = cv2.remap(
        photo_bgr.astype(np.float32),
        maps[..., 0],
        maps[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    pixel_deg = view.fov / side
    rgb, coverage = fill.push_pull(colour, weight, fill.levels(pixel_deg))
    frame = fill.frame(reached.reshape(side, side), round(fill.FILL_DEG / pixel_deg))
    alpha = fill.alpha(coverage) * fill.edge(frame, fill.EDGE_DEG / pixel_deg)
    shape = (side, side, *[1] * (view_bgr.ndim - 2))
    alpha = alpha.reshape(shape)
    laid = rgb.reshape(view_bgr.shape) * alpha + view_bgr * (1 - alpha)
    if np.issubdtype(view_bgr.dtype, np.integer):
        laid = np.clip(np.rint(laid), 0, 255)
    return laid.astype(view_bgr.dtype)


def _seen(
    s: Standing, origin, rays: np.ndarray, far: np.ndarray, plate: Box | None
) -> tuple[np.ndarray, np.ndarray]:
    """Where each view pixel's point falls in the photo (pixels), and whether the photo (its
    plate) shows it: the sky (infinitely far) by the camera's turn alone."""
    h, w = s.size
    lens = s._lens() @ cv2.Rodrigues(np.array(s.turn))[0]
    step = np.asarray(origin, float) - np.asarray(s.position, float)
    with np.errstate(divide="ignore"):
        nearness = 1 / far  # the sky 0, unknown NaN
    seen = rays @ lens.T + nearness[:, None] * (lens @ step)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = seen[:, :2] / seen[:, 2:]
    x0, y0, x1, y1 = plate or (0.0, 0.0, 1.0, 1.0)
    reached = (seen[:, 2] > 0) & np.all((at >= [x0 * w, y0 * h]) & (at < [x1 * w, y1 * h]), axis=1)
    return at, reached


def _hidden(
    s: Standing, scene: facades.Scene, rays: np.ndarray, far: np.ndarray, reached: np.ndarray
) -> np.ndarray:
    """The points the photographer could not see: a wall between the camera and them (nearer
    than their distance over HIDDEN), as the walk's shader tests it."""
    hidden = np.zeros(len(rays), bool)
    check = reached & np.isfinite(far)
    if not check.any() or not scene.walls:
        return hidden
    to = np.asarray(scene.origin, float) + rays[check] * far[check, None] - s.position
    distance = np.linalg.norm(to, axis=1)
    camera = facades.Scene(scene.panorama, tuple(s.position), scene.ground, scene.walls, (), 0)
    metres, met = facades.cast(camera, to / distance[:, None])
    hidden[np.flatnonzero(check)] = (met >= 0) & (metres < distance / HIDDEN)
    return hidden


def _on(at: np.ndarray, size: tuple[int, int], box: Box | None) -> np.ndarray:
    """Whether each point of the photo (pixels) falls in `box` (shares; none: nowhere)."""
    if box is None:
        return np.zeros(len(at), bool)
    h, w = size
    x0, y0, x1, y1 = box
    return np.all((at >= [x0 * w, y0 * h]) & (at <= [x1 * w, y1 * h]), axis=1)


def stretch_kept(at: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """How much of the photo laid in a view (`at`: each view pixel's place in the photo, side ×
    side × 2; `inside`: those it reaches) is kept where it is stretched: all up to FULL times its
    scale as a whole, none from GONE (a wall the photographer saw edge on, its few pixels smeared
    into streaks), smoothly in between."""
    with np.errstate(invalid="ignore"):
        du_dy, du_dx = np.gradient(np.where(inside, at[..., 0], np.nan))
        dv_dy, dv_dx = np.gradient(np.where(inside, at[..., 1], np.nan))
        # how little of the photo one view pixel spans, the way it is most squeezed: the lesser
        # singular value of the map's local derivative (a streak is squeezed one way only)
        square = du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2
        det = du_dx * dv_dy - du_dy * dv_dx
        least = np.sqrt(np.maximum(square - np.sqrt(np.maximum(square**2 - 4 * det**2, 0)), 0) / 2)
        scale = np.sqrt(np.abs(det))  # the photo's scale there, both ways
    known = np.isfinite(least) & (scale > 0)
    if not known.any():
        return np.ones(inside.shape, np.float32)
    stretch = np.where(known, np.median(scale[known]) / np.maximum(least, 1e-12), 1.0)
    t = np.clip((GONE - stretch) / (GONE - FULL), 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)
