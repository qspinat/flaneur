"""A photo laid in a view by its standing camera (see registration.Standing), as the 3D walk
draws it (web/src/sphere/photo.ts, standing.ts): each pixel of the view taken to today's point
along its ray (facades.background: walls, street, over the roofs, the sky), seen by the
photographer's camera. Nothing is cut hard: a point hidden from the photographer, or where the
photo would be smeared (a wall seen edge on), weighs less, and the layer is filled (fill: small
gaps interpolated, large ones fading to the view, the photo's edge soft). An Albert-Kahn photo:
its plate only, never the museum's label."""

from dataclasses import replace

import cv2
import numpy as np

from flaneur import facades, fill, match
from flaneur.crop import Crop
from flaneur.registration import Standing, ViewOf

# as the walk (standing.ts, photo.ts)
STRETCHED = 6.0  # a part of the photo laid this many times larger than the photo as a whole
HIDDEN = 1.1  # a point this much farther than a wall in the photographer's way is hidden
FAR_M = 1000.0  # a ray of the photo meeting nothing, framed as this far
FRAMED = 8  # the photo's rays framing where it is laid: this many steps across, each way
WHOLE: Crop = (0.0, 0.0, 1.0, 1.0)
HIDDEN_STEP = 2  # the hidden test is made every this many pixels (the fill spans ~2°)


def composite_standing(
    view_bgr: np.ndarray,
    photo_bgr: np.ndarray,
    s: Standing,
    scene: facades.Scene,
    view: ViewOf,
    depth: tuple[np.ndarray, np.ndarray] | None = None,
    flipped: bool = False,
    plate: Crop | None = None,
    label: Crop | None = None,
) -> np.ndarray:
    """The view (square, BGR, cut from the scene's panorama) with the photo laid in. `photo_bgr`:
    the scan as it is; `flipped`: shown mirrored (the camera was fitted on it so); `depth`: metres
    along each pixel's ray and whether it is the sky (see facades.background, made if not given);
    `plate`, `label`: what of the scan is shown and what never is (shares of the scan)."""
    side = view_bgr.shape[0]
    if view_bgr.shape[1] != side:
        raise ValueError("une vue est carrée")
    rays = match.view_directions(
        match.pixels_of((side, side)), view.heading, view.fov, side, view.pitch
    )
    metres, sky = facades.background(scene, rays) if depth is None else depth
    metres, sky = np.ravel(metres), np.ravel(sky)
    at, ahead = _seen(s, scene.origin, rays, metres)
    scan = np.column_stack([s.size[1] - at[:, 0], at[:, 1]]) if flipped else at
    reached = ahead & _on(scan, s.size, plate or WHOLE)
    shown = reached & ~_on(scan, s.size, label) & ~_hidden(s, scene, rays, metres, reached, side)
    grid = (side, side)
    kept = stretch_kept(
        at.reshape(*grid, 2), rays.reshape(*grid, 3), sky.reshape(grid), density(s, scene)
    )
    maps = np.where(reached[:, None], scan - 0.5, -1.0).astype(np.float32).reshape(*grid, 2)
    colour = cv2.remap(
        photo_bgr, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    rgb, alpha = fill.laid(
        colour, shown.reshape(grid) * kept, reached.reshape(grid), view.fov / side
    )
    laid = rgb * alpha[..., None] + view_bgr * (1 - alpha[..., None])
    if np.issubdtype(view_bgr.dtype, np.integer):
        laid = np.clip(np.rint(laid), 0, 255)
    return laid.astype(view_bgr.dtype)


def _seen(
    s: Standing, origin, rays: np.ndarray, metres: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Where each view pixel's point falls in the photo as shown (pixels), and whether it lies
    ahead of the camera: the sky (infinitely far) by the camera's turn alone."""
    kr = s.lens_turn().astype(np.float32)
    step = (np.asarray(origin, float) - np.asarray(s.position, float)).astype(np.float32)
    with np.errstate(divide="ignore"):
        nearness = (1 / metres).astype(np.float32)  # the sky 0
    seen = rays.astype(np.float32) @ kr.T + nearness[:, None] * (kr @ step)
    with np.errstate(divide="ignore", invalid="ignore"):
        return seen[:, :2] / seen[:, 2:], seen[:, 2] > 0


def _on(at: np.ndarray, size: tuple[int, int], box: Crop | None) -> np.ndarray:
    """Whether each point of the photo (pixels) falls in `box` (shares; none: nowhere)."""
    if box is None:
        return np.zeros(len(at), bool)
    h, w = size
    x0, y0, x1, y1 = box
    return np.all((at >= [x0 * w, y0 * h]) & (at <= [x1 * w, y1 * h]), axis=1)


def _hidden(
    s: Standing,
    scene: facades.Scene,
    rays: np.ndarray,
    metres: np.ndarray,
    reached: np.ndarray,
    side: int,
) -> np.ndarray:
    """The points the photographer could not see: a wall between the camera and them, nearer
    than their distance over HIDDEN (the walk's test). Made every HIDDEN_STEP pixels and spread
    to their neighbours (a hidden point only weighs nothing: the fill spans far more)."""
    hidden = np.zeros((side, side), bool)
    every = np.zeros((side, side), bool)
    every[::HIDDEN_STEP, ::HIDDEN_STEP] = True
    check = reached & np.isfinite(metres) & every.ravel()
    walls = _ahead_of(s, scene)
    if not check.any() or not walls:
        return hidden.ravel()
    to = np.asarray(scene.origin, float) + rays[check] * metres[check, None] - s.position
    distance = np.linalg.norm(to, axis=1)
    found, met = facades.cast(
        replace(scene, origin=tuple(s.position), walls=walls), to / distance[:, None]
    )
    hidden.ravel()[np.flatnonzero(check)] = (met >= 0) & (found < distance / HIDDEN)
    spread = np.repeat(
        np.repeat(hidden[::HIDDEN_STEP, ::HIDDEN_STEP], HIDDEN_STEP, 0), HIDDEN_STEP, 1
    )
    return spread[:side, :side].ravel() & reached


def _ahead_of(s: Standing, scene: facades.Scene) -> tuple[facades.Wall, ...]:
    """The walls not wholly behind the photographer (none behind can stand in his way)."""
    turn = cv2.Rodrigues(np.array(s.turn))[0]
    kept = []
    for w in scene.walls:
        top = w.top if w.top is not None else scene.ground + facades.DRAWN_ABOVE_M
        corners = np.array([[*end, z] for end in (w.a, w.b) for z in (scene.ground, top)])
        if np.any((corners - s.position) @ turn[2] > 0):
            kept.append(w)
    return tuple(kept)


def density(s: Standing, scene: facades.Scene) -> float:
    """The photo's pixels per radian as laid, seen from the panorama: the median over neighbouring
    rays across its width, on a grid FRAMED steps each way (the walk's density, photo.ts)."""
    h, w = s.size
    steps = np.arange(FRAMED + 1) / FRAMED
    pixels = np.array([[w * i, h * j] for i in steps for j in steps])
    rays = s.rays(pixels)
    camera = replace(scene, origin=tuple(s.position))
    metres = np.minimum(FAR_M, facades.background(camera, rays)[0])
    laid = (np.asarray(s.position, float) - scene.origin + rays * metres[:, None]).reshape(
        FRAMED + 1, FRAMED + 1, 3
    )
    unit = laid / np.linalg.norm(laid, axis=-1, keepdims=True)
    angle = np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=-1), -1, 1))
    found = (w / FRAMED) / angle[angle > 0]
    return float(np.median(found)) if len(found) else s.focal


def stretch_kept(
    at: np.ndarray, rays: np.ndarray, sky: np.ndarray, density_px: float
) -> np.ndarray:
    """How much of the photo laid in a view is kept where it would be smeared (the walk's rule,
    photo.ts): how little of the photo one view pixel spans the way it is most squeezed (`at`:
    each pixel's place in the photo, side × side × 2), against the photo laid as a whole
    (`density_px`, pixels per radian, over the radians a pixel spans: `rays`, side × side × 3) —
    all from STRETCHED times larger, none from twice that; the sky never (where two walls'
    backdrops meet, the photo folds there, a line, not a smear)."""
    with np.errstate(invalid="ignore"):
        du_dy, du_dx = np.gradient(at[..., 0])
        dv_dy, dv_dx = np.gradient(at[..., 1])
        square = du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2
        det = du_dx * dv_dy - du_dy * dv_dx
        least = np.sqrt(np.maximum(square - np.sqrt(np.maximum(square**2 - 4 * det**2, 0)), 0) / 2)
        spread = np.linalg.norm(np.gradient(rays, axis=1), axis=-1)
        ratio = least * STRETCHED / (density_px * spread)
    kept = fill.smooth(np.nan_to_num(ratio, nan=1.0), 0.5, 1.0)
    return np.where(sky, 1.0, kept).astype(np.float32)
