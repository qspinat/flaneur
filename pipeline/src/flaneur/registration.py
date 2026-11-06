"""An old photo registered onto a panorama: the directions, in the panorama's own picture, that
the photo's points show (a 3×3 matrix from the photo's [0, 1]² to directions in the camera frame
of the 360° picture, see match.camera_axes). Kept in the picture's own frame, it does not change
when the panorama is levelled again; it is worked on, checked and drawn in a perspective view cut
from the levelled panorama (match.perspective), where it is a homography. Exact for one plane (a
facade) or a photographer standing at the panorama; a street in depth lines up only roughly.

Points are in pixels with a pixel's corners at whole numbers (as a browser's clicks are), and
normalised to [0, 1] by the picture's width and height, x right, y down."""

from collections.abc import Callable
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from flaneur import match
from flaneur.sources import number

MIN_POINTS = 2  # clicked by hand: a camera's direction and zoom; each pair more refines it
SHIFTED_FROM = 4  # pairs clicked from which the lens's shift off the print's middle is fitted
MIN_SPREAD = 0.1  # share of the photo's diagonal the points clicked span: under it, bunched
CLICKED_OFF = 1 / 64  # a clicked pair farther than this (a share of the view's side: 10 px
# of 640) from the others' fit counts for less, whatever size the view is shown at
MIN_COVER = 0.25  # share of the photo inside the hull of the points matched: under it, bunched
MIN_STANDING = 4  # pairs with a depth a camera standing off the panorama is fitted from
STANDING_SHIFTED_FROM = 6  # and its lens shift too
STANDING_OFF = 0.01  # a pair this far (share of the photo's diagonal: 10 px of 1000) counts less
STANDING_FOCAL = (0.2, 5.0)  # times the photo's longer side, as the flat fit's: beyond, the
# camera slid along its axis (a longer lens traded for a farther stand)
HIDDEN = 1.1  # a point this much farther from the camera than the nearest it sees there
TRIPOD_M = 1.5  # a standing camera is first looked for this high above the street
MIN_FRAME = 0.10  # share of the view the warped photo covers (Blanc, Produit, Ingensand 2018)
MIN_MATCHED = 15  # matched automatically: as many agreeing (match.MIN_MATCHES for a view)
MATCHED_PX = 3.0  # a matched pair farther than this from the others' homography is left out
CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float32)
_CORNERS_1 = np.column_stack([CORNERS, np.ones(4)])


@dataclass(frozen=True)
class ViewOf:
    """A square perspective view cut from a levelled panorama (match.perspective's angles)."""

    heading: float
    pitch: float = match.VIEW_PITCH
    fov: float = match.VIEW_FOV


@dataclass(frozen=True)
class Registration:
    panorama: str
    m: tuple[float, ...]  # 9 numbers, row by row: the photo's [0, 1]² to directions
    view: ViewOf  # where it was worked on: where to look first

    @property
    def matrix(self) -> np.ndarray:
        return np.array(self.m).reshape(3, 3)


class NotFitted(ValueError):
    """Points that cannot place a photo (the message says why, in French)."""


def _to_camera(view: ViewOf, axes: np.ndarray) -> np.ndarray:
    """From a view's [0, 1]² to directions in the frame of the 360° picture of these `axes`:
    the rays of match.view_directions, whatever the view's size (in units of its side)."""
    return axes @ match._view_axes(view.heading, view.pitch).T @ _rays(view)


def _rays(view: ViewOf) -> np.ndarray:
    """From a view's [0, 1]² to directions in its own camera's frame (x right, y down)."""
    return np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, match.focal(view.fov, 1.0)]])


def _centre(shift, size: tuple[int, int]) -> np.ndarray:
    """Where a lens shifted by `shift` (shares of the photo's width, height) centres the photo
    of `size` (height, width), in its pixels."""
    return (0.5 + np.asarray(shift, float)) * [size[1], size[0]]


def _camera(x: np.ndarray, photo_size: tuple[int, int], view: ViewOf) -> np.ndarray:
    """From the photo's pixels to the view's [0, 1]²: a camera standing at the panorama, turned
    from the view's by the rotation vector x[:3], of focal length exp(x[3]) pixels, its lens
    shifted off the photo's middle by x[4:6] (shares of its width and height), else centred."""
    cx, cy = _centre(x[4:6] if len(x) > 4 else (0.0, 0.0), photo_size)
    rays = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, np.exp(x[3])]])
    return np.linalg.inv(_rays(view)) @ cv2.Rodrigues(x[:3])[0] @ rays


def _clicked(
    photo_px: np.ndarray, view_px: np.ndarray, photo_size: tuple[int, int], side: int, view: ViewOf
) -> np.ndarray:
    """The camera (see _camera) through pairs clicked by hand, from the photo's pixels to the
    view's: its lens shift fitted from SHIFTED_FROM pairs; a pair far from the others' fit (a
    detail near the camera, which shifts against those far off) counts for less."""
    h, w = photo_size
    seen = np.column_stack([view_px / side, np.ones(len(view_px))]) @ _rays(view).T
    seen /= np.linalg.norm(seen, axis=1, keepdims=True)

    def turned(focal: float) -> tuple[float, np.ndarray]:  # the rotation best for a focal
        rays = np.column_stack([photo_px - [w / 2, h / 2], np.full(len(photo_px), focal)])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        u, _, vt = np.linalg.svd(seen.T @ rays)
        r = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        return float(np.sum((seen - rays @ r.T) ** 2)), r

    focal = min(np.geomspace(0.2, 5.0, 25) * max(w, h), key=lambda f: turned(f)[0])
    start = np.append(cv2.Rodrigues(turned(focal)[1])[0].ravel(), np.log(focal))
    if len(photo_px) >= SHIFTED_FROM:
        start = np.append(start, [0.0, 0.0])

    clicked = view_px / side

    def off(x: np.ndarray) -> np.ndarray:  # in shares of the view's side
        return (
            cv2.perspectiveTransform(photo_px[None], _camera(x, photo_size, view))[0] - clicked
        ).ravel()

    found = least_squares(off, start, loss="soft_l1", f_scale=CLICKED_OFF, xtol=1e-12, ftol=1e-12)
    return np.diag([side, side, 1.0]) @ _camera(found.x, photo_size, view)


def of(panorama: str, m: np.ndarray, view: ViewOf, axes: np.ndarray) -> Registration:
    """A registration of unit size, its sign such that the photo's centre lies ahead of the
    view it was made in (a homogeneous matrix and its opposite show opposite directions)."""
    ahead = axes @ match._view_axes(view.heading, view.pitch)[2]
    m = m / np.linalg.norm(m) * np.sign(m @ [0.5, 0.5, 1.0] @ ahead)
    return Registration(panorama, tuple(float(x) for x in m.ravel()), view)


def fit(
    photo_px: np.ndarray,
    view_px: np.ndarray,
    photo_size: tuple[int, int],
    side: int,
    panorama: str,
    view: ViewOf,
    axes: np.ndarray,
    robust: bool = False,
) -> tuple[Registration, np.ndarray]:
    """The registration through pairs of points, in the photo (`photo_px`, of `photo_size`:
    height, width) and in a view (`view_px`, `side` pixels square) cut from a panorama of these
    `axes`; and which pairs it kept. Clicked by hand, a camera turned and zoomed through all of
    them (see _clicked), which never folds the photo; matched (`robust`), any flat warp through
    those that agree, the others left out."""
    photo_px, view_px = np.asarray(photo_px, float), np.asarray(view_px, float)
    photo = photo_px / [photo_size[1], photo_size[0]]
    if np.any((photo < 0) | (photo > 1)) or np.any((view_px < 0) | (view_px > side)):
        raise NotFitted("des points tombent hors de la photo ou de la vue")
    kept = np.ones(len(photo), bool)
    if robust:
        h = None
        if len(photo) >= 4:  # fitted in pixels: MAGSAC's threshold is in pixels
            h, mask = cv2.findHomography(photo_px, view_px, cv2.USAC_MAGSAC, MATCHED_PX)
            if mask is not None:
                kept = mask.ravel().astype(bool)
        if kept.sum() < MIN_MATCHED:
            raise NotFitted(f"il faut au moins {MIN_MATCHED} points qui s'accordent")
        if cv2.contourArea(cv2.convexHull(photo[kept].astype(np.float32))) < MIN_COVER:
            raise NotFitted("points trop groupés : les répartir sur toute la photo")
        if h is None:
            raise NotFitted("ces points ne donnent aucune déformation de la photo")
    else:
        if len(photo) < MIN_POINTS:
            raise NotFitted(f"il faut au moins {MIN_POINTS} paires de points")
        apart = np.linalg.norm(photo_px[:, None] - photo_px[None], axis=2).max()
        if apart < MIN_SPREAD * np.hypot(*photo_size):
            raise NotFitted("points trop groupés : les écarter sur la photo")
        h = _clicked(photo_px, view_px, photo_size, side, view)
    in_unit = np.diag([1 / side, 1 / side, 1.0]) @ h @ np.diag([photo_size[1], photo_size[0], 1.0])
    if not frame_ok(in_unit):
        raise NotFitted("la photo se replie, se retourne ou sort presque de la vue")
    return of(panorama, _to_camera(view, axes) @ in_unit, view, axes), kept


@dataclass(frozen=True)
class Standing:
    """An old photo's camera where the photographer stood (not at the panorama): its position
    (Lambert-93 x, y, and height as the panoramas', see cli._spheres), its turn (a rotation
    vector, from the world, east-north-up, to the camera: x right, y down, z ahead), its focal
    length (photo pixels), its lens shifted off the photo's middle (shares of its width,
    height), for a photo of `size` (height, width)."""

    position: tuple[float, float, float]
    turn: tuple[float, float, float]
    focal: float
    shift: tuple[float, float]
    size: tuple[int, int]

    def _lens(self) -> np.ndarray:
        cx, cy = _centre(self.shift, self.size)
        return np.array([[self.focal, 0.0, cx], [0.0, self.focal, cy], [0.0, 0.0, 1.0]])

    def project(self, world: np.ndarray) -> np.ndarray:
        """Where points of the world fall in the photo (pixels; behind it, NaN)."""
        seen = (world - self.position) @ cv2.Rodrigues(np.array(self.turn))[0].T @ self._lens().T
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(seen[:, 2:] > 0, seen[:, :2] / seen[:, 2:], np.nan)

    @property
    def heading(self) -> float:
        """The compass heading its camera looked (degrees from north, clockwise)."""
        ahead = cv2.Rodrigues(np.array(self.turn))[0][2]  # its z axis, in the world
        return float(np.degrees(np.arctan2(ahead[0], ahead[1])) % 360)

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        """The world directions (unit) of pixels of the photo."""
        local = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(self._lens()).T
        world = local @ cv2.Rodrigues(np.array(self.turn))[0]
        return world / np.linalg.norm(world, axis=1, keepdims=True)


def standing_to_json(s: Standing) -> dict:
    return asdict(s)


def standing_from_json(d: dict) -> Standing:
    """A standing camera as written (by a page too: checked)."""
    numbers = [*d["position"], *d["turn"], d["focal"], *d["shift"]]
    sizes = (len(d["position"]), len(d["turn"]), len(d["shift"]), len(d["size"]))
    whole = all(isinstance(n, int) and n > 0 for n in d["size"])
    if sizes != (3, 3, 2, 2) or not all(map(number, numbers)) or not whole:
        raise ValueError("une caméra debout a une position, une orientation, une focale, un cadre")
    return Standing(
        tuple(float(x) for x in d["position"]),
        tuple(float(x) for x in d["turn"]),
        float(d["focal"]),
        tuple(float(x) for x in d["shift"]),
        tuple(d["size"]),
    )


def start_of(r: Registration, axes: np.ndarray, size: tuple[int, int]) -> tuple:
    """The camera a flat registration is (standing at the panorama): its turn (rotation vector,
    world to camera), focal length and lens shift, as Standing keeps them."""
    h, w = size
    to_world = axes.T @ r.matrix @ np.diag([1 / w, 1 / h, 1.0])  # photo pixels to world rays
    _, lens, turn, *_ = cv2.RQDecomp3x3(np.linalg.inv(to_world))  # (angles, R, Q, ...)
    signs = np.diag(np.sign(np.diag(lens)))  # a positive focal length and depth
    lens, turn = lens @ signs, signs @ turn
    turn = turn * np.sign(np.linalg.det(turn))  # a rotation, not a mirror (lens: scaled below)
    lens = lens / lens[2, 2]
    focal = (lens[0, 0] + lens[1, 1]) / 2
    shift = (lens[0, 2] / w - 0.5, lens[1, 2] / h - 0.5)
    return cv2.Rodrigues(turn)[0].ravel(), float(focal), np.array(shift)


def fit_standing(
    photo_px: np.ndarray,
    world: np.ndarray,
    size: tuple[int, int],
    start: Registration,
    axes: np.ndarray,
    pin: tuple[float, float],
    ground: float,
) -> tuple[Standing, np.ndarray]:
    """The camera where the photographer stood, through pairs clicked by hand whose panorama
    point has a depth (`world`: those points; NaN rows, none: left out), and which pairs it
    used. It starts from the flat fit `start`, at the user's `pin` on a tripod; it stands within
    match.MOST_MOVED_M of the pin, match.STREET_HEIGHT_M above the `ground`; its lens shift is
    fitted from STANDING_SHIFTED_FROM pairs; a pair far from the others counts for less
    (STANDING_OFF of the photo's diagonal)."""
    photo_px, world = np.asarray(photo_px, float), np.asarray(world, float)
    inside = (photo_px >= 0) & (photo_px <= [size[1], size[0]])
    if not inside.all():
        raise NotFitted("des points tombent hors de la photo")
    used = ~np.isnan(world).any(axis=1)
    if used.sum() < MIN_STANDING:
        raise NotFitted(f"il faut au moins {MIN_STANDING} paires avec une profondeur")
    clicked, points = photo_px[used], world[used]
    diagonal = np.hypot(*size)
    turn, focal, shift = start_of(start, axes, size)
    low_z, high_z = (ground + m for m in match.STREET_HEIGHT_M)
    low_f, high_f = (np.log(k * max(size)) for k in STANDING_FOCAL)
    x0 = [*turn, np.clip(np.log(focal), low_f + 0.01, high_f - 0.01), *pin, ground + TRIPOD_M]
    low = [-np.inf] * 3 + [low_f, pin[0] - match.MOST_MOVED_M, pin[1] - match.MOST_MOVED_M, low_z]
    high = [np.inf] * 3 + [high_f, pin[0] + match.MOST_MOVED_M, pin[1] + match.MOST_MOVED_M, high_z]
    shifted = used.sum() >= STANDING_SHIFTED_FROM
    if shifted:
        x0, low, high = [*x0, *np.clip(shift, -0.45, 0.45)], [*low, -0.5, -0.5], [*high, 0.5, 0.5]

    def camera(x: np.ndarray) -> Standing:
        return Standing(
            tuple(x[4:7]),
            tuple(x[:3]),
            float(np.exp(x[3])),
            tuple(x[7:9]) if shifted else (0.0, 0.0),
            size,
        )

    def off(x: np.ndarray) -> np.ndarray:  # in shares of the photo's diagonal
        seen = camera(x).project(points)
        return (np.nan_to_num(seen, nan=1e6) - clicked).ravel() / diagonal

    found = least_squares(
        off, x0, bounds=(low, high), loss="soft_l1", f_scale=STANDING_OFF, xtol=1e-12, ftol=1e-12
    )
    at_limit = np.minimum(found.x[3:7] - low[3:7], np.array(high[3:7]) - found.x[3:7]) < 0.01
    if at_limit[0]:
        raise NotFitted("la focale trouvée bute sur ses limites : vérifier les paires")
    if at_limit[1:].any():
        raise NotFitted("la position trouvée bute sur ses limites : vérifier les paires")
    return camera(found.x), used


def errors_flat(
    r: Registration,
    photo_px: np.ndarray,
    view_px: np.ndarray,
    size: tuple[int, int],
    side: int,
    view: ViewOf,
    axes: np.ndarray,
) -> np.ndarray:
    """For each pair, how far from its photo point the flat fit puts its view point (shares of
    the photo's diagonal)."""
    h, w = size
    back = np.linalg.inv(in_view(r, view, axes))  # the view's [0, 1]² to the photo's
    seen = cv2.perspectiveTransform((np.asarray(view_px, float) / side)[None], back)[0] * [w, h]
    return np.linalg.norm(seen - photo_px, axis=1) / np.hypot(*size)


def errors_standing(s: Standing, photo_px: np.ndarray, world: np.ndarray) -> np.ndarray:
    """For each pair, how far from its photo point the standing camera sees its panorama point
    (shares of the photo's diagonal; NaN for a pair without depth)."""
    return np.linalg.norm(s.project(world) - photo_px, axis=1) / np.hypot(*s.size)


def _held_out(pairs: np.ndarray, refit: Callable, error: Callable) -> np.ndarray:
    """Each pair's error (`error(fitted, i)`) by the fit made without it (`refit(i)`): NaN for
    pairs not in `pairs`, infinite where that fit is refused (the worst, never left out)."""
    found = np.full(len(pairs), np.nan)
    for i in np.flatnonzero(pairs):
        try:
            found[i] = error(refit(i), i)
        except NotFitted:
            found[i] = np.inf
    return found


def held_out_flat(
    photo_px: np.ndarray,
    view_px: np.ndarray,
    size: tuple[int, int],
    side: int,
    view: ViewOf,
    axes: np.ndarray,
) -> np.ndarray:
    """Each pair's error by the flat fit made without it (infinite when it is refused)."""
    others = ~np.eye(len(photo_px), dtype=bool)
    return _held_out(
        np.ones(len(photo_px), bool),
        lambda i: fit(photo_px[others[i]], view_px[others[i]], size, side, "", view, axes)[0],
        lambda r, i: errors_flat(r, photo_px[[i]], view_px[[i]], size, side, view, axes)[0],
    )


def held_out_standing(
    photo_px: np.ndarray,
    world: np.ndarray,
    size: tuple[int, int],
    start: Registration,
    axes: np.ndarray,
    pin: tuple[float, float],
    ground: float,
) -> np.ndarray:
    """Each pair's error by the standing camera fitted without it (NaN without depth, infinite
    when that fit is refused)."""

    def without(i: int) -> Standing:
        rest = world.copy()
        rest[i] = np.nan
        return fit_standing(photo_px, rest, size, start, axes, pin, ground)[0]

    return _held_out(
        ~np.isnan(world).any(axis=1),
        without,
        lambda s, i: errors_standing(s, photo_px[[i]], world[[i]])[0],
    )


def own_directions(view_px: np.ndarray, view: ViewOf, axes: np.ndarray, side: int) -> np.ndarray:
    """The unit directions, in the panorama's own picture, of pixels of a view `side` pixels
    square (corners at whole numbers) cut from it levelled with `axes` (see view_pixels)."""
    unit = np.column_stack([np.asarray(view_px, float) / side, np.ones(len(view_px))])
    own = unit @ _to_camera(view, axes).T
    return own / np.linalg.norm(own, axis=1, keepdims=True)


def in_view(r: Registration, view: ViewOf, axes: np.ndarray) -> np.ndarray:
    """The homography from the photo's [0, 1]² to a view's, cut from the panorama levelled
    with these `axes`."""
    h = np.linalg.inv(_to_camera(view, axes)) @ r.matrix
    return h / abs(h[2, 2])  # its sign kept: points behind the view have w < 0


def directions(r: Registration, points: np.ndarray) -> np.ndarray:
    """The unit directions, in the panorama's own picture, of points of the photo ([0, 1]²)."""
    d = np.column_stack([points, np.ones(len(points))]) @ r.matrix.T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def view_pixels(own: np.ndarray, view: ViewOf, axes: np.ndarray, side: int) -> np.ndarray:
    """Where directions of the panorama's own picture fall in a view `side` pixels square."""
    q = own @ np.linalg.inv(_to_camera(view, axes)).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(q[:, 2:] > 0, q[:, :2] / q[:, 2:] * side, np.nan)


def frame_ok(h: np.ndarray) -> bool:
    """Whether the photo's frame, warped by `h` (photo to view, [0, 1]², whatever its scale),
    lies all on one side of the view's camera (then it is convex), not mirrored, and covers at
    least MIN_FRAME of the view."""
    warped = _CORNERS_1 @ h.T
    w = warped[:, 2] * np.sign(warped[0, 2])
    if np.any(w <= 0):
        return False
    frame = (warped[:, :2] / warped[:, 2:]).astype(np.float32)
    x, y = frame[:, 0], frame[:, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y) <= 0:  # turned the other way
        return False
    inside, _ = cv2.intersectConvexConvex(frame, CORNERS)
    return inside >= MIN_FRAME


def composite(
    view_bgr: np.ndarray, photo_bgr: np.ndarray, r: Registration, view: ViewOf, axes: np.ndarray
) -> np.ndarray:
    """The view (cut from the panorama levelled with `axes`) with the photo laid in where the
    registration puts it, opaque; as it was when the photo is not all ahead of the view."""
    if np.any(_CORNERS_1 @ in_view(r, view, axes).T[:, 2] <= 0):
        return view_bgr.copy()
    (ph, pw), (vh, vw) = photo_bgr.shape[:2], view_bgr.shape[:2]
    half = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])  # centres to corners
    h = (
        np.linalg.inv(half)
        @ np.diag([vw, vh, 1.0])
        @ in_view(r, view, axes)
        @ np.diag([1 / pw, 1 / ph, 1.0])
        @ half
    )
    warped = cv2.warpPerspective(photo_bgr, h, (vw, vh), borderMode=cv2.BORDER_REPLICATE)
    lands = cv2.warpPerspective(
        np.full((ph, pw), 255, np.uint8), h, (vw, vh), flags=cv2.INTER_NEAREST
    )
    laid = view_bgr.copy()
    cv2.copyTo(warped, lands, laid)
    return laid


def composite_standing(
    view_bgr: np.ndarray,
    photo_bgr: np.ndarray,
    s: Standing,
    origin: np.ndarray,
    view: ViewOf,
    metres: np.ndarray,
) -> np.ndarray:
    """The view (square, cut from the panorama at `origin`) with the photo laid in through
    depth: each pixel's point of today (`metres` along its ray; the sky infinitely far) seen by
    the standing camera, unless a nearer point hides it from the camera (HIDDEN times farther
    than the nearest seen in that part of the photo, as wide as a pixel of the view). Where the
    photo does not reach, or the depth is unknown (NaN), the view as it was."""
    side = view_bgr.shape[0]
    if view_bgr.shape[1] != side:
        raise ValueError("une vue est carrée")
    h, w = s.size
    pixels = match.pixels_of((side, side))
    world = match.view_directions(pixels, view.heading, view.fov, side, view.pitch)
    world = world.astype(np.float32)
    lens = (s._lens() @ cv2.Rodrigues(np.array(s.turn))[0]).astype(np.float32)
    step = np.asarray(origin, float) - s.position  # from the camera to the panorama
    far = np.asarray(metres, float).ravel()
    with np.errstate(divide="ignore"):
        nearness = (1 / far).astype(np.float32)  # the sky 0, unknown NaN
    # each pixel's point, from the camera, its distance along the panorama's ray divided out
    seen = world @ lens.T + nearness[:, None] * (lens @ step).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = seen[:, :2] / seen[:, 2:]
    inside = (seen[:, 2] > 0) & np.all((at >= 0) & (at < [w, h]), axis=1)
    # what the camera sees in each part of its photo is the nearest point there
    away = np.full(len(far), np.inf)
    finite = np.isfinite(far)
    away[finite] = np.linalg.norm(world[finite] * far[finite, None] + step, axis=1)
    cell = max(1, int(np.ceil(s.focal / match.focal(view.fov, side))))  # a view pixel, in it
    part = np.where(inside[:, None], at // cell, 0).astype(int)
    key = part[:, 1] * (w // cell + 1) + part[:, 0]
    nearest = np.full((h // cell + 1) * (w // cell + 1), np.inf)
    np.minimum.at(nearest, key[inside], away[inside])
    inside &= ~(away > HIDDEN * nearest[key])
    maps = np.where(inside[:, None], at - 0.5, -1.0).astype(np.float32).reshape(side, side, 2)
    warped = cv2.remap(
        photo_bgr, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return np.where(inside.reshape(side, side, *[1] * (view_bgr.ndim - 2)), warped, view_bgr)


def carried(
    r: Registration,
    view: ViewOf,
    axes: np.ndarray,
    onto: str,
    onto_view: ViewOf,
    onto_axes: np.ndarray,
    h_between: np.ndarray,
) -> Registration:
    """`r`, seen in `view` of its panorama, carried onto panorama `onto` by a homography from
    that view to `onto_view` of it ([0, 1]²)."""
    h = h_between @ in_view(r, view, axes)
    return of(onto, _to_camera(onto_view, onto_axes) @ h, onto_view, onto_axes)


def to_json(r: Registration) -> dict:
    return asdict(r)


def from_json(d: dict) -> Registration:
    """A registration as written (by a page too: checked)."""
    m = d["m"]
    if not (isinstance(m, list | tuple) and len(m) == 9 and all(number(x) for x in m)):
        raise ValueError("un enregistrement a 9 nombres")
    return Registration(str(d["panorama"]), tuple(float(x) for x in m), ViewOf(**d["view"]))
