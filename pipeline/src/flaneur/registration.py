"""An old photo registered onto a panorama: the directions, in the panorama's own picture, that
the photo's points show (a 3×3 matrix from the photo's [0, 1]² to directions in the camera frame
of the 360° picture, see match.camera_axes). Kept in the picture's own frame, it does not change
when the panorama is levelled again; it is worked on, checked and drawn in a perspective view cut
from the levelled panorama (match.perspective), where it is a homography. Exact for one plane (a
facade) or a photographer standing at the panorama; a street in depth lines up only roughly.

Points are in pixels with a pixel's corners at whole numbers (as a browser's clicks are), and
normalised to [0, 1] by the picture's width and height, x right, y down."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from flaneur import match
from flaneur.sources import number

MIN_POINTS = 6  # clicked by hand: Smapshot, Bae et al. and IGN settle at 6 to 8
MIN_COVER = 0.25  # share of the photo inside the hull of the points fitted: under it, bunched
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
    local = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, match.focal(view.fov, 1.0)]])
    return axes @ match._view_axes(view.heading, view.pitch).T @ local


def _of(panorama: str, m: np.ndarray, view: ViewOf, axes: np.ndarray) -> Registration:
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
    `axes`; and which pairs it kept. Clicked by hand, every pair as the user meant it; matched
    (`robust`), those that agree, the others left out."""
    photo_px, view_px = np.asarray(photo_px, float), np.asarray(view_px, float)
    photo = photo_px / [photo_size[1], photo_size[0]]
    if np.any((photo < 0) | (photo > 1)) or np.any((view_px < 0) | (view_px > side)):
        raise NotFitted("des points tombent hors de la photo ou de la vue")
    kept = np.ones(len(photo), bool)
    h = None
    if len(photo) >= 4:  # fitted in pixels: MAGSAC's threshold is in pixels
        method = cv2.USAC_MAGSAC if robust else 0
        h, mask = cv2.findHomography(photo_px, view_px, method, MATCHED_PX)
        if robust and mask is not None:
            kept = mask.ravel().astype(bool)
    if robust and kept.sum() < MIN_MATCHED:
        raise NotFitted(f"il faut au moins {MIN_MATCHED} points qui s'accordent")
    if kept.sum() < MIN_POINTS:
        raise NotFitted(f"il faut au moins {MIN_POINTS} paires de points")
    if cv2.contourArea(cv2.convexHull(photo[kept].astype(np.float32))) < MIN_COVER:
        raise NotFitted("points trop groupés : les répartir sur toute la photo")
    if h is None:
        raise NotFitted("ces points ne donnent aucune déformation de la photo")
    in_unit = np.diag([1 / side, 1 / side, 1.0]) @ h @ np.diag([photo_size[1], photo_size[0], 1.0])
    if not frame_ok(in_unit):
        raise NotFitted("la photo se replie, se retourne ou sort presque de la vue")
    return _of(panorama, _to_camera(view, axes) @ in_unit, view, axes), kept


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
    return _of(onto, _to_camera(onto_view, onto_axes) @ h, onto_view, onto_axes)


def to_json(r: Registration) -> dict:
    return asdict(r)


def from_json(d: dict) -> Registration:
    """A registration as written (by a page too: checked)."""
    m = d["m"]
    if not (isinstance(m, list | tuple) and len(m) == 9 and all(number(x) for x in m)):
        raise ValueError("un enregistrement a 9 nombres")
    return Registration(str(d["panorama"]), tuple(float(x) for x in m), ViewOf(**d["view"]))
