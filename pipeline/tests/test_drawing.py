"""A photo laid in a view by its standing camera, without holes (drawing.composite_standing)."""

import cv2
import numpy as np
import pytest

from flaneur import drawing, facades, match, registration
from flaneur.registration import ViewOf

SIDE = 320
SIZE = (600, 800)  # the photo: height, width
GROUND = -2.5  # the street, under the panorama's camera
NORTH = ViewOf(heading=0.0)  # the panorama's view up the street
FACING_NORTH = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # world to camera
STANDING = registration.Standing(
    position=(1.0, -2.0, GROUND + 1.5),  # 3 m from the panorama, on a tripod
    turn=tuple(cv2.Rodrigues(FACING_NORTH)[0].ravel()),
    focal=500.0,  # a wide lens: 77° across
    shift=(0.0, 0.0),
    size=SIZE,
)
ACROSS = facades.Wall((-60.0, 20.0), (60.0, 20.0), 40.0, "fitted")  # a facade 20 m north


def _scene(*walls: facades.Wall) -> facades.Scene:
    return facades.Scene("p", (0.0, 0.0, 0.0), GROUND, walls, (), 0)


def _coordinates() -> np.ndarray:
    """A photo whose every pixel holds its column, its row and 1: what lands where can be read,
    and the third channel, laid on a black view, is how much of the photo shows."""
    rows, columns = np.mgrid[0 : SIZE[0], 0 : SIZE[1]].astype(np.float32)
    return np.dstack([columns, rows, np.ones_like(rows)])


def _rays() -> np.ndarray:
    pixels = match.pixels_of((SIDE, SIDE))
    return match.view_directions(pixels, NORTH.heading, NORTH.fov, SIDE, NORTH.pitch)


def _within(at: np.ndarray) -> np.ndarray:
    """Points of the photo three pixels or more inside its edge (at the edge, its pixels repeat)."""
    return np.all((at >= 3) & (at <= [SIZE[1] - 3, SIZE[0] - 3]), axis=-1)


def _laid(scene, view=NORTH, metres=None, plate=None, label=None) -> np.ndarray:
    blank = np.zeros((SIDE, SIDE, 3), np.float32)
    return drawing.composite_standing(
        blank, _coordinates(), STANDING, scene, view, metres, plate, label
    )


def test_the_photo_is_laid_where_its_camera_sees_the_walls():
    scene = _scene(ACROSS)
    laid = _laid(scene)

    rays = _rays()
    points = rays * facades.background(scene, rays)[:, None]
    expected = STANDING.project(points).reshape(SIDE, SIDE, 2)
    full = (laid[..., 2] > 0.999) & _within(expected)  # wholly the photo, off its very edge
    assert full.mean() > 0.2
    assert np.abs(laid[..., :2][full] - (expected[full] - 0.5)).max() < 0.1  # pixel centres


def test_the_sky_is_laid_by_the_camera_turn_alone():
    laid = _laid(_scene(), metres=np.full((SIDE, SIDE), np.inf))  # all the sky

    far = (_rays() * 1e7).reshape(-1, 3)
    expected = STANDING.project(far + STANDING.position).reshape(SIDE, SIDE, 2)
    full = (laid[..., 2] > 0.999) & _within(expected)
    assert full.any() and np.abs(laid[..., :2][full] - (expected[full] - 0.5)).max() < 0.1


def test_a_view_the_photo_does_not_reach_is_left_as_it_was():
    laid = _laid(_scene(ACROSS), ViewOf(heading=180.0))

    assert (laid == 0).all()  # facing away from the photo


def test_beside_the_photo_s_frame_the_view_stays_and_inside_it_the_edge_fades():
    metres = np.full((SIDE, SIDE), 30.0)
    metres[:, : SIDE // 2] = np.nan  # depth unknown on the left: the photo does not reach there

    laid = _laid(_scene(), metres=metres)

    shown = laid[..., 2]
    assert (shown[:, : SIDE // 2] == 0).all()
    inner = shown[SIDE // 2, SIDE // 2 :]
    assert inner[0] < 0.2 and inner[40:].min() > 0.999  # soft over 1°, then whole


def test_a_point_hidden_from_the_photographer_weighs_nothing():
    pole = facades.Wall((0.0, 6.0), (0.1, 6.0), 40.0, "fitted")  # 10 cm wide, 6 m north
    scene = _scene(ACROSS, pole)
    rays = _rays()
    far = facades.background(scene, rays)
    at, reached = drawing._seen(STANDING, scene.origin, rays, far, None)

    hidden = drawing._hidden(STANDING, scene, rays, far, reached)

    on_pole = facades.cast(scene, rays)[1] == 1  # what the photographer sees of the pole
    assert hidden.any() and not (hidden & on_pole).any()
    assert not drawing._hidden(STANDING, _scene(ACROSS), rays, far, reached).any()


def test_a_point_hidden_from_the_photographer_is_filled_not_cut():
    pole = facades.Wall((0.0, 6.0), (0.1, 6.0), 40.0, "fitted")
    scene = _scene(ACROSS, pole)
    rays = _rays()
    far = facades.background(scene, rays)
    _, reached = drawing._seen(STANDING, scene.origin, rays, far, None)
    hidden = drawing._hidden(STANDING, scene, rays, far, reached).reshape(SIDE, SIDE)

    laid = _laid(scene)

    shown = laid[..., 2]
    around = hidden & (np.roll(shown, 10, axis=1) > 0.999)  # off the frame's soft edge
    assert around.sum() > 20 and shown[around].min() > 0.9  # no slit


def test_where_the_photo_would_be_stretched_it_fades_between_six_and_twelve_times():
    side = 64
    rows, cols = np.mgrid[0:side, 0:side].astype(np.float32)
    at = np.stack([cols * 0.6, rows * 0.6], axis=-1)  # the photo laid as a whole: 0.6 px a pixel
    at[:, 20:30, 0] = 20 * 0.6 + (cols[:, 20:30] - 20) * 0.075  # 8 times finer
    at[:, 30:, 0] = at[:, 29:30, 0] + (cols[:, 30:] - 29) * 0.6
    at[:, 52:, 0] = at[:, 51:52, 0] + (cols[:, 52:] - 51) * 0.02  # a wall seen edge on: 30 times

    kept = drawing.stretch_kept(at, np.ones((side, side), bool))

    assert (kept[:, 54:62] == 0).all()  # smeared: gone
    assert (kept[:, 3:17] == 1).all() and (kept[:, 33:48] == 1).all()
    assert 0 < kept[32, 24] < 1  # 8 times: fading


def test_an_albert_kahn_photo_shows_its_plate_and_never_its_label():
    plate, label = (0.1, 0.05, 0.95, 0.9), (0.1, 0.7, 0.3, 0.9)
    laid = _laid(_scene(ACROSS), plate=plate, label=label)

    full = laid[..., 2] > 0.999
    at = laid[..., :2][full] / laid[..., 2:][full]
    h, w = SIZE
    assert full.mean() > 0.1
    assert at[:, 0].min() >= plate[0] * w - 1 and at[:, 1].max() <= plate[3] * h + 1
    in_label = (
        (at[:, 0] > label[0] * w + 2)
        & (at[:, 0] < label[2] * w - 2)
        & (at[:, 1] > label[1] * h + 2)
        & (at[:, 1] < label[3] * h - 2)
    )
    assert not in_label.any()


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_the_view_keeps_its_kind(dtype):
    view = np.full((SIDE, SIDE, 3), 50, dtype)
    photo = np.full((*SIZE, 3), 200, dtype)

    laid = drawing.composite_standing(view, photo, STANDING, _scene(ACROSS), NORTH)

    assert laid.dtype == dtype and laid.max() == 200 and laid.min() == 50
