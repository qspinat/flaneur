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


def _read(laid: np.ndarray) -> np.ndarray:
    """Where in the photo each pixel of a layer laid on black was read from."""
    return laid[..., :2] / np.maximum(laid[..., 2:], 1e-9)


def _laid(scene, view=NORTH, depth=None, flipped=False, plate=None, label=None) -> np.ndarray:
    blank = np.zeros((SIDE, SIDE, 3), np.float32)
    return drawing.composite_standing(
        blank, _coordinates(), STANDING, scene, view, depth, flipped, plate, label
    )


def test_the_photo_is_laid_where_its_camera_sees_the_walls():
    scene = _scene(ACROSS)
    laid = _laid(scene)

    rays = _rays()
    points = rays * facades.background(scene, rays)[0][:, None]
    expected = STANDING.project(points).reshape(SIDE, SIDE, 2)
    full = (laid[..., 2] > 0.999) & _within(expected)  # wholly the photo, off its very edge
    assert full.mean() > 0.2
    assert np.abs(_read(laid)[full] - (expected[full] - 0.5)).max() < 0.1  # pixel centres


def test_the_sky_is_laid_by_the_camera_turn_alone():
    sky = np.ones((SIDE, SIDE), bool)
    laid = _laid(_scene(), depth=(np.full((SIDE, SIDE), np.inf), sky))  # all the sky

    far = (_rays() * 1e7).reshape(-1, 3)
    expected = STANDING.project(far + STANDING.position).reshape(SIDE, SIDE, 2)
    full = (laid[..., 2] > 0.999) & _within(expected)
    assert full.any() and np.abs(_read(laid)[full] - (expected[full] - 0.5)).max() < 0.1


def test_a_view_the_photo_does_not_reach_is_left_as_it_was():
    laid = _laid(_scene(ACROSS), ViewOf(heading=180.0))

    assert (laid == 0).all()  # facing away from the photo


def test_beside_the_photo_the_view_stays_and_across_its_edge_it_fades():
    metres = np.full((SIDE, SIDE), 30.0)
    metres[:, : SIDE // 2] = np.nan  # depth unknown on the left: the photo does not reach there
    one_degree = round(SIDE / NORTH.fov)

    laid = _laid(_scene(), depth=(metres, np.zeros((SIDE, SIDE), bool)))

    shown = laid[..., 2]
    assert shown[:, : SIDE // 2 - one_degree].max() < 0.01  # nothing 1° beyond its edge
    row = shown[SIDE // 2, SIDE // 2 - one_degree : SIDE // 2 + 3 * one_degree]
    assert ((row > 0.1) & (row < 0.9)).any() and row[-1] > 0.99  # soft across it
    assert np.all(np.diff(row) >= -1e-6)


def _pole_scene():
    pole = facades.Wall((0.0, 6.0), (0.1, 6.0), 40.0, "fitted")  # 10 cm wide, 6 m north
    scene = _scene(ACROSS, pole)
    rays = _rays()
    metres = facades.background(scene, rays)[0]
    _, ahead = drawing._seen(STANDING, scene.origin, rays, metres)
    hidden = drawing._hidden(STANDING, scene, rays, metres, ahead, SIDE)
    return scene, rays, hidden


def test_a_point_hidden_from_the_photographer_weighs_nothing():
    scene, rays, hidden = _pole_scene()

    on_pole = facades.cast(scene, rays)[1] == 1  # what the photographer sees of the pole
    assert hidden.any() and not (hidden & on_pole).any()


def test_a_point_hidden_from_the_photographer_is_filled_not_cut():
    scene, _, hidden = _pole_scene()

    shown = _laid(scene)[..., 2]

    hidden = hidden.reshape(SIDE, SIDE)
    around = hidden & (np.roll(shown, 10, axis=1) > 0.999)  # off the photo's soft edge
    assert around.sum() > 20 and shown[around].min() > 0.9  # no slit


def test_where_the_photo_would_be_smeared_it_fades():
    side = 64
    rows, cols = np.mgrid[0:side, 0:side].astype(np.float32)
    at = np.stack([cols * 0.6, rows * 0.6], axis=-1)  # 0.6 photo pixels a view pixel
    at[:, 32:, 0] = at[:, 31:32, 0] + (cols[:, 32:] - 31) * 0.02  # a wall seen edge on: 30 times
    rays = np.dstack([cols * 1e-3, np.ones_like(cols), rows * 1e-3])  # 1 mrad a pixel
    density = 600.0  # the photo as a whole: 600 pixels a radian, 0.6 a view pixel
    no_sky = np.zeros((side, side), bool)

    kept = drawing.stretch_kept(at, rays, no_sky, density)
    sky = drawing.stretch_kept(at, rays, np.ones((side, side), bool), density)

    assert (kept[:, 3:28] == 1).all() and (kept[:, 36:60] == 0).all()
    assert (sky == 1).all()  # the sky is never smeared


def test_a_mirrored_print_is_read_from_its_scan_its_plate_as_the_scan_s():
    plate = (0.0, 0.0, 0.5, 1.0)  # the scan's left half

    laid = _laid(_scene(ACROSS), flipped=True, plate=plate)

    full = laid[..., 2] > 0.999
    columns = laid[..., 0][full] / laid[..., 2][full]
    assert full.mean() > 0.05 and columns.max() <= 0.5 * SIZE[1] + 1  # only its left half


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


def test_the_museum_s_label_shows_the_view_not_a_fill_of_the_plate():
    label = (0.3, 0.4, 0.7, 0.8)  # a large label in the middle of the scan
    laid = _laid(_scene(ACROSS), label=label)

    rays = _rays()
    at = STANDING.project(rays * facades.background(_scene(ACROSS), rays)[0][:, None])
    h, w = SIZE
    middle = (np.abs(at[:, 0] - 0.5 * w) < 0.05 * w) & (np.abs(at[:, 1] - 0.6 * h) < 0.05 * h)
    assert middle.any() and laid[..., 2].ravel()[middle].max() < 0.1


def test_the_photo_is_laid_on_the_walls_the_walk_shows_their_corners_joined():
    gap = (
        facades.Wall((-30.0, 20.0), (-0.5, 20.0), 40.0, "fitted"),
        facades.Wall((0.5, 20.0), (30.0, 20.0), 40.0, "fitted"),  # 1 m short of the other
    )

    walked = drawing.as_walked(_scene(*gap))

    assert walked.walls[0].b == pytest.approx(walked.walls[1].a)  # one corner, no gap


def test_the_sky_is_never_hidden_from_the_photographer():
    pole = facades.Wall((0.0, 6.0), (0.1, 6.0), 40.0, "fitted")
    scene = _scene(pole)
    rays = _rays()
    metres = facades.background(scene, rays)[0]
    _, ahead = drawing._seen(STANDING, scene.origin, rays, metres)

    hidden = drawing._hidden(STANDING, scene, rays, metres, ahead, SIDE)

    assert not hidden[~np.isfinite(metres)].any()
