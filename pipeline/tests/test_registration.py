"""An old photo registered onto a panorama, worked on in a view cut from it."""

import cv2
import numpy as np
import pytest

from flaneur import match, registration
from flaneur.registration import ViewOf

SIDE = 640
VIEW = ViewOf(heading=140.0)
TILTED = match.camera_axes(198.0, -8.4, 6.9)  # a panorama leaning, as levelled
H = np.array([[0.6, 0.05, 0.2], [-0.02, 0.55, 0.25], [0.0001, 0.0002, 1.0]])
SPREAD = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9], [0.5, 0.4], [0.3, 0.7]])


def _camera(turn, focal: float, centre=(400.0, 300.0), size=(600, 800)) -> np.ndarray:
    """From the photo's [0, 1]² to VIEW's: a camera at the panorama turned by `turn` (a rotation
    vector) from the view's, of `focal` pixels, its lens centred on `centre` (pixels)."""
    h, w = size
    rays = np.array([[1.0, 0.0, -centre[0]], [0.0, 1.0, -centre[1]], [0.0, 0.0, focal]])
    view = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, match.focal(VIEW.fov, 1.0)]])
    m = np.linalg.inv(view) @ cv2.Rodrigues(np.array(turn, float))[0] @ rays @ np.diag([w, h, 1])
    return m / m[2, 2]


CAMERA = _camera([0.05, -0.1, 0.03], 900.0)  # looking a little aside, the lens centred
RAISED = _camera([0.05, -0.1, 0.03], 900.0, (430.0, 380.0))  # its lens raised and to the side


def _onto(points: np.ndarray, h: np.ndarray = CAMERA) -> np.ndarray:
    return cv2.perspectiveTransform(points[None], h)[0]


def _fit(points: np.ndarray, axes: np.ndarray = TILTED, view: ViewOf = VIEW, h=CAMERA, **kw):
    return registration.fit(
        points * [800, 600], _onto(points, h) * SIDE, (600, 800), SIDE, "p1", view, axes, **kw
    )


def _seen(r: registration.Registration) -> np.ndarray:
    h = registration.in_view(r, VIEW, TILTED)
    return h / h[2, 2]


def test_a_camera_is_found_back_from_points_clicked():
    r, kept = _fit(SPREAD, h=RAISED)

    assert kept.all()
    assert np.allclose(_seen(r), RAISED, atol=1e-6)


def test_two_pairs_place_a_photo_by_its_direction_and_zoom():
    r, _ = _fit(np.array([[0.2, 0.3], [0.8, 0.6]]))

    assert np.allclose(_seen(r), CAMERA, atol=1e-6)


def test_a_pair_clicked_off_moves_the_photo_but_never_folds_it():
    # a detail near the camera shifts against those far away: its pair agrees with none
    points = np.vstack([SPREAD, [[0.6, 0.8]]])
    seen = _onto(points) * SIDE
    seen[-1] += [70.0, -50.0]

    r, _ = registration.fit(points * [800, 600], seen, (600, 800), SIDE, "p1", VIEW, TILTED)

    assert registration.frame_ok(_seen(r))
    assert np.abs(_onto(SPREAD, _seen(r)) - _onto(SPREAD)).max() * SIDE < 40


def test_the_registration_does_not_depend_on_how_the_panorama_was_levelled():
    # the same details of the panorama's own picture, clicked in views cut under two levellings
    first, _ = _fit(SPREAD)
    seen = _onto(SPREAD, registration.in_view(first, VIEW, TILTED))  # where they lie in VIEW
    own = registration.directions(first, SPREAD)  # their directions in the panorama's picture
    relevelled = match.camera_axes(198.0, -6.0, 4.0)
    in_new_view = registration.view_pixels(own, VIEW, relevelled, SIDE)

    again, _ = registration.fit(
        SPREAD * [800, 600], in_new_view, (600, 800), SIDE, "p1", VIEW, relevelled
    )

    assert np.allclose(again.matrix / again.matrix[2, 2], first.matrix / first.matrix[2, 2])
    assert not np.allclose(in_new_view, seen * SIDE)  # the views did differ


def test_one_pair_is_refused():
    with pytest.raises(registration.NotFitted, match="2"):
        _fit(SPREAD[:1])


def test_pairs_too_close_together_are_refused():
    with pytest.raises(registration.NotFitted, match="groupés"):
        _fit(np.array([[0.5, 0.5], [0.53, 0.52], [0.51, 0.55]]))


def test_points_clicked_across_the_middle_of_a_mounted_print_are_fitted():
    # a print on its card: the details one can match lie in a band across its middle
    band = np.array([[0.3, 0.35], [0.7, 0.35], [0.7, 0.65], [0.3, 0.65], [0.5, 0.45], [0.4, 0.6]])

    r, kept = _fit(band, h=RAISED)

    assert kept.all()
    assert np.allclose(_seen(r), RAISED, atol=1e-6)


def test_points_matched_in_that_band_alone_are_still_refused():
    band = np.random.default_rng(2).uniform([0.3, 0.35], [0.7, 0.65], (20, 2))

    with pytest.raises(registration.NotFitted, match="groupés"):
        _fit(band, robust=True)


def test_matched_points_are_fitted_robustly_and_their_outliers_left_out():
    points = np.vstack([SPREAD, np.random.default_rng(1).uniform(0.05, 0.95, (30, 2))])
    view = _onto(points, H) * SIDE  # matched points: any flat warp
    view[:4] += 80  # four wrong matches

    r, kept = registration.fit(
        points * [800, 600], view, (600, 800), SIDE, "p1", VIEW, TILTED, robust=True
    )

    assert kept.sum() == len(points) - 4 and not kept[:4].any()
    assert np.allclose(registration.in_view(r, VIEW, TILTED), H, atol=1e-4)


@pytest.mark.parametrize(
    "h",
    [
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.6, 0.0, 1.0]]),  # the frame folds over
        np.diag([0.2, 0.2, 1.0]),  # 4% of the view
    ],
    ids=["folded", "tiny"],
)
def test_a_frame_that_folds_or_is_tiny_fails_the_shape_check(h):
    assert not registration.frame_ok(h)


def test_a_frame_well_inside_the_view_passes_the_shape_check():
    assert registration.frame_ok(H)


def test_a_registration_carried_by_a_shift_in_the_view_moves_with_it():
    r, _ = _fit(SPREAD)
    shift = np.array([[1.0, 0.0, -0.1], [0.0, 1.0, 0.05], [0.0, 0.0, 1.0]])
    there = match.camera_axes(10.0, 1.0, -2.0)
    onto = ViewOf(heading=150.0)

    carried = registration.carried(r, VIEW, TILTED, "p2", onto, there, shift)

    assert carried.panorama == "p2"
    assert np.allclose(
        _onto(SPREAD, registration.in_view(carried, onto, there)), _onto(SPREAD) + [-0.1, 0.05]
    )


def test_a_registration_survives_json():
    r, _ = _fit(SPREAD)

    assert registration.from_json(registration.to_json(r)) == r


def test_the_composite_lays_the_photo_where_the_registration_says():
    r, _ = _fit(SPREAD)
    photo = np.full((600, 800, 3), 255, np.uint8)
    view = np.zeros((SIDE, SIDE, 3), np.uint8)

    laid = registration.composite(view, photo, r, VIEW, TILTED)

    inside = (_onto(np.array([[0.5, 0.5]]))[0] * SIDE).astype(int)
    assert laid[inside[1], inside[0]].tolist() == [255, 255, 255]
    assert laid[5, 5].tolist() == [0, 0, 0]  # the corner of the view, outside the photo


def test_a_photo_below_the_horizon_points_where_it_is_seen():
    low = ViewOf(heading=140.0, pitch=-20.0)
    lower = np.array([[0.4, 0.0, 0.3], [0.0, 0.4, 0.55], [0.0, 0.0, 1.0]])  # the view's lower part
    r, _ = registration.fit(
        SPREAD * [800, 600], _onto(SPREAD, lower) * SIDE, (600, 800), SIDE, "p1", low, TILTED
    )

    seen = match.view_directions(np.array([[0.5, 0.75]]) * SIDE, 140.0, low.fov, SIDE, -20.0)
    own = seen @ TILTED.T  # that direction in the panorama's own picture
    assert registration.directions(r, np.array([[0.5, 0.5]]))[0] @ own[0] > 0.99


def test_the_views_rays_are_those_of_the_cut_views():
    r, _ = _fit(SPREAD)
    pixels = np.array([[100.0, 200.0], [500.0, 50.0]])

    own = match.view_directions(pixels - 0.5, VIEW.heading, VIEW.fov, SIDE, VIEW.pitch) @ TILTED.T

    assert np.allclose(registration.view_pixels(own, VIEW, TILTED, SIDE), pixels, atol=1e-6)
    assert r.panorama == "p1"


def test_a_direction_behind_the_view_falls_nowhere_in_it():
    forward = (
        match.view_directions(np.array([[319.5, 319.5]]), VIEW.heading, VIEW.fov, SIDE, VIEW.pitch)
        @ TILTED.T
    )

    assert np.isnan(registration.view_pixels(-forward, VIEW, TILTED, SIDE)).all()


def test_a_mirrored_frame_fails_the_shape_check():
    assert not registration.frame_ok(np.array([[-0.6, 0.0, 0.8], [0.0, 0.6, 0.2], [0.0, 0.0, 1.0]]))


def test_the_shape_check_does_not_depend_on_the_matrix_scale():
    assert registration.frame_ok(-3 * H)


def test_points_outside_the_photo_are_refused():
    points = SPREAD.copy()
    points[0] = [-0.5, -0.5]

    with pytest.raises(registration.NotFitted, match="hors"):
        _fit(points)


def test_matched_points_need_fifteen_that_agree():
    with pytest.raises(registration.NotFitted, match="15"):
        _fit(SPREAD, robust=True)


def test_points_whose_warp_fails_the_shape_check_are_refused():
    tiny = np.diag([0.2, 0.2, 1.0])
    with pytest.raises(registration.NotFitted, match="vue"):
        registration.fit(
            SPREAD * [800, 600], _onto(SPREAD, tiny) * SIDE, (600, 800), SIDE, "p1", VIEW, TILTED
        )


def test_the_composite_has_no_dark_seam_at_the_photos_edge():
    r, _ = _fit(SPREAD)
    laid = registration.composite(
        np.zeros((SIDE, SIDE, 3), np.uint8), np.full((600, 800, 3), 255, np.uint8), r, VIEW, TILTED
    )

    assert set(np.unique(laid)) <= {0, 255}


def test_a_photo_partly_behind_the_view_is_not_drawn():
    r, _ = _fit(SPREAD)
    away = ViewOf(heading=VIEW.heading + 120)
    view = np.zeros((SIDE, SIDE, 3), np.uint8)

    assert (
        registration.composite(view, np.full((600, 800, 3), 255, np.uint8), r, away, TILTED) == 0
    ).all()


@pytest.mark.parametrize(
    "m", [[1.0] * 8, ["1"] * 9, [float("nan")] + [1.0] * 8], ids=["eight", "texts", "not a number"]
)
def test_a_registration_written_wrong_is_refused(m):
    with pytest.raises(ValueError, match="9 nombres"):
        registration.from_json({"panorama": "p1", "m": m, "view": {"heading": 0.0}})
