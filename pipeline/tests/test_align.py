import cv2
import numpy as np
import pytest

from flaneur import align


def drawn_street(vanishing=(0.6, 0.55), size=(800, 1000)) -> np.ndarray:
    """Lines of facades and kerbs running to one point, as in a photo along a street."""
    w, h = size
    image = np.full((h, w, 3), 200, np.uint8)
    vx, vy = vanishing[0] * w, vanishing[1] * h
    for start in [(0, 50), (0, 250), (0, 900), (w, 30), (w, 300), (w, 950), (100, h), (700, h)]:
        t = 0.75  # stop before the point, as real lines do
        end = (start[0] + t * (vx - start[0]), start[1] + t * (vy - start[1]))
        cv2.line(image, start, tuple(int(c) for c in end), (30, 30, 30), 4)
    return image


def test_vanishing_point_of_a_street():
    found = align.vanishing_point(cv2.cvtColor(drawn_street((0.6, 0.55)), cv2.COLOR_BGR2GRAY))

    assert found == pytest.approx((0.6, 0.55), abs=0.02)


def test_street_direction_takes_the_facade_direction_nearest_the_guess():
    bearings = [(10, 30.0), (190, 20.0), (100, 25.0)]  # (bearing, metres of facade)

    assert align.street_direction(bearings, guess=170) == pytest.approx(190)
    assert align.street_direction(bearings, guess=85) == pytest.approx(100)
    assert align.street_direction(bearings, guess=145) is None  # nothing within 40°


def test_street_direction_adds_up_short_facades():
    main_street = [(90 + d, 6.0) for d in (-1, 0, 1, 0.5, -0.5)]  # 30 m in short pieces
    cross_street = [(120, 12.0)]

    assert align.street_direction(main_street + cross_street, guess=100) == pytest.approx(90, abs=1)


def test_orient_turns_towards_the_vanishing_point():
    f = 1 / 2 / np.tan(np.radians(53) / 2)  # focal length, in photo widths

    heading, horizon = align.orient(guess=95, vanishing=(0.75, 0.62), street=90, fov=53)

    assert heading == pytest.approx(90 - np.degrees(np.arctan(0.25 / f)))
    assert horizon == 0.62


def test_orient_gives_nothing_without_a_vanishing_point_or_a_street():
    assert align.orient(guess=95, vanishing=None, street=90, fov=53) is None
    assert align.orient(guess=95, vanishing=(0.5, 0.6), street=None, fov=53) is None


def test_orient_gives_nothing_when_the_lines_turn_too_far():
    assert align.orient(guess=0, vanishing=(0.5, 0.6), street=90, fov=53) is None


def test_a_camera_found_against_today_is_proposed_as_it_was_found():
    from flaneur.match import Situated

    walls = [{"line": [[1, 6], [21, 6]], "height": 18.5}]
    found = Situated(3.0, -2.0, 1.7, 14.0, 0.62, 45, 40, 3, 12.0, walls, "setup", 0.3)

    proposal = align.matched("pm:1", found)

    pose = {"x": 3.0, "y": -2.0, "z": 1.7, "heading": 14.0, "horizon": 0.62, "fov": 45}
    assert proposal == {"photo_id": "pm:1", "status": "matched", "pose": pose, "walls": walls}
