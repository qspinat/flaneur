"""Filling a weighted layer: small gaps interpolated, large ones fading, the frame's edge soft."""

import numpy as np
import pytest

from flaneur import fill


def _layer(side=200, value=100.0):
    colour = np.full((side, side, 3), value, np.float32)
    return colour, np.ones((side, side), np.float32)


def test_levels_follow_the_angle():
    assert fill.levels(90 / 700) == 5  # a sheet's view: 2° is 16 pixels
    assert fill.levels(90 / 1400) == 6  # twice as fine: one level more, the same angle


def test_a_slit_three_pixels_wide_is_filled():
    colour, weight = _layer()
    colour[:, :100], colour[:, 100:] = 80.0, 120.0
    weight[:, 99:102] = 0.0  # a slit

    rgb, coverage = fill.push_pull(colour, weight, 5)

    assert fill.alpha(coverage)[:, 99:102].min() >= 0.9
    assert 70 <= rgb[:, 99:102].min() and rgb[:, 99:102].max() <= 130


def test_a_gap_wider_than_the_reach_fades_smoothly():
    colour, weight = _layer(side=400)
    weight[:, 150:250] = 0.0  # 100 pixels: far more than 16

    _, coverage = fill.push_pull(colour, weight, 5)
    alpha = fill.alpha(coverage)

    assert alpha[:, 195:205].max() <= 0.1
    assert np.abs(np.diff(alpha[200])).max() <= 0.2  # no step


def test_the_frame_closes_notches_narrower_than_the_reach_and_keeps_wider_ones():
    reached = np.ones((200, 200), bool)
    reached[:30, 60:70] = False  # a notch 10 wide from the top edge
    reached[:30, 120:170] = False  # one 50 wide
    reached[90:100, 90:100] = False  # a hole inside

    frame = fill.frame(reached, 16)

    assert frame[:30, 60:70].all() and frame[90:100, 90:100].all()
    assert not frame[:20, 130:160].any()


def test_the_edge_fades_inside_the_frame_only():
    frame = np.zeros((100, 100), bool)
    frame[:, 20:] = True

    soft = fill.edge(frame, 8)

    assert (soft[:, :20] == 0).all()
    assert (soft[:, 21] < 0.2).all() and (soft[:, 40:] == 1).all()
    assert np.all(np.diff(soft[50, 20:40]) >= 0)  # rising into the photo


@pytest.mark.parametrize("coverage, expected", [(0.2, 0.0), (0.5, 0.5), (0.8, 1.0)])
def test_alpha_from_coverage(coverage, expected):
    assert fill.alpha(np.array([coverage]))[0] == pytest.approx(expected)
