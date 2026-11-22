"""Filling a weighted layer as a graphics chip would: small gaps interpolated, large ones
fading, the photo's edge soft."""

import numpy as np
import pytest

from flaneur import fill

PIXEL_DEG = 90 / 700  # a sheet's view: 2° is 16 pixels, 1° 8


def _layer(side=200, value=100.0):
    colour = np.full((side, side, 3), value, np.float32)
    return colour, np.ones((side, side), np.float32), np.ones((side, side), bool)


def test_levels_follow_the_angle():
    assert fill.levels(PIXEL_DEG) == 5
    assert fill.levels(PIXEL_DEG / 2) == 6  # twice as fine: one level more, the same angle
    assert fill.level(fill.EDGE_DEG, PIXEL_DEG) == 3  # texels of 8 pixels: about 1°


def test_a_slit_three_pixels_wide_is_filled():
    colour, weight, reached = _layer()
    colour[:, :100], colour[:, 100:] = 80.0, 120.0
    weight[:, 99:102] = 0.0  # a slit

    rgb, alpha = fill.laid(colour, weight, reached, PIXEL_DEG)

    assert alpha[:, 99:102].min() >= 0.9
    assert 70 <= rgb[:, 99:102].min() and rgb[:, 99:102].max() <= 130


def test_a_gap_wider_than_the_reach_fades_smoothly():
    colour, weight, reached = _layer(side=400)
    weight[:, 150:250] = 0.0  # 100 pixels, 13°: far wider than 2°

    _, alpha = fill.laid(colour, weight, reached, PIXEL_DEG)

    assert alpha[:, 195:205].max() <= 0.1
    assert np.abs(np.diff(alpha[200])).max() <= 0.2  # no step


def test_the_photo_s_edge_fades_across_it_and_nothing_shows_farther():
    colour, weight, reached = _layer()
    reached[:, :60] = False
    weight[:, :60] = 0.0

    _, alpha = fill.laid(colour, weight, reached, PIXEL_DEG)

    assert alpha[:, :52].max() < 0.01  # nothing 1° beyond the edge
    assert 0.3 < alpha[100, 60] < 0.7 and alpha[100, 70:].min() > 0.99  # soft across it
    assert np.all(np.diff(alpha[100, 50:80]) >= -1e-6)  # rising into the photo


def test_a_notch_narrower_than_the_edge_is_kept_a_wide_one_is_not():
    colour, weight, reached = _layer()
    for cols in (slice(60, 63), slice(120, 170)):  # notches from the top edge, 3 and 50 wide
        reached[:40, cols], weight[:40, cols] = False, 0.0

    _, alpha = fill.laid(colour, weight, reached, PIXEL_DEG)

    assert alpha[10:30, 60:63].min() > 0.5
    assert alpha[:20, 135:155].max() == 0


@pytest.mark.parametrize("x, expected", [(0.2, 0.0), (0.5, 0.5), (0.8, 1.0)])
def test_smooth_rises_between_its_ends(x, expected):
    assert fill.smooth(np.array([x]), 0.2, 0.8)[0] == pytest.approx(expected)
