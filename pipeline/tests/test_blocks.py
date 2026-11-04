"""Casting rays onto today's buildings."""

import numpy as np
import pytest

from flaneur.blocks import Blocks


def one_wall() -> Blocks:
    """A block 20 to 30 m north of the camera, its walls from 2 m to 17 m high."""
    return Blocks.of([([(-50, 20), (50, 20), (50, 30), (-50, 30)], 2.0, 15.0)])


def test_the_skyline_of_one_block():
    line = one_wall().skyline(0, 0, 2.0, np.radians([0.0, 180.0]))

    assert line[0] == pytest.approx(np.arctan2(15, 20))
    assert line[1] == -np.inf  # nothing that way


def test_a_ray_meets_the_first_wall_it_crosses():
    up_a_little = np.array([0.0, 1.0, 0.25]) / np.hypot(1, 0.25)
    rays = np.array([up_a_little, [0.0, -1.0, 0.0], [0.0, 1.0, 1.0]])

    points, edges = one_wall().hit(np.array([0.0, 0.0, 2.0]), rays)

    assert points[0] == pytest.approx([0, 20, 7])  # the near wall, 5 m above the camera
    assert edges[0] == 0
    assert np.isnan(points[1]).all() and edges[1] == -1  # nothing behind
    assert np.isnan(points[2]).all() and edges[2] == -1  # over the roof: the sky


def test_the_corners_seen_are_those_not_hidden_or_shared():
    blocks = Blocks.of(
        [
            ([(-10, 20), (0, 20), (0, 30), (-10, 30)], 0.0, 15.0),  # two houses side by side
            ([(0, 20), (10, 20), (10, 30), (0, 30)], 0.0, 15.0),
            ([(-5, 60), (5, 60), (5, 70), (-5, 70)], 0.0, 15.0),  # behind them, hidden
        ]
    )

    corners = blocks.corners(0, 0, 2.0, metres=100)

    seen = sorted((round(float(np.degrees(b))), round(float(d), 1)) for b, d, _, _ in corners)
    # the two outer front corners; not the shared one between the houses, nor those behind
    assert seen == [(-27, 22.4), (27, 22.4)]
    assert corners[0][2:] == pytest.approx([0.0, 15.0])


def test_rays_skylines_and_corners_come_out_the_same_in_batches(monkeypatch):
    rng = np.random.default_rng(0)
    rays = rng.normal(size=(500, 3))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    bearings = np.radians(np.arange(0, 360, 0.5))
    town = Blocks.of(
        [
            ([(x, y), (x + 8, y), (x + 8, y + 8), (x, y + 8)], 0.0, 12.0)
            for x in range(-40, 40, 16)
            for y in (12, 40)
        ]
    )

    def everything():
        return (
            *town.hit(np.array([0.0, 0.0, 2.0]), rays),
            town.skyline(0, 0, 2.0, bearings),
            town.corners(0, 0, 2.0, metres=100),
        )

    whole = everything()
    monkeypatch.setattr("flaneur.blocks.MOST_PAIRS", 7)  # a handful of rays at a time

    for batched, at_once in zip(everything(), whole, strict=True):
        np.testing.assert_array_equal(batched, at_once)
