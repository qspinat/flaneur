"""Lining a panorama up with today's buildings by its roofs, on built scenes."""

import numpy as np
import pytest

from flaneur import register
from flaneur.blocks import Blocks


def boxes() -> Blocks:
    """A square of four blocks around a crossroads, of different heights (metres NGF)."""
    blocks = [
        ([(5, 5), (60, 5), (60, 60), (5, 60)], 40.0, 18.0),
        ([(-60, 5), (-5, 5), (-5, 60), (-60, 60)], 40.0, 25.0),
        ([(-60, -60), (-5, -60), (-5, -5), (-60, -5)], 40.0, 12.0),
        ([(5, -60), (60, -60), (60, -5), (5, -5)], 40.0, 30.0),
    ]
    return Blocks.of(blocks)


def panorama(blocks: Blocks, x: float, y: float, z: float, azimuth: float) -> np.ndarray:
    """A 360° picture of the blocks: a blue sky over pale stone, cut at their roofs."""
    width, height = 1440, 720
    columns = np.arange(width)
    bearings = np.radians(azimuth + (columns + 0.5) * 360 / width - 180)
    roofs = blocks.skyline(x, y, z, bearings)
    roof_rows = np.where(np.isfinite(roofs), (0.5 - roofs / np.pi) * height, height / 2)
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = (150, 190, 215)  # stone (BGR)
    for c in columns:
        image[: int(roof_rows[c]), c] = (225, 170, 110)  # sky
    return image


def test_roof_edges_are_strongest_at_the_roofs():
    blocks = boxes()
    image = panorama(blocks, 0, 0, 41.6, 0)

    edges = register.picture_edges(image).across_rows

    roofs = blocks.skyline(0, 0, 41.6, np.radians((np.arange(720) + 0.5) / 2 - 180))
    seen = np.isfinite(roofs)  # down the streets, no roof
    rows = np.round((0.5 - roofs[seen] / np.pi) * register.ROWS).astype(int)
    assert edges[rows, np.flatnonzero(seen)].mean() > 0.9
    assert np.median(edges[: register.ROWS // 2]) < 0.1  # elsewhere, plain sky and stone


def test_register_finds_how_far_the_panorama_was_off():
    blocks = boxes()
    true_x, true_y, z, true_azimuth = 2.0, -1.0, 41.6, 30.0
    edges = register.picture_edges(panorama(blocks, true_x, true_y, z, true_azimuth))

    # recorded 5 m east, 4 m south and 9° off
    found = register.register(
        edges, true_azimuth - 9, true_x + 5, true_y - 4, lambda x, y: z, blocks
    )

    assert found.east == pytest.approx(-5, abs=1)
    assert found.north == pytest.approx(4, abs=1)
    assert found.turn == pytest.approx(9, abs=1)
    assert found.better


def walls_only(blocks: Blocks, x: float, y: float, z: float, azimuth: float) -> np.ndarray:
    """A 360° picture where each column takes the shade of the wall it sees, sky included (a
    white sky over white stone): no roof shows, only corners and the ends of facades."""
    width, height = 1440, 720
    bearings = np.radians(azimuth + (np.arange(width) + 0.5) * 360 / width - 180)
    level = np.column_stack([np.sin(bearings), np.cos(bearings), np.zeros(width)])
    _, wall = blocks.hit(np.array([x, y, z]), level)
    shade = np.where(wall >= 0, 60 + (wall * 37) % 150, 230).astype(np.uint8)
    return np.repeat(np.repeat(shade[None, :, None], height, axis=0), 3, axis=2)


def test_register_finds_the_panorama_by_its_corners_when_no_roof_shows():
    blocks = boxes()
    true_x, true_y, z, true_azimuth = 2.0, -1.0, 41.6, 30.0
    edges = register.picture_edges(walls_only(blocks, true_x, true_y, z, true_azimuth))

    found = register.register(
        edges, true_azimuth - 9, true_x + 5, true_y - 4, lambda x, y: z, blocks
    )

    assert (found.east, found.north, found.turn) == pytest.approx((-5, 4, 9), abs=1)


def test_a_panorama_without_edges_is_left_as_it_is():
    edges = register.picture_edges(np.full((720, 1440, 3), 128, np.uint8))  # a covered passage

    assert not register.register(edges, 0.0, 0.0, 0.0, lambda x, y: 41.6, boxes()).better


def test_a_panorama_moves_where_its_skyline_meets_the_buildings_better():
    from flaneur.panoramax import Picture

    here = Picture("p1", 2.347, 48.853, 90.0, True, "2025-05-01", "", "https://x/p1.jpg")
    there = Picture("p2", 2.348, 48.853, 10.0, True, "2025-05-01", "", "https://x/p2.jpg")

    def found(p: Picture, **fits: float) -> dict:
        recorded = [round(p.point.x, 2), round(p.point.y, 2), p.azimuth]
        return {"east": 3.0, "north": -2.0, "turn": -12.0, **fits, "recorded": recorded}

    corrections = {
        "p1": found(here, fit_before=0.3, fit_after=0.6),
        "p2": found(there, fit_before=0.3, fit_after=0.32),  # not worth it: left as it is
    }

    kept = register.kept(corrections, [here, there])
    moved, left = register.applied(
        [here, there], {i: (c.east, c.north, c.turn) for i, c in kept.items()}
    )

    assert (moved.point.x, moved.point.y) == pytest.approx(
        (here.point.x + 3, here.point.y - 2), abs=0.01
    )
    assert moved.azimuth == pytest.approx(78)
    assert left == there


def test_a_correction_found_for_another_recorded_pose_is_not_applied():
    from dataclasses import replace

    from flaneur.panoramax import Picture

    here = Picture("p1", 2.347, 48.853, 90.0, True, "2025-05-01", "", "https://x/p1.jpg")
    recorded = [round(here.point.x, 2), round(here.point.y, 2), 90.0]
    fixed = {"east": 3.0, "north": -2.0, "turn": -12.0, "fit_before": 0.3, "fit_after": 0.6}
    moved_since = replace(here, azimuth=120.0)  # harvested again since: a new heading

    assert register.kept({"p1": {**fixed, "recorded": recorded}}, [moved_since]) == {}
