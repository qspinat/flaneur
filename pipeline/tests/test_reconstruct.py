"""Today's panoramas placed by COLMAP, tied to the map: the pure parts, on built cases."""

import numpy as np
import pytest

from flaneur import reconstruct
from flaneur.panoramax import Picture


def test_tiles_cover_every_panorama_once_and_overlap():
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 600, (500, 2))

    tiles = reconstruct.tiles(points, side=250, overlap=50)

    owners = np.zeros(len(points), int)
    for tile in tiles:
        owners[tile.core] += 1
        assert set(tile.core) <= set(tile.members)  # a tile reconstructs a margin around its core
    assert (owners == 1).all()
    assert sum(len(t.members) for t in tiles) > len(points)  # the margins overlap


def test_the_heading_of_a_panorama_from_its_camera():
    # COLMAP's panorama camera: x right, y down, z towards the picture's centre; world up is
    # the model's z here, and the picture's centre faces 30° east of north
    r = np.radians(30)
    forward, right = np.array([np.sin(r), np.cos(r), 0]), np.array([np.cos(r), -np.sin(r), 0])
    down = np.array([0, 0, -1.0])
    cam_from_world = np.stack([right, down, forward])

    forward_found, up_found = reconstruct.axes(cam_from_world)

    assert forward_found == pytest.approx(forward)
    assert up_found == pytest.approx([0, 0, 1])


def test_a_reconstruction_is_levelled_and_fitted_to_the_gps():
    rng = np.random.default_rng(1)
    truth = np.column_stack([rng.uniform(0, 200, 60), rng.uniform(0, 40, 60)])  # a street
    # the model's own frame: scaled, turned, shifted, tipped over (its up is along y)
    scale, turn = 0.05, np.radians(40)
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    flat = truth @ rot.T * scale + [3.0, -1.0]
    # x east, y up, z south: right-handed, as any reconstruction is (a mirror cannot be fitted)
    model = np.column_stack([flat[:, 0], np.full(60, 0.7), -flat[:, 1]])
    up = np.tile([0.0, 1.0, 0.0], (60, 1)) + rng.normal(0, 0.02, (60, 3))
    gps = truth + rng.normal(0, 2.0, truth.shape)  # phones: a few metres off
    gps[:6] += 60  # and a few far off

    fit = reconstruct.fit_to_gps(model, up, gps)

    placed = fit.place(model)
    assert np.median(np.linalg.norm(placed - truth, axis=1)) < 1.5
    assert fit.far_off.sum() == 6 and fit.far_off[:6].all()


def test_a_crowded_tile_is_split():
    rng = np.random.default_rng(2)
    crowd = rng.uniform(100, 140, (900, 2))  # a square full of panoramas
    points = np.concatenate([crowd, rng.uniform(0, 600, (200, 2))])

    tiles = reconstruct.tiles(points, side=250, overlap=50, most=500)

    assert max(len(t.core) for t in tiles) <= 500
    assert max(len(t.members) for t in tiles) <= 750  # the margins shrink with the tiles
    owners = np.zeros(len(points), int)
    for tile in tiles:
        owners[tile.core] += 1
    assert (owners == 1).all()


def test_tiles_lie_on_a_fixed_grid():
    rng = np.random.default_rng(3)
    points = rng.uniform(1000, 1600, (300, 2))
    tile_of = {tuple(t.core) for t in reconstruct.tiles(points)}

    one_more = np.concatenate([points, [[700.0, 700.0]]])  # a new panorama far to the south-west

    assert tile_of <= {tuple(t.core) for t in reconstruct.tiles(one_more)}  # the others' tiles stay


def test_a_fit_where_nothing_agrees_places_no_panorama():
    model = np.zeros((10, 3))  # COLMAP put every panorama in one spot
    up = np.tile([0.0, 0.0, 1.0], (10, 1))
    gps = np.random.default_rng(4).uniform(0, 100, (10, 2))

    fit = reconstruct.fit_to_gps(model, up, gps)

    assert not fit.agreeing.any() and fit.far_off.all()


def in_run(id: str, run: str, rank: int, lon: float = 2.35) -> Picture:
    """A 360° picture, the `rank`th of its capture `run`, recorded at `lon`."""
    return Picture(id, lon, 48.85, 0.0, True, "2025-01-01", "", "", sequence=run, rank=rank)


def test_a_panorama_not_placed_moves_as_the_nearest_placed_in_its_run():
    pictures = [
        in_run("placed first", "run", 1),
        in_run("next", "run", 2),
        in_run("nearer the second", "run", 3),
        in_run("placed second", "run", 4),
        in_run("between", "run", 6),  # two ranks from each of two placed ones
        in_run("placed third", "run", 8),
        in_run("too far on", "run", 8 + reconstruct.BORROW_RANKS + 1),
        in_run("other run", "other", 2),
    ]
    fixes = {
        "placed first": (1.0, 2.0, 10.0),
        "placed second": (3.0, 4.0, 20.0),
        "placed third": (5.0, 6.0, 30.0),
    }

    found = reconstruct.borrowed(fixes, pictures)

    assert found["next"] == (1.0, 2.0, 10.0)
    assert found["nearer the second"] == (3.0, 4.0, 20.0)
    assert found["between"] == pytest.approx((4.0, 5.0, 25.0))
    assert found["placed first"] == (1.0, 2.0, 10.0)
    assert "too far on" not in found and "other run" not in found


def test_a_panorama_does_not_borrow_from_a_run_neighbour_taken_far_away():
    pictures = [in_run("placed", "run", 1, lon=2.35), in_run("after a jump", "run", 2, lon=2.36)]

    found = reconstruct.borrowed({"placed": (1.0, 2.0, 10.0)}, pictures)

    assert "after a jump" not in found  # 730 m east


def test_turns_either_side_of_south_borrow_as_south():
    pictures = [in_run("before", "run", 1), in_run("here", "run", 2), in_run("after", "run", 3)]

    found = reconstruct.borrowed(
        {"before": (0.0, 0.0, 179.0), "after": (0.0, 0.0, -179.0)}, pictures
    )

    assert abs(found["here"][2]) == pytest.approx(180.0)
