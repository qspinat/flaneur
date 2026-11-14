"""A model's depth of a panorama's views, scaled to today's walls where they say nothing."""

import threading

import cv2
import numpy as np
import pytest

from flaneur import depthmaps, facades, match, segment

SIDE = 32
VIEWS = len(match.VIEW_HEADINGS)


def _toward(heading: float, up: float = 0.0) -> np.ndarray:
    """A world direction (east, north, up) toward a heading, raised `up` degrees."""
    h, u = np.radians(heading), np.radians(up)
    return np.array([[np.sin(h) * np.cos(u), np.cos(h) * np.cos(u), np.sin(u)]])


def _walls(metres: float, views: list[int]) -> np.ndarray:
    """Walls met `metres` away on every cell of these views, none elsewhere."""
    walls = np.full((VIEWS, SIDE, SIDE), np.nan)
    walls[views] = metres
    return walls


def test_each_view_is_scaled_to_the_walls_it_sees():
    raw = np.ones((VIEWS, SIDE, SIDE))
    raw[1] = 2.0
    walls = _walls(10.0, [0, 1, 2])

    scales = depthmaps.scales(raw, walls)

    assert scales[0] == pytest.approx(10.0)
    assert scales[1] == pytest.approx(5.0)


def test_a_view_without_walls_takes_the_others_scale():
    raw = np.ones((VIEWS, SIDE, SIDE))

    scales = depthmaps.scales(raw, _walls(8.0, [0, 1, 2]))

    assert scales[6] == pytest.approx(8.0)


def test_too_few_views_with_walls_give_no_scale():
    raw = np.ones((VIEWS, SIDE, SIDE))

    assert depthmaps.scales(raw, _walls(8.0, [0, 1])) is None


def test_a_few_wall_cells_do_not_scale_a_view():
    raw = np.ones((VIEWS, SIDE, SIDE))
    walls = _walls(8.0, [0, 1, 2])
    walls[3, 0, :4] = 100.0  # four cells: too few to read a scale from

    assert depthmaps.scales(raw, walls)[3] == pytest.approx(8.0)


def test_a_ray_takes_the_depth_of_the_view_facing_it():
    maps = np.ones((VIEWS, SIDE, SIDE))
    maps[0] = 3.0
    rays = np.vstack([_toward(0.0, 5.0), _toward(90.0, 5.0)])

    assert depthmaps.filled(maps, rays).tolist() == [3.0, 1.0]


def test_a_ray_outside_every_view_has_no_depth():
    maps = np.ones((VIEWS, SIDE, SIDE))

    assert np.isnan(depthmaps.filled(maps, _toward(0.0, -89.0))).all()


def test_the_grid_s_rays_fall_on_their_own_cells():
    middle = slice(SIDE // 2 - 3, SIDE // 2 + 3)  # near the axis, even at the top: its own view's
    rays = depthmaps.grid_rays(SIDE)[:, :, middle]
    maps = np.arange(VIEWS * SIDE * SIDE, dtype=float).reshape(VIEWS, SIDE, SIDE)

    found = depthmaps.filled(maps, rays.reshape(-1, 3))

    assert found.tolist() == maps[:, :, middle].ravel().tolist()


def _panorama(tmp_path) -> match.Panorama:
    """A panorama whose picture is a plain grey 360°."""
    path = tmp_path / "p1.jpg"
    cv2.imwrite(str(path), np.full((64, 128, 3), 128, np.uint8))
    return match.Panorama("p1", 0.0, 0.0, 0.0, 0.0, path)


def test_maps_are_made_live_once_along_each_ray_and_kept_in_memory(tmp_path):
    calls = []

    def estimate(view: np.ndarray) -> np.ndarray:  # 4 m ahead everywhere along the axis
        calls.append(view.shape)
        return np.full(view.shape[:2], 4.0)

    maker = depthmaps.Depthmaps(threading.Lock(), estimate=estimate, side=SIDE)
    panorama = _panorama(tmp_path)

    first = maker.maps(panorama)
    again = maker.maps(panorama)

    assert len(calls) == VIEWS
    assert first.shape == (VIEWS, SIDE, SIDE)
    assert first[0, SIDE // 2, SIDE // 2] == pytest.approx(4.0, abs=0.05)  # the axis: as given
    assert first[0, 0, 0] > 4.0 * 1.2  # a corner's ray is longer than its depth
    assert again is first


def _kept_in(folder, calls: list, **given) -> depthmaps.Depthmaps:
    """A maker of 4 m everywhere, keeping its maps in `folder`, counting its `calls`."""

    def estimate(view: np.ndarray) -> np.ndarray:
        calls.append(1)
        return np.full(view.shape[:2], 4.0)

    return depthmaps.Depthmaps(
        threading.Lock(), estimate=estimate, side=SIDE, folder=folder, **given
    )


def test_maps_made_once_are_kept_on_disk_for_the_next_run(tmp_path):
    calls = []
    made = _kept_in(tmp_path / "kept", calls).maps(_panorama(tmp_path))

    again = _kept_in(tmp_path / "kept", calls).maps(_panorama(tmp_path))

    assert len(calls) == VIEWS  # made once, by the first
    np.testing.assert_allclose(again, made, rtol=1e-3)


def test_a_reader_that_does_not_make_has_only_the_maps_kept(tmp_path):
    calls = []

    assert _kept_in(tmp_path, calls, make=False).maps(_panorama(tmp_path)) is None
    assert calls == []


def test_maps_kept_are_named_as_their_panorama_is_levelled(tmp_path):
    calls = []
    level = {"name": "p1_0.0"}
    maker = _kept_in(tmp_path / "kept", calls, name=lambda p: level["name"])
    maker.maps(_panorama(tmp_path))
    level["name"] = "p1_2.0"  # levelled again: its views are not the same

    maker.maps(_panorama(tmp_path))

    assert len(calls) == 2 * VIEWS  # not the maps kept in memory for the old levelling
    assert sorted(f.name for f in (tmp_path / "kept").iterdir()) == ["p1_0.0.npz", "p1_2.0.npz"]


def test_maps_kept_by_another_model_or_resolution_are_made_again(tmp_path):
    calls = []
    np.savez_compressed(tmp_path / "p1.npz", depth=np.zeros((VIEWS, SIDE, SIDE)), model="old")

    made = _kept_in(tmp_path, calls).maps(_panorama(tmp_path))

    assert len(calls) == VIEWS and made[0, SIDE // 2, SIDE // 2] == pytest.approx(4.0, abs=0.05)


def test_an_unreadable_picture_has_no_maps(tmp_path):
    maker = depthmaps.Depthmaps(threading.Lock(), estimate=lambda v: np.ones(v.shape[:2]))
    missing = match.Panorama("gone", 0.0, 0.0, 0.0, 0.0, tmp_path / "gone.jpg")

    assert maker.maps(missing) is None


def test_moge_is_given_a_plain_rgb_view_at_low_resolution():
    torch = pytest.importorskip("torch")
    given = {}

    class Model:
        def infer(self, image, **options):
            given.update(options, shape=tuple(image.shape), dtype=image.dtype)
            return {"depth": torch.full(image.shape[-2:], 7.0)}

    maker = depthmaps.Depthmaps(threading.Lock())
    maker.loaded = (Model(), "cpu", torch.float32)

    depth = maker._moge(np.zeros((8, 8, 3), np.uint8)[:, ::-1])

    assert depth.shape == (8, 8) and depth[0, 0] == pytest.approx(7.0)
    assert given["shape"] == (3, 8, 8) and given["dtype"] == torch.float32
    assert given["fov_x"] == match.VIEW_FOV and given["resolution_level"] == depthmaps.LEVEL
    assert given["apply_mask"] is False


def test_the_sky_has_no_depth():
    scene = facades.Scene(
        "p1",
        (0.0, 0.0, 2.0),
        0.0,
        (
            facades.Wall((-4.0, -60.0), (-4.0, 60.0), 20.0, "fitted"),
            facades.Wall((4.0, -60.0), (4.0, 60.0), 20.0, "fitted"),
        ),
        (),
        0,
    )
    maps = np.ones((VIEWS, SIDE, SIDE))
    maps[:, : SIDE // 3] = 1000.0  # the upper third: as far as models put the sky

    fill = depthmaps.Filler(scene, maps)
    up, ahead = _toward(0.0, 30.0), _toward(0.0, 0.0)

    assert np.isnan(fill(up)).all() and np.isfinite(fill(ahead)).all()


def _scene() -> facades.Scene:
    """A street running north, its walls 4 m to either side of a camera 2 m above it."""
    return facades.Scene(
        "p1",
        (0.0, 0.0, 2.0),
        0.0,
        (
            facades.Wall((-4.0, -60.0), (-4.0, 60.0), 20.0, "fitted"),
            facades.Wall((4.0, -60.0), (4.0, 60.0), 20.0, "fitted"),
        ),
        (),
        0,
    )


def test_the_model_is_taken_only_on_what_stands():
    maps = np.full((VIEWS, SIDE, SIDE), 9.0)
    parts = np.full((VIEWS, SIDE, SIDE), segment.Part.BUILT, np.int8)
    parts[0, : SIDE // 2, :] = segment.Part.ELSE  # the upper half facing north: the sky
    up = _toward(0.0, 20.0)

    assert np.isnan(depthmaps.Filler(_scene(), maps, parts)(up)).all()
    assert np.isfinite(depthmaps.Filler(_scene(), maps)(up)).all()


def test_behind_what_passes_the_street_is_rebuilt():
    maps = np.full((VIEWS, SIDE, SIDE), 9.0)
    parts = np.full((VIEWS, SIDE, SIDE), segment.Part.GROUND, np.int8)
    parts[0, SIDE // 2 :, SIDE // 3 : 2 * SIDE // 3] = segment.Part.MOVING
    down = _toward(0.0, -20.0)  # meets the street 2 / tan(20°) ≈ 5.5 m ahead

    found = depthmaps.Filler(_scene(), maps, parts)(down)

    assert found[0] == pytest.approx(2.0 / np.sin(np.radians(20.0)), rel=0.05)


def test_a_model_that_fails_leaves_the_walls_alone_and_is_not_asked_again(tmp_path):
    calls = []

    def broken(view: np.ndarray) -> np.ndarray:
        calls.append(1)
        raise RuntimeError("no memory left")

    maker = depthmaps.Depthmaps(threading.Lock(), estimate=broken, side=SIDE)

    assert maker.maps(_panorama(tmp_path)) is None
    assert maker.maps(_panorama(tmp_path)) is None
    assert len(calls) == 1


def test_maps_kept_on_disk_are_still_read_after_the_model_failed(tmp_path):
    calls = []
    _kept_in(tmp_path, calls).maps(_panorama(tmp_path))  # kept on disk, by an earlier run

    def broken(view: np.ndarray) -> np.ndarray:
        raise RuntimeError("no memory left")

    maker = depthmaps.Depthmaps(threading.Lock(), estimate=broken, side=SIDE, folder=tmp_path)
    other = match.Panorama("p2", 0.0, 0.0, 0.0, 0.0, _panorama(tmp_path).path)
    assert maker.maps(other) is None  # made live: the model fails

    assert maker.maps(_panorama(tmp_path)) is not None
