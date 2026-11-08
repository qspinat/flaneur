"""A model's depth of a panorama's views, scaled to today's walls where they say nothing."""

import threading

import cv2
import numpy as np
import pytest

from flaneur import depthmaps, facades, match

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


def _panorama(tmp_path, name="p1") -> match.Panorama:
    """A panorama whose picture is a plain grey 360°."""
    path = tmp_path / f"{name}.jpg"
    cv2.imwrite(str(path), np.full((64, 128, 3), 128, np.uint8))
    return match.Panorama(name, 0.0, 0.0, 0.0, 0.0, path)


def test_maps_are_made_once_along_each_ray_and_kept(tmp_path):
    calls = []

    def estimate(view: np.ndarray) -> np.ndarray:  # 4 m ahead everywhere along the axis
        calls.append(view.shape)
        return np.full(view.shape[:2], 4.0)

    maker = depthmaps.Depthmaps(
        tmp_path / "depth", threading.Lock(), lambda p: p.id, estimate=estimate, side=SIDE
    )
    panorama = _panorama(tmp_path)

    first = maker.maps(panorama)
    again = depthmaps.Depthmaps(tmp_path / "depth", threading.Lock(), lambda p: p.id, make=False)

    assert len(calls) == VIEWS
    assert first.shape == (VIEWS, SIDE, SIDE)
    assert first[0, SIDE // 2, SIDE // 2] == pytest.approx(4.0, abs=0.05)  # the axis: as given
    assert first[0, 0, 0] > 4.0 * 1.2  # a corner's ray is longer than its depth
    assert again.kept(panorama)
    assert np.allclose(again.maps(panorama), first)


def test_without_making_a_missing_panorama_has_no_maps(tmp_path):
    reader = depthmaps.Depthmaps(tmp_path / "depth", threading.Lock(), lambda p: p.id, make=False)

    assert reader.maps(_panorama(tmp_path)) is None
    assert not reader.kept(_panorama(tmp_path))


def test_depth_pro_is_given_a_plain_rgb_picture(tmp_path):
    torch = pytest.importorskip("torch")
    given = {}

    class Processor:
        def __call__(self, images, **_):  # transformers' processors refuse reversed strides
            given["contiguous"] = images.flags.c_contiguous
            return _Inputs({"pixel_values": torch.zeros(1, 3, 4, 4)})

        def post_process_depth_estimation(self, found, target_sizes):
            return [{"predicted_depth": torch.ones(target_sizes[0])}]

    maker = depthmaps.Depthmaps(tmp_path, threading.Lock(), lambda p: p.id)
    maker.loaded = (lambda **_: None, Processor(), "cpu", torch.float32)

    depth = maker._depth_pro(np.zeros((8, 8, 3), np.uint8)[:, ::-1])

    assert given["contiguous"] and depth.shape == (8, 8)


class _Inputs(dict):
    def to(self, *_):
        return self


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
