import numpy as np
import pytest
from shapely.geometry import Point, box

from flaneur import scene


def feature(x: float, y: float, ground: float) -> dict:
    """A 10 m square BD TOPO building at (x, y) Lambert-93."""
    return {
        "geometry": box(x, y, x + 10, y + 10).__geo_interface__,
        "properties": {"hauteur": 18.0, "altitude_minimale_sol": ground, "date_d_apparition": None},
    }


def test_ground_follows_the_buildings_bases():
    features = [feature(0, 0, 30.0), feature(100, 0, 40.0)]

    ground = scene.Ground.from_buildings(features, area=box(0, 0, 110, 10), step=5)

    assert ground.at(Point(5, 5)) == pytest.approx(30, abs=0.5)
    assert ground.at(Point(105, 5)) == pytest.approx(40, abs=0.5)
    assert 30 < ground.at(Point(55, 5)) < 40


def test_buildings_stand_on_their_ground_relative_to_the_origin():
    features = [feature(0, 0, 30.0), feature(100, 0, 40.0)]
    ground = scene.Ground.from_buildings(features, area=box(0, 0, 110, 10), step=5)

    found = scene.buildings_of(features, origin=(50, 0, 35.0))

    assert found[1]["base"] == pytest.approx(5.0)
    assert found[1]["height"] == 18.0
    assert min(x for x, _ in found[1]["footprint"]) == pytest.approx(50)
    assert np.isfinite(ground.heights).all()


def test_a_camera_inside_today_s_buildings_is_not_shown():
    pose = {"x": 5, "y": 5, "z": 1.6, "heading": 0, "horizon": 0.6, "fov": 53}
    proposals = [{"photo_id": "a", "status": "lines", "pose": pose, "walls": []}]

    assert scene.shown(proposals, standing=box(0, 0, 10, 10)) == []
    assert len(scene.shown(proposals, standing=box(20, 20, 30, 30))) == 1


def test_each_building_has_a_core_that_hides_old_photos_behind_it():
    features = [
        feature(0, 0, 30.0),
        {**feature(100, 0, 40.0), "geometry": box(100, 0, 104, 4).__geo_interface__},
    ]

    cores = scene.cores_of(features, origin=(0, 0, 30.0))

    assert len(cores) == 1  # the 4 m house has no core left
    xs = [x for x, _ in cores[0]["footprint"]]
    assert min(xs) == pytest.approx(scene.CORE_INSET_M) and max(xs) == pytest.approx(
        10 - scene.CORE_INSET_M
    )


def test_the_walks_photos_are_given_in_the_scenes_frame():
    from flaneur.export import WalkPhoto

    me = WalkPhoto(
        id="pm:1",
        title="Rue X",
        caption=None,
        years=[1900, 1900],
        photographer="Atget",
        credit="CC0",
        record_url=None,
        lon=2.347,
        lat=48.853,
        heading=90.0,
        street=("Rue", "du Petit-Pont"),
        arrondissement=5,
        image_url="",
        crop=None,
    )
    here = scene.to_metres(Point(me.lon, me.lat))
    origin = (round(here.x) - 10, round(here.y) + 5, 35.0)

    [found] = scene.walk_of([me], origin)

    assert found["x"] == pytest.approx(here.x - origin[0], abs=0.01)
    assert found["y"] == pytest.approx(here.y - origin[1], abs=0.01)
    assert (found["lon"], found["lat"], found["heading"]) == (2.347, 48.853, 90.0)
    assert (found["street"], found["arrondissement"]) == (["Rue", "du Petit-Pont"], 5)


def test_the_ground_comes_out_the_same_in_batches(monkeypatch):
    features = [feature(x, y, 30.0 + x / 10) for x in range(0, 200, 20) for y in range(0, 100, 25)]
    area = box(0, 0, 210, 110)
    whole = scene.Ground.from_buildings(features, area, step=5).heights

    monkeypatch.setattr("flaneur.scene.MOST_PAIRS", 50)  # a few grid nodes at a time

    np.testing.assert_allclose(scene.Ground.from_buildings(features, area, step=5).heights, whole)
