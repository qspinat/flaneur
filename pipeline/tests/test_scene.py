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


def test_the_ground_comes_out_the_same_in_batches(monkeypatch):
    features = [feature(x, y, 30.0 + x / 10) for x in range(0, 200, 20) for y in range(0, 100, 25)]
    area = box(0, 0, 210, 110)
    whole = scene.Ground.from_buildings(features, area, step=5).heights

    monkeypatch.setattr("flaneur.scene.MOST_PAIRS", 50)  # a few grid nodes at a time

    np.testing.assert_allclose(scene.Ground.from_buildings(features, area, step=5).heights, whole)
