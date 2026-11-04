import numpy as np
import pytest

from flaneur import buildings


def test_prism_from_a_bd_topo_feature(load_fixture):
    feature = load_fixture("bdtopo_batiment.json")

    (prism,) = buildings.prisms(feature, origin=(651900, 6861700))

    assert prism.height == pytest.approx(feature["properties"]["hauteur"])
    assert prism.ground == pytest.approx(feature["properties"]["altitude_minimale_sol"])
    assert prism.appeared == 1870
    x, y = prism.footprint[0]
    assert 0 < x < 100 and 0 < y < 100  # metres from the origin


def test_missing_height_takes_the_default(load_fixture):
    feature = load_fixture("bdtopo_batiment.json")
    feature["properties"]["hauteur"] = None

    assert buildings.prisms(feature, origin=(0, 0))[0].height == buildings.DEFAULT_HEIGHT_M


def test_every_part_of_a_building_is_kept(load_fixture):
    feature = load_fixture("bdtopo_batiment.json")
    part = feature["geometry"]["coordinates"][0]
    moved = [[[x + 100, y, z] for x, y, z in ring] for ring in part]
    feature["geometry"]["coordinates"].append(moved)

    assert len(buildings.prisms(feature, origin=(651900, 6861700))) == 2


def test_a_camera_stands_outside_the_buildings_standing_that_year():
    from shapely.geometry import box, mapping

    def house(x0: float, date: str | None) -> dict:
        return {
            "geometry": mapping(box(x0, 0, x0 + 10, 10)),
            "properties": {"date_d_apparition": date},
        }

    features = [house(0, None), house(20, "1960-01-01"), house(40, "1820-01-01")]
    free = buildings.outside(buildings.standing(features, 1900))

    xs, ys = np.array([5.0, 25.0, 45.0, 10.5, 9.5, 15.0]), np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    # undated: standing; built 1960: not yet; 1820: standing; just off and on a facade; a street
    assert free(xs, ys).tolist() == [False, True, False, True, True, True]
