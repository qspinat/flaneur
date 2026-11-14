"""Views chosen to cover where an old photo's camera may stand, not the panoramas nearest it."""

import numpy as np

from flaneur import choice, match
from flaneur.blocks import Blocks

ROWS = [  # a street running north between two rows of houses, 12 m apart
    ([(-20, -200), (-6, -200), (-6, 200), (-20, 200)], 0.0, 15.0),
    ([(6, -200), (20, -200), (20, 200), (6, 200)], 0.0, 15.0),
]


def _in_street(x, y):
    return np.abs(np.asarray(x)) < 6


def _chooser(fixed, rows=ROWS) -> choice.Chooser:
    return choice.Chooser(frozenset(fixed), Blocks.of(rows), lambda x, y: 0 * x, _in_street)


def _panorama(name: str, y: float, x: float = 0.0) -> match.Panorama:
    return match.Panorama(name, x, y, 2.5, 0.0, path=None)


def _covers(chooser, views, cameras):
    return chooser.covering(views, cameras, chooser.aims(cameras))


def test_a_view_near_the_camera_facing_its_way_covers_it():
    near = _panorama("near", 3.0)
    cameras = np.array([[0.0, 0.0, 90.0], [0.0, 0.0, 270.0]])  # facing the east, the west row

    covers = _covers(_chooser(["near"]), [(near, 90.0), (near, 180.0)], cameras)

    assert covers[0].tolist() == [True, False]  # 90°: the east camera only
    assert not covers[1].any()  # facing south, 90° off both


def test_a_view_that_cannot_see_what_the_camera_sees_does_not_cover_it():
    kiosk = ([(1, 4), (5, 4), (5, 8), (1, 8)], 0.0, 4.0)  # between the panorama and the wall
    there = _panorama("there", 9.5, 3.0)
    camera = np.array([[3.0, 0.0, 90.0]])  # faces the east row at (6, 0)

    assert _covers(_chooser(["there"]), [(there, 90.0)], camera)[0, 0]
    assert not _covers(_chooser(["there"], [*ROWS, kiosk]), [(there, 90.0)], camera)[0, 0]


def test_views_are_chosen_where_the_camera_most_likely_stands():
    panoramas = [_panorama(f"p{k}", y) for k, y in enumerate((0.0, 15.0, 30.0, 45.0))]

    chosen = _chooser([p.id for p in panoramas]).views((0.0, 0.0), panoramas, budget=2)

    assert list(chosen) == ["p0"]
    east, west = sorted(chosen["p0"])
    assert 0 < east < 180 < west < 360  # one across the street each way


def test_a_panorama_not_placed_by_colmap_only_fills_a_hole():
    sure, unsure = _panorama("sure", 40.0), _panorama("unsure", 1.0)
    both = _chooser(["sure"]).views((0.0, 0.0), [unsure, sure], budget=4)
    only_unsure = _chooser([]).views((0.0, 0.0), [unsure], budget=4)

    assert list(both)[0] == "sure"  # the sure one first, though farther
    assert list(only_unsure) == ["unsure"]  # where no sure one sees: the unsure one


def test_a_place_s_choice_is_made_once_for_every_matcher():
    chooser = _chooser(["p"])
    pool = [_panorama("p", 0.0)]

    assert chooser.views((0.0, 0.0), pool) is chooser.views((0.0, 0.0), pool)


def test_what_a_camera_looks_at_does_not_depend_on_the_others_asked_with_it():
    far_wall = [([(-205, -50), (-200, -50), (-200, 50), (-205, 50)], 0.0, 15.0)]
    chooser = _chooser([], far_wall)
    west = np.array([[-60.0, 0.0, 270.0]])  # 140 m from the wall
    east = np.array([[x, 0.0, 90.0] for x in range(30, 61, 5)])  # pulling their mean away

    alone = chooser.aims(west)[0]
    together = chooser.aims(np.vstack([west, east]))[0]

    assert alone[0] == -200.0
    np.testing.assert_allclose(together, alone)
