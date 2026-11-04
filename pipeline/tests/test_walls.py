import pytest
from shapely.geometry import Point, box

from flaneur import walls


def test_facade_bearings_run_both_ways():
    found = walls.facade_bearings(walls.edges([box(0, 0, 10, 4)]), Point(5, -5))

    assert {round(b) for b, _ in found} >= {0, 90, 180, 270}


def street() -> list:
    """A street running north between two rows of houses, x = −5 and x = 7, closed at y = 60."""
    return walls.edges([box(-20, -10, -5, 60), box(7, -10, 20, 60), box(-20, 60, 20, 80)])


def xs(wall: walls.Wall) -> list[float]:
    return sorted(x for x, _ in wall.line)


def ys(wall: walls.Wall) -> list[float]:
    return sorted(y for _, y in wall.line)


def test_a_photo_along_a_street_gets_its_two_facades_and_the_one_closing_it():
    left, end, right = walls.facades(Point(0, 0), heading=5, street=0, fov=53, edges=street())

    inset = walls.INSET_M
    assert xs(left) == pytest.approx([-5 + inset] * 2)
    assert xs(right) == pytest.approx([7 - inset] * 2)
    assert ys(end) == pytest.approx([60 - inset] * 2)
    assert ys(left)[0] == pytest.approx(0)  # from where the photographer stood, not behind


def test_a_photo_facing_a_facade_gets_that_facade():
    (facade,) = walls.facades(Point(0, 0), heading=90, street=0, fov=53, edges=street())

    assert xs(facade) == pytest.approx([7 - walls.INSET_M] * 2)


def test_a_street_with_nothing_found_still_gets_facades():
    assert len(walls.facades(Point(0, 0), heading=0, street=None, fov=53, edges=[])) == 3


def test_the_width_ignores_a_house_jutting_out_for_a_few_metres():
    edges = street() + walls.edges([box(4, 8, 6, 12)])

    right = walls.facades(Point(0, 0), heading=0, street=0, fov=53, edges=edges)[2]

    assert xs(right) == pytest.approx([7 - walls.INSET_M] * 2)


def test_an_outline_the_camera_stands_on_is_not_a_facade():
    edges = street() + walls.edges([box(-0.5, -10, 0.5, 60)])

    left, _, right = walls.facades(Point(0, 0), heading=0, street=0, fov=53, edges=edges)

    assert xs(left) == pytest.approx([-5 + walls.INSET_M] * 2)
    assert xs(right) == pytest.approx([7 - walls.INSET_M] * 2)


def test_the_camera_keeps_room_on_both_sides():
    edges = walls.edges([box(-20, -10, -1.8, 60), box(7, -10, 20, 60)])

    left = walls.facades(Point(0, 0), heading=0, street=0, fov=53, edges=edges)[0]

    assert xs(left)[0] <= -walls.MIN_SIDE_M
