"""The depth of today's details, from a panorama and its neighbours matched together."""

from pathlib import Path

import numpy as np
import pytest

from flaneur import depth, match

SIDE = 640


def _street() -> np.ndarray:
    """Details on two walls 8 m apart (x = ±4) along a street (north, y), and its far end."""
    ys, zs = np.arange(2.0, 59.0, 2.0), np.arange(-2.0, 11.5, 1.5)
    sides = [[x, y, z] for x in (-4.0, 4.0) for y in ys for z in zs]
    end = [[x, 60.0, z] for x in np.arange(-3.5, 4.0, 0.5) for z in np.arange(-2.0, 11.5, 1.0)]
    return np.array(sides + end)


STREET = _street()


def _panorama(name: str, x: float, y: float) -> match.Panorama:
    return match.Panorama(name, x, y, 0.0, 0.0, Path(f"{name}.jpg"))


def _pixels(world: np.ndarray, heading: float) -> tuple[np.ndarray, np.ndarray]:
    """Where world directions fall in the view facing `heading`, and which fall in it."""
    local = world @ match._view_axes(heading, match.VIEW_PITCH).T
    ahead = local[:, 2] > 0
    f = match.focal(match.VIEW_FOV, SIDE)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = local[:, :2] / local[:, 2:] * f + SIDE / 2 - 0.5
    inside = ahead & np.all((px >= 0) & (px < SIDE), axis=1)
    return px, inside


class _Matcher:
    def match(self, a: match.Features, b: match.Features) -> tuple[np.ndarray, np.ndarray]:
        common, ia, ib = np.intersect1d(
            a.descriptors[:, 0], b.descriptors[:, 0], return_indices=True
        )
        return ia, ib


class _Views:
    """The street's details seen in each view of a panorama, each known by its number."""

    matcher = _Matcher()

    def of(self, p: match.Panorama) -> list[match.Features]:
        found = []
        for heading in match.VIEW_HEADINGS:
            seen = STREET - [p.x, p.y, p.z]
            px, inside = _pixels(seen / np.linalg.norm(seen, axis=1, keepdims=True), heading)
            ids = np.nonzero(inside)[0].astype(np.float32)
            descriptors = np.zeros((len(ids), 128), np.float32)
            descriptors[:, 0] = ids
            found.append(match.Features(px[inside].astype(np.float32), descriptors, (SIDE, SIDE)))
        return found


HOME, NEXT = _panorama("home", 0.0, 0.0), _panorama("next", 0.0, 3.0)


def test_neighbours_across_the_line_of_sight_come_first():
    along, across = _panorama("along", 0.0, 6.0), _panorama("across", 6.0, 0.0)

    found = depth.neighbours(HOME, [HOME, along, across], set(), heading=0.0)  # facing north

    assert [p.id for p in found] == ["across", "along"]


def test_neighbours_placed_by_colmap_come_first():
    near = [_panorama(f"p{d:g}", d, 0.0) for d in (3, 5, 8, 11)]

    found = depth.neighbours(HOME, [HOME, *near], {"p3"}, heading=0.0)

    assert [p.id for p in found] == ["p3", "p11", "p8", "p5"]


def test_neighbours_nearer_than_two_metres_or_beyond_fifteen_are_left_out():
    near = [_panorama(f"p{d:g}", d, 0.0) for d in (1.4, 2, 14, 16)]

    found = depth.neighbours(HOME, [HOME, *near], set(), heading=0.0)

    assert [p.id for p in found] == ["p14", "p2"]


def test_a_second_click_s_rays_give_its_depth():
    a, b, point = np.zeros(3), np.array([3.0, 0.0, 0.0]), np.array([1.0, 10.0, 2.0])

    metres = depth.meeting(a, _unit(point - a), b, _unit(point - b))

    assert metres == pytest.approx(np.linalg.norm(point))


def test_rays_that_do_not_meet_give_no_depth():
    a, b = np.zeros(3), np.array([3.0, 0.0, 0.0])
    point = np.array([1.0, 10.0, 2.0])

    passing_by = depth.meeting(a, _unit(point - a), b, _unit(point + [0, 0, 1.0] - b))
    behind = depth.meeting(a, _unit(a - point), b, _unit(b - point))
    alike = depth.meeting(a, _unit(point - a), b, _unit(point - a))  # parallel

    assert passing_by is None
    assert behind is None
    assert alike is None


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_details_seen_from_two_panoramas_get_their_depth():
    directions, metres = depth.triangulated(HOME, NEXT, _Views())

    points = directions * metres[:, None]
    on_a_wall = np.isclose(np.abs(points[:, 0]), 4.0, atol=0.01) | np.isclose(
        points[:, 1], 60, atol=0.01
    )
    assert len(metres) > 100 and on_a_wall.all()


def test_details_straight_ahead_of_both_give_no_depth():
    directions, metres = depth.triangulated(HOME, NEXT, _Views())

    points = directions * metres[:, None]
    assert not (points[:, 1] > 50).any()  # the far end, along the line between the two


def test_a_near_wall_seen_in_another_view_from_the_neighbour_gets_its_depth():
    far_next = _panorama("far", 0.0, 5.0)  # the wall at (-4, 4): 315° from home, 256° from it

    directions, metres = depth.triangulated(HOME, far_next, _Views())

    points = directions * metres[:, None]
    assert (
        np.isclose(points[:, 0], -4.0, atol=0.01) & np.isclose(points[:, 1], 4.0, atol=0.01)
    ).any()


class _Unreadable(_Views):
    def of(self, p: match.Panorama) -> list[match.Features]:
        return [] if p.id == "broken" else super().of(p)


def test_a_neighbour_whose_picture_cannot_be_read_is_passed_over():
    broken = _panorama("broken", 0.0, -3.0)

    _, metres = depth.triangulated(HOME, broken, _Unreadable())

    assert len(metres) == 0


def test_details_are_made_only_for_the_views_asked():
    _, only = depth.triangulated(HOME, NEXT, _Views(), only=[0])
    _, everywhere = depth.triangulated(HOME, NEXT, _Views())

    assert 0 < len(only) < len(everywhere)


def test_a_near_neighbour_across_beats_a_far_one_straight_ahead():
    ahead, aside = _panorama("ahead", 0.0, 15.0), _panorama("aside", 4.0, 0.0)

    found = depth.neighbours(HOME, [HOME, ahead, aside], set(), heading=0.0)

    assert [p.id for p in found] == ["aside", "ahead"]


def test_a_second_click_s_rays_need_not_be_unit():
    a, b, point = np.zeros(3), np.array([3.0, 0.0, 0.0]), np.array([1.0, 10.0, 2.0])

    metres = depth.meeting(a, 3 * (point - a), b, 0.5 * (point - b))

    assert metres == pytest.approx(np.linalg.norm(point))


def test_matched_rays_passing_close_but_a_little_askew_give_no_detail():
    a, b, point = np.zeros(3), np.array([3.0, 0.0, 0.0]), np.array([1.0, 10.0, 0.0])
    rays_a, rays_b = _unit(point - a)[None], _unit(point + [0, 0, 0.25] - b)[None]  # 0.25 m off

    points, _ = depth._met(a, b, rays_a, rays_b)

    assert len(points) == 0  # within 0.3 m, but 0.7° off its ray: a wrong match


def test_a_second_click_a_few_pixels_off_still_gives_its_depth():
    a, b, point = np.zeros(3), np.array([3.0, 0.0, 0.0]), np.array([1.0, 10.0, 0.0])

    clicked = depth.meeting(a, _unit(point - a), b, _unit(point + [0, 0, 0.1] - b))  # 0.3° off

    assert clicked == pytest.approx(np.linalg.norm(point), abs=0.1)  # a hand is not that sure
