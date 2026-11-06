"""The depth of today's details, from a panorama and its neighbours matched together."""

from pathlib import Path

import numpy as np

from flaneur import depth, match
from flaneur.registration import ViewOf

SIDE = 640
TOP, GROUND = 12.0, -2.5  # the walls' tops and the street, from the panoramas' height


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


def _hit(direction: np.ndarray) -> float:
    """How far a direction from HOME meets the street's walls or its end."""
    found = []
    if abs(direction[0]) > 1e-9:
        found.append(4.0 / abs(direction[0]))
    if direction[1] > 1e-9:
        found.append(60.0 / direction[1])
    t = min(found)
    return t if GROUND <= t * direction[2] <= TOP else np.inf


def test_neighbours_are_the_nearest_fixed_first_within_reach():
    near = [_panorama(f"p{d:g}", d, 0.0) for d in (0.5, 2, 5, 8, 11, 20)]

    found = depth.neighbours(HOME, [HOME, *near], {"p8"})

    assert [p.id for p in found] == ["p8", "p2", "p5", "p11"]


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


def test_a_direction_far_from_every_detail_has_no_depth():
    found = depth.of_panorama(HOME, [HOME, NEXT], set(), _Views())

    metres, *_ = depth.at(found, np.array([[0.0, 0.0, 1.0]]))  # straight up

    assert np.isnan(metres[0])


def test_a_depth_from_neighbours_colmap_did_not_fix_is_told():
    found = depth.of_panorama(HOME, [HOME, NEXT], set(), _Views())

    _, only_unfixed, _ = depth.at(found, found.directions[:1])

    assert only_unfixed[0]


def test_a_panorama_without_neighbours_has_no_depths():
    assert depth.of_panorama(HOME, [HOME], set(), _Views()) is None


def test_the_sky_of_a_view_is_infinitely_far_and_the_walls_are_not():
    found = depth.of_panorama(HOME, [HOME, NEXT], set(), _Views())
    view = ViewOf(heading=0.0)
    axes = HOME.axes

    metres = depth.in_view(found, view, axes, SIDE)

    rays = match.view_directions(np.array([[SIDE / 2, 20.0], [60.0, SIDE / 2]]), 0.0, side=SIDE)
    sky, wall = metres[20, SIDE // 2], metres[SIDE // 2, 60]
    assert np.isinf(sky) and np.isinf(_hit(rays[0]))
    assert abs(wall - _hit(rays[1])) < 0.05 * _hit(rays[1])


def test_a_near_wall_seen_in_another_view_from_the_neighbour_gets_its_depth():
    far_next = _panorama("far", 0.0, 5.0)  # the wall at (-4, 4): 315° from home, 256° from it

    directions, metres = depth.triangulated(HOME, far_next, _Views())

    points = directions * metres[:, None]
    assert (
        np.isclose(points[:, 0], -4.0, atol=0.01) & np.isclose(points[:, 1], 4.0, atol=0.01)
    ).any()


def test_a_direction_between_near_and_far_details_has_no_depth():
    near_and_far = depth.Depths(
        "home",
        np.array([[0.0, 1.0, 0.0]] * 4),
        np.array([5.0, 5.0, 30.0, 30.0]),  # a corner: a near wall's edge and the street beyond
        np.array(["next"] * 4),
        np.ones(4, bool),
    )

    metres, *_ = depth.at(near_and_far, np.array([[0.0, 1.0, 0.0]]))

    assert np.isnan(metres[0])


def test_a_depth_from_a_neighbour_colmap_fixed_is_not_flagged():
    found = depth.of_panorama(HOME, [HOME, NEXT], {"next"}, _Views())

    _, only_unfixed, _ = depth.at(found, found.directions[:1])

    assert not only_unfixed[0]


class _Unreadable(_Views):
    def of(self, p: match.Panorama) -> list[match.Features]:
        return [] if p.id == "broken" else super().of(p)


def test_a_neighbour_whose_picture_cannot_be_read_is_passed_over():
    broken = _panorama("broken", 0.0, -3.0)

    found = depth.of_panorama(HOME, [HOME, NEXT, broken], set(), _Unreadable())

    assert found is not None and set(found.seen_from.tolist()) == {"next"}


def test_a_view_with_too_few_details_has_no_depth_rather_than_sky():
    found = depth.of_panorama(HOME, [HOME, NEXT], set(), _Views())
    behind = ViewOf(heading=180.0)  # down the street, where nothing was matched

    metres = depth.in_view(found, behind, HOME.axes, SIDE)

    assert np.isnan(metres).all()


def test_details_in_a_line_give_a_depth_without_failing():
    up = np.radians(np.linspace(-10, 30, 12))  # one above the other: a line in the view
    in_a_line = depth.Depths(
        "home",
        np.column_stack([np.zeros(12), np.cos(up), np.sin(up)]),
        np.full(12, 10.0),
        np.array(["next"] * 12),
        np.ones(12, bool),
    )

    metres = depth.in_view(in_a_line, ViewOf(heading=0.0), HOME.axes, SIDE)

    assert np.isfinite(metres[SIDE // 2 + 20, SIDE // 2 + 40])


def test_a_view_takes_the_depth_of_the_three_views_nearest_its_way():
    assert sorted(depth.facing(140.0)) == [4, 5, 6]  # 150°, 120° and 180°: the nearest three
    assert sorted(depth.facing(355.0)) == [0, 1, 11]  # across north


def test_details_are_made_only_for_the_views_asked():
    only = depth.of_panorama(HOME, [HOME, NEXT], set(), _Views(), headings=[0])
    everywhere = depth.of_panorama(HOME, [HOME, NEXT], set(), _Views())

    assert 0 < len(only.metres) < len(everywhere.metres)


def test_depths_made_apart_are_joined():
    a = depth.of_panorama(HOME, [HOME, NEXT], set(), _Views(), headings=[0])
    b = depth.of_panorama(HOME, [HOME, NEXT], {"next"}, _Views(), headings=[1])

    both = depth.joined([a, b])

    assert len(both.metres) == len(a.metres) + len(b.metres) and both.fixed.sum() == len(b.metres)


def test_the_depth_around_a_view_is_its_facing_views_joined():
    made = []

    def view(k: int):
        made.append(k)
        return depth.of_panorama(HOME, [HOME, NEXT], set(), _Views(), headings=[k])

    found = depth.around(0.0, view)

    assert sorted(made) == sorted(depth.facing(0.0)) and len(found.metres) > 0


def test_no_depth_around_a_view_none_of_whose_views_has_any():
    assert depth.around(180.0, lambda k: None) is None
