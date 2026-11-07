"""The depth of today's details, from a panorama and its neighbours matched together."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

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


def test_candidates_placed_by_colmap_come_first_then_the_nearest():
    near = [_panorama(f"p{d:g}", d, 0.0) for d in (3, 5, 8, 11)]

    found = depth.candidates(HOME, [HOME, *near], {"p8"})

    assert [p.id for p in found] == ["p8", "p3", "p5", "p11"]


def test_candidates_nearer_than_two_metres_or_beyond_fifteen_are_left_out():
    near = [_panorama(f"p{d:g}", d, 0.0) for d in (1.4, 2, 14, 16)]

    found = depth.candidates(HOME, [HOME, *near], set())

    assert [p.id for p in found] == ["p2", "p14"]


def test_candidates_are_a_dozen_at_most():
    near = [_panorama(f"p{d:g}", 2.0 + d / 2, 0.0) for d in range(20)]

    assert len(depth.candidates(HOME, [HOME, *near], set())) == depth.MOST_CANDIDATES


def _bearings(*degrees: float) -> np.ndarray:
    """Level directions toward these bearings (clockwise from north)."""
    r = np.radians(degrees)
    return np.column_stack([np.sin(r), np.cos(r), np.zeros(len(r))])


def test_the_neighbour_giving_most_details_where_the_photo_looks_comes_first():
    found = {
        "few": _bearings(0, 5),
        "many": _bearings(0, 10, -10, 20),
        "aside": _bearings(*[90] * 9),
    }

    assert depth.aimed(found, heading=0.0) == ["many", "few", "aside"]


def test_details_behind_the_view_do_not_count():
    found = {"behind": _bearings(*[180] * 9), "ahead": _bearings(350)}

    assert depth.aimed(found, heading=0.0)[0] == "ahead"


def test_every_neighbour_is_ranked():
    found = {f"p{i:02d}": _bearings(*[0] * (i + 1)) for i in range(12)}

    assert depth.aimed(found, heading=0.0) == [f"p{i:02d}" for i in range(11, -1, -1)]


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
    """A vector, or each row, made unit."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _placed(home: match.Panorama, other: match.Panorama, views) -> tuple[np.ndarray, np.ndarray]:
    """The pair's details, matched, set right and placed (none if it cannot be set right)."""
    rays_a, rays_b = depth.matched(home, other, views)
    a, b = np.array([home.x, home.y, home.z]), np.array([other.x, other.y, other.z])
    pair = depth.set_right(a, b, rays_a, rays_b)
    return depth.placed(a, pair, rays_a, rays_b) if pair else (np.empty((0, 3)), np.empty(0))


def test_details_seen_from_two_panoramas_get_their_depth():
    directions, metres = _placed(HOME, NEXT, _Views())

    points = directions * metres[:, None]
    on_a_wall = np.isclose(np.abs(points[:, 0]), 4.0, atol=0.01) | np.isclose(
        points[:, 1], 60, atol=0.01
    )
    assert len(metres) > 100 and on_a_wall.all()


def test_details_straight_ahead_of_both_give_no_depth():
    directions, metres = _placed(HOME, NEXT, _Views())

    points = directions * metres[:, None]
    assert not (points[:, 1] > 50).any()  # the far end, along the line between the two


def test_a_near_wall_seen_in_another_view_from_the_neighbour_gets_its_depth():
    far_next = _panorama("far", 0.0, 5.0)  # the wall at (-4, 4): 315° from home, 256° from it

    directions, metres = _placed(HOME, far_next, _Views())

    points = directions * metres[:, None]
    assert (
        np.isclose(points[:, 0], -4.0, atol=0.01) & np.isclose(points[:, 1], 4.0, atol=0.01)
    ).any()


class _Unreadable(_Views):
    def of(self, p: match.Panorama) -> list[match.Features]:
        return [] if p.id == "broken" else super().of(p)


def test_a_neighbour_whose_picture_cannot_be_read_is_passed_over():
    broken = _panorama("broken", 0.0, -3.0)

    rays_a, rays_b = depth.matched(HOME, broken, _Unreadable())

    assert len(rays_a) == len(rays_b) == 0


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


def _turned(
    rays: np.ndarray, rotvec_deg: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Rays as a panorama whose orientation is off by this rotation records them, and that
    rotation (rays · R.T sets them right)."""
    rot = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    return rays @ rot, rot  # recorded = rot⁻¹ · true


def test_a_pair_off_by_a_few_degrees_is_set_right_from_its_own_matches():
    a, b = np.zeros(3), np.array([3.0, 6.0, 0.0])
    true_b = _unit(STREET - b)
    recorded, rot = _turned(true_b, (2.0, -1.0, 3.0))  # tilt and heading both off

    pair = depth.set_right(a, b, _unit(STREET - a), recorded)

    assert pair is not None
    assert np.degrees(np.abs(recorded @ pair.turn.T - true_b).max()) < 0.05
    assert np.linalg.norm(pair.at - b) < 0.05


def test_a_pair_set_right_places_its_details_where_they_stand():
    a, b = np.zeros(3), np.array([3.0, 6.0, 0.0])
    recorded, _ = _turned(_unit(STREET - b), (1.5, 0.5, -2.0))
    rays_a = _unit(STREET - a)

    pair = depth.set_right(a, b, rays_a, recorded)
    directions, metres = depth.placed(a, pair, rays_a, recorded)

    assert len(metres) > 0.5 * len(STREET)  # (the far ones meet too narrowly: see WIDE_DEG)
    assert np.abs(a + directions * metres[:, None] - STREET[_near(directions, rays_a)]).max() < 0.05


def test_wrong_matches_among_many_do_not_spoil_the_pair():
    a, b = np.zeros(3), np.array([3.0, 6.0, 0.0])
    rng = np.random.default_rng(0)
    rays_a, true_b = _unit(STREET - a), _unit(STREET - b)
    wrong = rng.choice(len(STREET), len(STREET) // 4, replace=False)
    true_b[wrong] = _unit(rng.normal(size=(len(wrong), 3)))  # a quarter matched at random
    recorded, _ = _turned(true_b, (0.0, 2.0, 1.0))

    pair = depth.set_right(a, b, rays_a, recorded)

    right = np.setdiff1d(np.arange(len(STREET)), wrong)
    assert np.degrees(np.abs(recorded[right] @ pair.turn.T - _unit(STREET[right] - b)).max()) < 0.05


def test_a_pair_whose_matches_agree_on_nothing_is_refused():
    a, b = np.zeros(3), np.array([3.0, 6.0, 0.0])
    rng = np.random.default_rng(1)

    pair = depth.set_right(a, b, _unit(STREET - a), _unit(rng.normal(size=STREET.shape)))

    assert pair is None


def test_a_pair_with_too_few_matches_is_refused():
    a, b = np.zeros(3), np.array([3.0, 6.0, 0.0])
    few = STREET[: depth.LEAST_MATCHES - 1]

    assert depth.set_right(a, b, _unit(few - a), _unit(few - b)) is None


def _near(directions: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """Which of `rays` each direction is (they are the same rays, some left out)."""
    return np.argmax(directions @ rays.T, axis=1)


def test_details_whose_rays_meet_too_narrowly_are_left_out():
    a, b = np.zeros(3), np.array([3.0, 0.0, 0.0])
    near, far = np.array([1.0, 10.0, 0.0]), np.array([1.0, 60.0, 0.0])  # 17 and 3 degrees apart
    seen = np.array([near, far])
    pair = depth.Pair(np.eye(3), b)

    _, metres = depth.placed(a, pair, _unit(seen - a), _unit(seen - b))

    assert metres == pytest.approx([np.linalg.norm(near)])  # the far one's depth smears along


def test_two_panoramas_at_one_spot_are_not_set_right():
    a = np.zeros(3)

    assert depth.set_right(a, a.copy(), _unit(STREET - a), _unit(STREET - a)) is None
