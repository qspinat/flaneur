"""Today's walls: fitted from the details panoramas see, drawn and removed by hand."""

import numpy as np
import pytest

from flaneur import facades
from flaneur.registration import ViewOf

GROUND = -2.5  # the street, under the panorama's camera (at 0)


def _street(gap: tuple[float, float] | None = None, seed: int = 1, count: int = 600) -> np.ndarray:
    """Details on two facades 8 m apart (x = ±4, y 2..58; `count` on each), the street's end
    (y = 60), the ground, 2 cm noise and 10 % strays, the whole a little off the round numbers;
    the left facade open over `gap` (a side street)."""
    rng = np.random.default_rng(seed)
    left = np.column_stack(
        [np.full(count, -4.0), rng.uniform(2, 58, count), rng.uniform(-2, 11, count)]
    )
    if gap:
        left = left[(left[:, 1] < gap[0]) | (left[:, 1] > gap[1])]
    right = np.column_stack(
        [np.full(count, 4.0), rng.uniform(2, 58, count), rng.uniform(-2, 11, count)]
    )
    n = count // 3
    end = np.column_stack([rng.uniform(-4, 4, n), np.full(n, 60.0), rng.uniform(-2, 11, n)])
    n = count // 2
    ground = np.column_stack([rng.uniform(-3.5, 3.5, n), rng.uniform(2, 58, n), np.full(n, GROUND)])
    points = np.vstack([left, right, end, ground]) + rng.normal(0, 0.02, (1, 3))
    points += rng.normal(0, 0.02, points.shape)
    strays = rng.uniform([-10, 0, -2], [10, 70, 12], (len(points) // 10, 3))
    return np.vstack([points, strays])


def _along(wall: facades.Wall) -> tuple[float, float]:
    """A wall's angle from north (degrees, 0–180) and its line's offset from the origin."""
    a, b = np.array(wall.a), np.array(wall.b)
    d = (b - a) / np.linalg.norm(b - a)
    return float(np.degrees(np.arctan2(d[0], d[1])) % 180), float(abs(d[0] * a[1] - d[1] * a[0]))


def _found(walls, angle: float, offset: float) -> list[facades.Wall]:
    """The walls within 2° and 0.2 m of the line at `angle` from north, `offset` from the origin."""
    return [
        w
        for w in walls
        if abs((_along(w)[0] - angle + 90) % 180 - 90) < 2 and abs(_along(w)[1] - offset) < 0.2
    ]


def _scene(walls) -> facades.Scene:
    return facades.Scene("p", (0.0, 0.0, 0.0), GROUND, tuple(walls), (), 0)


def test_the_walls_of_a_street_are_found_back():
    walls = facades.fit(_street(), GROUND)

    assert len(_found(walls, 0, 4.0)) == 2  # the left and right facades, x = ±4
    assert len(_found(walls, 90, 60.0)) == 1  # the street's end
    assert all(w.made == "fitted" and w.top > 10 for w in walls)


def test_a_side_street_between_two_facades_on_one_line_leaves_a_gap():
    walls = facades.fit(_street(gap=(20, 30)), GROUND)
    left = [w for w in _found(walls, 0, 4.0) if max(w.a[0], w.b[0]) < 0]

    through = np.array([[-4.0, 25.0, -2.0]])
    through /= np.linalg.norm(through)
    _, met = facades.cast(_scene(walls), through)

    assert len(left) == 2
    assert met[0] == facades.GROUND  # past the opening, onto the street


def test_a_sparse_facade_is_kept_whole():
    walls = facades.fit(_street(count=60), GROUND)  # a detail a metre
    right = [w for w in _found(walls, 0, 4.0) if min(w.a[0], w.b[0]) > 0]

    assert len(right) == 1  # not cut where details happen to be few
    assert abs(right[0].b[1] - right[0].a[1]) > 45  # of 56


def test_a_misplaced_bd_topo_wall_is_not_kept_without_details():
    walls = facades.fit(_street(), GROUND, [((6.0, 0.0), (6.0, 60.0)), ((4.3, 0.0), (4.3, 60.0))])

    assert not _found(walls, 0, 6.0)
    assert len(_found(walls, 0, 4.0)) == 2


def test_a_ray_meets_the_nearest_wall_then_the_ground_then_nothing():
    walls = [facades.Wall((4.0, 0.0), (4.0, 60.0), 12.0, "fitted")]
    rays = np.array([[4.0, 10.0, 0.0], [0.0, 1.0, -0.5], [-1.0, 0.0, 0.2]])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    metres, met = facades.cast(_scene(walls), rays)

    assert metres[0] == pytest.approx(np.hypot(4, 10)) and met[0] == 0
    assert metres[1] == pytest.approx(np.hypot(5, 2.5)) and met[1] == facades.GROUND
    assert np.isnan(metres[2]) and met[2] == facades.NOTHING


def test_the_sky_of_a_view_is_infinitely_far():
    walls = [facades.Wall((-4.0, 0.0), (-4.0, 60.0), 12.0, "fitted")]

    metres = facades.in_view(_scene(walls), ViewOf(heading=0.0), 64)

    assert np.isinf(metres[2, 32]) and np.isfinite(metres[32, 4])  # above; the left facade


def test_a_wall_drawn_replaces_the_fitted_one_and_a_removed_one_stays_removed():
    fitted = [
        facades.Wall((-4.0, 2.0), (-4.0, 58.0), 12.0, "fitted"),
        facades.Wall((4.0, 2.0), (4.0, 58.0), 12.0, "fitted"),
    ]
    drawn = [facades.Wall((-3.8, 0.0), (-3.8, 60.0), 20.0, "drawn")]

    kept = facades.with_drawn(fitted, drawn, [((4.0, 0.0), (4.0, 60.0))], GROUND)

    assert kept == drawn


def test_a_short_wall_drawn_aslant_replaces_only_the_stretch_it_covers():
    fitted = [facades.Wall((4.0, 2.0), (4.0, 58.0), 12.0, "fitted")]
    drawn = [facades.Wall((3.5, 28.0), (4.5, 33.0), 20.0, "drawn")]  # 11° off, 5 m long

    kept = facades.with_drawn(fitted, drawn, [], GROUND)

    pieces = sorted((w.a[1], w.b[1]) for w in kept if w.made == "fitted")
    assert pieces == [pytest.approx((2, 28)), pytest.approx((33, 58))]


def test_walls_drawn_near_two_panoramas_serve_both(tmp_path):
    path = tmp_path / "walls.jsonl"
    drawn = [facades.Wall((0.0, 5.0), (10.0, 5.0), 20.0, "drawn")]
    removed = [((0.0, -5.0), (10.0, -5.0))]
    facades.record(path, "a", drawn, removed)

    near_b = facades.saved_near(path, 10.0, 0.0)
    far = facades.saved_near(path, 100.0, 0.0)
    facades.record(path, "a", [], [])  # its walls taken back

    assert near_b == (drawn, removed)
    assert far == ([], [])
    assert facades.saved_near(path, 10.0, 0.0) == ([], [])


def test_a_wall_drawn_at_one_panorama_and_removed_at_another_stays_removed(tmp_path):
    path = tmp_path / "walls.jsonl"
    facades.record(path, "a", [facades.Wall((0.0, 5.0), (10.0, 5.0), 20.0, "drawn")], [])
    facades.record(path, "b", [], [((0.0, 5.0), (10.0, 5.0))])

    drawn, _ = facades.saved_near(path, 5.0, 0.0)

    assert drawn == []


def test_a_scene_without_details_has_no_walls():
    assert facades.fit(np.empty((0, 3)), GROUND) == []


def test_walls_are_found_in_lambert_93_far_from_its_origin():
    shift = np.array([651000.0, 6862000.0, 0.0])

    walls = facades.fit(_street() + shift, GROUND)

    back = [
        facades.Wall(
            tuple(np.subtract(w.a, shift[:2])), tuple(np.subtract(w.b, shift[:2])), w.top, w.made
        )
        for w in walls
    ]
    assert len(_found(back, 0, 4.0)) == 2
    assert len(_found(back, 90, 60.0)) == 1


def test_a_facade_dense_with_details_is_found():
    rng = np.random.default_rng(3)
    y, z = rng.uniform(0, 20, (2, 4000))
    points = np.column_stack([np.where(np.arange(4000) % 2, 4.0, -4.0), y, z * 0.6])
    points += rng.normal(0, 0.02, points.shape)

    assert len(_found(facades.fit(points, GROUND), 0, 4.0)) == 2


def test_a_wall_drawn_after_a_removal_near_it_stays(tmp_path):
    path = tmp_path / "walls.jsonl"
    facades.record(path, "b", [], [((4.5, 0.0), (4.5, 60.0))])  # a misplaced fitted wall removed
    drawn = [facades.Wall((4.0, 0.0), (4.0, 60.0), 20.0, "drawn")]  # then the right one drawn
    facades.record(path, "a", drawn, [])

    assert facades.saved_near(path, 0.0, 30.0)[0] == drawn


def test_a_removed_wall_fitted_again_a_little_longer_stays_removed():
    fitted = [facades.Wall((4.0, 0.0), (4.0, 60.0), 12.0, "fitted")]

    assert facades.with_drawn(fitted, [], [((4.0, 2.0), (4.0, 58.0))], GROUND) == []


def test_a_wall_drawn_again_after_its_removal_comes_back(tmp_path):
    path = tmp_path / "walls.jsonl"
    drawn = [facades.Wall((4.0, 0.0), (4.0, 60.0), 20.0, "drawn")]
    facades.record(path, "a", drawn, [])
    facades.record(path, "b", [], [((4.0, 0.0), (4.0, 60.0))])
    facades.record(path, "a", drawn, [])  # drawn there again

    assert facades.saved_near(path, 0.0, 30.0)[0] == drawn


def test_a_ray_barely_down_meets_no_street_far_off():
    walls = [facades.Wall((4.0, 0.0), (4.0, 60.0), 12.0, "fitted")]
    ray = np.array([[0.0, np.cos(np.radians(1)), -np.sin(np.radians(1))]])  # 143 m to the street

    metres, met = facades.cast(_scene(walls), ray)

    assert np.isnan(metres[0]) and met[0] == facades.NOTHING


def test_a_wall_drawn_without_its_top_takes_that_of_the_wall_it_replaces_else_twenty_metres():
    fitted = [facades.Wall((-4.0, 2.0), (-4.0, 58.0), 12.0, "fitted")]
    drawn = [
        facades.Wall((-3.8, 0.0), (-3.8, 60.0), None, "drawn"),
        facades.Wall((-5.0, 70.0), (5.0, 70.0), None, "drawn"),  # where none was fitted
    ]

    kept = facades.with_drawn(fitted, drawn, [], GROUND)

    assert [w.top for w in kept] == [12.0, GROUND + facades.DRAWN_ABOVE_M]  # over the street


def test_walls_drawn_near_say_where_they_were_drawn(tmp_path):
    path = tmp_path / "walls.jsonl"
    facades.record(path, "a", [facades.Wall((0.0, 5.0), (10.0, 5.0), None, "drawn")], [])

    drawn, _ = facades.saved_near(path, 0.0, 0.0)

    assert [w.at for w in drawn] == ["a"]


def test_a_line_kept_for_the_details_of_one_dropped_is_fitted_to_them():
    xy = np.column_stack([np.zeros(20), np.arange(20.0)])  # twenty details along x = 0
    same = (np.array([1.0, 0.0]), 0.0)
    tree = facades.cKDTree(xy)

    labels, lines = facades._shared(xy, tree, [same, same])  # the first dropped, the second holds

    assert len(lines) == 1 and (labels == 0).all()


def test_a_facade_seen_as_a_thick_band_is_one_wall():
    rng = np.random.default_rng(5)
    band = np.column_stack(
        [4.0 + rng.uniform(-0.25, 0.25, 2000), rng.uniform(2, 58, 2000), rng.uniform(-2, 11, 2000)]
    )

    walls = facades.fit(band, GROUND)

    assert len(walls) == 1  # details spread ±0.25 m across it: one facade, not slivers


def test_a_far_facade_seen_as_a_wider_band_is_one_wall():
    rng = np.random.default_rng(5)
    far = np.column_stack(  # 30 m off: its details spread 2 cm a metre of distance either side
        [30.0 + rng.uniform(-0.6, 0.6, 3000), rng.uniform(-20, 20, 3000), rng.uniform(-2, 11, 3000)]
    )

    walls = facades.fit(far, GROUND, origin=np.zeros(3))

    assert len(walls) == 1


def test_a_column_of_details_makes_no_wall():
    rng = np.random.default_rng(7)
    trunk = np.column_stack(  # a tree's trunk or a lamp post, 5 m off: details all up it
        [5.0 + rng.normal(0, 0.1, 400), 5.0 + rng.normal(0, 0.1, 400), rng.uniform(-2, 8, 400)]
    )

    walls = facades.fit(np.vstack([trunk, _street()]), GROUND, origin=np.zeros(3))

    assert not [w for w in walls if np.hypot(*(np.add(w.a, w.b) / 2 - 5.0)) < 2]


def test_walls_crossing_each_other_keep_the_one_with_more_details():
    rng = np.random.default_rng(6)
    facade = np.column_stack([np.full(900, 4.0), rng.uniform(0, 40, 900), rng.uniform(-2, 11, 900)])
    t = rng.uniform(0, 1, 120)  # a few details strung across it, 30° off: no facade does that
    across = np.column_stack(
        [t * 16 - 4, 10 + t * 16 * np.tan(np.radians(30)), rng.uniform(-2, 11, 120)]
    )

    walls = facades.fit(np.vstack([facade, across]), GROUND)

    assert len(_found(walls, 0, 4.0)) == 1
    assert len(walls) == 1


def test_walls_meeting_at_a_corner_are_both_kept():
    rng = np.random.default_rng(7)
    side = np.column_stack([np.full(600, 4.0), rng.uniform(0, 30, 600), rng.uniform(-2, 11, 600)])
    end = np.column_stack([rng.uniform(-10, 4, 300), np.full(300, 30.0), rng.uniform(-2, 11, 300)])

    walls = facades.fit(np.vstack([side, end]), GROUND)

    assert len(_found(walls, 0, 4.0)) == 1 and len(_found(walls, 90, 30.0)) == 1


def _side_street() -> np.ndarray:
    """A side street 5 m wide going west from x = -2 (facades at y = ±2.5, its far end at
    x = -40), and a column of details a metre inside each of its corners, on the line x = -2
    (pillars, drainpipes)."""
    xs, zs = np.arange(-40.0, -3.0, 0.5), np.arange(-2.0, 8.0, 0.5)
    sides = [[x, y, z] for y in (-2.5, 2.5) for x in xs for z in zs]
    end = [[-40.0, y, z] for y in np.arange(-2.0, 2.5, 0.5) for z in zs]
    corners = [[-2.0, y, z] for y in (-1.5, 1.5) for z in np.arange(-2.0, 10.0, 0.25)]
    return np.array(sides + end + corners)


def _across_mouth(walls) -> list[facades.Wall]:
    """The walls standing across the side street's mouth (on x = -2, spanning y = 0)."""
    return [
        w
        for w in walls
        if abs(w.a[0] + 2) < 0.5
        and abs(w.b[0] + 2) < 0.5
        and min(w.a[1], w.b[1]) < 0 < max(w.a[1], w.b[1])
    ]


MOUTH = [((-2.0, -4.0), (-2.0, 4.0))]  # BD TOPO's outline, closed across the side street


def test_a_wall_the_camera_sees_through_is_carved_away():
    points = _side_street()

    blind = facades.fit(points, GROUND, MOUTH)
    seeing = facades.fit(points, GROUND, MOUTH, origin=np.array([4.0, 0.0, 0.0]))

    assert _across_mouth(blind)  # the corners alone make a wall across the street
    assert not _across_mouth(seeing)  # but the street's far end is seen through it


def test_walls_the_camera_does_not_see_through_are_kept_whole():
    points = _side_street()

    seeing = facades.fit(points, GROUND, MOUTH, origin=np.array([4.0, 0.0, 0.0]))

    for side in (-2.5, 2.5):
        along = [w for w in seeing if abs(w.a[1] - side) < 0.3 and abs(w.b[1] - side) < 0.3]
        assert along and max(abs(w.a[0] - w.b[0]) for w in along) > 30


def test_a_short_wall_standing_alone_is_dropped():
    lone = facades.Wall((10.0, 10.0), (10.0, 13.0), 8.0, "fitted")  # a tree's crown, alone
    street = facades.Wall((-4.0, 0.0), (-4.0, 40.0), 12.0, "fitted")

    assert facades._linked([street, lone]) == [street]


def test_a_short_wall_meeting_another_is_kept():
    street = facades.Wall((-4.0, 0.0), (-4.0, 40.0), 12.0, "fitted")
    corner = facades.Wall((-4.5, 40.5), (-8.0, 40.5), 12.0, "fitted")  # round the corner

    assert facades._linked([street, corner]) == [street, corner]


def test_a_long_wall_standing_alone_is_kept():
    alone = facades.Wall((10.0, 0.0), (10.0, 20.0), 12.0, "fitted")

    assert facades._linked([alone]) == [alone]


def test_a_short_wall_crossing_a_facade_away_from_its_ends_is_not_linked_to_it():
    street = facades.Wall((-4.0, 0.0), (-4.0, 40.0), 12.0, "fitted")
    across = facades.Wall((-6.5, 20.0), (-1.5, 20.0), 8.0, "fitted")  # foliage, across its middle

    assert facades._linked([across, street]) == [street]


def test_a_facade_whose_details_spread_wide_is_one_wall():
    rng = np.random.default_rng(8)
    wide = np.column_stack(  # trees before it, balconies: its details spread 0.4 m (σ) across
        [6.0 + rng.normal(0, 0.4, 3000), rng.uniform(2, 40, 3000), rng.uniform(-2, 11, 3000)]
    )

    walls = facades.fit(wide, GROUND, origin=np.zeros(3))

    assert len(_found(walls, 0, 6.0)) == 1 and len(walls) == 1


def test_a_sparse_facade_beside_a_dense_one_crossing_its_line_keeps_its_band():
    rng = np.random.default_rng(9)
    dense = np.column_stack(  # a facade seen face on, by every neighbour, across its line
        [rng.uniform(-20, 20, 20000), np.full(20000, 15.0), rng.uniform(-2, 11, 20000)]
    )
    sparse = np.column_stack(  # one seen at a grazing angle: few details along it
        [-6.0 + rng.normal(0, 0.1, 250), rng.uniform(-30, -10, 250), rng.uniform(-2, 11, 250)]
    )
    points = np.vstack([dense, sparse])
    outline = [((-6.0, -32.0), (-6.0, -8.0))]  # BD TOPO's facade: the line is proposed

    walls = facades.fit(points, GROUND, outline, origin=np.zeros(3))

    assert _found(walls, 90, 15.0) and _found(walls, 0, 6.0)
