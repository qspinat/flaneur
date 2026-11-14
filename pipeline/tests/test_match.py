"""Finding an old photo's camera from points on today's buildings, on built scenes (the
matcher itself needs its models: it is checked on real photos, see `flaneur match`)."""

import numpy as np
import pytest

from flaneur import match


def level_camera(heading: float, fov: float, horizon: float, size: tuple[int, int]):
    """Projection of world points (east, north, up) by a level camera at the origin."""
    h, w = size
    f = w / 2 / np.tan(np.radians(fov) / 2)
    r = np.radians(heading)
    right, down, forward = (np.cos(r), -np.sin(r), 0), (0, 0, -1), (np.sin(r), np.cos(r), 0)

    def project(points: np.ndarray) -> np.ndarray:
        x, y, z = (points @ np.array(axis, float) for axis in (right, down, forward))
        return np.stack([w / 2 - 0.5 + f * x / z, horizon * h - 0.5 + f * y / z], axis=1)

    return project


def on_street(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """A flat street at height 0."""
    return np.zeros_like(xs)


def seen_from(n: int, panoramas: int) -> np.ndarray:
    """Which panoramas saw each of n points: one each, in turn."""
    return np.array([frozenset([f"pano:{k % panoramas}"]) for k in range(n)], dtype=object)


def test_a_view_and_its_rays_agree():
    equirectangular = np.zeros((1024, 2048, 3), np.uint8)
    equirectangular[300:310, 1200:1210] = 255  # a bright square, 31° right of the centre
    azimuth, heading = 40.0, 70.0

    view = match.perspective(equirectangular, match.camera_axes(azimuth), heading)

    row, column = np.argwhere(view.max(axis=2) > 128).mean(axis=0)
    [ray] = match.view_directions(np.array([[column, row]]), heading)
    bearing = np.degrees(np.arctan2(ray[0], ray[1])) % 360
    elevation = np.degrees(np.arcsin(ray[2]))
    assert bearing == pytest.approx(azimuth + (1205 / 2048 - 0.5) * 360, abs=0.5)
    assert elevation == pytest.approx((0.5 - 305 / 1024) * 180, abs=0.5)


def street(seed: int) -> np.ndarray:
    """Points on the facades on both sides of a street running north, and on its far end."""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            np.column_stack([np.full(60, -6.0), rng.uniform(5, 60, 60), rng.uniform(0, 18, 60)]),
            np.column_stack([np.full(60, 8.0), rng.uniform(5, 60, 60), rng.uniform(0, 18, 60)]),
            np.column_stack([rng.uniform(-6, 8, 30), np.full(30, 70.0), rng.uniform(0, 18, 30)]),
        ]
    )


def test_solve_finds_a_level_camera_among_wrong_matches():
    rng = np.random.default_rng(1)
    camera = np.array([3.0, -2.0, 1.6])
    walls = street(1)
    size = (800, 600)
    pixels = level_camera(10.0, 50.0, 0.6, size)(walls - camera)
    pixels[:40] = rng.uniform(0, 600, (40, 2))  # wrong matches
    sources = seen_from(len(walls), 3)

    found = match.solve(walls, pixels, sources, size, match.Where(on_street))

    assert found is not None
    assert (found.x, found.y, found.z) == pytest.approx(tuple(camera), abs=0.3)
    assert found.heading == pytest.approx(10, abs=1)
    assert found.fov == pytest.approx(50, abs=5)
    assert found.horizon == pytest.approx(0.6, abs=0.05)


def test_two_points_give_the_level_camera_that_sees_them():
    size = (800, 600)
    two = np.array([[-6.0, 20.0, 7.0], [8.0, 35.0, 3.0]])
    camera = np.array([3.0, -2.0, 1.6])
    pixels = level_camera(10.0, 50.0, 0.6, size)(two - camera)
    focal = 300 / np.tan(np.radians(25))

    found = match.upright_pairs(two[None], pixels[None], focal, (299.5, 0.6 * 800 - 0.5))

    [answers] = found
    assert any(np.allclose(a, [3.0, -2.0, 1.6, 10.0], atol=1e-6) for a in answers)


def test_solve_finds_a_camera_when_few_matches_are_right():
    rng = np.random.default_rng(3)
    camera = np.array([3.0, -2.0, 1.6])
    walls = np.concatenate([street(8), street(9)[:50]])  # 200 points
    size = (800, 600)
    pixels = level_camera(10.0, 50.0, 0.6, size)(walls - camera)
    wrong = rng.permutation(len(walls))[:175]
    pixels[wrong] = rng.uniform(0, 600, (175, 2))  # one match in eight is right

    found = match.solve(walls, pixels, seen_from(len(walls), 3), size, match.Where(on_street))

    assert found is not None and found.agree == 25
    assert (found.x, found.y, found.heading) == pytest.approx((3.0, -2.0, 10.0), abs=0.3)


def test_solve_keeps_the_camera_out_of_the_buildings_standing_then():
    camera = np.array([3.0, -2.0, 1.6])
    walls = street(4)
    size = (800, 600)
    pixels = level_camera(10.0, 50.0, 0.6, size)(walls - camera)
    sources = seen_from(len(walls), 2)

    def outside(xs, ys):  # a house then stood where the camera is
        return np.hypot(xs - 3.0, ys + 2.0) > 3.0

    assert match.solve(walls, pixels, sources, size, match.Where(on_street, outside)) is None
    assert match.solve(walls, pixels, sources, size, match.Where(on_street)) is not None


def test_solve_rejects_a_camera_that_does_not_stand_in_the_street():
    walls = street(2)
    size = (800, 600)
    pixels = level_camera(0.0, 50.0, 0.5, size)(walls - np.array([1.0, 0.0, 12.0]))  # 12 m up

    found = match.solve(walls, pixels, seen_from(len(walls), 2), size, match.Where(on_street))

    assert found is None


def test_solve_is_as_exact_in_lambert_93_metres():
    paris = np.array([652000.0, 6861000.0, 0.0])
    camera = np.array([3.0, -2.0, 1.6])
    walls = street(6)
    size = (800, 600)
    pixels = level_camera(0.0, 50.0, 0.5, size)(walls - camera)

    found = match.solve(
        walls + paris, pixels, seen_from(len(walls), 2), size, match.Where(on_street)
    )

    assert found.agree == len(walls)
    assert (found.x, found.y) == pytest.approx(tuple((camera + paris)[:2]), abs=0.02)


def test_each_detail_is_placed_once_where_its_rays_meet_or_on_its_walls():
    window, ledge = np.array([4.0, 20.0, 7.0]), np.array([-2.0, 15.0, 4.0])
    a, b = np.array([0.0, 0.0, 2.5]), np.array([6.0, 1.0, 2.5])

    def towards(target, origin):
        d = target - origin
        return d / np.linalg.norm(d)

    nowhere = [np.nan] * 3
    casts = [  # (detail, panorama, origin, direction, cast onto today's walls)
        (0, 0, a, towards(window, a), nowhere),  # 0: seen from two panoramas: their rays meet
        (0, 1, b, towards(window, b), nowhere),
        (1, 0, a, towards(window, a), nowhere),  # 1: two rays that miss each other, no wall
        (1, 1, b, towards(ledge, b), nowhere),
        (2, 0, a, towards(ledge, a), ledge),  # 2: one cast, on a wall
        (3, 1, b, towards(ledge, b), nowhere),  # 3: one cast, into the sky
        (4, 0, a, towards(ledge, a), ledge),  # 4: a duplicate capture, parallel rays; walls agree
        (4, 1, a, towards(ledge, a), ledge + 0.2),
        (5, 0, a, towards(ledge, a), ledge),  # 5: two views of one panorama, 5 m apart
        (5, 0, a, towards(window, a), window),
    ]
    keys, sources, origins, directions, spots = (np.array(c) for c in zip(*casts, strict=True))

    points, details, walls, seen = match.placed(keys, sources, origins, directions, spots)

    assert details.tolist() == [0, 2, 4]
    assert points == pytest.approx(np.array([window, ledge, ledge + 0.1]), abs=1e-6)
    assert walls.tolist() == [-1, 4, 6]  # the cast on whose wall each lies
    assert [sorted(s) for s in seen] == [[0, 1], [0], [0, 1]]


def painted(blocks, panorama, wall: tuple[int, int, int]) -> np.ndarray:
    """A 360° picture from the panorama (its centre facing north): walls in the colour `wall`,
    the street (height 0) dark grey, the sky black."""
    h, w = 256, 512
    rows, columns = np.mgrid[0:h, 0:w]
    bearing = np.radians((columns + 0.5) * 360 / w - 180)
    elevation = (0.5 - (rows + 0.5) / h) * np.pi
    rays = np.stack(
        [
            np.sin(bearing) * np.cos(elevation),
            np.cos(bearing) * np.cos(elevation),
            np.sin(elevation),
        ],
        axis=-1,
    ).reshape(-1, 3)
    origin = np.array([panorama.x, panorama.y, panorama.z])
    points, edge = blocks.hit(origin, rays)
    to_street = np.where(rays[:, 2] < 0, -origin[2] / np.minimum(rays[:, 2], -1e-9), np.inf)
    to_wall = np.where(edge >= 0, np.linalg.norm(points - origin, axis=1), np.inf)
    picture = np.zeros((h * w, 3), np.uint8)
    picture[to_street < to_wall] = 60
    picture[(edge >= 0) & (to_wall <= to_street)] = wall
    return picture.reshape(h, w, 3)


def test_today_is_seen_from_an_old_camera_through_the_panoramas_that_see_each_spot():
    from flaneur.blocks import Blocks

    blocks = Blocks.of(
        [
            ([(-50, 20), (50, 20), (50, 30), (-50, 30)], 0.0, 5.0),  # a low wall, 20 m north
            ([(1, 9), (4, 9), (4, 11), (1, 11)], 0.0, 8.0),  # a kiosk: hides it from `near`
        ]
    )
    near = match.Panorama("near", 5.0, 0.0, 2.5, 0.0, path=None)  # sees the wall's middle askew
    far = match.Panorama("far", -8.0, 0.0, 2.5, 0.0, path=None)  # more askew, but unhidden
    around = [
        (near, painted(blocks, near, (0, 0, 255))),
        (far, painted(blocks, far, (255, 255, 255))),
    ]
    camera = match.View(0.0, 0.0, 1.6, heading=0.0, horizon=0.6, fov=50.0)

    seen, source = match.today_from(camera, (300, 400), around, blocks, lambda x, y: 0 * x)

    ahead = int(0.6 * 300) - 10, 200
    assert seen[ahead].tolist() == [255, 255, 255] and source[ahead] == 1  # from `far`: it sees it
    assert seen[295, 200].tolist() == [60, 60, 60]  # just below: the street, not smeared
    assert seen[5, 200].max() < 30  # high up: the sky


def test_a_detail_seen_from_two_panoramas_is_placed_where_their_rays_meet():
    window = np.array([4.0, 20.0, 7.0])
    origins = np.array([[0.0, 0.0, 2.5], [6.0, 1.0, 2.5]])
    directions = (window - origins) / np.linalg.norm(window - origins, axis=1, keepdims=True)

    point, miss = match.meet(origins, directions)

    assert point == pytest.approx(window, abs=1e-6)
    assert miss == pytest.approx(0, abs=1e-6)


def test_a_level_camera_is_refined_from_points_it_sees():
    camera = np.array([3.0, -2.0, 1.6])
    walls = street(7)
    size = (800, 600)
    pixels = level_camera(10.0, 50.0, 0.6, size)(walls - camera)
    rough = match.View(4.0, -1.0, 2.5, 13.0, 0.5, 55.0)  # what a first fit might give

    found = match.refine_level(rough, walls, pixels, size)

    assert (found.x, found.y, found.z) == pytest.approx(tuple(camera), abs=0.05)
    assert (found.heading, found.fov, found.horizon) == pytest.approx((10, 50, 0.6), abs=0.1)


def test_a_street_rising_ahead_is_met_where_it_rises():
    from flaneur.blocks import Blocks

    h, w = 1440, 2880
    picture = np.zeros((h, w, 3), np.uint8)
    picture[h // 2 + 20 :] = (0, 0, 255)  # 2.5° and more below the panorama's horizon: red
    around = [(match.Panorama("here", 0.0, -1.0, 2.5, 0.0, path=None), picture)]
    camera = match.View(0.0, 0.0, 1.6, heading=0.0, horizon=0.5, fov=50.0)
    rising = lambda x, y: 0.08 * y  # noqa: E731  # steeper than the rays just below the horizon

    seen, source = match.today_from(camera, (100, 100), around, Blocks.of([]), rising)

    # met under 20 m ahead, well below the panorama's eye; not 80 m off or more, near its level
    assert (source[51:56, 50] == 0).all() and (seen[51:56, 50, 2] == 255).all()
    assert (source[:45, 50] == match.SKY).all()


def test_a_camera_seeing_one_far_facade_is_unsure_how_far_back_it_stood():
    rng = np.random.default_rng(5)
    camera = np.array([0.0, 0.0, 1.6])
    size = (800, 600)
    facade = np.column_stack(
        [rng.uniform(-10, 10, 60), np.full(60, 60.0), rng.uniform(0, 18, 60)]
    )  # one wall, 60 m ahead, square on
    deep = street(10)
    view = match.View(*camera, heading=0.0, horizon=0.6, fov=50.0)

    def seen(points):
        pixels = level_camera(0.0, 50.0, 0.6, size)(points - camera)
        return pixels + rng.normal(0, 1.0, pixels.shape)  # a pixel of noise

    unsure, _ = match.spread(view, facade, seen(facade), size)
    sure, _ = match.spread(view, deep, seen(deep), size)

    assert unsure > 0.5
    assert sure < 0.1


def test_solve_turns_down_a_camera_it_is_unsure_of():
    rng = np.random.default_rng(6)
    camera = np.array([0.0, 0.0, 1.6])
    size = (800, 600)
    facade = np.column_stack([rng.uniform(-10, 10, 60), np.full(60, 60.0), rng.uniform(0, 18, 60)])
    pixels = level_camera(0.0, 50.0, 0.6, size)(facade - camera)
    pixels += rng.normal(0, 1.0, pixels.shape)
    trace = match.Trace()

    assert (
        match.solve(facade, pixels, seen_from(60, 3), size, match.Where(on_street), trace) is None
    )
    assert trace.verdict == "aucune caméra"
    unsure = [c for c in trace.candidates if c.why == "position incertaine"]
    assert unsure and unsure[0].agree >= match.MIN_AGREE
    assert unsure[0].spread_m > match.MOST_SPREAD_M


def test_matches_with_no_two_view_geometry_count_as_none(monkeypatch):
    def fails(*args, **kwargs):  # OpenCV's estimator asserts on some degenerate sets
        raise match.cv2.error("!model.empty()")

    monkeypatch.setattr(match.cv2, "findFundamentalMat", fails)
    matcher = object.__new__(match.Matcher)  # no models: pairs is given
    matcher.pairs = lambda a, b: np.column_stack([np.arange(20), np.arange(20)])
    some = match.Features(
        np.random.default_rng(0).uniform(0, 100, (20, 2)).astype(np.float32),
        np.zeros((20, 128), np.float32),
        (100, 100),
    )

    mine, theirs = matcher.match(some, some)

    assert len(mine) == len(theirs) == 0


def test_solve_turns_down_a_photo_two_cameras_explain_as_well():
    size = (800, 600)
    camera = np.array([3.0, -2.0, 1.6])
    here = street(11)
    turn = np.radians(30)  # another street, turned 30° clockwise about the camera
    around = street(12) - camera
    there = camera + np.column_stack(
        [
            around[:, 0] * np.cos(turn) + around[:, 1] * np.sin(turn),
            -around[:, 0] * np.sin(turn) + around[:, 1] * np.cos(turn),
            around[:, 2],
        ]
    )
    one = level_camera(10.0, 50.0, 0.6, size)(here - camera)
    other = level_camera(40.0, 50.0, 0.6, size)(there - camera)  # seen the same, turned 30°
    walls, pixels = np.concatenate([here, there]), np.concatenate([one, other])
    sources = seen_from(len(walls), 3)

    assert (
        match.solve(walls, pixels, sources, size, match.Where(on_street)) is None
    )  # 150 against 150
    fewer = np.concatenate([here, there[:40]]), np.concatenate([one, other[:40]])
    found = match.solve(*fewer, seen_from(190, 3), size, match.Where(on_street))
    assert found is not None and found.heading == pytest.approx(10, abs=1)  # 150 against 40


def test_solve_keeps_the_camera_near_where_the_caption_puts_the_photo():
    camera = np.array([3.0, -2.0, 1.6])
    walls = street(13)
    size = (800, 600)
    pixels = level_camera(10.0, 50.0, 0.6, size)(walls - camera)
    sources = seen_from(len(walls), 2)

    near = match.solve(walls, pixels, sources, size, match.Where(on_street, around=(20.0, 30.0)))
    far = match.solve(walls, pixels, sources, size, match.Where(on_street, around=(60.0, 40.0)))

    assert near is not None and near.x == pytest.approx(3.0, abs=0.1)  # 38 m away
    assert far is None  # 72 m away: past what a caption gets wrong


def test_a_rival_explains_other_points_or_ties():
    def camera(x: float, inliers: range) -> match.Camera:
        found = np.array(inliers)
        return match.Camera(x, 0.0, 1.6, 0.0, 0.5, 60.0, len(found), 3, found)

    best = camera(0.0, range(100))

    assert not match._rivals(camera(4.0, range(90)), best)  # the same points, less well
    assert not match._rivals(camera(4.0, range(40, 110)), best)  # 10 of its own
    assert match._rivals(camera(4.0, range(100, 170)), best)  # 70 of its own: another story
    assert match._rivals(camera(4.0, range(97)), best)  # a tie
    assert not match._rivals(camera(1.0, range(100, 170)), best)  # too near to be another


def test_a_view_with_no_sky_is_still_seen():
    from flaneur.blocks import Blocks

    blocks = Blocks.of([([(-50, 3), (50, 3), (50, 13), (-50, 13)], 0.0, 200.0)])  # a cliff ahead
    here = match.Panorama("here", 0.0, -1.0, 2.5, 0.0, path=None)
    around = [(here, painted(blocks, here, (0, 0, 255)))]
    camera = match.View(0.0, 0.0, 1.6, heading=0.0, horizon=0.9, fov=30.0)

    seen, source = match.today_from(camera, (60, 80), around, blocks, lambda x, y: 0 * x)

    assert (source != match.SKY).all()
    assert seen.shape == (60, 80, 3)


def test_a_camera_found_is_traced_as_found():
    camera = np.array([3.0, -2.0, 1.6])
    size = (800, 600)
    walls = street(1)
    pixels = level_camera(10.0, 50.0, 0.6, size)(walls - camera)
    trace = match.Trace()

    found = match.solve(
        walls, pixels, seen_from(len(walls), 3), size, match.Where(on_street), trace
    )

    assert found is not None and trace.verdict == "située"
    assert any(c.why == "" and c.agree == found.agree for c in trace.candidates)


def test_too_few_points_are_traced_before_any_camera():
    trace = match.Trace()

    match.solve(
        np.zeros((5, 3)),
        np.zeros((5, 2)),
        seen_from(5, 2),
        (800, 600),
        match.Where(on_street),
        trace,
    )

    assert trace.verdict == "trop peu de points" and trace.points == 5


def sight_of(camera: np.ndarray, heading: float, fov: float, landmark: tuple[float, float]):
    """Where a landmark stands in the photo of a level camera: a sight of it."""
    [[u, _]] = level_camera(heading, fov, 0.5, (100, 100))(np.array([[*landmark, 1.6]]) - camera)
    return match.Sight(points=(landmark,), column=(u + 0.5) / 100)


def test_a_landmark_is_seen_where_the_photo_shows_it():
    camera = np.array([3.0, -2.0, 1.6])
    sight = sight_of(camera, 10.0, 50.0, (60.0, 300.0))
    view = match.View(*camera, heading=10.0, horizon=0.6, fov=50.0)

    assert sight.seen_by(view)
    assert not sight.seen_by(match.View(*camera, heading=20.0, horizon=0.6, fov=50.0))
    assert match.Sight(points=((-60.0, 300.0), (60.0, 300.0)), column=sight.column).seen_by(view)


def test_a_landmark_of_twin_towers_is_seen_where_either_of_them_stands():
    camera = np.array([0.0, 0.0, 1.6])
    one_tower = sight_of(camera, 0.0, 50.0, (18.0, 300.0))  # its point: midway, 18 m from each
    view = match.View(*camera, heading=0.0, horizon=0.6, fov=50.0)
    midway = ((0.0, 300.0),)

    assert match.Sight(points=midway, column=one_tower.column, spreads=(18.0,)).seen_by(view)
    assert not match.Sight(points=midway, column=one_tower.column).seen_by(view)


def test_a_wide_landmark_near_is_seen_as_far_off_as_its_tops_can_stand():
    camera = np.array([0.0, 0.0, 1.6])
    tower = (22.7, 49.6)  # 25 m from the point, seen 24.6° off it: asin(25 / 60)
    sight = sight_of(camera, 10.0, 50.0, tower)
    view = match.View(*camera, heading=10.0, horizon=0.6, fov=50.0)

    assert match.Sight(((0.0, 60.0),), sight.column, spreads=(19.0,)).seen_by(view)


def test_each_alternative_of_a_landmark_keeps_its_own_spread():
    camera = np.array([0.0, 0.0, 1.6])
    beside = sight_of(camera, 0.0, 50.0, (26.0, 300.0))  # 5° off the spire, 300 m away
    view = match.View(*camera, heading=0.0, horizon=0.6, fov=50.0)
    spire, towers = (0.0, 300.0), (-900.0, 300.0)

    either = match.Sight((spire, towers), beside.column, spreads=(0.0, 18.6))
    assert not either.seen_by(view)


def test_a_landmark_tells_two_cameras_that_explain_a_photo_as_well_apart():
    size = (800, 600)
    camera = np.array([3.0, -2.0, 1.6])
    here = street(11)
    turn = np.radians(30)
    around = street(12) - camera
    there = camera + np.column_stack(
        [
            around[:, 0] * np.cos(turn) + around[:, 1] * np.sin(turn),
            -around[:, 0] * np.sin(turn) + around[:, 1] * np.cos(turn),
            around[:, 2],
        ]
    )
    walls = np.concatenate([here, there])
    pixels = np.concatenate(
        [
            level_camera(10.0, 50.0, 0.6, size)(here - camera),
            level_camera(40.0, 50.0, 0.6, size)(there - camera),
        ]
    )
    sighted = match.Where(on_street, sights=(sight_of(camera, 10.0, 50.0, (60.0, 300.0)),))

    found = match.solve(walls, pixels, seen_from(len(walls), 3), size, sighted)

    assert found is not None and found.heading == pytest.approx(10, abs=1)


def test_a_camera_that_does_not_see_the_landmark_is_turned_down():
    camera = np.array([3.0, -2.0, 1.6])
    walls, size = street(1), (800, 600)
    pixels = level_camera(10.0, 50.0, 0.6, size)(walls - camera)
    elsewhere = match.Where(on_street, sights=(sight_of(camera, 40.0, 50.0, (60.0, 300.0)),))
    trace = match.Trace()

    assert match.solve(walls, pixels, seen_from(len(walls), 3), size, elsewhere, trace) is None
    assert all(c.why for c in trace.candidates)  # none that sees it there is checked through


def test_a_landmark_does_not_stand_in_for_a_second_panorama():
    camera = np.array([3.0, -2.0, 1.6])
    walls, size = street(1), (800, 600)
    pixels = level_camera(10.0, 50.0, 0.6, size)(walls - camera)
    sighted = match.Where(on_street, sights=(sight_of(camera, 10.0, 50.0, (60.0, 300.0)),))
    trace = match.Trace()

    assert match.solve(walls, pixels, seen_from(len(walls), 1), size, sighted, trace) is None
    assert {c.why for c in trace.candidates} >= {"vue d'une seule source"}


def test_a_zoomed_view_is_aimed_at_the_landmarks_middle_and_sees_it_whole():
    panorama = match.Panorama("p", 0.0, 0.0, 0.0, 0.0, None)

    heading, pitch, fov = match.aimed(panorama, (0.0, 100.0), 50.0)

    assert heading == pytest.approx(0.0)
    assert pitch == pytest.approx(np.degrees(np.arctan(25 / 100)))
    assert fov == pytest.approx(2 * np.degrees(np.arctan(35 / 100)))
    assert match.aimed(panorama, (0.0, 3000.0), 50.0)[2] == match.ZOOM_FOV[0]  # far: narrowest


def test_a_ray_of_a_zoomed_view_meets_the_landmark_as_far_as_it_stands():
    origin = np.array([0.0, 0.0, 1.5])
    east = np.radians(10)
    rays = np.array([[0.0, 1.0, 0.0], [np.sin(east), np.cos(east), 0.0], [0.0, -1.0, 0.0]])

    met = match.onto(origin, rays, (0.0, 100.0))

    assert met[0] == pytest.approx([0.0, 100.0, 1.5])
    assert met[1] == pytest.approx([100 * np.tan(east), 100.0, 1.5])
    assert np.isnan(met[2]).all()  # turned away from it


def test_the_band_zoomed_on_maps_back_onto_the_photo_as_matched():
    photo = np.zeros((1000, 2000, 3), np.uint8)

    band, left, scale = match._band(photo, 0.5)

    assert band.shape[:2] == (600, 520)  # the upper part, 13 % of the width either side
    assert left == 740
    assert scale == pytest.approx(match.PHOTO_SIDE / 2000)


def facade_seen_from_the_origin(n: int = 60):
    """Points on a facade 20 m north, and where a level camera 1.6 m up at the origin, facing
    north, sees them (photo of 600 × 800)."""
    rng = np.random.default_rng(1)
    world = np.column_stack([rng.uniform(-10, 10, n), np.full(n, 20.0), rng.uniform(0.5, 12, n)])
    world[:, 1] += rng.uniform(-3, 3, n)  # balconies, cornices: not one plane
    project = level_camera(0.0, 60.0, 0.5, (600, 800))
    return world, project(world - [0, 0, 1.6])


def old_photo_rays(world: np.ndarray, at: tuple[float, float, float], name: str) -> match.Rays:
    """The rays of an old photo placed at `at`, to the points it shares with the photo."""
    origin = np.array(at)
    ahead = (world - origin) / np.linalg.norm(world - origin, axis=1, keepdims=True)
    n = len(world)
    return match.Rays(
        np.arange(n),
        np.full(n, f"photo:{name}", dtype=object),
        np.tile(origin, (n, 1)),
        ahead,
        np.full((n, 3), np.nan),  # never cast: an old photo's walls may since have gone
        np.full((n, 6), np.nan),
    )


def nothing_from_today(pixels: np.ndarray) -> match.Observed:
    return match.Observed(pixels, (600, 800), match.Rays.of([]))


def test_a_photo_matched_with_one_old_photo_only_is_not_placed():
    world, pixels = facade_seen_from_the_origin()
    trace = match.Trace()

    found = match.place(
        nothing_from_today(pixels),
        match.Where(on_street),
        trace,
        old_photo_rays(world, (6.0, -4.0, 1.6), "a"),
    )

    assert found.camera is None  # its rays alone tell directions, never how far
    assert trace.points == 0


def test_rays_of_two_old_photos_that_meet_place_a_photo():
    world, pixels = facade_seen_from_the_origin()
    both = match.Rays.of(
        [
            old_photo_rays(world, (6.0, -4.0, 1.6), "a"),
            old_photo_rays(world, (-7.0, -2.0, 1.6), "b"),
        ]
    )

    found = match.place(nothing_from_today(pixels), match.Where(on_street), match.Trace(), both)

    assert found.camera is not None
    assert np.hypot(found.camera.x, found.camera.y) < 0.2
    assert found.camera.panoramas == 2
    assert {"photo:a", "photo:b"} <= frozenset().union(*found.points.seen)


def test_points_placed_already_keep_the_pictures_that_saw_them():
    world, pixels = facade_seen_from_the_origin()
    seen = np.array([frozenset({"pano:1", "pano:2", "photo:a"})] * len(world), dtype=object)

    found = match.place(
        nothing_from_today(pixels),
        match.Where(on_street),
        match.Trace(),
        known=match.Known(np.arange(len(world)), world, seen),
    )

    assert found.camera is not None
    assert found.points.seen[0] == {"pano:1", "pano:2", "photo:a"}
    assert found.points.world == pytest.approx(
        world[np.isin(np.arange(len(world)), found.camera.inliers)]
    )


def test_a_wall_point_the_model_puts_far_off_is_left_out():
    from flaneur.blocks import Blocks

    wall = Blocks.of([([(-50, 10), (50, 10), (50, 12), (-50, 12)], 0.0, 30.0)])  # 10 m north
    here = match.Panorama("here", 0.0, 0.0, 2.5, 0.0, path=None)
    ahead = np.array([[0.0, 1.0, 0.0]] * 3)
    model = np.array([30.0, 11.0, np.nan])  # beyond the wall; on it; nothing said

    rays = match._cast(here, np.arange(3), ahead, wall, model=model)

    kept = ~np.isnan(rays.casts).any(axis=1)
    assert kept.tolist() == [False, True, True]  # the model three times as far: left out
