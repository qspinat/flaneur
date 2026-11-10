"""Today's panoramas set right on the map against two surveys (BD TOPO, LiDAR HD)."""

import numpy as np
import pytest

from flaneur import poses

ORIGIN = np.array([1000.0, 2000.0])
# a crossroads: two facades along the street, one across it ahead, one behind (on the plan, metres)
FACADES = np.array(
    [
        [[995.0, 1970.0], [995.0, 2030.0]],
        [[1005.0, 1970.0], [1005.0, 2030.0]],
        [[980.0, 2025.0], [995.0, 2025.0]],
        [[1005.0, 1985.0], [1020.0, 1985.0]],
    ]
)


def _details(count: int = 3000, seed: int = 0) -> np.ndarray:
    """Points on the facades (as a panorama standing at ORIGIN sees them, if placed right)."""
    rng = np.random.default_rng(seed)
    k = rng.integers(len(FACADES), size=count)
    t = rng.uniform(0, 1, count)[:, None]
    a, b = FACADES[k, 0], FACADES[k, 1]
    return a + t * (b - a) + rng.normal(0, 0.05, (count, 2))


def test_a_panorama_placed_off_is_set_right_against_the_facades():
    true = _details()
    seen = poses.moved(true, ORIGIN, poses.Correction(-1.2, 2.5, 4.0))  # as placed: off

    found = poses.fitted(seen, ORIGIN, lambda p: poses.segment_distance(p, FACADES))

    back = poses.moved(seen, ORIGIN, found)
    assert np.median(np.linalg.norm(back - true, axis=1)) < 0.1


def test_too_few_details_by_the_facades_set_nothing():
    far = _details(count=3000) + [200.0, 0.0]  # nowhere near any facade

    assert poses.fitted(far, ORIGIN, lambda p: poses.segment_distance(p, FACADES)) is None


def test_two_surveys_agreeing_on_a_large_error_correct_it():
    a, b = poses.Correction(2.0, -3.0, 1.0), poses.Correction(2.4, -3.6, 0.6)

    found = poses.agreed(a, b, _details(), ORIGIN)

    assert list(found) == pytest.approx([2.2, -3.3, 0.8])


def test_two_surveys_disagreeing_correct_nothing():
    details = _details()
    a, b = poses.Correction(2.0, -3.0, 0.0), poses.Correction(-1.0, -3.0, 0.0)
    assert poses.agreed(a, b, details, ORIGIN) is None
    a, b = poses.Correction(0.0, 0.0, 7.0), poses.Correction(0.0, 0.0, 2.0)
    assert poses.agreed(a, b, details, ORIGIN) is None


def test_an_error_as_small_as_the_surveys_own_is_left():
    small = poses.Correction(0.6, -0.7, 0.6)

    assert poses.agreed(small, small, _details(), ORIGIN) is None


def test_corrections_moving_the_details_alike_agree_whatever_their_numbers():
    rng = np.random.default_rng(0)
    cluster = np.array([1010.0, 2015.0]) + rng.normal(0, 1.5, (800, 2))  # 18 m off, a few m wide
    centre = cluster.mean(axis=0)
    a = poses.Correction(3.0, 0.0, -14.0)
    turned_a = poses.moved(centre[None], ORIGIN, a)[0]
    turned_b = poses.moved(centre[None], ORIGIN, poses.Correction(0.0, 0.0, -13.0))[0]
    shift = turned_a - turned_b  # the second turns 1° less, about the panorama, and shifts more
    b = poses.Correction(float(shift[0]), float(shift[1]), -13.0)

    assert poses.agreed(a, b, cluster, ORIGIN) is not None


def test_lidar_outlines_are_found_where_a_building_stands_above_the_street():
    heights = np.full((40, 40), 33.0)  # 0.5 m cells, the street at 33 m
    heights[:, 20:] = 50.0  # a building east of x = 10 m
    distance = poses.outline_distance(heights, x0=0.0, y1=20.0, street=33.0)

    near, off = distance(np.array([[10.0, 10.0], [7.0, 10.0]]))
    assert near < 0.6 and off == pytest.approx(3.0, abs=0.6)


def test_corrections_are_kept_and_read_back(tmp_path):
    path = tmp_path / "poses.json"
    kept = {
        "p1": poses.Verdict(
            poses.Correction(2.2, -3.3, 0.8),
            poses.Correction(2.0, -3.0, 1.0),
            poses.Correction(2.4, -3.6, 0.6),
            900,
            (652000.0, 6861000.0, 12.5),
        ),
        "p2": poses.Verdict(None, poses.Correction(0.1, 0.0, 0.2), None, 40, (1.0, 2.0, 3.0)),
    }

    poses.save(path, kept)

    assert poses.load(path) == kept
    assert poses.corrections(poses.load(path)) == {"p1": poses.Correction(2.2, -3.3, 0.8)}


def test_a_panorama_both_surveys_place_off_alike_is_corrected():
    true = _details()
    seen = poses.moved(true, ORIGIN, poses.Correction(-2.0, 2.5, 0.0))
    survey = lambda p: poses.segment_distance(p, FACADES)  # noqa: E731

    found = poses.judged(seen, ORIGIN, survey, survey)

    assert list(found.correction) == pytest.approx([2.0, -2.5, 0.0], abs=0.1)
    assert found.details == len(seen)


def test_a_panorama_the_surveys_place_apart_is_left():
    seen = poses.moved(_details(), ORIGIN, poses.Correction(-2.0, 2.5, 0.0))
    elsewhere = FACADES + [3.0, 3.0]  # a second survey 4 m off the first, both ways

    found = poses.judged(
        seen,
        ORIGIN,
        lambda p: poses.segment_distance(p, FACADES),
        lambda p: poses.segment_distance(p, elsewhere),
    )

    assert found.correction is None and found.bdtopo is not None and found.lidar is not None


def test_a_panorama_placed_off_is_set_right_against_lidar_outlines():
    heights = np.full((160, 160), 33.0)  # 0.5 m cells, x 960..1040, y 1960..2040; the street 33 m
    heights[:, :70] = heights[:, 90:] = 50.0  # buildings west of x = 995 and east of x = 1005
    heights[:30, 70:90] = 50.0  # and one closing the street, north of y = 2025
    lidar = poses.outline_distance(heights, x0=960.0, y1=2040.0, street=33.0)
    rng = np.random.default_rng(0)
    along = rng.uniform(1975, 2025, 2000)
    true = np.vstack(
        [
            np.column_stack([np.full(1000, 995.0), along[:1000]]),
            np.column_stack([np.full(1000, 1005.0), along[1000:]]),
            np.column_stack([rng.uniform(995, 1005, 600), np.full(600, 2025.0)]),
        ]
    )
    seen = poses.moved(true, ORIGIN, poses.Correction(1.0, -1.5, 0.0))

    found = poses.fitted(seen, ORIGIN, lidar)

    back = poses.moved(seen, ORIGIN, found)
    assert np.median(np.linalg.norm(back - true, axis=1)) < 0.3


def test_along_a_lone_facade_the_shift_it_cannot_tell_is_left():
    rng = np.random.default_rng(0)
    facade = np.array([[[1005.0, 1970.0], [1005.0, 2030.0]]])  # one facade, along north
    true = np.column_stack([np.full(2000, 1005.0), rng.uniform(1975, 2025, 2000)])
    seen = true + [-1.0, 0.0]  # off across it only

    found = poses.fitted(seen, ORIGIN, lambda p: poses.segment_distance(p, facade))

    assert found.east == pytest.approx(1.0, abs=0.1)
    assert abs(found.north) < 0.2 and abs(found.turn) < 0.3  # nothing to tell them: left


def test_a_panorama_turned_and_shifted_far_off_is_found_as_the_eye_finds_it():
    true = _details()
    seen = poses.moved(true, ORIGIN, poses.Correction(-6.0, 5.0, -12.0))  # 12 degrees, 8 m off
    survey = lambda p: poses.segment_distance(p, FACADES)  # noqa: E731

    found = poses.fitted(seen, ORIGIN, survey)
    nudged = poses.fitted(seen, ORIGIN, survey, start=poses.Correction(0.0, 0.0, 0.0))

    back = poses.moved(seen, ORIGIN, found)
    assert np.median(np.linalg.norm(back - true, axis=1)) < 0.1
    pushed = poses.moved(seen, ORIGIN, nudged) if nudged else seen  # from where it stands: lost
    assert np.median(np.linalg.norm(pushed - true, axis=1)) > 1.0


def test_along_a_lone_facade_turned_off_the_turn_and_the_shift_across_are_found():
    rng = np.random.default_rng(0)
    facade = np.array([[[1005.0, 1970.0], [1005.0, 2030.0]]])  # one facade, along north
    true = np.column_stack([np.full(2000, 1005.0), rng.uniform(1975, 2025, 2000)])
    seen = poses.moved(true, ORIGIN, poses.Correction(-3.0, 0.0, 6.0))

    found = poses.fitted(seen, ORIGIN, lambda p: poses.segment_distance(p, facade))

    back = poses.moved(seen, ORIGIN, found)
    assert np.median(np.abs(back[:, 0] - 1005.0)) < 0.1  # laid on the facade, turned straight
    assert abs(found.north) < 0.2  # along it, nothing to tell: left
    assert abs(np.degrees(np.arctan2(*np.ptp(back, axis=0)[::-1])) - 90.0) < 0.5


def test_details_matching_two_places_alike_set_nothing():
    corner = np.array([[[1004.0, 2004.0], [1004.0, 2016.0]], [[1004.0, 2004.0], [1016.0, 2004.0]]])
    twins = np.concatenate([corner, corner + [-10.0, 0.0]])  # the same corner 10 m west
    rng = np.random.default_rng(0)
    k = rng.integers(2, size=2000)
    t = rng.uniform(0, 1, 2000)[:, None]
    seen = corner[k, 0] + t * (corner[k, 1] - corner[k, 0]) + [-5.0, 0.0]  # halfway: either one

    assert poses.fitted(seen, ORIGIN, lambda p: poses.segment_distance(p, twins)) is None


def test_a_panorama_both_surveys_place_far_off_alike_is_corrected():
    true = _details()
    seen = poses.moved(true, ORIGIN, poses.Correction(-6.0, 5.0, -12.0))  # beyond a nudge
    survey = lambda p: poses.segment_distance(p, FACADES)  # noqa: E731

    found = poses.judged(seen, ORIGIN, survey, survey)

    back = poses.moved(seen, ORIGIN, found.correction)
    assert np.median(np.linalg.norm(back - true, axis=1)) < 0.1


def test_details_behind_something_raised_are_not_lidars_to_fit():
    heights = np.full((160, 160), 33.0)  # 0.5 m cells, x 960..1040, y 1960..2040
    heights[78:82, 100:104] = 45.0  # a tree crown 10-12 m east of the panorama, at its height
    behind, clear = np.array([[1015.0, 2000.0]]), np.array([[1000.0, 2015.0]])

    seen = poses.clear_of(
        np.vstack([behind, clear]), ORIGIN, heights, x0=960.0, y1=2040.0, street=33.0
    )

    assert seen.tolist() == [False, True]


def test_details_seen_inside_buildings_by_a_panorama_placed_off_are_still_lidars():
    heights = np.full((160, 160), 33.0)  # 0.5 m cells, x 960..1040, y 1960..2040
    heights[:, :70] = heights[:, 90:] = 50.0  # buildings west of x = 995 and east of x = 1005
    heights[:30, 70:90] = 50.0  # and one closing the street, north of y = 2025
    lidar = poses.outline_distance(heights, x0=960.0, y1=2040.0, street=33.0)
    rng = np.random.default_rng(0)
    true = np.vstack(  # seen: the east facade (2 m inside, as placed) and the street's end
        [
            np.column_stack([np.full(1500, 1005.0), rng.uniform(1975, 2025, 1500)]),
            np.column_stack([rng.uniform(995, 1005, 600), np.full(600, 2025.0)]),
        ]
    )
    seen = poses.moved(true, ORIGIN, poses.Correction(2.0, 0.0, 0.0))

    def sees(points, origin):
        return poses.clear_of(points, origin, heights, x0=960.0, y1=2040.0, street=33.0)

    found = poses.judged(seen, ORIGIN, lidar, lidar, sees)

    assert found.correction is not None and found.correction.east == pytest.approx(-2.0, abs=0.3)


def test_surveys_both_placing_a_panorama_far_off_agree_relative_to_how_far():
    details = _details()
    a, b = poses.Correction(-4.0, 3.0, 0.0), poses.Correction(-4.0, 4.6, 0.0)  # 5 m off, 1.6 apart

    assert poses.agreed(a, b, details, ORIGIN) is not None
    near, apart = poses.Correction(-1.6, 0.0, 0.0), poses.Correction(-1.6, 1.2, 0.0)
    assert poses.agreed(near, apart, details, ORIGIN) is None  # 1.2 m apart for 1.6 m off: no


def test_surveys_disagreeing_on_the_turn_correct_nothing_even_near_the_panorama():
    rng = np.random.default_rng(0)
    near = ORIGIN + rng.normal(0, 3.0, (800, 2))  # details close by: a turn moves them little
    a, b = poses.Correction(-2.0, 0.0, 1.8), poses.Correction(-2.0, 0.0, 4.8)  # turns 3° apart

    assert poses.agreed(a, b, near, ORIGIN) is None
    big = poses.Correction(-4.0, 3.0, -14.6), poses.Correction(-4.0, 3.0, -13.5)  # 1.1° of 14
    assert poses.agreed(*big, near, ORIGIN) is not None


def test_a_panorama_is_judged_from_its_details_both_surveys_and_the_street(tmp_path):
    n = int(2 * poses.HALF_M / poses.CELL_M)  # the patch kept around (1000, 2000), as fetched
    heights = np.full((n, n), 33.0)
    col = 955.0 + (np.arange(n) + 0.5) * poses.CELL_M
    row = 2045.0 - (np.arange(n) + 0.5) * poses.CELL_M
    heights[:, (col < 995) | (col > 1005)] = 50.0  # buildings either side of a street
    heights[np.ix_(row > 2025, (col >= 995) & (col <= 1005))] = 50.0  # and one closing it
    np.save(tmp_path / "mns_1000_2000.npy", heights.astype("<f4"))
    facades = [
        ((995.0, 1975.0), (995.0, 2025.0)),
        ((1005.0, 1975.0), (1005.0, 2025.0)),
        ((995.0, 2025.0), (1005.0, 2025.0)),
    ]
    rng = np.random.default_rng(0)
    plan = np.vstack(
        [
            np.column_stack([np.full(800, 995.0), rng.uniform(1976, 2024, 800)]),
            np.column_stack([np.full(800, 1005.0), rng.uniform(1976, 2024, 800)]),
            np.column_stack([rng.uniform(996, 1004, 400), np.full(400, 2025.0)]),
        ]
    )
    seen = poses.moved(plan, ORIGIN, poses.Correction(2.0, -1.5, 0.0))
    details = np.column_stack([seen, rng.uniform(3, 12, len(seen))])  # heights over the street 0

    found = poses.judge(details, np.array([1000.0, 2000.0, 2.5]), 0.0, facades, tmp_path, None)

    assert list(found.correction) == pytest.approx([-2.0, 1.5, 0.0], abs=0.3)


def test_along_what_the_details_cannot_tell_the_surveys_are_not_compared_nor_followed():
    rng = np.random.default_rng(0)
    along = np.column_stack([np.full(2000, 1005.0), rng.uniform(1975, 2025, 2000)])  # one facade
    a, b = poses.Correction(-2.0, 2.6, 0.0), poses.Correction(-2.0, 5.2, 0.0)  # apart along it

    found = poses.agreed(a, b, along, ORIGIN, free=[np.array([0.0, 1.0])])  # north: not pinned

    assert found is not None and list(found) == pytest.approx([-2.0, 0.0, 0.0])
    assert poses.agreed(a, b, along, ORIGIN) is None  # compared as they stand: 2.6 m apart


def test_a_direction_either_survey_cannot_pin_is_neither_compared_nor_followed():
    rng = np.random.default_rng(0)
    along = np.column_stack([np.full(2000, 1005.0), rng.uniform(1975, 2025, 2000)])
    a, b = poses.Correction(-2.0, 0.0, 0.0), poses.Correction(-2.0, 3.0, 0.0)  # apart along it
    free = [np.array([0.0, 1.0]), np.array([0.0, -1.0])]  # the same direction, from each fit

    assert list(poses.agreed(a, b, along, ORIGIN, free=free)) == pytest.approx([-2.0, 0.0, 0.0])
    across = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]  # nothing pinned by both: a turn alone
    assert poses.agreed(a, b, along, ORIGIN, free=across) is None


def test_something_raised_right_over_the_panorama_hides_nothing():
    heights = np.full((160, 160), 33.0)  # 0.5 m cells, x 960..1040, y 1960..2040
    heights[78:82, 80:82] = 45.0  # an awning or a crown 0-1 m east of the panorama
    ahead = np.array([[1015.0, 2000.0]])

    assert poses.clear_of(ahead, ORIGIN, heights, x0=960.0, y1=2040.0, street=33.0).tolist() == [
        True
    ]


class _Answer:
    def __init__(self, content, kind="image/x-bil;bits=32"):
        self.content, self.headers = content, {"content-type": kind}

    def raise_for_status(self):
        pass


class _Client:
    def __init__(self, answer):
        self.answer = answer

    def get(self, url, timeout):
        return self.answer


def test_lidar_cells_without_data_are_unknown_and_a_wrong_answer_is_refused(tmp_path):
    n = int(2 * poses.HALF_M / poses.CELL_M)
    heights = np.full((n, n), 35.0, "<f4")
    heights[0, :10] = -99999.0  # no data there
    found, _, _ = poses.surface(1000.0, 2000.0, tmp_path, _Client(_Answer(heights.tobytes())))

    assert np.isnan(found[0, :10]).all() and np.isfinite(found[1:]).all()
    with pytest.raises(ValueError):
        poses.surface(
            3000.0, 2000.0, tmp_path, _Client(_Answer(b"<ServiceException/>", "text/xml"))
        )
    assert not (tmp_path / "mns_3000_2000.npy").exists()
