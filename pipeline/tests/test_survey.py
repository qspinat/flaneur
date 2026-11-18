import math

import numpy as np
import pytest

from flaneur import survey


def _rot(deg: float, v) -> np.ndarray:
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]) @ np.asarray(
        v, float
    )


class _Street:
    """Panoramas whose true place and heading error are known: `truth` id → (x, y, turn): the
    turn (degrees, counter-clockwise) that sets each right, `recorded` where each stands now.
    A pair says what depth.set_right says: how far `other` turns to agree with `home` as both
    stand, and where `other` stands seen from `home` (its recorded distance kept)."""

    def __init__(self, truth: dict, recorded: dict):
        self.truth, self.recorded = truth, recorded

    def seen(self, home: str, other: str) -> survey.Seen:
        (hx, hy, hc), (ox, oy, oc) = self.truth[home], self.truth[other]
        toward = np.array([ox - hx, oy - hy])
        toward = _rot(-hc, toward / np.linalg.norm(toward))  # as `home` sees it, turned wrong
        at = self.recorded[home]
        far = math.dist(self.recorded[home], self.recorded[other])
        return survey.Seen(oc - hc, tuple(np.asarray(at) + far * toward))


def _set_right(street: _Street, survey_ids, neighbours, in_street=lambda x, y: True):
    refs = {i: street.recorded[i] for i in survey_ids}
    others = {i: p for i, p in street.recorded.items() if i not in refs}
    return survey.set_right(refs, others, lambda i: neighbours[i], street.seen, in_street)


def test_a_volunteer_seen_by_two_survey_panoramas_gets_their_heading_and_its_place():
    truth = {"s1": (0, 0, 0), "s2": (10, 0, 0), "v": (5, 8, 12.0)}
    recorded = {"s1": (0, 0), "s2": (10, 0), "v": (6, 9)}  # 1.4 m off
    found = _set_right(_Street(truth, recorded), ["s1", "s2"], {"v": ["s1", "s2"]})

    v = found["v"]
    assert v.turn == pytest.approx(12.0) and (v.hops, v.references) == (1, 2)
    assert v.shift == pytest.approx((-1.0, -1.0), abs=1e-6)


def test_survey_panoramas_that_disagree_set_nothing():
    street = _Street(
        {"s1": (0, 0, 0), "s2": (10, 0, 0), "v": (5, 8, 12.0)},
        {"s1": (0, 0), "s2": (10, 0), "v": (5, 8)},
    )
    wrong = street.seen

    def seen(home, other):
        s = wrong(home, other)
        return s._replace(turn=s.turn + 5) if home == "s2" else s  # 5 degrees apart

    found = survey.set_right(
        {"s1": (0, 0), "s2": (10, 0)},
        {"v": (5, 8)},
        lambda i: ["s1", "s2"],
        seen,
        lambda x, y: True,
    )

    assert "v" not in found


def test_one_survey_neighbour_sets_the_heading_but_passes_nothing_on():
    truth = {"s": (0, 0, 0), "v": (6, 0, 9.0), "w": (12, 0, 9.0)}
    street = _Street(truth, {k: (x, y) for k, (x, y, _) in truth.items()})

    found = _set_right(street, ["s"], {"v": ["s"], "w": ["v"]})

    assert found["v"].turn == pytest.approx(9.0) and found["v"].references == 1
    assert found["v"].shift is None  # one bearing: no crossing
    assert "w" not in found


def test_a_firm_correction_is_passed_on_up_to_the_hops_measured():
    # a ladder: two survey panoramas, then volunteers two by two, each seen by the two before
    truth = {"a0": (0, 0, 0.0), "b0": (0, 6, 0.0)}
    for k in range(1, survey.MOST_HOPS + 2):
        truth[f"a{k}"], truth[f"b{k}"] = (8 * k, 0, 20.0), (8 * k, 6, 20.0)
    street = _Street(truth, {i: (x, y) for i, (x, y, _) in truth.items()})
    neighbours = {
        f"{s}{k}": [f"a{k - 1}", f"b{k - 1}"] for k in range(1, survey.MOST_HOPS + 2) for s in "ab"
    }

    found = _set_right(street, ["a0", "b0"], neighbours)

    assert {i: f.hops for i, f in found.items() if i.startswith("a")} == {
        f"a{k}": k for k in range(1, survey.MOST_HOPS + 1)
    }  # the one beyond: none
    assert all(f.turn == pytest.approx(20.0) for f in found.values())


def test_a_big_move_is_kept_only_where_the_new_place_is_in_the_street():
    truth = {"s1": (0, 0, 0), "s2": (10, 0, 0), "v": (5, 8, 6.0), "w": (15, 8, 6.0)}
    recorded = {"s1": (0, 0), "s2": (10, 0), "v": (5, 16), "w": (15, 16)}  # 8 m off each
    street = _Street(truth, recorded)

    found = _set_right(
        street, ["s1", "s2"], {"v": ["s1", "s2"], "w": ["s1", "s2"]}, in_street=lambda x, y: x < 15
    )

    assert found["v"].shift == pytest.approx((0.0, -8.0), abs=1e-6)
    assert found["w"].shift is None and found["w"].turn == pytest.approx(6.0)


def test_only_turns_beyond_the_survey_s_tolerance_are_applied_the_compass_the_other_way():
    found = {
        "turned": survey.SetRight(10.0, (1.0, -2.0), 1, 2),
        "right": survey.SetRight(0.4, (0.3, 0.1), 1, 3),
        "kept": survey.SetRight(-6.0, None, 2, 2),
    }

    assert survey.fixes(found) == {"turned": (1.0, -2.0, -10.0), "kept": (0.0, 0.0, 6.0)}
