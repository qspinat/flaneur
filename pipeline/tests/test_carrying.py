"""A placed photo carried to neighbouring panoramas, and to old photos sharing its details."""

import cv2
import numpy as np
import pytest

from flaneur import carrying, match, registration
from flaneur.registration import ViewOf

SIDE = 640
HOME = match.camera_axes(198.0, -8.4, 6.9)
THERE = match.camera_axes(30.0, 1.0, -2.0)
H = np.array([[0.6, 0.05, 0.2], [-0.02, 0.55, 0.25], [0.0001, 0.0002, 1.0]])
SPREAD = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9], [0.5, 0.4], [0.3, 0.7]])
VIEW = ViewOf(heading=150.0)


@pytest.fixture
def placed():
    seen = cv2.perspectiveTransform(SPREAD[None], H)[0] * SIDE
    return registration.fit(SPREAD * [800, 600], seen, (600, 800), SIDE, "home", VIEW, HOME)[0]


def _onto(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points[None].astype(float), h)[0]


def _today_pairs(shift_px: tuple[float, float], n: int = 80, wrong: int = 0):
    rng = np.random.default_rng(5)
    here = rng.uniform(150, 450, (n, 2))  # details under the photo, in the home view
    there = here + shift_px
    there[:wrong] = rng.uniform(0, SIDE, (wrong, 2))
    return here, there


def test_a_neighbour_seeing_the_same_patch_carries_it(placed):
    here, there = _today_pairs((-40.0, 12.0))
    onto = ViewOf(heading=150.0)

    carried = carrying.carried_today(placed, VIEW, HOME, here, there, SIDE, "next", onto, THERE)

    assert carried.panorama == "next"
    expected = _onto(SPREAD, H) + np.array([-40.0, 12.0]) / SIDE
    assert np.allclose(
        _onto(SPREAD, registration.in_view(carried, onto, THERE)), expected, atol=1e-3
    )


def test_carrying_stops_where_too_few_points_agree(placed):
    here, there = _today_pairs((-40.0, 12.0), n=60, wrong=30)  # 30 agree, under 40

    assert (
        carrying.carried_today(placed, VIEW, HOME, here, there, SIDE, "next", VIEW, THERE) is None
    )


def test_a_carried_frame_that_leaves_the_view_is_dropped(placed):
    here, there = _today_pairs((-600.0, 0.0))  # the photo carried out of the view

    assert (
        carrying.carried_today(placed, VIEW, HOME, here, there, SIDE, "next", VIEW, THERE) is None
    )


@pytest.mark.parametrize(("metres", "strength"), [(2.0, 1.0), (5.0, 1.0), (12.5, 0.5), (20.0, 0.0)])
def test_strength_fades_from_five_to_twenty_metres(metres, strength):
    assert carrying.strength(metres) == pytest.approx(strength)


def test_an_old_photo_matching_a_placed_one_is_proposed_on_its_panorama(placed):
    rng = np.random.default_rng(2)
    mine = rng.uniform(0.05, 0.95, (150, 2))  # in the other old photo, [0, 1]²
    between = np.array([[0.9, 0.02, 0.05], [0.01, 0.95, 0.03], [0.0, 0.0, 1.0]])  # to the placed

    proposed = carrying.carried_old(
        placed, HOME, _onto(mine, between) * [800, 600], (600, 800), mine * [700, 500], (500, 700)
    )

    assert proposed.panorama == "home"
    expected = registration.in_view(placed, VIEW, HOME) @ between
    assert np.allclose(
        registration.in_view(proposed, VIEW, HOME), expected / expected[2, 2], atol=1e-3
    )


def test_too_few_shared_details_propose_nothing(placed):
    mine = np.random.default_rng(2).uniform(0.05, 0.95, (40, 2))

    assert (
        carrying.carried_old(
            placed, HOME, mine * [800, 600], (600, 800), mine * [700, 500], (500, 700)
        )
        is None
    )


def _features(points: np.ndarray, size) -> match.Features:
    return match.Features(points.astype(np.float32), np.zeros((len(points), 128), np.float32), size)


def _shared(placed_row_photo, other):
    rng = np.random.default_rng(2)
    mine = rng.uniform(0.05, 0.95, (150, 2))
    between = np.array([[0.9, 0.02, 0.05], [0.01, 0.95, 0.03], [0.0, 0.0, 1.0]])
    features = {
        placed_row_photo: _features(_onto(mine, between) * [800, 600], (600, 800)),
        other: _features(mine * [700, 500], (500, 700)),
    }
    pairs = {(placed_row_photo, other): np.column_stack([np.arange(150)] * 2)}
    return features, pairs


def _row(r, answer="fits"):
    return {
        "answer": answer,
        "registration": r and registration.to_json(r),
        "flipped": False,
        "inputs": "ia",
        "lon": 2.3,
        "lat": 48.8,
        "heading": 0.0,
    }


def test_old_photos_sharing_a_placed_ones_details_are_proposed_from_it(placed):
    features, pairs = _shared("a", "b")

    found = carrying.proposals(
        {"a": _row(placed)},
        {"a": {"b"}},
        lambda a, b: pairs.get((a, b)),
        features.get,
        {"home": HOME}.get,
        {"a": "ia", "b": "ib"},
    )

    assert set(found) == {"b"} and found["b"]["carried_from"] == "a"
    assert found["b"]["registration"].panorama == "home"


@pytest.mark.parametrize(
    ("held", "why"),
    [
        (lambda p: {"a": _row(p), "b": _row(None, "far")}, "b answered"),
        (lambda p: {"a": _row(None, "far")}, "a too far now"),
    ],
    ids=["the other answered", "the placed one answered too far"],
)
def test_no_proposal_from_or_to_a_photo_answered_otherwise(placed, held, why):
    features, pairs = _shared("a", "b")

    found = carrying.proposals(
        held(placed),
        {"a": {"b"}, "b": {"a"}},
        lambda a, b: pairs.get((a, b)),
        features.get,
        {"home": HOME}.get,
        {"a": "ia", "b": "ib"},
    )

    assert found == {}, why


def test_a_copy_is_tried_outward_and_not_past_a_panorama_that_does_not_see_it():
    near = {"home": [("p1", 4.0)], "p1": [("home", 4.0), ("p2", 6.0)], "p2": [("p1", 6.0)]}
    tried = []

    def carry(panorama):
        tried.append(panorama)
        return None  # p1 does not see it

    assert (
        carrying.outward("home", lambda p: near[p], {"home": 0.0, "p1": 4.0, "p2": 10.0}.get, carry)
        == []
    )
    assert tried == ["p1"]  # p2 lies only beyond p1


def test_a_photo_placed_flipped_proposes_its_neighbours_the_right_way_round(placed):
    rng = np.random.default_rng(2)
    mine = rng.uniform(0.05, 0.95, (150, 2))
    between = np.array([[0.9, 0.02, 0.05], [0.01, 0.95, 0.03], [0.0, 0.0, 1.0]])
    shown = _onto(mine, between)  # where they lie in the placed photo as shown, flipped
    kept = np.column_stack([1 - shown[:, 0], shown[:, 1]])  # in its image as kept, unflipped
    features = {
        "a": _features(kept * [800, 600], (600, 800)),
        "b": _features(mine * [700, 500], (500, 700)),
    }
    row = {**_row(placed), "flipped": True, "inputs": "ia"}

    found = carrying.proposals(
        {"a": row},
        {"a": {"b"}},
        lambda a, b: np.column_stack([np.arange(150)] * 2),
        features.get,
        {"home": HOME}.get,
        {"a": "ia", "b": "ib"},
    )

    expected = registration.in_view(placed, VIEW, HOME) @ between
    assert np.allclose(
        registration.in_view(found["b"]["registration"], VIEW, HOME),
        expected / expected[2, 2],
        atol=1e-3,
    )


def test_a_photo_placed_on_an_image_it_no_longer_has_proposes_nothing(placed):
    features, pairs = _shared("a", "b")

    found = carrying.proposals(
        {"a": {**_row(placed), "inputs": "before"}},
        {"a": {"b"}},
        lambda a, b: pairs.get((a, b)),
        features.get,
        {"home": HOME}.get,
        {"a": "now", "b": "x"},
    )

    assert found == {}


def test_a_proposal_says_where_its_source_was_placed(placed):
    features, pairs = _shared("a", "b")
    row = {**_row(placed), "lon": 2.3467, "lat": 48.8521, "heading": 40.0}

    found = carrying.proposals(
        {"a": row},
        {"a": {"b"}},
        lambda a, b: pairs.get((a, b)),
        features.get,
        {"home": HOME}.get,
        {"a": "ia", "b": "ib"},
    )

    assert (found["b"]["lon"], found["b"]["lat"], found["b"]["heading"]) == (2.3467, 48.8521, 40.0)


def test_old_photos_near_each_other_are_compared_once_and_never_copies_of_one_negative():
    where = {
        "a": (0.0, 0.0),
        "a2": (1.0, 0.0),
        "b": (50.0, 0.0),
        "c": (0.0, 59.0),
        "far": (200.0, 0.0),
    }
    copies = {"a": {"a", "a2"}, "a2": {"a", "a2"}}
    kept = {("a", "c")}

    found = carrying.to_compare(where, 60.0, copies, lambda a, b: (a, b) in kept)

    assert found == [("a", "b"), ("a2", "b"), ("a2", "c")]  # b and c: 77 m apart


def test_no_photos_have_no_pairs_to_compare():
    assert carrying.to_compare({}, 60.0, {}, lambda a, b: False) == []
