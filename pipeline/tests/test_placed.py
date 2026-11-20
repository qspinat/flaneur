import numpy as np
import pytest
from test_bridging import AT, A, B, _seen
from test_match import facade_seen_from_the_origin

from flaneur import match, placed, registration

HERE = match.Panorama("here", 0.0, 0.0, 1.6, 30.0, None, roll=1.0, pitch=-2.0)  # at the camera
WITH_HERE = {**AT, "here": HERE}


def _in_the_photo():
    """The facade's points the photo (800 × 600) shows."""
    world, pixels = facade_seen_from_the_origin(120)
    inside = ((pixels >= 0) & (pixels <= [800, 600])).all(axis=1)
    return world[inside], pixels[inside]


def _off_in_view(r: registration.Registration, sphere, world, pixels) -> float:
    """How far (view pixels, worst) the registration puts the photo's points from where the
    panorama sees them."""
    h = registration.in_view(r, r.view, sphere.axes)
    unit = np.column_stack([pixels / [800, 600], np.ones(len(pixels))]) @ h.T
    at = unit[:, :2] / unit[:, 2:]  # in the view, its side 1
    seen = world - [sphere.x, sphere.y, sphere.z]
    own = seen / np.linalg.norm(seen, axis=1, keepdims=True) @ sphere.axes.T
    truth = registration.view_pixels(own, r.view, sphere.axes, 1)
    return float(np.abs(at - truth).max())


def test_a_placement_is_registered_exactly_in_a_panorama_that_fits():
    world, pixels = _in_the_photo()
    points, observed = _seen(world, pixels, {"here": list(range(len(world)))}, WITH_HERE)

    r = placed.registered(points, observed, [HERE])

    assert r.panorama == "here" and _off_in_view(r, HERE, world, pixels) < 1e-3


def test_a_panorama_without_enough_pairs_gives_way_to_the_next():
    world, pixels = _in_the_photo()
    points, observed = _seen(
        world, pixels, {"b": list(range(10)), "here": list(range(len(world)))}, WITH_HERE
    )

    assert placed.registered(points, observed, [B, HERE]).panorama == "here"


def test_no_panorama_that_fits_registers_nothing():
    world, pixels = _in_the_photo()
    points, observed = _seen(world, pixels, {"a": list(range(10)), "b": list(range(10))})

    assert placed.registered(points, observed, [A, B]) is None


def test_the_panorama_where_most_pairs_agree_wins_over_the_first():
    world, pixels = _in_the_photo()
    points, observed = _seen(
        world, pixels, {"a": list(range(20)), "here": list(range(len(world)))}, WITH_HERE
    )

    assert placed.registered(points, observed, [A, HERE]).panorama == "here"


REG = {"panorama": "p", "m": [1.0] * 9, "view": {"heading": 0.0, "pitch": 5.0, "fov": 75.0}}


def _matched(verdict=None, area_checked=True, marks=None, unsure=False) -> placed.Matched:
    return placed.Matched({"x": 0.0}, verdict, "img", area_checked, marks, unsure)


STANDING = {"position": [1.0, 2.0, 30.0]}


def _build(matched, answers=None, far=frozenset(), registered=None):
    """The list from `matched` placements of match, the page's `answers`, those answered
    `far`; each registered in panorama "p", standing (the photos asked to register noted in
    `registered`)."""

    def register(placements: dict) -> dict:
        if registered is not None:
            registered.update(placements)
        return {k: placed.Registered(REG, STANDING) for k in placements}

    return placed.build(answers or {}, set(far), matched, register)


def test_a_yes_is_checked_a_near_is_near_an_unseen_one_is_automatic():
    found = _build({"y": _matched("yes"), "n": _matched("near"), "u": _matched()})

    assert {k: p.status for k, p in found.items()} == {
        "y": placed.CHECKED,
        "n": placed.NEAR,
        "u": placed.AUTOMATIC,
    }


def test_one_on_panoramas_nothing_checks_keeps_its_matrix_alone():
    unsure = _build({"x": _matched(unsure=True)})["x"]
    seen = _build({"x": _matched("yes", unsure=True)})["x"]  # the note is for those not seen

    assert (unsure.registration, unsure.standing) == (REG, None)
    assert seen.standing == STANDING


def test_a_no_leaves_the_list():
    assert _build({"x": _matched("no")}) == {}


def test_a_verdict_not_known_is_an_error_not_a_photo_dropped():
    with pytest.raises(KeyError):
        _build({"x": _matched("unsure")})


def test_an_answer_on_the_placing_page_wins_and_is_checked():
    answer = {"registration": {**REG, "panorama": "q"}, "inputs": "img", "flipped": True}

    found = _build({"x": _matched()}, answers={"x": answer})["x"]

    assert (found.status, found.source, found.registration["panorama"], found.flipped) == (
        placed.CHECKED,
        "page",
        "q",
        True,
    )


def test_an_answer_far_takes_the_match_placement_out():
    assert _build({"x": _matched()}, far={"x"}) == {}


def test_only_the_placements_kept_are_registered():
    registered = {}
    answer = {"registration": REG, "inputs": "img", "flipped": False}

    _build(
        {"a": _matched(), "f": _matched(), "n": _matched("no"), "k": _matched("near")},
        answers={"a": answer},
        far={"f"},
        registered=registered,
    )

    assert set(registered) == {"k"}


def test_an_automatic_one_says_whether_its_area_passes_and_its_marks_are_checked():
    passes = _build({"x": _matched(marks="checked")})["x"].for_scene()
    not_yet = _build({"x": _matched(area_checked=False, marks="proposed")})["x"].for_scene()
    seen = _build({"x": _matched("near", area_checked=False, marks="proposed")})["x"]

    assert (passes["unchecked_area"], passes["proposed_marks"]) == (False, False)
    assert (not_yet["unchecked_area"], not_yet["proposed_marks"]) == (True, True)
    assert (seen.unchecked_area, seen.proposed_marks) == (False, False)  # notes: for unseen


def test_an_automatic_one_not_seen_says_it_rests_on_panoramas_nothing_checks():
    def note(unsure, verdict=None):
        m = placed.Matched({"x": 0.0}, verdict, "img", True, None, unsure_sources=unsure)
        return _build({"x": m})["x"].unsure_sources

    assert note(True) and not note(False)
    assert not note(True, "near")  # seen by the user: the note is for those not seen


def _points_seen_by(*seen) -> match.Points:
    n = len(seen)
    return match.Points(np.zeros((n, 2)), np.zeros((n, 3)), np.array(seen, dtype=object))


def test_a_placement_is_unsure_under_half_its_sightings_by_sure_panoramas():
    known = {"sure", "gps"}
    quarter = _points_seen_by({"sure", "gps"}, {"gps"}, {"gps", "photo:x"}, {"photo:y"})
    half = _points_seen_by({"sure"}, {"gps"}, {"photo:x"})

    assert placed.unsure(quarter, {"sure"}, known)  # 1 of 4 sightings: old photos left out
    assert not placed.unsure(half, {"sure"}, known)


def test_a_placement_no_panorama_saw_is_not_said_unsure():
    assert not placed.unsure(_points_seen_by({"photo:y"}), {"sure"}, {"sure"})


def test_points_bunched_on_one_part_of_the_photo_still_register_it_whole():
    world, pixels = _in_the_photo()
    corner = (pixels[:, 0] < 360) & (pixels[:, 1] < 300)  # a fifth of the photo, as on facades
    world, pixels = world[corner], pixels[corner]
    points, observed = _seen(world, pixels, {"here": list(range(len(world)))}, WITH_HERE)

    r = placed.registered(points, observed, [HERE])

    assert r is not None and _off_in_view(r, HERE, world, pixels) < 1e-3
