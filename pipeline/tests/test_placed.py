import numpy as np
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


def _build(**given):
    """The list from a few photos: `standing` placements of match (matcher LoMa-G), the
    verdicts given, the page's answers; images all 'img'; area 'cite' checked for LoMa-G."""
    standing = given.get("standing", {})
    return placed.build(
        answers=given.get("answers", {}),
        far=given.get("far", set()),
        standing={k: {"matcher": "lomag"} for k in standing},
        verdicts=given.get("verdicts", {}),
        registrations={k: REG for k in standing},
        images={k: "img" for k in [*standing, *given.get("answers", {})]},
        checked_areas=given.get("checked_areas", {"cite": ["lomag"]}),
        area_of={k: "cite" for k in standing},
        marks=given.get("marks", {}),
    )


def test_a_yes_is_checked_a_near_is_near_an_unseen_one_is_automatic():
    found = _build(standing=["y", "n", "u"], verdicts={"y": "yes", "n": "near"})

    assert {k: p.status for k, p in found.items()} == {
        "y": placed.CHECKED,
        "n": placed.NEAR,
        "u": placed.AUTOMATIC,
    }


def test_a_no_leaves_the_list():
    assert _build(standing=["x"], verdicts={"x": "no"}) == {}


def test_an_answer_on_the_placing_page_wins_and_is_checked():
    answer = {"registration": {**REG, "panorama": "q"}, "inputs": "img", "flipped": True}

    found = _build(standing=["x"], answers={"x": answer})["x"]

    assert (found.status, found.source, found.registration["panorama"], found.flipped) == (
        placed.CHECKED,
        "page",
        "q",
        True,
    )


def test_an_answer_far_takes_the_match_placement_out():
    assert _build(standing=["x"], far={"x"}) == {}


def test_an_answer_on_an_older_image_gives_way_to_match():
    answer = {"registration": {**REG, "panorama": "q"}, "inputs": "old", "flipped": False}

    found = _build(standing=["x"], answers={"x": answer})["x"]

    assert (found.source, found.registration["panorama"]) == ("match", "p")


def test_an_automatic_one_says_whether_its_area_passes_and_its_marks_are_checked():
    passes = _build(standing=["x"], marks={"x": "checked"})["x"].for_scene()
    not_yet = _build(standing=["x"], checked_areas={"cite": []}, marks={"x": "proposed"})["x"]

    assert (passes["unchecked_area"], passes["proposed_marks"]) == (False, False)
    assert (not_yet.for_scene()["unchecked_area"], not_yet.for_scene()["proposed_marks"]) == (
        True,
        True,
    )
