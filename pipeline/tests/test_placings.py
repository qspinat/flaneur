"""The user's answers on the placing page, kept with the other overrides."""

import pytest

from flaneur import placings
from flaneur.registration import Registration, Standing, ViewOf, standing_to_json

R = Registration("p1", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), ViewOf(heading=40.0))
SPOT = {"lon": 2.346, "lat": 48.852, "heading": 40.0}


def test_the_latest_answer_holds_and_earlier_rows_stay_in_the_file(tmp_path):
    path = tmp_path / "placings.jsonl"
    placings.record(path, "pm:1", "far", 9.0, **SPOT)
    placings.record(path, "pm:1", "fits", 12.5, **SPOT, registration=R)

    row = placings.holding(path)["pm:1"]

    assert row["answer"] == "fits" and placings.registration_of(row) == R
    assert len(path.read_text().splitlines()) == 2


def test_a_skip_leaves_the_answer_that_held(tmp_path):
    path = tmp_path / "placings.jsonl"
    placings.record(path, "pm:1", "fits", 12.5, **SPOT, registration=R)
    placings.record(path, "pm:1", "skip", 2.0)
    placings.record(path, "pm:2", "skip", 3.0)

    given = placings.state(path)
    assert given.held["pm:1"]["answer"] == "fits" and "pm:2" not in given.held
    assert given.skipped == {"pm:2"} and given.answers == ["pm:1"]


@pytest.mark.parametrize("answer", ["fits", "points"])
def test_a_photo_placed_in_its_panorama_needs_its_registration(tmp_path, answer):
    with pytest.raises(ValueError, match="enregistrement"):
        placings.record(tmp_path / "p.jsonl", "pm:1", answer, 5.0, **SPOT)


def test_too_far_keeps_the_place_set_on_the_map_and_no_registration(tmp_path):
    row = placings.record(tmp_path / "p.jsonl", "pm:1", "far", 5.0, **SPOT)

    assert row["registration"] is None and row["lon"] == SPOT["lon"]


def test_a_photo_answered_on_needs_its_place_and_a_skip_none(tmp_path):
    with pytest.raises(ValueError, match="place"):
        placings.record(tmp_path / "p.jsonl", "pm:1", "far", 5.0)
    with pytest.raises(ValueError, match="place"):
        placings.record(tmp_path / "p.jsonl", "pm:1", "skip", 5.0, **SPOT)


def test_an_answer_outside_the_four_is_refused(tmp_path):
    with pytest.raises(ValueError, match="réponse"):
        placings.record(tmp_path / "p.jsonl", "pm:1", "maybe", 5.0, **SPOT)


@pytest.mark.parametrize("seconds", [-1, True, "3"])
def test_seconds_are_a_positive_number(tmp_path, seconds):
    with pytest.raises(ValueError, match="secondes"):
        placings.record(tmp_path / "p.jsonl", "pm:1", "skip", seconds)


def test_accepted_photos_carried_from_a_photo_stay_when_it_is_answered_again(tmp_path):
    path = tmp_path / "placings.jsonl"
    placings.record(path, "pm:1", "fits", 5.0, **SPOT, registration=R)
    placings.record(path, "pm:2", "fits", 5.0, **SPOT, registration=R, carried_from="pm:1")
    placings.record(path, "pm:1", "far", 5.0, **SPOT)  # the photo it came from, answered again

    assert placings.carried(placings.holding(path)) == {"pm:1": ["pm:2"]}


@pytest.mark.parametrize(
    "spot",
    [{"lon": 2.3}, {"heading": 40.0}, SPOT],
    ids=["half a place", "a direction", "a place"],
)
def test_a_skip_carries_no_place_at_all(tmp_path, spot):
    with pytest.raises(ValueError, match="place"):
        placings.record(tmp_path / "p.jsonl", "pm:1", "skip", 5.0, **spot)


@pytest.mark.parametrize(
    "spot",
    [
        {**SPOT, "lon": "2.3"},
        {**SPOT, "lat": float("nan")},
        {**SPOT, "heading": 720.0},
        {**SPOT, "heading": None},
    ],
    ids=["a text", "not a number", "past a turn", "no direction"],
)
def test_a_photo_placed_in_3d_needs_a_place_and_direction_that_are_numbers(tmp_path, spot):
    with pytest.raises(ValueError, match="place|direction"):
        placings.record(tmp_path / "p.jsonl", "pm:1", "fits", 5.0, **spot, registration=R)


def test_flipped_is_yes_or_no(tmp_path):
    with pytest.raises(ValueError, match="retournée"):
        placings.record(tmp_path / "p.jsonl", "pm:1", "far", 5.0, **SPOT, flipped="false")


@pytest.mark.parametrize("seconds", [float("nan"), float("inf")])
def test_seconds_that_are_not_finite_are_refused(tmp_path, seconds):
    with pytest.raises(ValueError, match="secondes"):
        placings.record(tmp_path / "p.jsonl", "pm:1", "skip", seconds)


@pytest.mark.parametrize(
    ("answer", "carried_from"),
    [("points", "pm:9"), ("far", "pm:9"), ("fits", "pm:1")],
    ids=["clicked by hand", "too far", "itself"],
)
def test_only_an_accepted_proposal_is_carried_from_another_photo(tmp_path, answer, carried_from):
    r = R if answer != "far" else None
    with pytest.raises(ValueError, match="portée"):
        placings.record(
            tmp_path / "p.jsonl",
            "pm:1",
            answer,
            5.0,
            **SPOT,
            registration=r,
            carried_from=carried_from,
        )


# --- the pairs clicked, and the camera standing where the photographer stood

PAIRS = {
    "photo_px": [
        [100.0, 80.0],
        [500.0, 90.0],
        [300.0, 400.0],
        [120.0, 380.0],
        [400.0, 300.0],
        [650.0, 200.0],
        [700.0, 500.0],
    ],
    "own": [
        [0.1, 0.0, 1.0],
        [0.4, 0.0, 1.0],
        [0.25, 0.3, 1.0],
        [0.1, 0.3, 1.0],
        [0.3, 0.2, 1.0],
        [0.5, 0.1, 1.0],
        [0.55, 0.35, 1.0],
    ],
    "view": {"heading": 40.0, "pitch": 5.0, "fov": 75.0},
    "side": 1280,
}
CAMERA = Standing((652_000.0, 6_861_000.0, -1.0), (0.1, 0.2, 0.3), 700.0, (0.0, -0.05), (600, 800))
STANDING = {
    "camera": standing_to_json(CAMERA),
    "metres": [12.0, 30.5, None, 8.2, 15.0, 20.0, 9.5],  # six with a depth
    "source": ["wall", "drawn", None, "ground", "wall", "click", "wall"],
    "neighbours": ["p2", "p3"],
}


def test_pairs_are_kept_with_an_answer_by_points(tmp_path):
    path = tmp_path / "placings.jsonl"
    placings.record(path, "pm:1", "points", 30.0, **SPOT, registration=R, pairs=PAIRS)

    row = placings.holding(path)["pm:1"]

    assert row["pairs"] == PAIRS and row["fit"] == "flat" and placings.standing_of(row) is None


@pytest.mark.parametrize("answer", ["fits", "far"])
def test_pairs_are_refused_on_an_answer_that_is_not_by_points(tmp_path, answer):
    registration = R if answer == "fits" else None
    with pytest.raises(ValueError, match="paires"):
        placings.record(
            tmp_path / "p.jsonl",
            "pm:1",
            answer,
            3.0,
            **SPOT,
            registration=registration,
            pairs=PAIRS,
        )


@pytest.mark.parametrize(
    "pairs",
    [
        {**PAIRS, "own": PAIRS["own"][:3]},  # not as many on each side
        {**PAIRS, "photo_px": [[1.0, 2.0]], "own": [[0.0, 0.0, 1.0]]},  # one pair
        {**PAIRS, "photo_px": [[1.0, "a"]] * 4},  # not numbers
        {k: v for k, v in PAIRS.items() if k != "view"},  # where they were clicked
        {**PAIRS, "view": {"heading": 40.0, "pitch": 5.0, "fov": 0.0}},
        {**PAIRS, "side": True},
        ["not", "a", "dict"],
    ],
    ids=["uneven", "one pair", "not numbers", "no view", "no field of view", "no size", "a list"],
)
def test_pairs_written_wrong_are_refused(tmp_path, pairs):
    with pytest.raises(ValueError, match="paires"):
        placings.record(
            tmp_path / "p.jsonl", "pm:1", "points", 3.0, **SPOT, registration=R, pairs=pairs
        )


def test_a_parallax_answer_keeps_its_camera_and_its_flat_registration(tmp_path):
    path = tmp_path / "placings.jsonl"
    placings.record(
        path,
        "pm:1",
        "points",
        30.0,
        **SPOT,
        registration=R,
        pairs=PAIRS,
        standing=STANDING,
    )

    row = placings.holding(path)["pm:1"]

    assert placings.registration_of(row) == R and row["fit"] == "parallax"
    assert placings.standing_of(row) == CAMERA and row["standing"]["metres"][2] is None


@pytest.mark.parametrize(
    "standing",
    [
        {"metres": STANDING["metres"], "source": STANDING["source"]},
        {**STANDING, "camera": {"focal": 1}},
    ],
    ids=["no camera", "a camera half written"],
)
def test_a_parallax_camera_written_wrong_is_refused(tmp_path, standing):
    with pytest.raises(ValueError):
        placings.record(
            tmp_path / "p.jsonl",
            "pm:1",
            "points",
            3.0,
            **SPOT,
            registration=R,
            pairs=PAIRS,
            standing=standing,
        )


def test_a_camera_whose_depths_do_not_match_the_pairs_is_refused(tmp_path):
    with pytest.raises(ValueError, match="paire"):
        placings.record(
            tmp_path / "p.jsonl",
            "pm:1",
            "points",
            3.0,
            **SPOT,
            registration=R,
            pairs=PAIRS,
            standing={**STANDING, "metres": [12.0]},
        )


def test_answers_written_before_pairs_still_load(tmp_path):
    path = tmp_path / "placings.jsonl"
    path.write_text(
        '{"photo_id": "pm:1", "answer": "points", "seconds": 5.0, "lon": 2.3, "lat": 48.8, '
        '"heading": 10.0, "registration": {"panorama": "p1", "m": [1, 0, 0, 0, 1, 0, 0, 0, 1], '
        '"view": {"heading": 40.0}}, "flipped": false, "carried_from": null, "inputs": "x"}\n'
    )

    row = placings.holding(path)["pm:1"]

    assert placings.registration_of(row) == R and placings.standing_of(row) is None


@pytest.mark.parametrize(
    "standing",
    [
        {
            **STANDING,
            "metres": [12.0, None, None, None, 3.0, 20.0, None],  # three
            "source": ["wall", None, None, None, "wall", "wall", None],
        },
        {**STANDING, "source": ["wall", "drawn", None, None, "wall", "click", "wall"]},
        {**STANDING, "source": ["wall", "satellite", None, "ground", "wall", "click", "wall"]},
        {**STANDING, "neighbours": "p2"},
        {**STANDING, "camera": {**STANDING["camera"], "size": [60, 80]}},  # clicks off its photo
    ],
    ids=[
        "too few depths",
        "a depth with no source",
        "a source not known",
        "neighbours not a list",
        "photo too small",
    ],
)
def test_a_parallax_camera_that_does_not_go_with_its_pairs_is_refused(tmp_path, standing):
    with pytest.raises(ValueError):
        placings.record(
            tmp_path / "p.jsonl",
            "pm:1",
            "points",
            3.0,
            **SPOT,
            registration=R,
            pairs=PAIRS,
            standing=standing,
        )


@pytest.mark.parametrize(
    "pairs",
    [
        {**PAIRS, "own": [[0.0, 0.0, 0.0]] + PAIRS["own"][1:]},
        {**PAIRS, "view": {"heading": 40.0, "pitch": 120.0, "fov": 75.0}},
    ],
    ids=["no direction", "pitch past the vertical"],
)
def test_pairs_with_impossible_directions_are_refused(tmp_path, pairs):
    with pytest.raises(ValueError):
        placings.record(
            tmp_path / "p.jsonl", "pm:1", "points", 3.0, **SPOT, registration=R, pairs=pairs
        )


def test_only_an_answer_by_points_says_which_fit_it_kept(tmp_path):
    path = tmp_path / "placings.jsonl"
    placings.record(path, "pm:1", "far", 3.0, **SPOT)

    assert placings.holding(path)["pm:1"]["fit"] is None


def test_a_row_with_an_empty_camera_has_none(tmp_path):
    assert placings.standing_of({"standing": {}}) is None


@pytest.mark.parametrize(
    "start", [{"lon": "a", "lat": 48.8}, {"lon": 2.3}], ids=["not a number", "no lat"]
)
def test_a_parallax_camera_s_starting_pin_written_wrong_is_refused(tmp_path, start):
    with pytest.raises(ValueError):
        placings.record(
            tmp_path / "p.jsonl",
            "pm:1",
            "points",
            3.0,
            **SPOT,
            registration=R,
            pairs=PAIRS,
            standing={**STANDING, "start": start},
        )


def test_an_answer_kept_in_parallax_under_the_old_rule_is_saved_again(tmp_path):
    four = {  # made when four depths sufficed, each from the neighbours that saw it
        "camera": STANDING["camera"],
        "metres": [12.0, 30.5, None, 8.2, 15.0, None, None],
        "from": [["p2"], ["p2", "p3"], None, ["p3"], ["p2"], None, None],
    }

    row = placings.record(
        tmp_path / "p.jsonl",
        "pm:1",
        "points",
        30.0,
        **SPOT,
        registration=R,
        pairs=PAIRS,
        standing=four,
    )

    assert row["fit"] == "parallax"


@pytest.mark.parametrize(
    "seconds",
    [
        [None] * 6,  # one short
        [None] * 6 + [{"neighbour": "p2"}],  # no direction
        [None] * 6 + [{"neighbour": 2, "own": [0.1, 0.2, 1.0]}],  # not a panorama
    ],
    ids=["one short", "no direction", "not a panorama"],
)
def test_second_clicks_that_do_not_go_with_their_pairs_are_refused(tmp_path, seconds):
    with pytest.raises(ValueError, match="second clic"):
        placings.record(
            tmp_path / "p.jsonl",
            "pm:1",
            "points",
            3.0,
            **SPOT,
            registration=R,
            pairs={**PAIRS, "second_clicks": seconds},
        )


def test_second_clicks_are_kept_with_their_pairs(tmp_path):
    seconds = [None] * 6 + [{"neighbour": "p2", "own": [0.1, 0.2, 1.0]}]

    row = placings.record(
        tmp_path / "p.jsonl",
        "pm:1",
        "points",
        3.0,
        **SPOT,
        registration=R,
        pairs={**PAIRS, "second_clicks": seconds},
    )

    assert row["pairs"]["second_clicks"][6]["neighbour"] == "p2"
