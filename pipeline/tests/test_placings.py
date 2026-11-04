"""The user's answers on the placing page, kept with the other overrides."""

import pytest

from flaneur import placings
from flaneur.registration import Registration, ViewOf

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
