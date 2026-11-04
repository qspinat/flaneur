import json

import pytest
import typer
from typer.testing import CliRunner

from flaneur import cli, confirming, match
from flaneur.cli import app


def test_help_lists_the_pipeline():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Pipeline Flâneur" in result.output


def test_a_trace_tells_which_camera_turned_down_was_the_right_one():
    trace = match.Trace(points=40, verdict="aucune caméra")
    trace.candidates += [
        match.Candidate(1.0, 0.5, 92.0, 60.0, 0.5, 30, 0.9, "position incertaine"),
        match.Candidate(40.0, 0.0, 10.0, 60.0, 0.5, 22, None, "ne peut se tenir là"),
    ]
    truth = (match.Panorama("p", 0.0, 0.0, 2.5, 0.0, path=None), 90.0)

    row = cli._traced(trace, truth)

    right, wrong = row["candidates"]
    assert right["right"] and right["off_m"] == pytest.approx(1.12, abs=0.01)
    assert not wrong["right"] and wrong["spread_m"] is None
    assert cli._funnel([row, row, {"verdict": "située"}]) == "aucune caméra 2 ; située 1"


def test_placements_judged_wrong_by_eye_are_set_aside_the_last_word_holding(tmp_path, monkeypatch):
    path = tmp_path / "eye_checks.jsonl"
    rows = [
        {"photo_id": "a", "inputs": "1", "eye": "wrong", "why": "the tower falls left"},
        {"photo_id": "a", "inputs": "2", "eye": "right"},  # another placement: the first holds
        {"photo_id": "b", "inputs": "1", "eye": "wrong"},
        {"photo_id": "b", "inputs": "1", "eye": "doubtful"},  # looked at again
        {"photo_id": "c", "eye": "wrong"},  # no placement named: nothing set aside
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(cli, "EYE_CHECKS", path)

    assert cli._set_aside() == {("a", "1"): "the tower falls left"}


def test_photos_placed_near_but_not_at_their_camera_are_no_anchors(tmp_path, monkeypatch):
    path = tmp_path / "anchors.jsonl"
    for photo_id, verdict in [("a", "yes"), ("b", "no"), ("c", "near")]:
        confirming.record(path, photo_id, verdict, "1", 2)
    monkeypatch.setattr(cli, "ANCHORS", path)

    assert cli._refused() == {("b", "1"), ("c", "1")}


def test_photos_the_user_turned_down_stay_down_until_a_yes(tmp_path, monkeypatch):
    path = tmp_path / "anchors.jsonl"
    for photo_id, verdict, inputs in [
        ("a", "no", "1"),
        ("b", "near", "1"),
        ("c", "no", "1"),
        ("c", "yes", "2"),  # placed otherwise since, and taken
        ("d", "yes", "1"),
    ]:
        confirming.record(path, photo_id, verdict, inputs, 2)
    monkeypatch.setattr(cli, "ANCHORS", path)

    assert cli._turned_down() == {"a", "b"}


def test_a_photo_placed_again_without_any_anchor_is_its_placement_against_today():
    assert cli._placed_from("abc", []) == "abc"  # a "no" to it still holds


def test_the_photos_chained_from_one_are_found_through_every_round():
    results = {
        "b": {"from": ["a"]},
        "c": {"from": ["b"]},
        "d": {"from": ["c", "x"]},
        "e": {"from": ["x"]},
    }

    assert cli._chained_from(results, "a") == {"b", "c", "d"}
    assert cli._chained_from(results, "c") == {"d"}


def test_the_panoramas_are_not_used_when_their_tilts_were_measured_otherwise(monkeypatch):
    monkeypatch.setattr(cli.level, "current", lambda path: False)

    with pytest.raises(typer.BadParameter, match="flaneur level"):
        cli._lined_up([])
