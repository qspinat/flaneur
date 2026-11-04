"""The confirmation page: the user's yes or no to each photo placed."""

import json

import pytest

from flaneur import confirming


def test_a_verdict_is_appended_with_what_the_photo_was_placed_from(tmp_path):
    path = tmp_path / "anchors.jsonl"

    confirming.record(path, "pm:1", "yes", "abc", 3.24)
    confirming.record(path, "pm:1", "no", "abc", 1)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["photo_id"], r["verdict"], r["inputs"], r["seconds"]) for r in rows] == [
        ("pm:1", "yes", "abc", 3.2),
        ("pm:1", "no", "abc", 1.0),
    ]


@pytest.mark.parametrize(("verdict", "seconds"), [("maybe", 1), ("yes", -1), ("yes", True)])
def test_a_verdict_written_wrong_is_refused(tmp_path, verdict, seconds):
    with pytest.raises(ValueError):
        confirming.record(tmp_path / "anchors.jsonl", "pm:1", verdict, "abc", seconds)


def test_the_photos_not_judged_as_placed_now_come_first_round_by_round():
    placed = [
        {"id": "a", "round": 0, "inputs": "1"},
        {"id": "b", "round": 2, "inputs": "1"},
        {"id": "c", "round": 1, "inputs": "1"},
        {"id": "d", "round": 0, "inputs": "2"},
    ]
    verdicts = {"a": {"verdict": "yes", "inputs": "1"}, "d": {"verdict": "no", "inputs": "1"}}

    queue = confirming.queue(placed, verdicts)

    assert [(p["id"], p["verdict"]) for p in queue] == [
        ("d", None),  # judged when placed otherwise: to judge again
        ("c", None),
        ("b", None),
        ("a", "yes"),
    ]


def test_photos_the_others_disagree_with_come_first_among_those_to_judge():
    placed = [
        {"id": "a", "round": 0, "inputs": "1"},
        {"id": "b", "round": 1, "inputs": "1", "doubt": "bouge de 3.0 m"},
    ]

    assert [p["id"] for p in confirming.queue(placed, {})] == ["b", "a"]
