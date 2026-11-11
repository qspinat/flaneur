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
    verdicts = {"a": {"verdict": "yes", "inputs": "1"}}  # the word that holds for each

    queue = confirming.queue(placed, verdicts)

    assert [(p["id"], p["verdict"]) for p in queue] == [
        ("d", None),
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


def test_photos_to_check_closely_come_first_among_those_to_judge():
    placed = [
        {"id": "a", "round": 0, "inputs": "1", "by": "LoMa-G"},
        {"id": "b", "round": 0, "inputs": "1", "by": "DISK", "check": "LoMa-G n'a rien trouvé"},
    ]

    assert [p["id"] for p in confirming.queue(placed, {})] == ["b", "a"]


def test_a_no_keeps_the_reason_it_was_given(tmp_path):
    path = tmp_path / "anchors.jsonl"

    confirming.record(path, "pm:1", "no", "abc", 1, reason="reproduction")

    assert json.loads(path.read_text())["reason"] == "reproduction"


@pytest.mark.parametrize(("verdict", "reason"), [("yes", "reproduction"), ("no", "because")])
def test_a_reason_goes_only_with_a_no_and_is_one_of_the_page_s(tmp_path, verdict, reason):
    with pytest.raises(ValueError):
        confirming.record(tmp_path / "anchors.jsonl", "pm:1", verdict, "abc", 1, reason=reason)


def test_a_no_comes_back_with_its_reason():
    placed = [{"id": "a", "round": 0, "inputs": "1"}, {"id": "b", "round": 0, "inputs": "1"}]
    verdicts = {"a": {"verdict": "no", "inputs": "1", "reason": "mirrored"}}  # b: none holds

    queue = {p["id"]: p for p in confirming.queue(placed, verdicts)}

    assert (queue["a"]["reason"], queue["b"]["reason"]) == ("mirrored", None)
