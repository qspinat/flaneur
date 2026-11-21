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


def test_the_last_mark_on_a_pair_holds_for_its_placement_only(tmp_path):
    path = tmp_path / "wrong_pairs.jsonl"
    confirming.mark_pair(path, "pm:1", "abc", 3, True)
    confirming.mark_pair(path, "pm:1", "abc", 5, True)
    confirming.mark_pair(path, "pm:1", "abc", 5, False)  # taken back
    confirming.mark_pair(path, "pm:1", "other", 7, True)  # found again from other inputs

    assert confirming.marked(path) == {("pm:1", "abc"): {3}, ("pm:1", "other"): {7}}


@pytest.mark.parametrize(
    ("index", "wrong"), [(-1, True), (1.5, True), (True, True), ("3", True), (None, True), (3, "1")]
)
def test_a_mark_on_no_pair_or_neither_wrong_nor_right_is_refused(tmp_path, index, wrong):
    with pytest.raises(ValueError):
        confirming.mark_pair(tmp_path / "wrong_pairs.jsonl", "pm:1", "abc", index, wrong)


def test_the_pairs_shown_say_which_are_marked_wrong():
    scene = {"pairs": [{"index": 3}, {"index": 4}], "plan": {}}

    shown = confirming.with_marks(scene, {3})

    assert [p["wrong"] for p in shown["pairs"]] == [True, False]
    assert confirming.with_marks({"pairs": None}, {3}) == {"pairs": None}


def test_a_pair_marked_on_the_page_comes_back_marked_and_judges_nothing(tmp_path, ask):
    placed = [{"id": "pm:1", "inputs": "abc", "round": 0}]
    anchors, marks = tmp_path / "anchors.jsonl", tmp_path / "wrong_pairs.jsonl"
    scene = {"pairs": [{"index": 3}, {"index": 4}]}
    server = confirming.serve(
        placed, anchors, marks, dict, lambda *a, **k: None, lambda *a, **k: scene, 0
    )

    (marked, _), (refused, _), (_, shown) = ask(
        server,
        ("POST", "/api/pair", {"photo_id": "pm:1", "index": 4, "wrong": True}),
        ("POST", "/api/pair", {"photo_id": "pm:2", "index": 4, "wrong": True}),
        ("GET", "/scene/pm:1"),
    )

    assert (marked, refused) == (200, 400)
    assert [p["wrong"] for p in json.loads(shown)["pairs"]] == [False, True]
    assert not anchors.exists()  # a mark is not a verdict
