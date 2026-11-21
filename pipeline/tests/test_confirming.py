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
    confirming.mark_pair(path, "pm:1", "abc", [3.0, 4.5], True)
    confirming.mark_pair(path, "pm:1", "abc", [5.0, 1.0], True)
    confirming.mark_pair(path, "pm:1", "abc", [5.0, 1.0], False)  # taken back
    confirming.mark_pair(path, "pm:1", "other", [7.0, 2.0], True)  # found again from other inputs

    assert confirming.marked(path) == {
        ("pm:1", "abc"): {(3.0, 4.5)},
        ("pm:1", "other"): {(7.0, 2.0)},
    }


@pytest.mark.parametrize("wrong", ["1", 1, None])
def test_a_mark_neither_wrong_nor_right_is_refused(tmp_path, wrong):
    with pytest.raises(ValueError):
        confirming.mark_pair(tmp_path / "wrong_pairs.jsonl", "pm:1", "abc", (1.0, 2.0), wrong)


def test_a_points_key_is_its_pixel_to_the_thousandth_the_same_wherever_made():
    assert confirming.point_key([84.9015, 2.0]) == confirming.point_key((84.9015, 2.0))
    assert confirming.point_key([1.00049, 2.0]) == (1.0, 2.0)


def test_the_pairs_shown_say_which_are_marked_wrong():
    scene = {"pairs": [{"index": 3, "point": [1.0, 2.0]}, {"index": 4, "point": [3.0, 4.0]}]}

    shown = confirming.with_marks(scene, {(1.0, 2.0)})

    assert [p["wrong"] for p in shown["pairs"]] == [True, False]
    assert confirming.with_marks({"pairs": None}, {(1.0, 2.0)}) == {"pairs": None}


def test_a_pair_marked_on_the_page_comes_back_marked_and_judges_nothing(tmp_path, ask):
    placed = [{"id": "pm:1", "inputs": "abc", "round": 0}]
    anchors, marks = tmp_path / "anchors.jsonl", tmp_path / "wrong_pairs.jsonl"
    scene = {"pairs": [{"index": 3, "point": [1.0, 2.0]}, {"index": 4, "point": [3.0, 4.0]}]}
    asked = []
    server = confirming.serve(
        placed,
        anchors,
        marks,
        dict,
        lambda *a, **k: None,
        lambda photo_id, source=False: asked.append(source) or scene,
        0,
    )

    (marked, _), (unshown, _), (unknown, _), (not_a_number, _), (_, shown) = ask(
        server,
        ("POST", "/api/pair", {"photo_id": "pm:1", "index": 4, "wrong": True, "source": True}),
        ("POST", "/api/pair", {"photo_id": "pm:1", "index": 99, "wrong": True}),  # not shown
        ("POST", "/api/pair", {"photo_id": "pm:2", "index": 4, "wrong": True}),
        ("POST", "/api/pair", {"photo_id": "pm:1", "index": True, "wrong": True}),
        ("GET", "/scene/pm:1"),
    )

    assert (marked, unshown, unknown, not_a_number) == (200, 400, 400, 400)
    assert asked[0] is True  # only the view the page showed
    assert [p["wrong"] for p in json.loads(shown)["pairs"]] == [False, True]
    assert confirming.marked(marks) == {("pm:1", "abc"): {(3.0, 4.0)}}  # by its point
    assert not anchors.exists()  # a mark is not a verdict


def test_a_marks_file_written_wrong_leaves_the_pairs_shown_unmarked(tmp_path, ask):
    placed = [{"id": "pm:1", "inputs": "abc", "round": 0}]
    marks = tmp_path / "wrong_pairs.jsonl"
    marks.write_text("{not json\n")
    scene = {"pairs": [{"index": 3, "point": [1.0, 2.0]}]}
    server = confirming.serve(
        placed, tmp_path / "a.jsonl", marks, dict, lambda *a, **k: None, lambda *a, **k: scene, 0
    )

    ((status, shown),) = ask(server, ("GET", "/scene/pm:1"))

    assert status == 200 and json.loads(shown)["pairs"][0]["wrong"] is False
