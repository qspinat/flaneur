"""The order photos come to the placing page in."""

import numpy as np

from flaneur import chain, queueing


def _where(n: int, east_m: float = 0.0) -> tuple[float, float]:
    return 652_000.0 + east_m, 6_861_500.0


def _order(ids, **kw):
    base = {
        "where": {i: _where(0) for i in ids},
        "held": {},
        "skipped": set(),
        "ahead": {},
        "graph": {},
        "nearest_m": {i: 5.0 for i in ids},
        "every": 0,
    }
    return queueing.order(ids, **{**base, **kw})


def _held(*ids, answer="fits", east_m=0.0):
    return {i: {"answer": answer, "x": 652_000.0 + east_m, "y": 6_861_500.0} for i in ids}


def test_photos_that_carry_most_old_photos_not_placed_yet_come_first():
    graph = {"a": {"b", "c", "x"}, "b": {"a"}, "c": {"a", "d"}, "d": {"c"}, "x": {"a"}}

    found = _order(["a", "b", "c", "d"], graph=graph, held=_held("x"))

    assert [i for i, _ in found] == ["a", "c", "b", "d"]  # a: b, c; c: d (and a); b: a; d: c


def test_a_photo_whose_neighbours_are_all_placed_carries_nothing_more():
    graph = {"a": {"x", "y"}, "b": {"c"}, "c": {"b"}}

    found = _order(["a", "b", "c"], graph=graph, held=_held("x", "y"))

    assert found[0][0] in ("b", "c")


def test_among_equals_the_nearest_to_a_panorama_comes_first():
    found = _order(["a", "b"], nearest_m={"a": 12.0, "b": 3.0})

    assert [i for i, _ in found] == ["b", "a"]


def test_every_fourth_answer_brings_the_photo_farthest_from_those_placed():
    ids = [f"p{n}" for n in range(1, 9)]
    where = {f"p{n}": _where(n, east_m=10.0 * n) for n in range(1, 9)}
    graph = {f"p{n}": {f"q{n}{k}" for k in range(9 - n)} for n in range(1, 9)}

    def head(answered: int):
        held = _held(*[f"x{k}" for k in range(answered)])  # placed, out of these photos' way
        return _order(ids, where=where, graph=graph, every=4, held=held, answered=answered)[0]

    assert [head(n)[1] for n in range(4)] == ["voisins", "voisins", "voisins", "isolée"]
    assert head(3)[0] == "p8"  # 80 m from those placed, the farthest


def test_photos_answered_are_left_out_and_skipped_ones_come_back_after_the_others():
    found = _order(["a", "b", "c"], held=_held("a", answer="far"), skipped={"b"})

    assert found == [("c", "proche"), ("b", "passée")]


def test_photos_with_no_panorama_near_come_last_but_skipped_ones():
    found = _order(["a", "b", "c"], nearest_m={"a": np.inf, "b": 5.0, "c": 5.0}, skipped={"c"})

    assert [i for i, _ in found] == ["b", "a", "c"]


def test_photos_asked_ahead_come_first_with_why():
    found = _order(["a", "b", "c"], ahead={"c": "revoir", "b": "portée"}, graph={"a": {"z"}})

    assert found[:2] == [("c", "revoir"), ("b", "portée")]


def test_old_photos_are_neighbours_when_the_pairs_kept_for_their_images_share_enough(tmp_path):
    images = {"pm:1": "aa", "pm:2": "bb", "pm:3": "cc"}
    full, half = tmp_path / "old_pairs_full", tmp_path / "old_pairs_half"
    full.mkdir(), half.mkdir()
    np.save(full / chain.pair_name(("pm:1", "aa"), ("pm:2", "bb")), np.zeros((150, 2), np.int32))
    np.save(half / chain.pair_name(("pm:1", "aa"), ("pm:2", "bb")), np.zeros((150, 2), np.int32))
    np.save(full / chain.pair_name(("pm:1", "aa"), ("pm:3", "cc")), np.zeros((40, 2), np.int32))
    np.save(full / chain.pair_name(("pm:2", "old"), ("pm:3", "cc")), np.zeros((500, 2), np.int32))

    assert chain.old_neighbours(tmp_path, images, 100) == {"pm:1": {"pm:2"}, "pm:2": {"pm:1"}}


def test_a_photo_carrying_no_one_comes_for_its_panorama_not_its_neighbours():
    found = _order(["a", "b"], graph={"a": {"b"}, "b": {"a"}}, held={})

    assert found == [("a", "voisins"), ("b", "voisins")]
    assert _order(["a"], graph={}) == [("a", "proche")]


def test_the_rhythm_counts_every_answer_given_reopened_ones_too():
    ids = [f"p{n}" for n in range(1, 5)]
    where = {i: _where(0, east_m=10.0 * n) for n, i in enumerate(ids)}

    head = _order(ids, where=where, every=4, answered=3)[0]

    assert head[1] == "isolée"


def test_odd_files_among_the_pairs_kept_are_left_aside(tmp_path):
    folder = tmp_path / "old_pairs_full"
    folder.mkdir()
    np.save(folder / "a__b__c.npy", np.zeros((150, 2), np.int32))
    (folder / "x__y.npy").write_bytes(b"not an array")

    assert chain.old_neighbours(tmp_path, {"a": "h"}, 100) == {}
