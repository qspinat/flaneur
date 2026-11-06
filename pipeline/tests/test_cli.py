import json
import threading

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from flaneur import cli, confirming, depth, facades, match
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


def test_panoramas_fixed_by_colmap_are_told_apart():
    from flaneur.panoramax import Picture

    here = Picture("p1", 2.3466, 48.852, 40.0, True, "2024-01-01", "", "")
    moved = Picture("p2", 2.3467, 48.852, 40.0, True, "2024-01-01", "", "")
    fixes = {
        "p1": {"fix": [0.1, 0.2, 0.3], "recorded": [here.point.x, here.point.y, 40.0]},
        "p2": {"fix": [0.1, 0.2, 0.3], "recorded": [0.0, 0.0, 40.0]},  # recorded elsewhere then
    }

    assert cli._fixed_by_colmap([here, moved], fixes) == {"p1"}


LEFT = [[-4.0, y, z] for y in np.arange(2.0, 40.0, 0.5) for z in (-1.0, 3.0, 8.0)]
STREET = np.array(LEFT + [[4.0, y, z] for _, y, z in LEFT])  # two facades along y
METRES = np.linalg.norm(STREET, axis=1)


def _scenes(monkeypatch, tmp_path):
    """Scenes around a panorama with a neighbour beside it (and one too far), every view's
    matches with it being the street's two facades; and the matches asked."""
    home = match.Panorama("home", 0.0, 0.0, 0.0, 0.0, tmp_path / "h.jpg")
    beside = match.Panorama("beside", 2.0, 1.0, 0.0, 0.0, tmp_path / "b.jpg")  # in the street
    far = match.Panorama("far", 40.0, 0.0, 0.0, 0.0, tmp_path / "f.jpg")
    asked = []

    def triangulated(a, b, views, only):
        asked.append((a.id, b.id, tuple(only)))
        return STREET / METRES[:, None], METRES

    monkeypatch.setattr(depth, "triangulated", triangulated)
    spheres = [home, beside, far]
    scenes = cli._Scenes(
        spheres, set(), lambda: None, threading.Lock(), lambda x, y: [], tmp_path / "w.jsonl"
    )
    return asked, scenes, home


def test_scenes_match_each_view_with_a_neighbour_once_and_keep_the_walls(monkeypatch, tmp_path):
    asked, scenes, _ = _scenes(monkeypatch, tmp_path)

    first = scenes.scene("home", None)
    again = scenes.scene("home", None)
    chosen = scenes.scene("home", ("beside",))

    assert len(asked) == len(match.VIEW_HEADINGS)  # all around, once each
    assert again == first
    assert len(first.walls) == 2
    assert first.neighbours == ("beside",) and first.chosen is None
    assert chosen.walls == first.walls and chosen.chosen == ("beside",)  # the same matches
    assert scenes.scene("nowhere", None) is None


def _square(x0: float, x1: float) -> dict:
    ring = [[x0, 0.0, 30.0], [x1, 0.0, 30.0], [x1, 10.0, 30.0], [x0, 10.0, 30.0], [x0, 0.0, 30.0]]
    return {"geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {}}


def test_bd_topo_s_street_facades_leave_the_walls_between_buildings_out():
    near = cli._street_facades([_square(0.0, 10.0), _square(10.2, 20.0)])  # heights kept: 3D

    found = near(10.0, -5.0)

    inside = [s for s in found if all(5.0 < x < 15.0 and 1.0 < y < 9.0 for x, y in s)]
    assert found and not inside  # the facades, not the party wall at x = 10


def test_walls_drawn_change_the_scene_without_matching_again(monkeypatch, tmp_path):
    asked, scenes, home = _scenes(monkeypatch, tmp_path)
    fitted = scenes.scene("home", None)
    left = next(w for w in fitted.walls if w.a[0] < 0)

    facades.record(tmp_path / "w.jsonl", "home", [], [(left.a, left.b)])

    assert len(scenes.scene("home", None).walls) == len(fitted.walls) - 1
    assert len(asked) == len(match.VIEW_HEADINGS)


def test_neighbours_chosen_by_hand_are_matched_however_far(monkeypatch, tmp_path):
    asked, scenes, _ = _scenes(monkeypatch, tmp_path)

    found = scenes.scene("home", ("far",))  # 40 m off: never picked by geometry

    assert found.neighbours == ("far",)
    assert len(asked) == len(match.VIEW_HEADINGS)


def test_walls_drawn_give_a_scene_where_no_neighbour_is_matched(monkeypatch, tmp_path):
    asked, scenes, _ = _scenes(monkeypatch, tmp_path)
    drawn = facades.Wall((-4.0, 0.0), (-4.0, 30.0), 20.0, "drawn")
    facades.record(tmp_path / "w.jsonl", "home", [drawn], [])

    found = scenes.scene("home", ("nowhere",))  # a choice of none known

    assert found.walls == (drawn,) and found.details == 0 and not asked


def test_no_neighbour_chosen_matches_none(monkeypatch, tmp_path):
    asked, scenes, _ = _scenes(monkeypatch, tmp_path)

    found = scenes.scene("home", ())  # all left out by hand: only walls drawn

    assert found.neighbours == () and found.chosen == () and not asked
