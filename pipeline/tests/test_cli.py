import json
import threading

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from flaneur import cli, confirming, depth, facades, match, segment
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


def _units(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _scenes(monkeypatch, tmp_path, garbled: frozenset = frozenset(), seen=None):
    """Scenes around a panorama with a neighbour beside it (and one too far), their matches
    the street's two facades seen from each (those with a neighbour `garbled`, at random); and
    the pairs matched."""
    home = match.Panorama("home", 0.0, 0.0, 0.0, 0.0, tmp_path / "h.jpg")
    beside = match.Panorama("beside", 2.0, 1.0, 0.0, 0.0, tmp_path / "b.jpg")  # in the street
    far = match.Panorama("far", 40.0, 0.0, 0.0, 0.0, tmp_path / "f.jpg")
    asked = []

    def matched(a, b, views):
        asked.append((a.id, b.id))
        seen_b = _units(STREET - [b.x, b.y, b.z])
        if b.id in garbled:
            seen_b = _units(np.random.default_rng(0).normal(size=STREET.shape))
        return _units(STREET - [a.x, a.y, a.z]), seen_b

    monkeypatch.setattr(depth, "matched", matched)
    spheres = [home, beside, far]
    scenes = cli._Scenes(
        spheres,
        set(),
        lambda: None,
        threading.Lock(),
        lambda x, y: [],
        tmp_path / "w.jsonl",
        tmp_path / "pairs",
        seen=seen,
    )
    return asked, scenes, home


def test_scenes_match_each_neighbour_once_and_keep_the_walls(monkeypatch, tmp_path):
    asked, scenes, _ = _scenes(monkeypatch, tmp_path)

    first = scenes.scene("home", None)
    again = scenes.scene("home", None)
    chosen = scenes.scene("home", ("beside",))

    assert asked == [("home", "beside")]  # the one 2-15 m off, once
    assert again == first
    assert len(first.walls) == 2
    assert first.neighbours == ("beside",) and first.chosen is None
    assert chosen.walls == first.walls and chosen.chosen == ("beside",)  # the same matches
    assert scenes.scene("nowhere", None) is None


def test_a_pair_s_matches_are_kept_on_disk(monkeypatch, tmp_path):
    asked, scenes, _ = _scenes(monkeypatch, tmp_path)
    scenes.scene("home", None)

    _, again, _ = _scenes(monkeypatch, tmp_path)  # the page started again
    found = again.scene("home", None)

    assert asked == [("home", "beside")] and len(found.walls) == 2


def test_a_neighbour_that_cannot_be_set_right_gives_no_walls(monkeypatch, tmp_path):
    _, scenes, _ = _scenes(monkeypatch, tmp_path, garbled=frozenset({"beside"}))

    found = scenes.scene("home", None)

    assert found.walls == () and found.neighbours == ()


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
    assert asked == [("home", "beside")]


def test_neighbours_chosen_by_hand_are_matched_however_far(monkeypatch, tmp_path):
    asked, scenes, _ = _scenes(monkeypatch, tmp_path)

    found = scenes.scene("home", ("far",))  # 40 m off: never a candidate

    assert found.neighbours == ("far",)
    assert asked == [("home", "far")]


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


def test_a_neighbour_whose_pictures_cannot_be_read_is_tried_again(monkeypatch, tmp_path):
    asked, scenes, _ = _scenes(monkeypatch, tmp_path)
    real = depth.matched
    unread = [True]

    def flaky(a, b, views):  # its pictures not read the first time
        if unread.pop() if unread else False:
            asked.append((a.id, b.id))
            return np.empty((0, 3)), np.empty((0, 3))
        return real(a, b, views)

    monkeypatch.setattr(depth, "matched", flaky)
    first = scenes.scene("home", None)
    again = scenes.scene("home", None)

    assert first.neighbours == () and again.neighbours == ("beside",)


def test_only_details_seen_on_buildings_make_walls(monkeypatch, tmp_path):
    def west_only(panorama, rays):  # the left facade (x = -4) a building, the right one trees
        return np.where(rays[:, 0] < 0, segment.BUILT, segment.OTHER)

    _, scenes, _ = _scenes(monkeypatch, tmp_path, seen=west_only)

    found = scenes.scene("home", None)

    assert len(found.walls) == 1 and found.walls[0].a[0] < 0


def test_walls_are_topped_where_the_home_sees_the_sky(monkeypatch, tmp_path):
    def skyline(panorama, rays):  # both faces (x = ±4) building up to 12 m, the sky above
        z = 4.0 / np.abs(rays[:, 0]) * rays[:, 2]
        return np.where(z <= 12.0, segment.BUILT, segment.SKY)

    _, scenes, _ = _scenes(monkeypatch, tmp_path, seen=skyline)

    found = scenes.scene("home", None)

    assert len(found.walls) == 2 and all(w.top == pytest.approx(12.0, abs=0.3) for w in found.walls)


def test_a_failing_model_is_asked_once_and_the_walls_made_as_before(monkeypatch, tmp_path):
    asked = []

    def failing(self, panorama):
        asked.append(panorama.id)
        raise RuntimeError("not fetched")

    monkeypatch.setattr(segment.Segmenter, "kinds", failing)
    seen = cli._seeing(make=True)
    panorama = match.Panorama("p", 0.0, 0.0, 0.0, 0.0, tmp_path / "p.jpg")
    rays = np.array([[1.0, 0.0, 0.0]])

    found = [seen(panorama, rays).tolist() for _ in range(3)]

    assert asked == ["p"] and found == [[segment.BUILT]] * 3


def test_details_from_the_pairs_already_made_never_wake_the_matcher(monkeypatch, tmp_path):
    asked, scenes, home = _scenes(monkeypatch, tmp_path)
    candidates = depth.candidates(home, scenes.spheres, scenes.fixed)

    before = scenes.details(home, candidates, made_only=True)
    scenes.scene("home", None)  # the pair made (and kept on disk)
    after = scenes.details(home, candidates, made_only=True)

    assert len(before) == 0 and asked == [("home", "beside")] and len(after) > 0


def _kept(**more) -> dict:
    fields = dict(
        x=1.0,
        y=2.0,
        above_street=1.5,
        heading=90.0,
        horizon=0.5,
        fov=60.0,
        agree=30,
        panoramas=2,
        moved_m=3.0,
        walls=[],
        inputs="i",
        outline=0.1,
    )
    return {**fields, **more}


def test_a_placement_kept_with_its_matcher_is_read_as_it_was_found():
    situated = cli._situation(_kept(matcher="lomag"))

    assert situated == match.Situated(**_kept())


def test_a_placement_s_inputs_name_its_matcher_unless_disk_s_as_before():
    assert cli._placed_inputs("i", "disk") == cli._placed_inputs("i", None) == "i"
    assert cli._placed_inputs("i", "lomag") == "i:lomag"


@pytest.mark.parametrize(
    ("kept", "refused", "left"),
    [
        (None, False, ["lomag", "disk"]),  # never matched
        ({"inputs": "other"}, False, ["lomag", "disk"]),  # matched from other inputs
        ({"inputs": "i"}, False, ["lomag"]),  # left unplaced by DISK alone, before LoMa-G
        ({"inputs": "i", "tried": ["lomag"]}, False, ["disk"]),  # stopped before DISK
        ({"inputs": "i", "tried": ["disk"]}, False, ["lomag"]),  # LoMa-G's check had not passed
        ({"inputs": "i", "tried": ["lomag", "disk"]}, False, []),
        (_kept(), False, []),  # placed by DISK before LoMa-G: kept
        (_kept(), True, ["lomag"]),  # ... unless turned down: LoMa-G's turn
        (_kept(matcher="lomag", inputs="i:lomag", tried=["lomag"]), False, []),
        (_kept(matcher="lomag", inputs="i:lomag", tried=["lomag"]), True, ["disk"]),
        (_kept(matcher="disk", tried=["lomag", "disk"]), True, []),
        (_kept(matcher="lomag", inputs="other:lomag", tried=["lomag"]), False, ["lomag", "disk"]),
    ],
)
def test_a_photo_is_matched_again_only_by_the_matchers_left_to_try(kept, refused, left):
    assert cli._left_to_try(kept, "i", refused) == left


def test_only_current_placements_by_lomag_are_left_out_of_chaining():
    matches = {
        "a": _kept(),
        "b": _kept(matcher="disk"),
        "c": _kept(matcher="lomag"),
        "d": None,
        "e": _kept(matcher="lomag"),  # stale or turned down: not situated
    }

    assert cli._not_chained(matches, {"a", "b", "c"}) == {"c"}


def test_a_photo_placed_against_today_is_shown_so_rather_than_as_last_chained():
    situated = {"a": "placed now"}
    chained = {
        "photos": {
            "a": {"situated": _kept(), "round": 2, "from": ["b"]},
            "c": {"situated": _kept(), "round": 1, "from": ["a"]},
        }
    }

    shown, rounds = cli._with_chained(situated, chained, {"a", "c"})

    assert shown["a"] == "placed now" and rounds["a"] == {"round": 0, "from": []}
    assert shown["c"] == match.Situated(**_kept()) and rounds["c"] == {"round": 1, "from": ["a"]}


def test_each_matcher_has_its_own_known_answer_check(tmp_path):
    path = tmp_path / "match_check.json"

    assert cli._of_matcher(path, "disk") == path
    assert cli._of_matcher(path, "lomag") == tmp_path / "match_check_lomag.json"


@pytest.mark.parametrize(
    ("kept", "found"),
    [
        (_kept(matcher="lomag", tried=["lomag"]), {"by": "LoMa-G"}),
        (
            _kept(matcher="disk", tried=["lomag", "disk"]),
            {"by": "DISK", "check": "LoMa-G n'a rien trouvé de juste"},
        ),
        (_kept(matcher="disk", tried=["disk"]), {"by": "DISK"}),  # LoMa-G not tried yet
        (_kept(), {"by": "DISK (avant LoMa-G)"}),
    ],
)
def test_the_page_tells_which_matcher_placed_a_photo_and_whether_to_check_it(kept, found):
    assert cli._found_by(kept) == found


def test_lomag_s_check_goes_stale_with_its_code_disk_s_with_the_panoramas_only():
    assert cli._check_setup("disk", "s") == "s"
    assert cli._check_setup("lomag", "s").startswith("s:")
    assert cli._check_setup("lomag", "s") != cli._check_setup("lomag", "t")


def test_the_old_photo_s_outlines_are_drawn_on_today_s_view_in_amber():
    old = np.full((400, 400, 3), 30, np.uint8)
    old[120:280, 120:280] = 220  # a light square: its outline is the street's only edge
    old[12:14] = old[:, 386:388] = 255  # the print's border, a few pixels in, as a crop leaves it
    today = np.full((100, 100, 3), 100, np.uint8)  # the sheet's size: a quarter of the photo's

    drawn = cli._outlines(old, today)

    amber = (drawn == cli.OUTLINE_COLOUR).all(axis=2)
    assert amber[29:32, 50].any()  # the square's top edge, at the sheet's size
    assert not amber[45:55, 45:55].any()  # inside it
    assert not amber[:6].any() and not amber[:, 94:].any()  # the print's border is not drawn
    assert (drawn[50, 50] == 80).all()  # today, in grey, dimmed under the outlines
