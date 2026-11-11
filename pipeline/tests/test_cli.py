import json
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from flaneur import cli, confirming, depth, facades, match, poses, segment
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

    assert cli._turned_down() == {"b", "c"}


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


def test_panoramas_stand_corrected_except_for_recaler(tmp_path, monkeypatch):
    from flaneur.panoramax import Picture

    here = Picture("p1", 2.3466, 48.852, 40.0, True, "2024-01-01", "", "")
    at = poses.at(here.point.x, here.point.y, 40.0)
    verdicts = {"p1": poses.Verdict(poses.Correction(2.0, -1.0, 4.0), None, 900, at)}
    poses.save(tmp_path / "poses.json", verdicts)
    monkeypatch.setattr(cli, "POSES", tmp_path / "poses.json")
    monkeypatch.setattr(cli, "FIXES", tmp_path / "fixes.json")  # none: as recorded
    monkeypatch.setattr(poses, "APPLIED", "all")

    (corrected,) = cli._placed([here])
    (as_colmap,) = cli._placed([here], corrected=False)

    assert corrected != here and as_colmap == here


def test_nothing_applied_reads_no_verdict(tmp_path, monkeypatch):
    from flaneur.panoramax import Picture

    here = Picture("p1", 2.3466, 48.852, 40.0, True, "2024-01-01", "", "")
    (tmp_path / "poses.json").write_text("not json")
    monkeypatch.setattr(cli, "POSES", tmp_path / "poses.json")
    monkeypatch.setattr(cli, "FIXES", tmp_path / "fixes.json")
    monkeypatch.setattr(poses, "APPLIED", "none")

    assert cli._placed([here]) == [here]


def test_homes_with_pairs_are_counted_under_their_own_names(tmp_path):
    a = match.Panorama("a", 0.0, 0.0, 0.0, 10.0, tmp_path / "a.jpg")
    b = match.Panorama("b", 5.0, 0.0, 0.0, 20.0, tmp_path / "b.jpg")
    for n in range(3):
        (tmp_path / f"{cli._levelled_name(a)}__x{n}_0.0_0.00_0.00.npz").touch()
        (tmp_path / f"b_99.0_0.00_0.00__x{n}_0.0_0.00_0.00.npz").touch()  # b as turned since

    assert cli._paired_homes([a, b], tmp_path) == [a]


def test_parallaxe_with_nothing_to_compare_says_so_before_loading_the_world(tmp_path, monkeypatch):
    def world(**_):
        raise AssertionError("the world loaded for nothing")

    monkeypatch.setattr(cli, "_placing_world", world)

    result = CliRunner().invoke(app, ["parallaxe", "--answers", str(tmp_path / "none.jsonl")])

    assert result.exit_code == 0 and "aucune réponse par points" in result.output


def test_panoramas_used_by_more_photos_come_first(monkeypatch):
    near = [
        match.Panorama(n, x, 0.0, 0.0, 0.0, None)
        for n, x in [("a", 0.0), ("b", 30.0), ("c", 500.0)]
    ]
    walk = [
        SimpleNamespace(id="pm:1", place=(10.0, 0.0)),
        SimpleNamespace(id="pm:2", place=(75.0, 0.0)),
        SimpleNamespace(id="pm:3", place=(500.0, 0.0)),  # not exported: match leaves it
    ]
    monkeypatch.setattr(cli, "_place_of", lambda me: me.place)
    monkeypatch.setattr(cli, "_exported", lambda i: SimpleNamespace(exists=lambda: i != "pm:3"))

    used = cli._used_by_matching(walk, near)

    assert [p.id for p in used] == ["b", "a"]  # b near both; c near none that match works on


def test_recaler_refuses_prefixes_and_photos_together(monkeypatch):
    monkeypatch.setattr(cli, "_placing_world", lambda **_: pytest.fail("loaded for nothing"))

    result = CliRunner().invoke(app, ["recaler", "1a2b", "--photos"])

    assert result.exit_code != 0 and "--photos" in result.output


@pytest.mark.parametrize(
    "known_at, holds",
    [
        ([1.0, 2.0, 30.0], True),
        ([1.0, 2.0, 33.0], False),  # its panorama turned since
        (None, False),  # made before poses were kept: made again
    ],
)
def test_a_proposal_holds_while_its_panorama_stands_where_it_was_made(monkeypatch, known_at, holds):
    monkeypatch.setattr(cli, "_image", lambda photo_id: "h")
    known = {"panorama": "p1", "inputs": "h", **({"at": known_at} if known_at else {})}

    assert cli._still_proposed(known, "pm:1", "p1", (1.0, 2.0, 30.0)) is holds


def test_copies_carried_from_panoramas_moved_since_are_made_again(tmp_path, monkeypatch):
    row = {"registration": {"panorama": "a"}, "inputs": "i", "flipped": False}
    made = cli._answered_as(row)
    kept = {
        "pm:1": {"made": made, "copies": [], "at": {"a": [0.0, 0.0, 0.0], "b": [5.0, 0.0, 0.0]}},
        "pm:2": {"made": made, "copies": [], "at": {"a": [0.0, 0.0, 0.0], "c": [9.0, 0.0, 0.0]}},
        "pm:3": {"made": made, "copies": [], "at": {"d": None}},  # its home at hand since
    }
    (tmp_path / "carried.json").write_text(json.dumps(kept))
    monkeypatch.setattr(cli, "CARRIED", tmp_path / "carried.json")
    now = {"a": (0.0, 0.0, 0.0), "b": (5.0, 0.0, 0.0), "c": (9.0, 0.0, 4.0), "d": (1.0, 1.0, 0.0)}

    assert set(cli._carried({"pm:1": row, "pm:2": row, "pm:3": row}, now)) == {"pm:1"}


def test_recaler_keeps_clear_of_the_page_only_when_it_wakes_the_matcher(monkeypatch):
    guarded = []
    monkeypatch.setattr(cli, "_page_closed", lambda page: guarded.append(page))

    def world(**_):
        raise RuntimeError("world")

    monkeypatch.setattr(cli, "_placing_world", world)

    CliRunner().invoke(app, ["recaler"])  # pairs on disk only: no model
    assert guarded == []
    CliRunner().invoke(app, ["recaler", "--photos"])
    assert guarded == [8772]


def test_copies_are_carried_from_panoramas_as_they_stand_and_lean():
    a = match.Panorama("a", 1.0, 2.0, 0.0, 30.0, None, 1.5, -0.5)

    assert cli._poised([a]) == {"a": (1.0, 2.0, 30.0, 1.5, -0.5)}


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


def _judged(tmp_path, *rows):
    path = tmp_path / "anchors.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return cli._Judged(path)


def _said(verdict, inputs, x=0.0, y=0.0, heading=90.0):
    return {
        "photo_id": "a",
        "verdict": verdict,
        "inputs": inputs,
        "camera": {"x": x, "y": y, "heading": heading},
    }


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

    situated = {k: SimpleNamespace(inputs="i") for k in ("a", "b", "c")}
    situated["d"] = SimpleNamespace(inputs="i:proposed-abc")  # said yes to: no points to chain by

    assert cli._not_chained(matches, situated) == {"c", "d"}


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
    disk = cli._check_setup("disk", "cite", [])

    assert cli._check_setup("lomag", "cite", []).startswith(f"{disk}:")  # and its own code
    assert cli._check_setup("disk", "saint-severin", []) != disk  # each area its own


def test_the_old_photo_s_outlines_are_drawn_on_today_s_view_in_amber():
    old = np.full((400, 400, 3), 30, np.uint8)
    old[120:280, 120:280] = 220  # a light square: its outline is the street's only edge
    old[12:14] = old[:, 386:388] = 255  # the print's border, a few pixels in, as a crop leaves it
    grain = np.random.default_rng(0).integers(0, 255, (30, 27))
    old[300:390, 20:100] = np.kron(grain, np.ones((3, 3)))[:90, :80, None]  # fine texture
    today = np.full((300, 300, 3), 100, np.uint8)  # the sheet's size

    drawn = cli._outlines(old, today)

    amber = (drawn == cli.OUTLINE_COLOUR).all(axis=2)
    assert amber[88:93, 150].any()  # the square's top edge, at the sheet's size
    assert not amber[130:170, 130:170].any()  # inside it
    assert not amber[:15].any() and not amber[:, 285:].any()  # the print's border is not drawn
    assert amber[230:290, 20:70].mean() < 0.6  # texture stays lines, not a solid patch
    assert (drawn[150, 150] == 80).all()  # today, in grey, dimmed under the outlines


def test_no_view_of_today_is_shown_from_a_panorama_too_far_from_the_camera(tmp_path, monkeypatch):
    photo = tmp_path / "old.jpg"
    cv2.imwrite(str(photo), np.full((80, 60, 3), 128, np.uint8))
    monkeypatch.setattr(cli, "_exported", lambda photo_id: photo)
    far = match.Panorama("p", cli.SHEET_MOST_M + 5, 0.0, 2.5, 0.0, path=tmp_path / "p.jpg")
    m = match.Situated(0.0, 0.0, 1.5, 90.0, 0.5, 60.0, 30, 2, 3.0, [], "i", 0.1)
    heights = type("H", (), {"street": staticmethod(lambda x, y: 0.0)})()

    row = cli._sheet_row("pm:1", m, heights, [far], lambda path: np.zeros((512, 1024, 3), np.uint8))

    assert row is None


def test_a_photo_and_today_s_view_are_framed_alike(tmp_path, monkeypatch):
    photo = tmp_path / "old.jpg"
    cv2.imwrite(str(photo), np.full((80, 60, 3), 128, np.uint8))
    monkeypatch.setattr(cli, "_exported", lambda photo_id: photo)
    near = match.Panorama("p", 1.0, 0.0, 2.5, 0.0, path=tmp_path / "p.jpg")
    m = match.Situated(0.0, 0.0, 1.5, 90.0, 0.5, 60.0, 30, 2, 3.0, [], "i", 0.1)
    heights = type("H", (), {"street": staticmethod(lambda x, y: 0.0)})()

    framed = cli._framed(
        "pm:1", m, heights, [near], lambda path: np.zeros((512, 1024, 3), np.uint8), 30
    )

    assert framed.old.shape == framed.today.shape == (40, 30, 3) and framed.away_m == 1.0


def _match_world(tmp_path, monkeypatch, previous: dict, places) -> None:
    """`match` over two photos pm:1 and pm:2 with everything faked: `previous` as matches.json
    held, `places(matcher)` the camera each matcher finds (None: nothing)."""
    photos = [SimpleNamespace(id="pm:1"), SimpleNamespace(id="pm:2")]
    exported = tmp_path / "old.jpg"
    cv2.imwrite(str(exported), np.zeros((8, 8, 3), np.uint8))
    heights = SimpleNamespace(street=lambda x, y: 0.0, origin_z=0.0)

    for name, value in {
        "MATCHES": tmp_path / "matches.json",
        "MATCH_TRACE": tmp_path / "trace.json",
        "ANCHORS": tmp_path / "anchors.jsonl",  # the user's verdicts: none unless a test says
        "_matching": lambda: (heights, [], lambda year: SimpleNamespace(blocks=None)),
        "_matches": lambda: previous,
        "_situated": lambda: {},
        "_exported": lambda photo_id: exported,
        "_inputs": lambda scene, photo_id: "i",
        "_image": lambda photo_id: "h",
        "_place_of": lambda me: (0.0, 0.0),
        "_where": lambda me, heights, now: None,
        "_then_year": lambda me: 1900,
        "_visit": lambda me: me.id,
        "_traced": lambda trace: {"verdict": trace.verdict},
        "_grey": lambda image: image,
        "_views": lambda kept=None, matcher="disk": SimpleNamespace(matcher=matcher),
    }.items():
        monkeypatch.setattr(cli, name, value)
    monkeypatch.setattr(cli.db, "connect", lambda path: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli.export, "photos", lambda *args: photos)
    nothing = lambda *a: None  # noqa: E731
    no_store = SimpleNamespace(features=nothing, save_observed=nothing, save_points=nothing)
    monkeypatch.setattr(cli.chain, "Store", lambda folder, matcher: no_store)
    monkeypatch.setattr(cli.outlines, "contrast", lambda *args: 0.1)
    monkeypatch.setattr(
        cli.match, "observe", lambda image, near, blocks, views, *rest: views.matcher
    )
    monkeypatch.setattr(
        cli.match, "place", lambda observed, where, trace: (places(observed), [], None)
    )


CAMERA = match.Camera(1.0, 2.0, 1.5, 90.0, 0.5, 60.0, agree=30, panoramas=2)


def test_match_tries_lomag_then_disk_on_the_photos_left_and_keeps_what_each_did(
    tmp_path, monkeypatch
):
    _match_world(  # LoMa-G finds nothing; DISK places both
        tmp_path, monkeypatch, {}, lambda matcher: CAMERA if matcher == "disk" else None
    )

    cli.match_photos(photo=None, again=False, force=True)

    kept = json.loads((tmp_path / "matches.json").read_text())
    assert {k: (m["matcher"], m["tried"]) for k, m in kept.items()} == {
        "pm:1": ("disk", ["lomag", "disk"]),
        "pm:2": ("disk", ["lomag", "disk"]),
    }


def test_a_redo_that_finds_nothing_keeps_the_placement_made_from_these_inputs(
    tmp_path, monkeypatch
):
    before = _kept(matcher="disk", tried=["disk"])  # placed from today's inputs ("i")
    _match_world(tmp_path, monkeypatch, {"pm:1": dict(before)}, lambda matcher: None)

    cli.match_photos(photo=["pm:1"], again=False, force=True)

    assert json.loads((tmp_path / "matches.json").read_text())["pm:1"] == before


@pytest.mark.parametrize(
    "before, said",
    [
        (_kept(inputs="other", matcher="disk", tried=["disk"]), []),  # placed from others
        (_kept(matcher="disk", tried=["disk"]), [_said("no", "i", x=1.0, y=2.0)]),  # turned down
    ],
)
def test_a_run_that_finds_nothing_replaces_a_stale_or_refused_placement(
    tmp_path, monkeypatch, before, said
):
    _match_world(tmp_path, monkeypatch, {"pm:1": dict(before)}, lambda matcher: None)
    (tmp_path / "anchors.jsonl").write_text(
        "".join(json.dumps({**r, "photo_id": "pm:1"}) + "\n" for r in said)
    )

    cli.match_photos(photo=None, again=False, force=True)

    kept = json.loads((tmp_path / "matches.json").read_text())["pm:1"]
    assert kept == {"inputs": "i", "tried": ["lomag", "disk"]}


def _candidate(agree, why, x=10.0):
    return {
        "x": x,
        "y": 0.0,
        "heading": 90.0,
        "fov": 50.0,
        "horizon": 0.5,
        "agree": agree,
        "spread_m": None,
        "why": why,
    }


def test_a_camera_turned_down_with_many_points_agreeing_is_proposed():
    trace = {
        "a": {"candidates": [_candidate(30, "x"), _candidate(60, "position incertaine")]},
        "b": {"candidates": [_candidate(40, "position incertaine")]},  # too few agree
        "c": {"candidates": [_candidate(80, "ne peut se tenir là")]},  # placed by a matcher
        "d": {"candidates": [_candidate(80, "rivale")]},  # traced from another image
    }
    matches = {
        "a": {"inputs": "i", "tried": ["lomag"]},
        "b": {"inputs": "i", "tried": ["lomag"]},
        "c": {"x": 0.0, "inputs": "i:lomag", "matcher": "lomag"},
        "d": {"inputs": "old", "tried": ["lomag"]},
    }

    proposed = cli._proposals(trace, matches, lambda k: (10.0, 3.0), lambda k: "i", 2.4)

    assert set(proposed) == {"a"}
    a = proposed["a"]
    assert (a["agree"], a["why"], a["above_street"], a["moved_m"]) == (
        60,
        "position incertaine",
        2.4,
        3.0,
    )
    assert a["inputs"].startswith("i:proposed-")


@pytest.mark.parametrize(
    ("verdict", "inputs_now", "placed"),
    [("yes", "i", True), ("no", "i", False), (None, "i", False), ("yes", "j", False)],
)
def test_a_proposed_camera_is_a_placement_once_said_yes_to(
    tmp_path, monkeypatch, verdict, inputs_now, placed
):
    photo = tmp_path / "old.jpg"
    cv2.imwrite(str(photo), np.zeros((8, 8, 3), np.uint8))
    proposal = cli._proposals(
        {"a": {"candidates": [_candidate(60, "position incertaine")]}},
        {"a": {"inputs": "s:i", "tried": ["lomag"]}},
        lambda k: (0.0, 0.0),
        lambda k: "s:i",
        2.4,
    )
    (tmp_path / "proposed.json").write_text(json.dumps(proposal))
    monkeypatch.setattr(cli, "PROPOSED_CAMERAS", tmp_path / "proposed.json")
    monkeypatch.setattr(cli, "_matches", dict)
    monkeypatch.setattr(cli, "_exported", lambda photo_id: photo)
    monkeypatch.setattr(cli, "_scenes_now", lambda: {"a": "s"})
    monkeypatch.setattr(cli, "_photo_part", lambda photo_id: inputs_now)  # the photo now
    a = proposal["a"]
    camera = {"x": a["x"], "y": a["y"], "heading": a["heading"]}
    said = [{"photo_id": "a", "verdict": verdict, "inputs": a["inputs"], "camera": camera}]
    (tmp_path / "anchors.jsonl").write_text(json.dumps(said[0]) + "\n" if verdict else "")
    monkeypatch.setattr(cli, "ANCHORS", tmp_path / "anchors.jsonl")

    assert ("a" in cli._situated()) == placed


def test_a_yes_is_to_the_camera_proposed_not_to_the_next_one(tmp_path, monkeypatch):
    photo = tmp_path / "old.jpg"
    photo.write_bytes(b"x")
    run = {"a": {"inputs": "i", "tried": ["lomag"]}}
    first = cli._proposals(
        {"a": {"candidates": [_candidate(60, "rivale", x=10.0)]}},
        run,
        lambda k: (0, 0),
        lambda k: "i",
        2,
    )
    later = cli._proposals(
        {"a": {"candidates": [_candidate(70, "rivale", x=50.0)]}},
        run,
        lambda k: (0, 0),
        lambda k: "i",
        2,
    )
    monkeypatch.setattr(cli, "PROPOSED_CAMERAS", tmp_path / "proposed.json")
    monkeypatch.setattr(cli, "_matches", dict)
    monkeypatch.setattr(cli, "_exported", lambda photo_id: photo)
    monkeypatch.setattr(cli, "_scenes_now", lambda: {"a": "s"})
    monkeypatch.setattr(cli, "_inputs", lambda scene, photo_id: "i")
    a = first["a"]
    yes = {
        "photo_id": "a",
        "verdict": "yes",
        "inputs": a["inputs"],
        "camera": {"x": a["x"], "y": a["y"], "heading": a["heading"]},
    }
    (tmp_path / "anchors.jsonl").write_text(json.dumps(yes) + "\n")
    monkeypatch.setattr(cli, "ANCHORS", tmp_path / "anchors.jsonl")

    (tmp_path / "proposed.json").write_text(json.dumps(first))
    assert "a" in cli._situated()  # the camera said yes to
    (tmp_path / "proposed.json").write_text(json.dumps(later))
    assert "a" not in cli._situated()  # another, 40 m away


def test_a_trace_made_from_another_image_proposes_nothing():
    trace = {"a": {"inputs": "old", "candidates": [_candidate(60, "rivale")]}}

    assert (
        cli._proposals(
            trace, {"a": {"inputs": "i", "tried": []}}, lambda k: (0, 0), lambda k: "i", 2
        )
        == {}
    )


def test_a_no_to_a_proposed_camera_leaves_the_photo_to_chaining(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_verdicts",
        lambda: {
            "a": {"verdict": "no", "inputs": "i:proposed-abc"},
            "b": {"verdict": "no", "inputs": "i"},
        },
    )

    assert cli._turned_down() == {"b"}


def test_a_proposed_camera_judged_already_is_not_offered_again(tmp_path):
    def camera(n):
        return {"inputs": f"s:i:proposed-{n}", "x": 10.0 * n, "y": 0.0, "heading": 90.0}

    proposed = {"a": camera(1), "b": camera(2), "c": camera(3)}
    judged = _judged(
        tmp_path,
        {**_said("no", "s:i:proposed-1", x=10.0), "photo_id": "a"},  # that very camera
        {**_said("no", "s:i:proposed-0", x=0.0), "photo_id": "b"},  # another, 20 m away
    )

    assert set(cli._to_judge(proposed, judged)) == {"b", "c"}


def test_a_seeds_image_is_read_where_it_was_kept_and_never_published(monkeypatch, tmp_path):
    from flaneur import cli, export

    monkeypatch.setattr(cli, "WEB_DATA", tmp_path / "site")
    monkeypatch.setattr(cli, "SEEDED", tmp_path / "seeded")
    for folder, photo_id in (("site", "pm:shown"), ("seeded", "pm:seed")):
        path = tmp_path / folder / export.image_name(photo_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    assert cli._exported("pm:shown") == tmp_path / "site" / export.image_name("pm:shown")
    assert cli._exported("pm:seed") == tmp_path / "seeded" / export.image_name("pm:seed")


def _panorama(n, x, y=0.0, roll=0.0):
    from pathlib import Path

    from flaneur import match

    return match.Panorama(f"p{n}", x, y, 0.0, 0.0, Path(f"p{n}.jpg"), roll, 0.0)


def _scene(near_by, here=(0.0, 0.0), code="code"):
    from flaneur import cli

    return cli._scene_of(cli._compared_with(near_by, here), here, code)


def test_a_photos_fingerprint_holds_when_panoramas_it_does_not_compare_are_added():
    from flaneur import cli

    near_by = [_panorama(n, 2.0 * n) for n in range(cli.MATCH_PANORAMAS)]  # all within reach

    assert _scene(near_by) == _scene(near_by + [_panorama(99, 500.0)])  # far away
    assert _scene(near_by) == _scene(near_by + [_panorama(98, 50.0)])  # in reach, not among 16


def test_a_photos_fingerprint_changes_with_what_its_match_is_found_from():
    near_by = [_panorama(n, 10.0 * n) for n in range(1, 4)]
    before = _scene(near_by)

    assert _scene(near_by + [_panorama(9, 5.0)]) != before  # a nearer panorama
    assert _scene([_panorama(1, 10.5), *near_by[1:]]) != before  # one moved
    assert _scene([_panorama(1, 10.0, roll=1.0), *near_by[1:]]) != before  # one levelled again
    assert _scene(near_by, here=(0.5, 0.0)) != before  # the photo's place moved
    assert _scene(near_by, code="other") != before  # the code that finds it changed


def test_the_page_shows_only_the_word_that_holds_for_each_photo_as_placed(monkeypatch, tmp_path):
    _judged(tmp_path, _said("yes", "s:img"))
    monkeypatch.setattr(cli, "ANCHORS", tmp_path / "anchors.jsonl")
    here = {"x": 0.3, "y": 0.0, "heading": 90.5}

    def placed(inputs, camera):
        return [{"id": "a", "inputs": inputs, "camera": camera}]

    assert cli._said_on(placed("t:img", here))["a"]["verdict"] == "yes"  # its camera, hardly moved
    assert cli._said_on(placed("t:img", {**here, "x": 5.0})) == {}  # placed elsewhere: judge again


def test_each_photos_scene_is_found_from_the_panoramas_match_compares_it_with(monkeypatch):
    from types import SimpleNamespace

    from flaneur import cli

    rng = np.random.default_rng(0)
    near_by = [_panorama(n, *rng.integers(-60, 60, 2).astype(float)) for n in range(300)]
    near_by += [_panorama(400 + n, 3.0, 4.0) for n in range(20)]  # ties, more than 16
    walk = [
        SimpleNamespace(id=str(n), here=(float(x), float(y)))
        for n, (x, y) in enumerate([(0.0, 0.0), (3.0, 4.0), (50.0, -50.0), (500.0, 0.0)])
    ]
    monkeypatch.setattr(cli, "_place_of", lambda me: me.here)
    monkeypatch.setattr(cli, "_code", lambda: "code")

    assert cli._scenes(walk, near_by) == {me.id: _scene(near_by, me.here) for me in walk}


def test_a_verdict_follows_its_camera_found_again_from_other_panoramas(tmp_path):
    judged = _judged(tmp_path, _said("yes", "oldscene:img:lomag"))

    def placed(inputs, x=0.0, y=0.0, heading=90.0):
        return {"inputs": inputs, "x": x, "y": y, "heading": heading}

    assert (
        judged.on("a", placed("newscene:img", 0.6, 0.6, 91.5))["verdict"] == "yes"
    )  # hardly moved
    assert judged.on("a", placed("newscene:img", x=1.2)) is None  # moved
    assert judged.on("a", placed("newscene:img", heading=93.0)) is None  # turned
    assert judged.on("a", placed("newscene:other")) is None  # another crop of the photo
    assert judged.on("a", placed("newscene:img:ab12cd34")) is None  # landmarks marked since
    assert judged.on("a", {"inputs": "newscene:img"}) is None  # not placed: its inputs only


def test_the_last_word_on_a_camera_holds(tmp_path):
    judged = _judged(tmp_path, _said("yes", "s1:img"), _said("no", "s2:img", x=0.3))

    assert (
        judged.on("a", {"inputs": "s3:img", "x": 0.1, "y": 0.0, "heading": 90.0})["verdict"] == "no"
    )
    assert (
        judged.on("a", {"inputs": "s1:img", "x": 50.0, "y": 0.0, "heading": 0.0})["verdict"]
        == "yes"
    )


def test_a_yes_holds_while_the_photo_is_the_same(tmp_path):
    judged = _judged(tmp_path, _said("yes", "oldscene:img"))
    placed = {"inputs": "oldscene:img", "x": 0.0, "y": 0.0, "heading": 90.0}

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(cli, "_photo_part", lambda photo_id: "img")
        assert judged.holds("a", placed)
        assert not _judged(tmp_path, _said("near", "oldscene:img")).holds("a", placed)
        patched.setattr(cli, "_photo_part", lambda photo_id: "recropped")
        assert not judged.holds("a", placed)


def _placed_now(monkeypatch, tmp_path, kept, *said, photo="img"):
    """What is placed now (`_situated`, `_confirmed`) of one photo "a" stored as `kept`, its
    scene now "new" and the photo itself now `photo`, the user having `said` these."""
    image = tmp_path / "old.jpg"
    cv2.imwrite(str(image), np.zeros((8, 8, 3), np.uint8))
    anchors = tmp_path / "anchors.jsonl"
    anchors.write_text("".join(json.dumps(r) + "\n" for r in said))
    monkeypatch.setattr(cli, "_matches", lambda: {"a": kept})
    monkeypatch.setattr(cli, "_exported", lambda photo_id: image)
    monkeypatch.setattr(cli, "_scenes_now", lambda: {"a": "new"})
    monkeypatch.setattr(cli, "_photo_part", lambda photo_id: photo)
    monkeypatch.setattr(cli, "PROPOSED_CAMERAS", tmp_path / "proposed.json")
    monkeypatch.setattr(cli, "ANCHORS", anchors)
    return set(cli._situated()), set(cli._confirmed())


def test_a_placement_said_yes_to_stands_while_the_photo_is_the_same(monkeypatch, tmp_path):
    kept, yes = _kept(inputs="old:img"), _said("yes", "old:img", x=1.0, y=2.0)

    assert _placed_now(monkeypatch, tmp_path, kept, yes) == ({"a"}, {"a"})
    assert _placed_now(monkeypatch, tmp_path, kept, yes, photo="recropped") == (set(), set())
    assert _placed_now(monkeypatch, tmp_path, kept) == (set(), set())  # stale, not judged


def test_a_verdict_given_to_a_camera_holds_for_it_found_again(monkeypatch, tmp_path):
    found_again = _kept(inputs="new:img", x=1.3)  # 0.3 m from the camera judged

    no = _said("no", "old:img", x=1.0, y=2.0)
    yes = _said("yes", "old:img", x=1.0, y=2.0)

    assert _placed_now(monkeypatch, tmp_path, found_again) == ({"a"}, set())  # to judge
    assert _placed_now(monkeypatch, tmp_path, found_again, no) == (set(), set())
    assert _placed_now(monkeypatch, tmp_path, found_again, yes) == ({"a"}, {"a"})


ELSEWHERE = match.Camera(30.0, 2.0, 1.5, 90.0, 0.5, 60.0, agree=30, panoramas=2)  # 29 m away


@pytest.mark.parametrize(
    ("photo", "again", "found"),
    [(None, True, ELSEWHERE), (["pm:1"], False, ELSEWHERE), (["pm:1"], False, None)],
)
def test_matching_never_takes_away_a_placement_said_yes_to(
    tmp_path, monkeypatch, photo, again, found
):
    before = _kept(inputs="old:h", matcher="disk", tried=["disk"])  # other panoramas then
    _match_world(tmp_path, monkeypatch, {"pm:1": dict(before)}, lambda matcher: found)
    yes = {**_said("yes", "old:h", x=1.0, y=2.0), "photo_id": "pm:1"}
    (tmp_path / "anchors.jsonl").write_text(json.dumps(yes) + "\n")

    cli.match_photos(photo=photo, again=again, force=True)

    assert json.loads((tmp_path / "matches.json").read_text())["pm:1"] == before


def test_an_areas_check_holds_while_panoramas_are_added_elsewhere():
    from flaneur import pilot

    home, other = pilot.AREAS["saint-severin"].centroid, pilot.AREAS["cite"].centroid
    near_by = [_panorama(n, home.x + 10.0 * n, home.y) for n in range(3)]
    before = cli._check_setup("disk", "saint-severin", near_by)

    assert (
        cli._check_setup("disk", "saint-severin", [*near_by, _panorama(9, other.x, other.y)])
        == before
    )
    assert (
        cli._check_setup("disk", "saint-severin", [*near_by, _panorama(8, home.x, home.y + 5)])
        != before
    )


def test_a_photo_belongs_to_the_area_it_stands_in_else_the_nearest():
    from flaneur import pilot

    cite = pilot.AREAS["cite"].centroid

    assert cli._area_of((cite.x, cite.y)) == "cite"
    assert cli._area_of((cite.x + 2000.0, cite.y)) == "cite"  # east of both: the Cité is nearer


@pytest.mark.parametrize(("passed_in", "placed"), [("cite", True), ("saint-severin", False)])
def test_a_photo_is_matched_only_by_matchers_checked_in_its_area(
    tmp_path, monkeypatch, passed_in, placed
):
    _match_world(tmp_path, monkeypatch, {}, lambda matcher: CAMERA)
    monkeypatch.setattr(cli, "MATCH_CHECK", tmp_path / "match_check.json")
    monkeypatch.setattr(cli, "_area_of", lambda here: "cite")  # where the photos stand
    check = {passed_in: {"passed": True, "setup": cli._check_setup("disk", passed_in, [])}}
    (tmp_path / "match_check.json").write_text(json.dumps(check))  # DISK's, checked there only

    cli.match_photos(photo=None, again=False, force=False)

    kept = json.loads((tmp_path / "matches.json").read_text()) if placed else {}
    assert ("x" in kept.get("pm:1", {})) == placed
    assert kept.get("pm:1", {}).get("matcher", "disk") == "disk"
