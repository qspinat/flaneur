import json
import subprocess
import threading
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from flaneur import (
    chain,
    cli,
    confirming,
    depth,
    facades,
    match,
    placed,
    placing,
    poses,
    registration,
    segment,
)
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


def test_a_chained_photo_keeps_its_points_apart_from_match_s(tmp_path, monkeypatch):
    camera = match.Camera(1.0, 2.0, 1.5, 90.0, 0.5, 60.0, agree=30, panoramas=2)
    points = match.Points(
        np.zeros((1, 2)), np.zeros((1, 3)), np.array([frozenset({"photo:a"})], dtype=object)
    )
    anchor, me = SimpleNamespace(id="a", inputs="ia"), SimpleNamespace(id="b")
    work = SimpleNamespace(
        round_zero={"a": anchor},
        walk=[me],
        elsewhere=set(),
        turned_down=set(),
        refused=set(),
        anchors=lambda me, placed: [anchor],
        observed=lambda me: "observed",
        place=lambda me, observed, anchors, trace: match.Found(camera, np.zeros((0, 6)), points),
        scenes={"b": "scene"},
        by_id={"b": me},
        where={"b": SimpleNamespace(around=(0.0, 0.0))},
        heights=SimpleNamespace(street=lambda x, y: 0.0),
        then=lambda year: SimpleNamespace(blocks=None),
        anchor=lambda photo_id, camera, points, inputs: chain.Anchor(
            photo_id, camera, None, None, points, inputs
        ),
        store=SimpleNamespace(),  # match's DISK store: never written by chaining
    )
    for name, value in {
        "_inputs": lambda scene, photo_id: "i",
        "_exported": lambda photo_id: "old.jpg",
        "_grey": lambda image: image,
        "_then_year": lambda me: 1900,
    }.items():
        monkeypatch.setattr(cli, name, value)
    monkeypatch.setattr(cli.cv2, "imread", lambda path: np.zeros((8, 8, 3), np.uint8))
    monkeypatch.setattr(cli.outlines, "contrast", lambda *args: 0.1)
    monkeypatch.setattr(cli, "CHAINED_POINTS", tmp_path)

    results, placed = cli._rounds(work, set())
    assert not any(tmp_path.iterdir())  # not while chaining: a pass may set it aside
    cli._for_pages(work, results, placed)

    inputs = results["b"]["situated"]["inputs"]
    assert cli.chain.Store(tmp_path, None).points("b", inputs).pixels.tolist() == [[0.0, 0.0]]
    assert results["b"]["situated"]["outline"] == 0.1


def test_a_photo_placed_without_anchors_keeps_nothing_of_its_own(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "CHAINED_POINTS", tmp_path)
    points = match.Points(np.ones((1, 2)), np.ones((1, 3)), np.array([frozenset({"p"})], object))

    cli._keep_chained("b", "i", points)  # its inputs: those of a placement against today

    assert not any(tmp_path.iterdir())


def test_a_chained_placements_points_are_read_where_the_chaining_kept_them(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "CHAINED_POINTS", tmp_path)
    m = replace(_situated_at_origin(), inputs="i:from-ab12cd34")
    points = match.Points(np.ones((1, 2)), np.ones((1, 3)), np.array([frozenset({"p"})], object))
    cli._keep_chained("pm:1", m.inputs, points)

    store, inputs = cli._kept_store(m, {"matcher": cli.LOMAG})  # match's other try: aside

    assert store.points("pm:1", inputs).pixels.tolist() == [[1.0, 1.0]]
    assert cli._unchained(inputs) == "i"  # its observation: the photo's own, as matched


def test_a_photo_chained_without_anchors_shows_no_points_not_match_s(tmp_path, monkeypatch):
    disk = cli.MATCHERS[cli.DISK]._replace(photos=tmp_path)
    monkeypatch.setattr(cli, "MATCHERS", {**cli.MATCHERS, cli.DISK: disk})
    m = replace(_situated_at_origin(), inputs="i")  # as match's own placement, turned down
    points = match.Points(np.ones((1, 2)), np.ones((1, 3)), np.array([frozenset({"p"})], object))
    cli.chain.Store(tmp_path, None).save_points("pm:1", "i", points)

    assert cli._placement_points("pm:1", m, None, chained=True) is None
    assert cli._placement_points("pm:1", m, None, chained=False) is not None


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


def test_a_pair_kept_under_the_other_s_name_counts_for_both(tmp_path):
    a = match.Panorama("a", 0.0, 0.0, 0.0, 10.0, tmp_path / "a.jpg")
    for n in range(2):
        (tmp_path / f"x{n}_0.0_0.00_0.00__{cli._levelled_name(a)}.npz").touch()
    (tmp_path / f"{cli._levelled_name(a)}__x2_0.0_0.00_0.00.npz").touch()

    assert cli._paired_homes([a], tmp_path) == [a]


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


def _scenes(monkeypatch, tmp_path, garbled: frozenset = frozenset(), seen=None, z=0.0, **options):
    """Scenes around a panorama with a neighbour beside it (and one too far), their matches
    the street's two facades seen from each (those with a neighbour `garbled`, at random); and
    the pairs matched."""
    home = match.Panorama("home", 0.0, 0.0, z, 0.0, tmp_path / "h.jpg")
    beside = match.Panorama("beside", 2.0, 1.0, z, 0.0, tmp_path / "b.jpg")  # in the street
    far = match.Panorama("far", 40.0, 0.0, z, 0.0, tmp_path / "f.jpg")
    asked = []

    def matched(a, b, views):
        asked.append((a.id, b.id))
        street = STREET + [0.0, 0.0, z]  # the panoramas' frame
        seen_b = _units(street - [b.x, b.y, b.z])
        if b.id in garbled:
            seen_b = _units(np.random.default_rng(0).normal(size=STREET.shape))
        return _units(street - [a.x, a.y, a.z]), seen_b

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
        **options,
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


def test_a_pair_matched_one_way_serves_the_other_panorama(monkeypatch, tmp_path):
    asked, scenes, home = _scenes(monkeypatch, tmp_path)
    own = scenes.scene("home", None)  # home with beside: matched, kept as home__beside

    seen_back = scenes.scene("beside", None)  # beside with home: the same pair, read back
    beside = scenes.by_id["beside"]

    assert asked == [("home", "beside")]
    assert scenes._made(beside, home) and len(seen_back.walls) == len(own.walls) == 2
    assert len(scenes.details(beside, [home], made_only=True)) > 0


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


def _camera(x=0.0, y=0.0, heading=90.0, fov=60.0, horizon=0.5, above_street=1.5):
    return {
        "x": x,
        "y": y,
        "heading": heading,
        "fov": fov,
        "horizon": horizon,
        "above_street": above_street,
    }


def _said(verdict, inputs, x=0.0, y=0.0, heading=90.0, fov=60.0, horizon=0.5):
    return {
        "photo_id": "a",
        "verdict": verdict,
        "inputs": inputs,
        "camera": _camera(x, y, heading, fov, horizon),
    }


def _at(inputs, x=0.0, y=0.0, heading=90.0, fov=60.0, horizon=0.5, above_street=1.5):
    """A placement (its inputs and camera) as a verdict is read on it."""
    return {"inputs": inputs, **_camera(x, y, heading, fov, horizon, above_street)}


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


def test_a_photo_placed_against_today_is_shown_so_unless_only_its_chained_camera_is_said_yes():
    today = match.Situated(**_kept(inputs="i"))
    situated = {"a": today, "d": today}
    row = lambda frm: {"situated": _kept(inputs="i:from-ab12cd34"), "round": 1, "from": frm}  # noqa: E731
    chained = {"photos": {"a": row(["b"]), "c": row(["a"]), "d": row(["b"])}}
    judged = SimpleNamespace(said_yes=lambda k, p: k == "d" and p["inputs"] == "i:from-ab12cd34")

    shown, rounds = cli._with_chained(situated, chained, {"a", "c", "d"}, judged)

    assert shown["a"] == today and rounds["a"] == {"round": 0, "from": []}
    assert shown["c"] == match.Situated(**_kept(inputs="i:from-ab12cd34"))
    assert rounds["c"] == {"round": 1, "from": ["a"]}
    assert rounds["d"]["round"] == 1  # its chained camera said yes to, not today's


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

    views = cli._viewed_from([far], m, (), set())
    row = cli._sheet_row("pm:1", m, heights, views, lambda path: np.zeros((512, 1024, 3), np.uint8))

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
        "MATCH_CHECK": tmp_path / "match_check.json",  # none passed unless a test says
        "CHECKED_AREAS": tmp_path / "checked_areas.json",
        "_matching": lambda: (heights, [], lambda year: SimpleNamespace(blocks=None)),
        "_colmap_fixed": set,
        "_wall_scenes": lambda near_by, fixed: "scenes",
        "_wall_depth": lambda scenes, but=frozenset(): f"walls of {scenes}",
        "_photo_walls": lambda scenes, near_by: lambda here: "walls",
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
    monkeypatch.setattr(
        cli.match, "focus", lambda photo, observed, found, *rest: (observed, found, None)
    )


CAMERA = match.Camera(1.0, 2.0, 1.5, 90.0, 0.5, 60.0, agree=30, panoramas=2)


def test_match_tries_lomag_then_disk_on_the_photos_left_and_keeps_what_each_did(
    tmp_path, monkeypatch
):
    _match_world(  # LoMa-G finds nothing; DISK places both
        tmp_path, monkeypatch, {}, lambda matcher: CAMERA if matcher == "disk" else None
    )

    cli.match_photos(photo=None, again=False)

    kept = json.loads((tmp_path / "matches.json").read_text())
    assert {k: (m["matcher"], m["tried"]) for k, m in kept.items()} == {
        "pm:1": ("disk", ["lomag", "disk"]),
        "pm:2": ("disk", ["lomag", "disk"]),
    }


def test_a_match_stopped_midway_keeps_the_trace_of_the_photos_it_went_through(
    tmp_path, monkeypatch
):
    calls = []

    def places(matcher):
        calls.append(matcher)
        if len(calls) > 1:  # the night's hour comes during the second photo
            raise KeyboardInterrupt
        return None

    _match_world(tmp_path, monkeypatch, {}, places)

    with pytest.raises(KeyboardInterrupt):
        cli.match_photos(photo=None, again=False)

    assert list(json.loads((tmp_path / "trace.json").read_text())) == ["pm:1"]


def test_a_redo_that_finds_nothing_keeps_the_placement_made_from_these_inputs(
    tmp_path, monkeypatch
):
    before = _kept(matcher="disk", tried=["disk"])  # placed from today's inputs ("i")
    _match_world(tmp_path, monkeypatch, {"pm:1": dict(before)}, lambda matcher: None)

    cli.match_photos(photo=["pm:1"], again=False)

    kept = json.loads((tmp_path / "matches.json").read_text())["pm:1"]
    assert kept == {**before, "fitted_from": "walls"}  # tried with these walls


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

    cli.match_photos(photo=None, again=False)

    kept = json.loads((tmp_path / "matches.json").read_text())["pm:1"]
    assert kept == {"inputs": "i", "tried": ["lomag", "disk"], "fitted_from": "walls"}


def _alone(tmp_path, monkeypatch, before: dict) -> list[str]:
    """`match` over pm:1 alone, placed as `before`, no matcher finding anything: the matchers
    tried, in order."""
    tries = []
    _match_world(tmp_path, monkeypatch, {"pm:1": dict(before)}, lambda m: tries.append(m))
    monkeypatch.setattr(cli.export, "photos", lambda *args: [SimpleNamespace(id="pm:1")])
    return tries


def test_a_near_placement_stays_when_the_other_matcher_finds_nothing(tmp_path, monkeypatch):
    before = _kept(inputs="i:lomag", matcher="lomag", tried=["lomag"], fitted_from="walls")
    tries = _alone(tmp_path, monkeypatch, before)
    said = _said("near", "i:lomag", x=1.0, y=2.0)
    (tmp_path / "anchors.jsonl").write_text(json.dumps({**said, "photo_id": "pm:1"}) + "\n")

    cli.match_photos(photo=None, again=False)
    kept = json.loads((tmp_path / "matches.json").read_text())["pm:1"]
    monkeypatch.setattr(cli, "_matches", lambda: {"pm:1": kept})
    cli.match_photos(photo=None, again=False)  # the next night

    assert (kept["x"], kept["tried"]) == (before["x"], ["lomag", "disk"])  # still near, shown
    assert tries == ["disk"]  # DISK tried once, not every night


def test_a_placement_marked_since_stays_when_its_redo_finds_nothing(tmp_path, monkeypatch):
    image = "1234567890abcdef"
    before = _kept(inputs=f"s:{image}", matcher="disk", tried=["disk"], fitted_from="walls")
    tries = _alone(tmp_path, monkeypatch, before)
    monkeypatch.setattr(cli, "_inputs", lambda scene, photo_id: f"s:{image}:ab12cd34")

    cli.match_photos(photo=None, again=False)
    kept = json.loads((tmp_path / "matches.json").read_text())["pm:1"]
    monkeypatch.setattr(cli, "_matches", lambda: {"pm:1": kept})
    cli.match_photos(photo=None, again=False)  # the next night

    assert (kept["x"], kept["inputs"]) == (before["x"], before["inputs"])
    assert tries == ["lomag", "disk"]  # tried with the marks once, not every night


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


def test_a_proposed_camera_judged_already_is_not_offered_again(monkeypatch, tmp_path):
    def camera(n):
        return {"inputs": f"s:i:proposed-{n}", "matcher": f"proposed-{n}", **_camera(10.0 * n)}

    proposed = {"a": camera(1), "b": camera(2), "c": camera(3)}
    (tmp_path / "proposed.json").write_text(json.dumps(proposed))
    monkeypatch.setattr(cli, "PROPOSED_CAMERAS", tmp_path / "proposed.json")
    monkeypatch.setattr(cli, "_exported", lambda photo_id: tmp_path / "proposed.json")
    monkeypatch.setattr(cli, "_inputs", lambda scene, photo_id: "s:i")
    judged = _judged(
        tmp_path,
        {**_said("no", "s:i:proposed-1", x=10.0), "photo_id": "a"},  # that very camera
        {**_said("no", "s:i:proposed-0", x=0.0), "photo_id": "b"},  # another, 20 m away
    )

    assert set(cli._proposed_now(dict.fromkeys(proposed, "s"), judged)) == {"b", "c"}


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

    return cli._scene_of(cli._reach(near_by, cli.SCENE_REACH_M, None)(here), here, code)


def test_a_photos_fingerprint_holds_when_panoramas_out_of_its_reach_are_added():
    from flaneur import cli

    near_by = [_panorama(n, 2.0 * n) for n in range(cli.MATCH_PANORAMAS)]  # all within reach

    assert _scene(near_by) == _scene(near_by + [_panorama(99, 500.0)])  # far away
    assert _scene(near_by) == _scene(near_by + [_panorama(98, 70.0)])  # beyond its focusing pass


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
    here = _camera(x=0.3, heading=90.5)

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

    assert judged.on("a", _at("newscene:img", 0.6, 0.6, 91.5, 61.5, 0.51))["verdict"] == "yes"
    assert judged.on("a", _at("newscene:img", x=1.2)) is None  # moved
    assert judged.on("a", _at("newscene:img", heading=93.0)) is None  # turned
    assert judged.on("a", _at("newscene:img", fov=63.0)) is None  # wider: framed otherwise
    assert judged.on("a", _at("newscene:img", horizon=0.53)) is None  # tilted
    assert judged.on("a", _at("newscene:img", above_street=3.0)) is None  # a floor up
    assert judged.on("a", _at("newscene:other")) is None  # another crop of the photo
    assert judged.on("a", _at("newscene:img:ab12cd34")) is None  # landmarks marked since
    assert judged.on("a", {"inputs": "newscene:img"}) is None  # not placed: its inputs only


def test_a_verdict_does_not_follow_its_inputs_to_another_camera(tmp_path):
    judged = _judged(tmp_path, _said("no", "s1:img"))  # new walls since: the same inputs

    assert judged.on("a", _at("s1:img", 0.6, 0.6))["verdict"] == "no"  # found again
    assert judged.on("a", _at("s1:img", x=4.0)) is None  # another camera: shown again


def test_a_verdict_kept_without_its_cameras_width_holds_for_its_inputs_only(tmp_path):
    old = _said("yes", "s1:img")
    old["camera"] = {"x": 0.0, "y": 0.0, "heading": 90.0}  # as kept before 11/11, evening
    judged = _judged(tmp_path, old)

    assert judged.on("a", _at("s1:img"))["verdict"] == "yes"
    assert judged.on("a", _at("s2:img")) is None


def test_a_chained_placements_anchors_are_not_taken_for_the_landmarks_marked():
    anchor = SimpleNamespace(id="b", inputs="s:other")

    assert cli._photo_of(cli._placed_from("s:img", [anchor])) == "img"
    assert cli._photo_of(cli._placed_from("s:img:ab12cd34", [anchor])) == "img:ab12cd34"


def test_the_last_word_on_a_camera_holds(tmp_path):
    judged = _judged(tmp_path, _said("yes", "s1:img"), _said("no", "s2:img", x=0.3))

    assert judged.on("a", _at("s3:img", x=0.1))["verdict"] == "no"
    assert judged.on("a", _at("s1:img", x=50.0, heading=0.0))["verdict"] == "yes"  # its inputs


def test_a_yes_taken_back_beside_its_camera_no_longer_follows_it(tmp_path):
    judged = _judged(tmp_path, _said("yes", "s1:img"), _said("no", "s2:img", x=0.5))

    assert judged.on("a", _at("s3:img", x=-0.6)) is None  # 0.6 m from the yes, 1.1 from the no
    assert judged.on("a", _at("s1:img"))["verdict"] == "no"  # the newest word reaching it


def test_a_yes_holds_while_the_photo_is_the_same(tmp_path):
    judged = _judged(tmp_path, _said("yes", "oldscene:img"))
    placed = _at("oldscene:img")

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
NUDGED = match.Camera(1.9, 2.0, 1.5, 90.0, 0.5, 60.0, agree=30, panoramas=2)  # 0.9 m: one by one


@pytest.mark.parametrize(
    ("photo", "again", "found"),
    [
        (None, True, ELSEWHERE),
        (["pm:1"], False, ELSEWHERE),
        (["pm:1"], False, NUDGED),
        (["pm:1"], False, None),
    ],
)
def test_matching_never_takes_away_a_placement_said_yes_to(
    tmp_path, monkeypatch, photo, again, found
):
    before = _kept(inputs="old:h", matcher="disk", tried=["disk"])  # other panoramas then
    _match_world(tmp_path, monkeypatch, {"pm:1": dict(before)}, lambda matcher: found)
    yes = {**_said("yes", "old:h", x=1.0, y=2.0), "photo_id": "pm:1"}
    (tmp_path / "anchors.jsonl").write_text(json.dumps(yes) + "\n")

    cli.match_photos(photo=photo, again=again)

    written = tmp_path / "matches.json"  # none when nothing was matched
    assert not written.exists() or json.loads(written.read_text())["pm:1"] == before


def test_matching_never_redoes_a_photo_whose_proposed_camera_was_said_yes_to(tmp_path, monkeypatch):
    unplaced = {"inputs": "old:h", "tried": ["lomag", "disk"]}
    proposal = _kept(inputs="old:h:proposed-12345678", x=20.0)
    _match_world(tmp_path, monkeypatch, {"pm:1": dict(unplaced)}, lambda matcher: CAMERA)
    (tmp_path / "proposed.json").write_text(json.dumps({"pm:1": proposal}))
    monkeypatch.setattr(cli, "PROPOSED_CAMERAS", tmp_path / "proposed.json")
    yes = {**_said("yes", proposal["inputs"], x=20.0, y=2.0), "photo_id": "pm:1"}
    (tmp_path / "anchors.jsonl").write_text(json.dumps(yes) + "\n")

    cli.match_photos(photo=["pm:1", "pm:2"], again=True)

    kept = json.loads((tmp_path / "matches.json").read_text())
    assert kept["pm:1"] == unplaced
    assert kept["pm:2"]["x"] == CAMERA.x  # the other, matched


def test_a_proposed_camera_said_yes_to_is_proposed_still_whatever_its_panoramas(
    tmp_path, monkeypatch
):
    held = _kept(inputs="old:img:proposed-12345678")
    (tmp_path / "proposed.json").write_text(json.dumps({"a": held, "b": _kept(inputs="old:x")}))
    (tmp_path / "anchors.jsonl").write_text(json.dumps(_said("yes", held["inputs"], 1.0, 2.0)))
    for name, value in {
        "PROPOSED_CAMERAS": tmp_path / "proposed.json",
        "ANCHORS": tmp_path / "anchors.jsonl",
        "MATCH_TRACE": tmp_path / "trace.json",
        "_scenes_now": dict,
        "_matches": dict,
        "_photo_part": lambda photo_id: "img",
    }.items():
        monkeypatch.setattr(cli, name, value)
    monkeypatch.setattr(cli.db, "connect", lambda path: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli.export, "photos", lambda *args: [])

    cli.propose_turned_down()  # none proposed anew: the panoramas changed

    assert json.loads((tmp_path / "proposed.json").read_text()) == {"a": held}


def test_a_proposed_camera_said_yes_to_wins_over_a_placement_not_judged(monkeypatch, tmp_path):
    proposal = _kept(inputs="new:img:proposed-12345678", x=20.0)
    (tmp_path / "proposed.json").write_text(json.dumps({"a": proposal}))
    yes = _said("yes", proposal["inputs"], x=20.0, y=2.0)

    _placed_now(monkeypatch, tmp_path, _kept(inputs="new:img"), yes)  # found since, not judged

    assert cli._situated()["a"].x == 20.0


def test_an_areas_check_passes_only_on_enough_views():
    assert cli._wanting(10, 10, 0.3, 0.5, 0, 30) == []
    assert cli._wanting(3, 3, 0.3, 0.5, 0, 9) == [f"au moins {cli.CHECK_LEAST} vues"]


def test_an_area_without_decoys_is_left_unchecked_and_the_others_checked(tmp_path, monkeypatch):
    from flaneur import pilot

    home, other = pilot.AREAS["saint-severin"].centroid, pilot.AREAS["cite"].centroid
    near_by = [_panorama(n, home.x + n, home.y) for n in range(3)]  # all within FAR_M
    near_by += [_panorama(10 + n, other.x + 230.0 * (n % 2), other.y + n) for n in range(4)]
    near_by += [_panorama(20, other.x + 115.0, other.y)]  # between: no decoy, not drawn
    heights = SimpleNamespace(streets=None, origin_z=0.0)
    for name, value in {
        "MATCH_CHECK": tmp_path / "check.json",
        "CHECK_TRACE": tmp_path / "trace.json",
        "_matching": lambda: (
            heights,
            near_by,
            lambda year: SimpleNamespace(blocks=None, free=None),
        ),
        "_views": lambda matcher="disk": SimpleNamespace(matcher=SimpleNamespace(features=len)),
        "_colmap_fixed": lambda: {p.id for p in near_by},
        "_cut_view": lambda panorama, heading: np.zeros((2, 2, 3), np.uint8),
        "_check_view": lambda panorama, heading: None,
        "_marked_by_hand": lambda view, shown, rng: (),
        "_wall_scenes": lambda near_by, fixed: None,
        "_wall_depth": lambda walls, but=frozenset(): None,
    }.items():
        monkeypatch.setattr(cli, name, value)
    monkeypatch.setattr(cli.db, "connect", lambda path: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli.panoramas, "load", lambda conn: [])
    monkeypatch.setattr(cli.landmarks, "among", lambda blocks: ())
    monkeypatch.setattr(cli.fakes, "mixed", lambda a, b, span: a)
    monkeypatch.setattr(cli.fakes, "aged", lambda image, rng: image)
    monkeypatch.setattr(cli.match, "Where", lambda *args: None)
    monkeypatch.setattr(cli.match, "observe", lambda *args: None)
    monkeypatch.setattr(cli.match, "place", lambda *args: (None, None, None))
    monkeypatch.setattr(
        cli.match, "focus", lambda photo, observed, found, *rest: (observed, found, None)
    )

    cli.match_check(tries=20, seed=0, matcher="disk", area=None)

    assert set(json.loads((tmp_path / "check.json").read_text())) == {"cite"}


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


def test_a_photo_in_an_area_no_check_passes_is_matched_with_every_matcher(tmp_path, monkeypatch):
    _match_world(tmp_path, monkeypatch, {}, lambda matcher: CAMERA if matcher == "disk" else None)

    cli.match_photos(photo=None, again=False)  # no check passed anywhere

    kept = json.loads((tmp_path / "matches.json").read_text())["pm:1"]
    assert (kept["matcher"], kept["tried"]) == ("disk", ["lomag", "disk"])


def test_match_writes_the_areas_checked(tmp_path, monkeypatch):
    _match_world(tmp_path, monkeypatch, {}, lambda matcher: None)
    check = {"cite": {"passed": True, "setup": cli._check_setup("disk", "cite", [])}}
    (tmp_path / "match_check.json").write_text(json.dumps(check))  # DISK's, in the Cité only

    cli.match_photos(photo=None, again=False)

    written = json.loads((tmp_path / "checked_areas.json").read_text())
    assert written == {"saint-severin": [], "cite": ["disk"]}


def test_match_check_stale_only_checks_the_areas_whose_setup_changed(tmp_path, monkeypatch):
    checked = []
    heights = SimpleNamespace(streets=None, origin_z=0.0)
    for name, value in {
        "MATCH_CHECK": tmp_path / "check.json",
        "CHECKED_AREAS": tmp_path / "checked_areas.json",
        "_matching": lambda: (heights, [], lambda year: SimpleNamespace(blocks=None)),
        "_views": lambda matcher="disk": None,
        "_colmap_fixed": set,
        "_wall_scenes": lambda near_by, fixed: None,
        "_check_setup": lambda matcher, area, near_by: "now",
        "_check_area": lambda name, *rest: checked.append(name),
    }.items():
        monkeypatch.setattr(cli, name, value)
    monkeypatch.setattr(cli.landmarks, "among", lambda blocks: ())
    kept = {"cite": {"passed": False, "setup": "now"}, "saint-severin": {"setup": "before"}}
    (tmp_path / "check.json").write_text(json.dumps(kept))  # DISK's

    cli.match_check(tries=20, seed=0, matcher="disk", area=None, stale=True)

    assert checked == ["saint-severin"]  # the Cité's check is as things are: kept, failed or not
    assert json.loads((tmp_path / "checked_areas.json").read_text()) == {
        "saint-severin": [],
        "cite": [],
    }  # what the list says of each area follows the check


def test_match_check_stale_loads_no_matcher_when_every_check_is_current(tmp_path, monkeypatch):
    heights = SimpleNamespace(streets=None, origin_z=0.0)

    def no_matcher(matcher="disk"):
        raise AssertionError("the matcher was loaded for nothing")

    for name, value in {
        "MATCH_CHECK": tmp_path / "check.json",
        "CHECKED_AREAS": tmp_path / "checked_areas.json",
        "_matching": lambda: (heights, [], lambda year: SimpleNamespace(blocks=None)),
        "_views": no_matcher,
        "_check_setup": lambda matcher, area, near_by: "now",
    }.items():
        monkeypatch.setattr(cli, name, value)
    current = {a: {"passed": True, "setup": "now"} for a in ("cite", "saint-severin")}
    (tmp_path / "check.json").write_text(json.dumps(current))

    cli.match_check(tries=20, seed=0, matcher="disk", area=None, stale=True)


def _wall_ahead(distance: float = 20.0, height: float = 5.0):
    """A long wall `distance` m north of the origin (its base on a level street)."""
    from flaneur import blocks

    footprint = [
        (-500.0, distance),
        (500.0, distance),
        (500.0, distance + 10),
        (-500.0, distance + 10),
    ]
    return blocks.Blocks.of([(footprint, 0.0, height)])


def _level(xs, ys):  # the street, level at 0
    return np.zeros(len(xs))


EYE = match.View(0.0, 0.0, 1.5, 0.0, 0.5, 60.0)  # a camera facing north, as found


SHRUNK = 1.0 * match.PHOTO_SIDE / 120  # a photo 120 wide, matched at PHOTO_SIDE


def _points_along_a_wall() -> match.Points:
    """30 points on a wall 20 m north of EYE, seen by q, at their pixels of a 120 px photo."""
    world = np.array([[x, 20.0, 1.5] for x in np.linspace(-8.0, 8.0, 30)])
    pixels = np.array([[(x / 20.0 * 51.96 + 60) * SHRUNK, 60 * SHRUNK] for x, _, _ in world])
    return match.Points(pixels, world, np.array([frozenset({"q"})] * 30, dtype=object))


def test_a_placement_s_pairs_are_drawn_on_the_photo_and_where_the_panorama_sees_them():
    points = _points_along_a_wall()
    world, pixels, shrunk = points.world, points.pixels, SHRUNK
    aside = match.Panorama("p", 3.0, 0.0, 1.5, 0.0, path=None)

    pairs = cli._pairs_shown(points, (120, 120), (60, 60), cli._toward(EYE, aside))

    assert len(pairs) == cli.PAIRS_SHOWN
    first = pairs[0]
    expected = match.project(cli._toward(EYE, aside), world[[first["index"]]], (60, 60))[0]
    assert first["today"] == pytest.approx(list(expected), abs=0.01)
    assert first["old"] == pytest.approx([pixels[first["index"], 0] / shrunk / 2, 30.0], abs=0.01)


def test_the_plan_puts_the_camera_at_its_centre_and_the_panorama_at_its_distance():
    m = match.Situated(10.0, 20.0, 1.5, 90.0, 0.5, 60.0, 30, 2, 3.0, [], "i", 0.1)
    shown = match.Panorama("p", 13.0, 24.0, 2.5, 0.0, path=None)
    near = [[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)], [(900.0, 900.0), (905.0, 900.0), (905.0, 905.0)]]

    plan = cli._plan(m, shown, [(10.0, 30.0)], (12.0, 15.0), near, [(15.0, 26.04, 3.0)])

    assert plan["camera"] == {"heading": 90.0, "fov": 60.0} and plan["reach"] == cli.PLAN_M
    assert plan["points"] == [[5.0, 6.0]]  # where the camera's points stand
    assert plan["panorama"] == {"at": [3.0, 4.0], "away_m": 5.0}
    assert plan["sources"] == [[0.0, 10.0]] and plan["caption"] == [2.0, -5.0]
    assert plan["buildings"] == [[[-10.0, -20.0], [-5.0, -20.0], [-5.0, -15.0]]]  # the far one left


ASIDE = match.Panorama("p", 3.0, 0.0, 1.5, 0.0, path=None)  # 3 m east of the camera
F600 = 300 / np.tan(np.radians(30))  # the focal of a 600 px view 60° wide


def test_what_today_s_view_shows_stands_on_a_wall_else_on_the_street_else_far_off():
    seen = cli._depth_of_view(cli._toward(EYE, ASIDE), (600, 600), None, _wall_ahead(), _level, 100)

    ahead, low, high = seen[2, 2], seen[5, 2], seen[0, 2]  # rows at 49.5, 249.5... 549.5 px
    assert ahead[1] == pytest.approx(20.0, abs=0.01)  # the wall, 20 m north
    assert low[2] == pytest.approx(0.0, abs=0.05) and low[1] < 20.0  # the street
    assert np.hypot(*(high[:2] - ASIDE.x)) > 1e4  # over the wall: far off


def test_the_panorama_s_own_walls_come_before_bd_topo_s():
    fitted = facades.Wall((-500.0, 17.0), (500.0, 17.0), 30.0, "fitted")  # BD TOPO's, set back 3 m
    walls = facades.Scene("p", (ASIDE.x, ASIDE.y, ASIDE.z), 0.0, (fitted,), (), 0)
    view = cli._toward(EYE, ASIDE)

    seen = cli._depth_of_view(view, (600, 600), walls, _wall_ahead(height=40.0), _level, 100)

    assert seen[2, 2, 1] == pytest.approx(17.0, abs=0.01)


def test_the_photo_s_outlines_are_drawn_where_the_panorama_sees_them():
    old = np.zeros((600, 600, 3), np.uint8)
    old[150:330, 440:470] = 255  # an upright on the wall (above its foot, 4° below the horizon)
    today = np.zeros((600, 600, 3), np.uint8)
    wall = _wall_ahead(height=40.0)

    plain = cli._outlines(old, today)
    moved = cli._parallax_outlines(old, today, EYE, cli._toward(EYE, ASIDE), None, wall, _level)

    def column(drawn):  # where the amber lines stand
        return float(np.median(np.nonzero((drawn == cli.OUTLINE_COLOUR).all(axis=2))[1]))

    east = (column(plain) + 0.5 - 300) / F600 * 20.0  # where the camera sees the upright, 20 m off
    seen_from = 299.5 + F600 * (east - ASIDE.x) / 20.0  # and so the panorama, 3 m east of it
    assert column(moved) == pytest.approx(seen_from, abs=3.0)
    assert seen_from < column(plain) - 40


def test_an_outline_against_the_sky_stays_where_it_is_as_anything_far_off():
    old = np.zeros((600, 600, 3), np.uint8)
    old[40:120, 290:310] = 255  # a spire over the roofs
    today = np.zeros((600, 600, 3), np.uint8)
    low = _wall_ahead(height=5.0)

    plain = (cli._outlines(old, today) == cli.OUTLINE_COLOUR).all(axis=2)
    moved = cli._parallax_outlines(old, today, EYE, cli._toward(EYE, ASIDE), None, low, _level)
    moved = (moved == cli.OUTLINE_COLOUR).all(axis=2)

    near = cv2.dilate(plain.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert plain.any() and moved[near].sum() >= 0.9 * moved.sum() >= 0.9 * plain.sum() * 0.5


def test_a_ray_down_meets_bd_topo_s_wall_before_the_street_beyond_the_fitted_walls():
    aside = facades.Wall((-500.0, 60.0), (-400.0, 60.0), 30.0, "fitted")  # far to the left
    walls = facades.Scene("p", (ASIDE.x, ASIDE.y, ASIDE.z), 0.0, (aside,), (), 0)
    low = [[299.5, 299.5 + 0.035 * F600]]  # 2° down: the street 43 m off, a facade at 10 m first
    view = cli._toward(EYE, ASIDE)

    seen, _ = cli._where_shown(view, (600, 600), np.array(low), walls, _wall_ahead(10.0), _level)

    assert seen[0, 1] == pytest.approx(10.0, abs=0.05)


@pytest.mark.parametrize("height", [600, 602, 603])
def test_a_map_sampled_every_few_pixels_is_spread_to_every_pixel_where_it_belongs(height):
    step = 4
    rows = np.arange(step / 2 - 0.5, height, step)
    columns = np.arange(step / 2 - 0.5, 600, step)
    at = np.stack(np.meshgrid(columns, rows), axis=-1)  # each sample looks where it stands

    to_x, to_y = cli._spread(at, (height, 600), step)

    inner = (slice(2, height - 3), slice(2, 597))  # within the samples' span
    assert np.abs(to_y - np.arange(height)[:, None])[inner].max() < 0.01
    assert np.abs(to_x - np.arange(600)[None, :])[inner].max() < 0.01


def test_a_map_is_not_blended_across_a_break():
    at = np.zeros((10, 10, 2))
    at[:, :5, 0], at[:, 5:, 0] = 100.0, 500.0  # a wall's edge against the sky beyond it
    at[:, :, 1] = 50.0
    at[0, 0] = np.nan  # a sample seen nowhere (behind the camera)

    to_x, _ = cli._spread(at, (40, 40), 4)

    found = np.unique(np.round(to_x))
    assert set(found) <= {100.0, 500.0} | set(found[found < 0])  # nothing in between


def test_the_plan_keeps_a_long_building_whose_corners_are_all_off_it():
    m = match.Situated(0.0, 0.0, 1.5, 0.0, 0.5, 60.0, 30, 2, 3.0, [], "i", 0.1)
    shown = match.Panorama("p", 2.0, 0.0, 2.5, 0.0, path=None)
    block = [(-100.0, 8.0), (100.0, 8.0), (100.0, 18.0), (-100.0, 18.0)]  # 8 m ahead, 200 m long

    plan = cli._plan(m, shown, [], (0.0, 0.0), [block])

    assert len(plan["buildings"]) == 1


def _situated_at_origin():
    return match.Situated(0.0, 0.0, 1.5, 0.0, 0.5, 60.0, 30, 2, 3.0, [], "i", 0.1)


def _standing_at(name, x, y):
    return match.Panorama(name, x, y, 2.5, 0.0, path=None)


def _seen_by(*seen) -> match.Points:
    """Points seen by these sets of panoramas (where they stand: no matter)."""
    n = len(seen)
    return match.Points(np.zeros((n, 2)), np.zeros((n, 3)), np.array(seen, dtype=object))


def test_the_panoramas_a_camera_was_found_from_come_most_points_first_then_nearest():
    east = {"a": 9.0, "b": 3.0, "c": 6.0, "far": 1.0}
    by_id = {name: _standing_at(name, x, 0.0) for name, x in east.items()}
    points = _seen_by({"a"}, {"a", "c"}, {"b"}, {"c"}, {"photo:x"})  # an old photo: no panorama

    first = cli._sources_first(points, by_id, _situated_at_origin(), set())

    assert [p.id for p in first] == ["c", "a", "b"]  # c 2 points, a 2 (farther), b 1
    assert cli._sources_first(None, by_id, _situated_at_origin(), set()) == []


def test_a_source_colmap_placed_comes_before_one_that_saw_more_if_it_saw_half_as_many():
    by_id = {"gps": _standing_at("gps", 2.0, 0.0), "colmap": _standing_at("colmap", 5.0, 0.0)}
    points = _seen_by({"gps"}, {"gps", "colmap"}, {"gps", "colmap"}, {"gps"})  # 4 and 2

    first = cli._sources_first(points, by_id, _situated_at_origin(), {"colmap"})

    assert [p.id for p in first] == ["colmap", "gps"]


def test_a_source_colmap_placed_that_saw_few_points_comes_after_the_one_that_saw_them():
    by_id = {"gps": _standing_at("gps", 2.0, 0.0), "colmap": _standing_at("colmap", 5.0, 0.0)}
    points = _seen_by(*[{"gps"}] * 33, {"gps", "colmap"})  # 34 and 1 (paris_musees:587154)

    first = cli._sources_first(points, by_id, _situated_at_origin(), {"colmap"})

    assert [p.id for p in first] == ["gps", "colmap"]


def test_today_s_view_comes_from_a_source_however_far_before_any_nearer_panorama():
    source = _standing_at("source", cli.SHEET_MOST_M + 5, 0.0)  # it matched what the photo shows
    near_by = [_standing_at("gps", 1.0, 0.0), source]

    views = cli._viewed_from(near_by, _situated_at_origin(), [source], set())

    assert [p.id for p in views] == ["source", "gps"]


def test_without_a_source_today_s_view_comes_from_a_panorama_colmap_placed():
    near_by = [_standing_at("gps", 1.0, 0.0), _standing_at("colmap", 9.0, 0.0)]

    views = cli._viewed_from(near_by, _situated_at_origin(), (), {"colmap"})

    assert [p.id for p in views] == ["colmap", "gps"]  # the other kept, last


def _observed(pixels, rows):
    """An observation whose rays are `rows` of (key, source, origin, direction)."""
    keys, sources, origins, directions = (np.array(c) for c in zip(*rows, strict=True))
    casts, walls = np.full((len(rows), 3), np.nan), np.full((len(rows), 6), np.nan)
    rays = match.Rays(keys, sources.astype(object), origins, directions, casts, walls)
    return match.Observed(np.array(pixels, np.float32), (100, 100), rays)


def test_a_pair_s_today_end_is_where_the_view_s_panorama_matched_it():
    pixels = [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]
    p = _standing_at("p", 0.0, 0.0)
    ahead = [0.0, 1.0, 0.0]  # north, level: the view's centre
    rays = [(0, "p", [0.0, 0.0, 2.5], ahead), (1, "q", [5.0, 0.0, 2.5], ahead)]
    observed = _observed(pixels, rays)
    points = match.Points(  # their pixels as kept on disk: within a rounding of the details'
        np.array([[20.0004, 20.0], [10.0, 9.9996]]),
        np.array([[5.0, 30.0, 2.5], [3.0, 30.0, 2.5]]),
        np.array([{"q"}, {"p", "q"}], dtype=object),
    )
    seen_from = match.View(p.x, p.y, p.z, 0.0, 0.5, 60.0)

    matched = cli._matched_in(points, observed, p, seen_from, (60, 60))

    assert np.isnan(matched[0]).all()  # seen by q only
    assert matched[1] == pytest.approx([29.5, 29.5], abs=0.01)  # p's own ray: straight ahead


def test_pairs_matched_in_the_view_are_shown_alone_and_said_so():
    points = _points_along_a_wall()
    matched = np.full((30, 2), np.nan)
    matched[[4, 9]] = [[11.0, 30.0], [22.0, 30.0]]

    pairs = cli._pairs_shown(points, (120, 120), (60, 60), cli._toward(EYE, ASIDE), matched)

    assert sorted(p["index"] for p in pairs) == [4, 9] and all(p["matched"] for p in pairs)
    assert {tuple(p["today"]) for p in pairs} == {(11.0, 30.0), (22.0, 30.0)}
    unmatched = cli._pairs_shown(points, (120, 120), (60, 60), cli._toward(EYE, ASIDE))
    assert len(unmatched) == cli.PAIRS_SHOWN and not any(p["matched"] for p in unmatched)


def test_a_pair_matched_in_the_view_is_shown_wherever_its_point_would_project():
    points = _points_along_a_wall()
    points.world[4] = [0.0, -20.0, 1.5]  # its point behind the view: it would project nowhere
    matched = np.full((30, 2), np.nan)
    matched[4] = [11.0, 30.0]

    pairs = cli._pairs_shown(points, (120, 120), (60, 60), cli._toward(EYE, ASIDE), matched)

    assert [p["index"] for p in pairs] == [4]


def _by_elevation(*bands):
    """What rays see by how high they look: the first of `bands` (up to this elevation, degrees:
    kind) they look below, else the sky."""

    def kinds(rays):
        up = np.degrees(np.arcsin(rays[:, 2]))
        return np.select([up <= top for top, _ in bands], [kind for _, kind in bands], segment.SKY)

    return kinds


def _column_ahead(kinds):
    """The middle column of a 6 × 6 map of today's view from ASIDE, a BD TOPO wall 20 m north
    topped at 6 m (its rows look 25.7°, 16.1°, 5.5° up, then down)."""
    seen = cli._depth_of_view(
        cli._toward(EYE, ASIDE), (600, 600), None, _wall_ahead(height=6.0), _level, 100, kinds
    )
    return seen[:, 3]


def test_a_wall_rises_as_far_as_its_panorama_sees_building_over_it():
    column = _column_ahead(_by_elevation((20.0, segment.BUILT)))  # a roof up to 20° over a 13° wall

    assert column[1, 1] == pytest.approx(20.0, abs=0.01)  # 16°: on the wall's face, 20 m off


def test_the_sky_right_over_a_wall_stands_on_its_face():
    column = _column_ahead(_by_elevation((10.0, segment.BUILT)))  # sky from 10° up

    assert column[0, 1] == pytest.approx(20.0, abs=0.01)
    assert column[1, 1] == pytest.approx(20.0, abs=0.01)


def test_trees_over_a_wall_stay_far():
    column = _column_ahead(_by_elevation((10.0, segment.BUILT), (30.0, segment.OTHER)))

    assert column[1, 1] > 1000 and column[0, 1] > 1000


def test_a_building_seen_over_the_sky_over_a_wall_stays_far():
    seen = _by_elevation((10.0, segment.BUILT), (20.0, segment.SKY), (30.0, segment.BUILT))
    column = _column_ahead(seen)

    assert column[1, 1] == pytest.approx(20.0, abs=0.01) and column[0, 1] > 1000  # 26°: a spire


def test_a_panorama_not_segmented_leaves_the_walls_as_they_were():
    column = _column_ahead(lambda rays: np.full(len(rays), segment.BUILT))  # building everywhere

    assert column[1, 1] > 1000 and column[0, 1] > 1000


def test_today_s_view_comes_from_the_panorama_next_to_the_camera_when_well_nearer(tmp_path):
    def at(name, y, read=True):
        path = tmp_path / f"{name}.jpg"
        if read:
            path.write_bytes(b"jpg")
        return match.Panorama(name, 0.0, y, 2.5, 0.0, path)

    camera = SimpleNamespace(x=0.0, y=0.0)
    source, next_to, unread = at("source", 37.0), at("next", 2.0), at("unread", 1.0, read=False)

    def shown(sources, near_by, fixed, first=True):
        views = cli._viewed_from(near_by, camera, sources, fixed, nearest_first=first)
        return cli._first_read(views).id

    assert shown([source], [source, next_to], {"source", "next"}) == "next"  # 35 m nearer
    assert shown([source], [source, next_to], {"source", "next"}, first=False) == "source"
    assert shown([source], [source, next_to], {"source"}) == "source"  # not COLMAP's
    assert shown([source], [source, unread], {"source", "unread"}) == "source"  # not at hand
    assert shown([at("near", 8.0)], [next_to], {"near", "next"}) == "near"  # 6 m: kept


def test_the_other_view_says_how_far_it_stands(tmp_path):
    def at(name, y):
        (tmp_path / f"{name}.jpg").write_bytes(b"jpg")
        return match.Panorama(name, 0.0, y, 2.5, 0.0, tmp_path / f"{name}.jpg")

    camera = SimpleNamespace(x=0.0, y=0.0)
    source, next_to = at("source", 37.7), at("next", 2.2)

    assert cli._other_view([source, next_to], next_to, camera) == {"away_m": 37.7}
    assert cli._other_view([next_to], next_to, camera) is None  # the one shown


def test_a_photo_is_placed_where_its_focusing_pass_puts_it_and_the_pass_traced(
    tmp_path, monkeypatch
):
    _match_world(tmp_path, monkeypatch, {}, lambda matcher: None)
    compared = match.Panorama("compared", 3.0, 2.0, 2.5, 0.0, None)
    heights = SimpleNamespace(street=lambda x, y: 0.0, origin_z=0.0)
    monkeypatch.setattr(
        cli, "_matching", lambda: (heights, [compared], lambda year: SimpleNamespace(blocks=None))
    )
    nearer = match.Camera(4.0, 2.0, 1.5, 92.0, 0.5, 60.0, agree=40, panoramas=1)
    left_out = []

    def focus(photo, observed, found, here, panoramas, fixed, but, *rest):
        left_out.append(but)
        return observed, (nearer, [], None), {"panoramas": 1, "moved_m": 3.0}

    monkeypatch.setattr(cli.match, "focus", focus)

    cli.match_photos(photo=["pm:1"], again=False)

    kept = json.loads((tmp_path / "matches.json").read_text())["pm:1"]
    assert (kept["x"], kept["heading"]) == (4.0, 92.0)
    assert left_out == [{"compared"}]  # the panoramas compared already: not matched again
    trace = json.loads((tmp_path / "trace.json").read_text())["pm:1"]
    assert trace["focus"] == {"panoramas": 1, "moved_m": 3.0}


def test_a_panorama_the_focusing_pass_may_reach_is_part_of_a_photo_s_scene():
    near_by = [_panorama(n, 1.0 * n) for n in range(cli.MATCH_PANORAMAS)]  # those compared

    # not compared (beyond MATCH_RADIUS_M), but 12 m from a camera found 50 m off
    assert _scene([*near_by, _panorama(90, 0.0, 62.0)]) != _scene(near_by)


def test_an_areas_check_follows_the_panoramas_its_focusing_pass_may_reach():
    from flaneur import pilot

    area = pilot.AREAS["saint-severin"]
    home = area.centroid
    near_by = [_panorama(n, home.x + 10.0 * n, home.y) for n in range(3)]
    before = cli._check_setup("disk", "saint-severin", near_by)
    out = area.buffer(62.5).exterior.coords[0]  # a view's camera 50 m off: 12.5 m from it

    assert cli._check_setup("disk", "saint-severin", [*near_by, _panorama(7, *out)]) != before


class _Masks:
    """Segmentation masks kept for some panoramas (see segment.Segmenter.kept)."""

    def __init__(self, *kept: str):
        self.ids = set(kept)

    def kept(self, panorama: match.Panorama) -> bool:
        return panorama.id in self.ids


def _never(*args):
    raise AssertionError("refitted: the pairs were set right again")


def test_fitted_walls_are_read_back_without_pairing_while_their_stamp_holds(monkeypatch, tmp_path):
    masks = _Masks("home", "beside")
    _, scenes, _ = _scenes(monkeypatch, tmp_path, kept=tmp_path / "fitted", masks=masks)
    scenes.scene("home", None)  # the pairs made
    first = scenes.scene("home", None, made_only=True)

    _, again, _ = _scenes(monkeypatch, tmp_path, kept=tmp_path / "fitted", masks=masks)
    monkeypatch.setattr(again, "pair", _never)

    assert again.scene("home", None, made_only=True) == first


def _unmask_beside(scenes, again):  # masks are read once a run: the next run's
    return again(_Masks("home"))


def _move_beside(scenes, again):
    scenes.spheres = [replace(p, x=2.5) if p.id == "beside" else p for p in scenes.spheres]
    return scenes


def _other_facades(scenes, again):
    scenes.segments = lambda x, y: [((-4.0, 0.0), (-4.0, 40.0))]
    return scenes


def _another_version(scenes, again):
    cli.WALLS_VERSION += 1
    return scenes


@pytest.mark.parametrize("change", [_unmask_beside, _move_beside, _other_facades, _another_version])
def test_a_mask_a_position_the_facades_or_the_version_changed_refits_them(
    change, monkeypatch, tmp_path
):
    monkeypatch.setattr(cli, "WALLS_VERSION", cli.WALLS_VERSION)

    def again(masks):
        return _scenes(monkeypatch, tmp_path, masks=masks)[1]

    _, scenes, home = _scenes(monkeypatch, tmp_path, masks=_Masks("home", "beside"))
    scenes.scene("home", None)  # the pairs made
    before = scenes.stamp(home)

    assert change(scenes, again).stamp(home) != before


def test_no_pair_on_disk_counts_in_the_stamp_until_it_is_made(monkeypatch, tmp_path):
    _, scenes, home = _scenes(monkeypatch, tmp_path, made=frozenset())
    before = scenes.stamp(home)
    scenes.scene("home", None)  # pairs made: on disk now, not in the snapshot

    assert scenes.stamp(home) == before
    assert scenes.scene("home", None, made_only=True).neighbours == ()


def test_a_cache_written_before_the_area_grew_gives_the_same_heights(monkeypatch, tmp_path):
    masks = _Masks("home", "beside")
    _, scenes, _ = _scenes(monkeypatch, tmp_path, kept=tmp_path / "fitted", masks=masks)
    scenes.scene("home", None)  # the pairs made
    low = scenes.scene("home", None, made_only=True)

    _, risen, _ = _scenes(monkeypatch, tmp_path, z=5.0, kept=tmp_path / "fitted", masks=masks)
    monkeypatch.setattr(risen, "pair", _never)  # every panorama 5 m higher: the frame moved
    high = risen.scene("home", None, made_only=True)

    assert high.ground == pytest.approx(low.ground + 5.0)
    assert [w.top for w in high.walls] == pytest.approx([w.top + 5.0 for w in low.walls])


def test_a_ray_meeting_a_fitted_wall_gets_its_point_and_its_wall_row(monkeypatch, tmp_path):
    _, scenes, home = _scenes(monkeypatch, tmp_path, masks=_Masks("home", "beside"))
    scenes.scene("home", None)  # the pairs made
    ahead = _units(np.array([[-4.0, 10.0, 0.0]]))

    points, rows = cli._wall_depth(scenes)(home, ahead)

    assert points[0][0] == pytest.approx(-4.0, abs=0.3)  # on the left facade
    assert rows[0][0] == pytest.approx(-4.0, abs=0.3) and rows[0][2] == pytest.approx(-4.0, abs=0.3)


def test_a_panorama_without_pairs_or_masks_keeps_bd_topo_and_wakes_no_matcher(
    monkeypatch, tmp_path
):
    ahead = _units(np.array([[-4.0, 10.0, 0.0]]))
    _, unpaired, home = _scenes(monkeypatch, tmp_path, made=frozenset(), masks=_Masks("home"))
    unpaired.views = _never
    _, unmasked, _ = _scenes(monkeypatch, tmp_path, masks=_Masks("home"))  # not "beside"
    unmasked.scene("home", None)  # the pairs made

    assert np.isnan(cli._wall_depth(unpaired)(home, ahead)[0]).all()
    assert np.isnan(cli._wall_depth(unmasked)(home, ahead)[0]).all()


def test_match_s_walls_are_those_of_the_placing_page_and_the_confirmation_page(monkeypatch):
    facades_shown = object()
    monkeypatch.setattr(cli, "_street_segments", lambda: facades_shown)
    near_by = [_panorama(n, 3.0 * n) for n in range(3)]

    scenes = cli._wall_scenes(near_by, set())
    shown = cli._ViewWalls(near_by, set(), None)

    assert scenes.segments is facades_shown and shown.scenes.segments is facades_shown
    assert scenes.kept == shown.scenes.kept == cli.WALLS_FITTED


def test_match_places_from_the_walls_fitted(tmp_path, monkeypatch):
    _match_world(tmp_path, monkeypatch, {}, lambda matcher: None)
    given = []

    def observe(image, near, blocks, views, where, trace, mine, origin_z, depth):
        given.append(depth)
        return views.matcher

    def focus(photo, observed, found, *rest):
        given.append(rest[-1])
        return observed, found, None

    monkeypatch.setattr(cli.match, "observe", observe)
    monkeypatch.setattr(cli.match, "focus", focus)

    cli.match_photos(photo=["pm:1"], again=False)

    assert given and set(given) == {"walls of scenes"}


def test_the_code_is_hashed_once_a_command(monkeypatch):
    hashed = []
    real = cli._hash_of
    monkeypatch.setattr(cli, "_hash_of", lambda *modules: hashed.append(1) or real(*modules))
    cli._code.cache_clear()

    assert cli._code() == cli._code() and len(hashed) == 1


def test_a_panorama_paired_but_not_segmented_has_no_walls_of_its_own(monkeypatch, tmp_path):
    _, unmasked, home = _scenes(monkeypatch, tmp_path)
    unmasked.scene("home", None)  # the pairs made
    _, scenes, _ = _scenes(monkeypatch, tmp_path, masks=_Masks("beside"))  # not "home"

    assert unmasked.has_pairs(home) and not scenes.has_pairs(home)


def test_new_walls_redo_a_photo_in_match_but_never_hide_its_placement(tmp_path, monkeypatch):
    before = _kept(matcher="disk", tried=["lomag", "disk"], fitted_from="old walls")
    _match_world(tmp_path, monkeypatch, {"pm:1": dict(before)}, lambda matcher: CAMERA)

    assert cli._stands("pm:1", before, "i", cli._Judged())  # shown meanwhile: same inputs
    cli.match_photos(photo=None, again=False)

    assert json.loads((tmp_path / "matches.json").read_text())["pm:1"]["fitted_from"] == "walls"


def test_the_same_walls_leave_a_placement_be(tmp_path, monkeypatch):
    before = _kept(matcher="disk", tried=["lomag", "disk"], fitted_from="walls")
    _match_world(tmp_path, monkeypatch, {"pm:1": dict(before)}, lambda matcher: NUDGED)

    cli.match_photos(photo=None, again=False)

    assert json.loads((tmp_path / "matches.json").read_text())["pm:1"] == before


def test_a_placement_kept_when_new_walls_find_nothing_is_not_redone_again(tmp_path, monkeypatch):
    before = _kept(matcher="disk", tried=["lomag", "disk"], fitted_from="old walls")
    _match_world(tmp_path, monkeypatch, {"pm:1": dict(before)}, lambda matcher: None)

    cli.match_photos(photo=None, again=False)

    kept = json.loads((tmp_path / "matches.json").read_text())["pm:1"]
    assert kept == {**before, "fitted_from": "walls"}  # kept, tried with these walls


def test_a_photo_said_yes_is_not_redone_for_new_walls(tmp_path, monkeypatch):
    before = _kept(inputs="old:h", matcher="disk", tried=["disk"], fitted_from="old walls")
    _match_world(tmp_path, monkeypatch, {"pm:1": dict(before)}, lambda matcher: NUDGED)
    yes = {**_said("yes", "old:h", x=1.0, y=2.0), "photo_id": "pm:1"}
    (tmp_path / "anchors.jsonl").write_text(json.dumps(yes) + "\n")

    cli.match_photos(photo=None, again=False)

    written = tmp_path / "matches.json"
    assert not written.exists() or json.loads(written.read_text())["pm:1"] == before


def test_the_check_never_sees_walls_fitted_with_or_drawn_from_the_panorama_its_view_is_cut_from(
    monkeypatch, tmp_path
):
    _, scenes, home = _scenes(monkeypatch, tmp_path, masks=_Masks("home", "beside"))
    scenes.scene("home", None)  # the pairs made
    drawn = facades.Wall((-1.0, 3.0), (1.0, 3.0), None, "drawn")
    facades.record(scenes.walls_path, "beside", [drawn], [])  # drawn while at "beside"
    ahead = _units(np.array([[0.0, 1.0, 0.0], [-4.0, 10.0, 0.0]]))

    seen = cli._wall_depth(scenes)(home, ahead)[0]
    cut = cli._wall_depth(scenes, but=frozenset({"beside"}))(home, ahead)[0]

    assert not np.isnan(seen).any()  # the wall drawn, the facade fitted with "beside"
    assert np.isnan(cut).all()


def test_a_photo_s_walls_change_with_the_walls_it_may_meet_not_with_their_code(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cli, "WALLS_VERSION", cli.WALLS_VERSION)

    def walls(z=0.0):
        _, scenes, _ = _scenes(monkeypatch, tmp_path, z=z, kept=tmp_path / "fitted")
        scenes.scene("home", None)  # the pairs made
        return cli._photo_walls(scenes, scenes.spheres)((0.0, 0.0))

    before = walls()
    cli.WALLS_VERSION += 1  # the code changed: the walls fitted again, the same
    same = walls()
    risen = walls(z=5.0)  # the frame of heights moved: the same walls
    wall = facades.Wall((-1.0, 3.0), (1.0, 3.0), None, "drawn")
    facades.record(tmp_path / "w.jsonl", "home", [wall], [])

    assert before == same == risen != walls()  # a wall drawn near


def test_a_photo_s_walls_are_told_apart_when_two_share_their_ends_one_without_a_top():
    home = match.Panorama("home", 0.0, 0.0, 0.0, 0.0, path=None)
    walls = (
        facades.Wall((-1.0, 3.0), (1.0, 3.0), None, "drawn"),
        facades.Wall((-1.0, 3.0), (1.0, 3.0), 9.0, "fitted"),
    )

    class Scenes:
        def scene(self, panorama, chosen, made_only=False):
            return facades.Scene("home", (0.0, 0.0, 0.0), -2.5, walls, (), 0)

    assert cli._photo_walls(Scenes(), [home])((0.0, 0.0))


def test_a_redo_for_new_walls_cut_short_is_taken_up_again(tmp_path, monkeypatch):
    before = _kept(matcher="disk", tried=["lomag", "disk"], fitted_from="old walls")

    def stopped(matcher):
        if matcher == "disk":
            raise KeyboardInterrupt  # the run stopped before DISK's turn
        return None

    _match_world(tmp_path, monkeypatch, {"pm:1": dict(before)}, stopped)

    with pytest.raises(KeyboardInterrupt):
        cli.match_photos(photo=None, again=False)

    written = tmp_path / "matches.json"
    assert not written.exists() or json.loads(written.read_text())["pm:1"] == before


def test_a_pair_listed_at_the_start_the_other_way_round_is_made(monkeypatch, tmp_path):
    name = cli._levelled_name
    home, beside = (match.Panorama(i, 0.0, 0.0, 0.0, 0.0, tmp_path / f"{i}.jpg") for i in "hb")
    _, scenes, _ = _scenes(
        monkeypatch, tmp_path, made=frozenset({f"{name(beside)}__{name(home)}.npz"})
    )

    assert scenes._made(home, beside) and not scenes._made(beside, scenes.by_id["far"])


def test_a_registration_is_made_again_only_when_its_placement_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "REGISTERED", tmp_path / "registered.json")
    monkeypatch.setattr(cli, "_registering_world", lambda: ({}, set()))
    made = []
    monkeypatch.setattr(cli, "_register_one", lambda k, m, by_id, fixed: made.append(k) or None)
    placed_at = _kept(matcher="lomag")

    cli._registrations({"pm:1": placed_at})
    cli._registrations({"pm:1": placed_at})  # the same: read back
    cli._registrations({"pm:1": {**placed_at, "heading": placed_at["heading"] + 5}})

    assert made == ["pm:1", "pm:1"]


def _registered(tmp_path, monkeypatch, placements: dict, walls: dict, stood: dict) -> list:
    """REGISTERED holding `placements` registered in panorama "q" (made as now), the walls
    around "q" (`walls`: one top a call, see facades.Wall), the standing cameras made by
    `stood` (photo id: (flat, parallax)); what was stood, in order."""
    monkeypatch.setattr(cli, "REGISTERED", tmp_path / "registered.json")
    row = {"panorama": "q", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    cli.write_json(
        cli.REGISTERED,
        {k: {"made": cli._registered_as(m), "registration": row} for k, m in placements.items()},
    )
    origin = (10.0, 20.0, 35.0)

    def scene(panorama):
        wall = facades.Wall((0.0, 5.0), (9.0, 5.0), walls["top"], "fitted")
        sphere = SimpleNamespace(azimuth=10.0, roll=0.5, pitch=-0.5)
        return cli._Home(sphere, facades.Scene(panorama, origin, 32.5, (wall,), (), 8))

    camera = registration.Standing(
        (11.0, 20.5, 35.2), (0.1, 0.2, 0.3), 700.0, (0.0, 0.05), (60, 80)
    )
    made = []

    def stand(k, m, home):
        made.append(k)
        if isinstance(stood[k], Exception):
            raise stood[k]
        return placing.Stood(camera, *stood[k])

    monkeypatch.setattr(cli, "_standing_home", scene)
    monkeypatch.setattr(cli, "_stand_one", stand)
    return made


def _copies(tmp_path, monkeypatch, walls: dict, stood: dict) -> list:
    """Two photos placed, each carried to panoramas "a" and "b" (pm:flat not standing at its
    home); the walls around every panorama (`walls`: their top); the copies' standing cameras
    made by `stood` ((photo, panorama): (flat, parallax, held)); what was stood, in order."""
    monkeypatch.setattr(cli, "CARRIED_STANDING", tmp_path / "carried_standing.json")
    monkeypatch.setattr(cli, "REGISTERED", tmp_path / "registered.json")
    cli.write_json(
        cli.REGISTERED, {k: {"made": cli._registered_as(_kept())} for k in ("pm:1", "pm:flat")}
    )
    origin = (10.0, 20.0, 35.0)

    def home(panorama):
        wall = facades.Wall((0.0, 5.0), (9.0, 5.0), walls["top"], "fitted")
        sphere = SimpleNamespace(azimuth=10.0, roll=0.5, pitch=-0.5)
        return cli._Home(sphere, facades.Scene(panorama, origin, 32.5, (wall,), (), 8))

    camera = registration.Standing(
        (11.0, 20.5, 35.2), (0.1, 0.2, 0.3), 700.0, (0.0, 0.05), (60, 80)
    )
    made = []

    def stand(photo_id, kept, home, against, again, least):
        made.append((photo_id, home.scene.panorama))
        return placing.Stood(camera, *stood[made[-1]])

    monkeypatch.setattr(cli, "_standing_home", home)
    monkeypatch.setattr(cli, "_stand_one", stand)
    monkeypatch.setattr(cli, "_photo_store", lambda: None)
    monkeypatch.setattr(cli, "_matches", lambda: {"pm:1": _kept(), "pm:flat": _kept()})
    return made


def _held_and_carried():
    row = {"registration": {"panorama": "q"}, "inputs": "i", "flipped": False}
    held = {
        "pm:1": {**row, "standing": {"position": [0, 0, 0]}},
        "pm:flat": {**row, "standing": None},
    }
    copies = [
        {"registration": {"panorama": p, "m": [1] * 9, "view": {"heading": 0.0}}, "strength": 0.8}
        for p in "ab"
    ]
    return held, {k: {"made": "x", "copies": copies, "at": {}} for k in held}


def test_a_copy_stands_where_it_beats_its_matrix_on_enough_held_out_pairs(monkeypatch, tmp_path):
    stood = {("pm:1", "a"): (0.006, 0.003, 30), ("pm:1", "b"): (0.006, 0.0055, 30)}
    made = _copies(tmp_path, monkeypatch, {"top": 40.0}, stood)
    held, carried = _held_and_carried()

    cli._copies_standing(held, carried)

    kept = json.loads((tmp_path / "carried_standing.json").read_text())
    assert set(kept) == {"pm:1"} and made == [("pm:1", "a"), ("pm:1", "b")]  # not pm:flat
    assert cli._copy_standing(kept["pm:1"]["a"])["position"] == [1.0, 0.5, 0.2]  # from "a"
    assert cli._copy_standing(kept["pm:1"]["b"]) is None  # 0.05 % better: within the margin


def test_a_copy_judged_on_too_few_held_out_pairs_keeps_its_matrix(monkeypatch, tmp_path):
    stood = {("pm:1", "a"): (0.006, 0.001, 9), ("pm:1", "b"): (0.006, 0.001, 10)}
    _copies(tmp_path, monkeypatch, {"top": 40.0}, stood)
    held, carried = _held_and_carried()

    cli._copies_standing(held, carried)

    kept = json.loads((tmp_path / "carried_standing.json").read_text())
    assert cli._copy_standing(kept["pm:1"]["a"]) is None
    assert cli._copy_standing(kept["pm:1"]["b"]) is not None


def test_a_copy_stands_again_only_when_its_walls_change(monkeypatch, tmp_path):
    stood = {("pm:1", p): (0.006, 0.003, 30) for p in "ab"}
    walls = {"top": 40.0}
    made = _copies(tmp_path, monkeypatch, walls, stood)
    held, carried = _held_and_carried()

    cli._copies_standing(held, carried)
    cli._copies_standing(held, carried)  # read back
    walls["top"] = 45.0
    cli._copies_standing(held, carried)

    assert made == [("pm:1", "a"), ("pm:1", "b")] * 2


def test_a_standing_camera_is_shown_only_where_it_lays_the_photo_better(monkeypatch, tmp_path):
    placements = {"pm:1": _kept(), "pm:2": _kept(inputs="j")}
    _registered(
        tmp_path,
        monkeypatch,
        placements,
        {"top": 40.0},
        {"pm:1": (0.02, 0.01), "pm:2": (0.01, 0.02)},
    )

    shown = cli._registrations(placements)

    assert shown["pm:1"].standing["position"] == [1.0, 0.5, 0.2]  # metres off the panorama
    assert shown["pm:1"].registration["panorama"] == "q"
    assert shown["pm:2"].standing is None


def test_a_standing_camera_that_fails_leaves_the_others_and_is_tried_again(
    monkeypatch, tmp_path, capsys
):
    placements = {"bad": _kept(), "good": _kept(inputs="j")}
    stood = {"bad": np.linalg.LinAlgError("singular"), "good": (0.02, 0.01)}
    made = _registered(tmp_path, monkeypatch, placements, {"top": 40.0}, stood)

    shown = cli._registrations(placements)
    cli._registrations(placements)

    assert shown["bad"].registration and shown["bad"].standing is None
    assert shown["good"].standing is not None
    assert made == ["bad", "good", "bad"] and "pas debout" in capsys.readouterr().err


def test_errors_not_measured_show_no_standing_camera_and_stay_json(monkeypatch, tmp_path):
    placements = {"pm:1": _kept()}
    _registered(tmp_path, monkeypatch, placements, {"top": 40.0}, {"pm:1": (np.inf, np.inf)})

    shown = cli._registrations(placements)

    assert shown["pm:1"].standing is None
    assert "Infinity" not in (tmp_path / "registered.json").read_text()


def test_a_standing_camera_is_made_again_only_when_its_walls_change(monkeypatch, tmp_path):
    placements = {"pm:1": _kept()}
    walls = {"top": 40.0}
    made = _registered(tmp_path, monkeypatch, placements, walls, {"pm:1": (0.02, 0.01)})

    cli._registrations(placements)
    cli._registrations(placements)  # the same walls: read back
    walls["top"] = 45.0  # a wall drawn again
    cli._registrations(placements)

    assert made == ["pm:1", "pm:1"]


def test_the_placed_list_is_written_from_the_answers_verdicts_and_matches(monkeypatch, tmp_path):
    exported = tmp_path / "old.jpg"
    exported.touch()
    answer = {"registration": {"panorama": "q"}, "inputs": "h", "flipped": False}
    said = [("pm:1", "near", "i"), ("pp", "yes", "i:proposed-ab12cd34")]  # pp: a camera refused
    said.append(("pm:3", "yes", "i:from-ab12cd34"))  # its chained camera said yes to
    said.append(("pm:1", "yes", "i:from-ab12cd34"))  # match's said near, the chained one yes
    said.append(("pn", "near", "i:from-ab12cd34"))  # a chained one said near: listed as near
    (tmp_path / "anchors.jsonl").write_text(
        "".join(json.dumps({"photo_id": k, "verdict": v, "inputs": i}) + "\n" for k, v, i in said)
    )
    marked = [("pm:3", "h"), ("pm:4", "before")]  # proposed on today's crop, on an older one
    (tmp_path / "sights.jsonl").write_text(
        "".join(
            json.dumps(
                {"photo_id": k, "image": i, "landmarks": ["notre-dame-tours"], "column": 0.5}
            )
            + "\n"
            for k, i in marked
        )
    )
    for name, value in {
        "PLACED_PHOTOS": tmp_path / "placed.json",
        "CHECKED_AREAS": tmp_path / "checked_areas.json",
        "SIGHTS": tmp_path / "sights.jsonl",
        "ANCHORS": tmp_path / "anchors.jsonl",
        "_placing_answers": lambda: ({}, {"pa": answer}, set()),
        "_scenes_now": lambda: {k: "s" for k in ("pm:1", "pm:2", "pm:3", "pm:4", "pp", "pc", "pn")},
        "_matches": lambda: {
            "pm:1": _kept(),
            "pm:2": _kept(inputs="before"),
            "pm:3": _kept(),
            "pm:4": _kept(),
        },
        "_proposed_now": lambda scenes, judged, marks=True: {
            "pp": _kept(inputs="i:proposed-ab12cd34")
        },
        "_inputs": lambda scene, photo_id: "i",
        "_photo_part": lambda photo_id: "",  # each placed from the photo as it is (see "i")
        "_exported": lambda photo_id: exported,
        "_image": lambda photo_id: "h",
        "_registrations": lambda standing: {
            k: placed.Registered({"panorama": "p"}) for k in standing
        },
        "_kept_store": lambda m, kept: (SimpleNamespace(points=lambda k, i: seen_by.get(k)), "i"),
        "_sure_panoramas": lambda: ({"sure"}, {"sure", "gps"}),  # sure of their pose, known
        "_chained": lambda scenes, judged, anchors: {
            "photos": {
                k: {"situated": _kept(inputs="i:from-ab12cd34")}
                for k in ("pc", "pm:1", "pm:3", "pn")
            }
        },
    }.items():
        monkeypatch.setattr(cli, name, value)
    seen_by = {"pm:3": _seen_by({"sure"}, {"gps"}), "pm:4": _seen_by({"gps"}, {"gps", "photo:x"})}

    cli._placed_photos(write=True)

    written = json.loads((tmp_path / "placed.json").read_text())
    assert {k: (p["status"], p["source"], p["proposed_marks"]) for k, p in written.items()} == {
        "pa": ("checked", "page", False),
        "pm:1": ("checked", "chain", False),  # pm:2 was placed from what the photo no longer is
        "pm:3": ("checked", "chain", False),  # a yes wins over match's not judged
        "pm:4": ("automatic", "match", False),  # its marks were made on another crop
        "pp": ("checked", "match", False),
        "pn": ("near", "chain", False),  # pc, chained and not judged: proposed on the page only
    }
    assert {k for k, p in written.items() if p["unsure_sources"]} == {"pm:4"}  # gps only


def test_only_the_photos_placed_with_a_registration_go_to_3d(monkeypatch):
    reg = {"panorama": "p1", "m": [1.0] * 9, "view": {"heading": 0.0}}
    monkeypatch.setattr(
        cli,
        "_placed_photos",
        lambda write=False, answering=None: {
            "pm:1": cli.placed.Placed("pm:1", "automatic", "match", reg, "h"),
            "pm:2": cli.placed.Placed("pm:2", "checked", "match", None, "h"),  # map only
        },
    )

    assert cli._shown_in_3d() == {
        "pm:1": {
            "registration": reg,
            "inputs": "h",
            "flipped": False,
            "status": "automatic",
            "unchecked_area": False,
            "proposed_marks": False,
            "unsure_sources": False,
            "standing": None,
        }
    }


def test_a_match_placement_reaches_the_3d_walk_with_its_status(monkeypatch, tmp_path):
    from test_scene3d import REG, _photo, _picture

    (tmp_path / "walk.json").write_text(json.dumps({"photos": [{"id": "pm:1", "in_3d": True}]}))
    (tmp_path / "panoramas").mkdir()
    (tmp_path / "panoramas" / "p1.jpg").touch()
    row = {"registration": REG, "inputs": "h", "flipped": False, "status": "near"}
    for name, value in {
        "WEB_DATA": tmp_path,
        "PANORAMAS": tmp_path / "panoramas",
        "_placing_answers": lambda: ({}, {}, set()),
        "_walk": lambda conn, places: [_photo("pm:1")],
        "_lined_up": lambda pictures: [_picture("p1", 2.3466)],
        "_panoramas": lambda pictures, heights: [],
        "_poised": lambda spheres: {},
        "_carried": lambda shown, now: {},
        "_shown_in_3d": lambda answering=None: {
            "pm:1": {
                **row,
                "unchecked_area": False,
                "proposed_marks": False,
                "unsure_sources": False,
                "standing": None,
            }
        },
    }.items():
        monkeypatch.setattr(cli, name, value)
    monkeypatch.setattr(cli.db, "connect", lambda path: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli.panoramas, "load", lambda conn: [])

    cli.write_scene()

    shown = json.loads((tmp_path / "scene.json").read_text())["photos"]
    assert [(p["id"], p["status"]) for p in shown] == [("pm:1", "near")]


def test_porter_carries_a_placement_of_match_too(monkeypatch, tmp_path):
    from test_scene3d import REG

    row = {"registration": REG, "inputs": "h", "flipped": False, "status": "automatic"}
    home = match.Panorama("p1", 0.0, 0.0, 0.0, 0.0, tmp_path / "p1.jpg")
    for name, value in {
        "CARRIED": tmp_path / "carried.json",
        "_shown_in_3d": lambda answering=None: {"pm:1": row},
        "_lined_up": lambda pictures: [],
        "_panoramas": lambda pictures, heights: [home],
        "_poised": lambda spheres: {},
        "_views": lambda kept=None, matcher="disk": None,
        "_to_neighbours": lambda r, home, spheres, street, views: (["copy"], set()),
    }.items():
        monkeypatch.setattr(cli, name, value)
    monkeypatch.setattr(cli.db, "connect", lambda path: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli.panoramas, "load", lambda conn: [])
    monkeypatch.setattr(cli.panoramas, "links", lambda pictures: {})

    cli.carry_placed()

    carried = json.loads((tmp_path / "carried.json").read_text())
    assert carried["pm:1"]["copies"] == ["copy"]
    assert carried["pm:1"]["made"] == cli._answered_as(row)


def test_the_export_checks_a_hand_placement_against_the_image_it_writes(monkeypatch):
    answers = {"pa": {"inputs": "h"}, "pm:1": {"inputs": "old"}}  # pa: its image not yet written
    shown = {"pm:1": {"inputs": "new"}}  # pm:1's answer was on an old image: match's instead
    monkeypatch.setattr(cli, "_shown_in_3d", lambda answering=None: shown)

    assert cli._placed_on(({}, answers, set())) == {"pa": "h", "pm:1": "new"}


def test_the_areas_message_tells_a_check_not_made_from_one_that_failed(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli, "MATCH_CHECK", tmp_path / "check.json")
    monkeypatch.setattr(cli, "_check_setup", lambda matcher, area, near_by: "now")
    failed = {a: {"passed": False, "setup": "now"} for a in ("cite", "saint-severin")}
    (tmp_path / "check.json").write_text(json.dumps(failed))  # DISK's: done, not passed
    # LoMa-G's: never made

    cli._checked_in([])

    said = capsys.readouterr().out
    assert "contrôle non passé : DISK (saint-severin), DISK (cite)" in said
    assert "`flaneur match-check --perimes --matcher lomag`" in said
    assert "--matcher disk" not in said  # done for these panoramas: doing it again changes nothing


def test_walls_go_near_the_placements_first_then_the_areas_then_the_rest():
    from shapely.geometry import box

    spheres = [_panorama(n, 100.0 * n, 0.0) for n in range(4)]  # 0, 100, 200, 300 m east
    places = [(5.0 + 100.0 * n, 0.0) for n in range(3)]  # a photo by each but the last
    near_placed = [(210.0, 0.0)]
    in_area = [box(90.0, -10.0, 110.0, 10.0)]

    homes = cli._wall_homes(spheres, places, near_placed, in_area)

    assert [p.id for p in homes] == [spheres[2].id, spheres[1].id, spheres[0].id]


def test_walls_masks_only_segments_what_is_missing(monkeypatch):
    home, beside = _panorama(1, 0.0, 0.0), _panorama(2, 5.0, 0.0)
    monkeypatch.setattr(cli.depth, "candidates", lambda home, spheres, fixed: [beside])
    made = []
    masks = SimpleNamespace(kept=lambda p: p is home, made=lambda p: made.append(p.id))

    cli._segment_homes([home], [home, beside], set(), masks)

    assert made == [beside.id]


def test_walls_made_are_cached_for_match(monkeypatch, tmp_path):
    asked, scenes, home = _scenes(monkeypatch, tmp_path, kept=tmp_path / "fitted")

    cli._make_walls([home], scenes)

    assert asked == [("home", "beside")]  # the pair made
    assert (tmp_path / "fitted" / f"{cli._levelled_name(home)}.json").exists()  # its walls kept


def test_walls_made_again_pair_nothing_and_read_what_was_kept(monkeypatch, tmp_path):
    _, scenes, home = _scenes(monkeypatch, tmp_path, kept=tmp_path / "fitted")
    cli._make_walls([home], scenes)
    made = frozenset(p.name for p in (tmp_path / "pairs").glob("*.npz"))
    _, again, _ = _scenes(monkeypatch, tmp_path, kept=tmp_path / "fitted", made=made)
    monkeypatch.setattr(again, "pair", _never)  # nothing new: no pair set right, no fit
    first = scenes.scene("home", None, made_only=True)

    cli._make_walls([home], again)

    assert again.scene("home", None, made_only=True).walls == first.walls  # as match reads them


def test_a_panorama_that_cannot_be_segmented_leaves_the_others_segmented(monkeypatch):
    broken, home, beside = (_panorama(n, 5.0 * n, 0.0) for n in range(3))
    monkeypatch.setattr(cli.depth, "candidates", lambda home, spheres, fixed: [])
    made = []

    def make(p):
        if p is broken:
            raise MemoryError("out of memory")
        made.append(p.id)

    masks = SimpleNamespace(kept=lambda p: False, made=make)

    cli._segment_homes([broken, home, beside], [broken, home, beside], set(), masks)

    assert made == [home.id, beside.id]


def test_the_light_update_lists_on_the_images_the_export_wrote(monkeypatch, tmp_path):
    image = tmp_path / "p.jpg"
    image.write_bytes(b"before the crop")
    monkeypatch.setattr(cli, "REFRESH_LOCK", tmp_path / ".lock")
    monkeypatch.setattr(cli, "_exported", lambda photo_id: image)
    monkeypatch.setattr(cli, "_placing_answers", lambda: "answers")
    done = []

    def export():
        cli._image("p")  # the export reads the image it is about to replace
        image.write_bytes(b"cropped since")
        done.append("export")

    monkeypatch.setattr(cli, "export_walk", export)
    monkeypatch.setattr(
        cli, "_placed_photos", lambda write, answering: done.append(("list", cli._image("p")))
    )
    monkeypatch.setattr(cli, "_write_scene", lambda answering: done.append(("scene", answering)))
    cli._image.cache_clear()
    try:
        cli._update_light()
    finally:
        cli._image.cache_clear()

    assert done == ["export", ("list", cli.landmarks.image_hash(image)), ("scene", "answers")]


def _ran(outcomes: dict | None = None, clock=None):
    """A night's runner noting each step run (its arguments), returning its outcome (an exit
    code, or an exception raised), advancing `clock` by a step's hour."""
    ran = []

    def run(command, timeout=None):
        step = tuple(command[3:])  # past "python -m flaneur"
        ran.append((step, timeout))
        if clock is not None:
            clock[0] += 3600.0
        outcome = (outcomes or {}).get(step, 0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(returncode=outcome)

    return ran, run


def test_the_night_runs_its_steps_in_order_and_goes_on_after_one_fails():
    ran, run = _ran({("murs", "--masques"): 1})

    cli._run_night(None, run=run, clock=lambda: 0.0)

    assert [step for step, _ in ran] == [*cli.NIGHT_STEPS, ("mettre-a-jour",)]
    assert ran[0][1] is None  # no hour: no timeout


def test_no_night_step_starts_after_the_hour():
    clock = [0.0]
    stopped = subprocess.TimeoutExpired("murs", 1.0)
    ran, run = _ran({cli.NIGHT_STEPS[2]: stopped}, clock)

    cli._run_night(2.5 * 3600.0, run=run, clock=lambda: clock[0])

    steps = [step for step, _ in ran]
    assert steps == [*cli.NIGHT_STEPS[:3], ("mettre-a-jour",)]  # stopped at the hour; the light
    assert ran[2][1] == pytest.approx(0.5 * 3600.0)  # the third may run until the hour only


def test_following_updates_once_after_answers_settle():
    now = [0.0]
    writes = {5.0: 1, 10.0: 2}  # the page writes twice, 5 s apart; then quiet
    updates = []

    def sleep(seconds):
        now[0] += seconds
        if now[0] > 60.0:
            raise _Stop

    def stamp(paths):
        return max((n for at, n in writes.items() if at <= now[0]), default=0)

    with pytest.raises(_Stop):
        cli._follow(
            lambda: updates.append(now[0]), [], sleep=sleep, clock=lambda: now[0], stamp=stamp
        )

    assert updates == [20.0]  # once, 10 s after the last write


class _Stop(Exception):
    pass


def test_a_second_update_waits_for_the_first(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "REFRESH_LOCK", tmp_path / ".lock")
    done = []
    monkeypatch.setattr(cli, "_placing_answers", lambda: None)
    monkeypatch.setattr(cli, "_placed_photos", lambda write, answering: done.append("list"))
    monkeypatch.setattr(cli, "export_walk", lambda: None)
    monkeypatch.setattr(cli, "_write_scene", lambda answering: None)

    with cli._refreshing():  # the first, under way
        second = threading.Thread(target=cli._update_light)
        second.start()
        second.join(0.3)
        waited = second.is_alive() and not done
    second.join(5.0)

    assert waited and done == ["list"]


def test_walls_made_at_night_load_the_matcher_once(monkeypatch):
    monkeypatch.setattr(cli, "_street_segments", lambda: None)
    loaded = []

    scenes = cli._wall_scenes([_panorama(1, 0.0)], set(), lambda: loaded.append(1) or "views")
    scenes.views(), scenes.views()

    assert loaded == [1]


def test_a_mark_made_since_leaves_a_match_placement_shown_not_standing_for_match(tmp_path):
    image, other = "1234567890abcdef", "fedcba0987654321"
    yes = {"photo_id": "y", "verdict": "yes", "inputs": f"old:{image}"}
    (tmp_path / "anchors.jsonl").write_text(json.dumps(yes) + "\n")
    judged = cli._Judged(tmp_path / "anchors.jsonl")
    kept, marked = _kept(inputs=f"scene:{image}"), f"scene:{image}:ab12cd34"

    assert cli._found_as_now("p", kept, marked, judged, marks=False)  # shown
    assert not cli._found_as_now("p", kept, marked, judged)  # match redoes it
    assert not cli._found_as_now("p", kept, f"scene:{other}:ab12cd34", judged, marks=False)
    said = _kept(inputs=f"old:{image}")
    assert cli._found_as_now("y", said, f"new:{image}:ab12cd34", judged, marks=False)


def test_a_proposed_camera_and_a_confirmed_place_stay_once_marked(monkeypatch, tmp_path):
    image, fp = "1234567890abcdef", "proposed-0a1b2c3d"
    proposed = _kept(inputs=f"s:{image}:{fp}", matcher=fp)
    (tmp_path / "proposed.json").write_text(json.dumps({"pp": proposed}))
    yes = {"photo_id": "pp", "verdict": "yes", "inputs": proposed["inputs"]}
    (tmp_path / "anchors.jsonl").write_text(json.dumps(yes) + "\n")
    exported = tmp_path / "pp.jpg"
    exported.touch()
    for name, value in {
        "PROPOSED_CAMERAS": tmp_path / "proposed.json",
        "ANCHORS": tmp_path / "anchors.jsonl",
        "_exported": lambda photo_id: exported,
        "_inputs": lambda scene, photo_id: f"s:{image}:ab12cd34",  # marked since
        "_scenes_now": lambda: {"pp": "s"},
        "_matches": lambda: {},
    }.items():
        monkeypatch.setattr(cli, name, value)
    judged = cli._Judged()

    assert set(cli._proposed_now({"pp": "s"}, judged, marks=False)) == {"pp"}
    assert set(cli._confirmed()) == {"pp"}  # the walk keeps its place


def test_no_test_reaches_the_pipeline_s_data(tmp_path):
    written = [cli.CHECKED_AREAS, cli.PLACED_PHOTOS, cli.REGISTERED, cli.MATCHES, cli.WEB_DATA]

    assert all(path.is_relative_to(tmp_path) for path in written)


def test_a_panorama_whose_walls_fail_leaves_the_others_made(capsys):
    broken, home = _panorama(1, 0.0, 0.0), _panorama(2, 5.0, 0.0)
    made = []

    def make(p):
        if p is broken:
            raise MemoryError("out of memory")
        made.append(p.id)

    cli._make_walls([broken, home], SimpleNamespace(make=make))

    assert made == [home.id] and "laissé" in capsys.readouterr().err


def test_a_registration_is_made_again_when_the_registering_code_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "REGISTERED", tmp_path / "registered.json")
    monkeypatch.setattr(cli, "_registering_world", lambda: ({}, set()))
    made = []
    monkeypatch.setattr(cli, "_register_one", lambda k, m, by_id, fixed: made.append(k) or None)
    placed_at = _kept(matcher="lomag")

    cli._registrations({"pm:1": placed_at})
    monkeypatch.setattr(cli, "_registering_code", lambda: "fixed since")
    cli._registrations({"pm:1": placed_at})

    assert made == ["pm:1", "pm:1"]


def test_proposed_marks_are_said_only_of_a_placement_made_with_them(monkeypatch, tmp_path):
    exported = tmp_path / "old.jpg"
    exported.touch()
    (tmp_path / "anchors.jsonl").write_text("")
    (tmp_path / "sights.jsonl").write_text(
        "".join(
            json.dumps(
                {"photo_id": k, "image": "h", "landmarks": ["notre-dame-tours"], "column": 0.5}
            )
            + "\n"
            for k in ("with", "before")
        )
    )
    for name, value in {
        "PLACED_PHOTOS": tmp_path / "placed.json",
        "CHECKED_AREAS": tmp_path / "checked_areas.json",
        "SIGHTS": tmp_path / "sights.jsonl",
        "ANCHORS": tmp_path / "anchors.jsonl",
        "_placing_answers": lambda: ({}, {}, set()),
        "_scenes_now": lambda: {"with": "s", "before": "s"},
        "_matches": lambda: {
            "with": _kept(inputs="s:h:abcd1234"),  # placed with today's marks
            "before": _kept(inputs="s:h"),  # placed before they were proposed
        },
        "_proposed_now": lambda scenes, judged, marks=True: {},
        "_inputs": lambda scene, photo_id: "s:h:abcd1234",
        "_photo_part": lambda photo_id: "h:abcd1234",
        "_exported": lambda photo_id: exported,
        "_image": lambda photo_id: "h",
        "_registrations": lambda standing: {
            k: placed.Registered({"panorama": "p"}) for k in standing
        },
    }.items():
        monkeypatch.setattr(cli, name, value)

    found = cli._placed_photos()

    assert (found["with"].proposed_marks, found["before"].proposed_marks) == (True, False)


def test_a_placement_kept_after_new_marks_is_registered_from_what_it_was_tried_from(monkeypatch):
    seen = SimpleNamespace(rays="from the photo as it is now")
    store = SimpleNamespace(
        points=lambda k, inputs: "points" if inputs == "s:img" else None,
        observed=lambda k, inputs: seen if inputs == "s:img:ab12cd34" else None,  # rewritten
    )
    monkeypatch.setattr(cli, "_kept_store", lambda m, kept: (store, "s:img"))
    monkeypatch.setattr(cli, "_sources_first", lambda points, by_id, m, fixed: ["sphere"])
    monkeypatch.setattr(cli.placed, "registered", lambda p, o, spheres: (p, o.rays))
    kept = {**_kept(inputs="s:img", matcher="disk"), "tried_from": "s:img:ab12cd34"}

    found = cli._register_one("p", kept, {}, set())

    assert found == ("points", "from the photo as it is now")


def test_a_registration_that_fails_leaves_the_others_and_is_tried_again(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "REGISTERED", tmp_path / "registered.json")
    monkeypatch.setattr(cli, "_registering_world", lambda: ({}, set()))

    def register(k, m, by_id, fixed):
        if k == "bad":
            raise ValueError("residuals are not finite")
        return None

    monkeypatch.setattr(cli, "_register_one", register)

    found = cli._registrations({"bad": _kept(), "good": _kept(x=3.0)})

    kept = json.loads((tmp_path / "registered.json").read_text())
    assert found == {"bad": (None, None), "good": (None, None)} and set(kept) == {"good"}


def test_no_test_watches_or_reads_the_pipeline_s_answers(tmp_path):
    assert all(path.is_relative_to(tmp_path) for path in cli.ANSWER_FILES)
    assert cli._placing_answers.__defaults__[0] is None  # PLACINGS, read when called


def _picture(i: str, lon: float, azimuth: float = 90.0):
    from flaneur.panoramax import Picture

    return Picture(i, lon, 48.853, azimuth, True, "2025-07-10", "", "")


def test_survey_corrections_are_read_once_applied_where_the_panorama_still_stands(monkeypatch):
    from flaneur import survey

    pictures = [_picture("v", 2.3450), _picture("moved", 2.3460), _picture("s", 2.3470)]
    where = cli._standing(pictures)
    found = {
        "v": survey.SetRight(10.0, (1.0, -2.0), 1, 2),
        "moved": survey.SetRight(10.0, None, 1, 2),
    }
    judged = {**where, "moved": (0.0, 0.0, 0.0)}  # set right where it no longer stands
    survey.save(cli.SURVEYED, found, judged)

    monkeypatch.setattr(survey, "APPLIED", "none")
    off = cli._placed(pictures)
    monkeypatch.setattr(survey, "APPLIED", "all")
    on = {p.id: p for p in cli._placed(pictures)}

    assert [p.azimuth for p in off] == [90.0, 90.0, 90.0]  # switched off: none
    assert on["v"].azimuth == pytest.approx(80.0)  # turned counter-clockwise: the compass back
    assert cli._standing([on["v"]])["v"][:2] == pytest.approx(
        (where["v"][0] + 1.0, where["v"][1] - 2.0), abs=0.02
    )
    assert (on["moved"].azimuth, on["s"].azimuth) == (90.0, 90.0)


def test_the_night_sets_panoramas_right_on_the_survey_first_then_segments():
    assert cli.NIGHT_STEPS[:2] == (("releve",), ("segmenter",))


def test_bd_topo_corrects_nothing_the_survey_answered(monkeypatch):
    from flaneur import survey

    pictures = [_picture("v", 2.3450), _picture("bd", 2.3460)]
    where = cli._standing(pictures)
    survey.save(cli.SURVEYED, {"v": survey.SetRight(0.5, None, 1, 2)}, where)  # found right
    turned = poses.Correction(0.0, 0.0, 8.0)
    poses.save(cli.POSES, {i: poses.Verdict(turned, turned, 400, where[i]) for i in ("v", "bd")})
    monkeypatch.setattr(survey, "APPLIED", "all")
    monkeypatch.setattr(poses, "APPLIED", "all")

    placed = {p.id: p.azimuth for p in cli._placed(pictures)}

    assert placed == {"v": 90.0, "bd": pytest.approx(82.0)}


def test_a_lidar_patch_that_cannot_be_read_keeps_the_panorama_where_it_stands(monkeypatch):
    def broken(x, y, folder, client):
        raise cli.httpx.ConnectError("no network")

    monkeypatch.setattr(cli.poses, "in_street", broken)

    assert cli._in_street_or_not(None)(1.0, 2.0) is False


def test_a_photo_matched_again_has_its_features_extracted_when_not_kept(monkeypatch):
    matcher = object()
    monkeypatch.setattr(cli, "_carrying_views", lambda: SimpleNamespace(matcher=matcher))

    assert cli._photo_store().matcher is matcher  # not None: it could not extract them


def test_a_photo_matched_again_keeps_the_pairs_one_flat_warp_agrees_on(monkeypatch):
    rng = np.random.default_rng(3)
    mine = rng.uniform(0, 600, (40, 2))
    theirs = mine * 1.5 + [40.0, 20.0]
    theirs[:8] = rng.uniform(0, 1280, (8, 2))  # matched wrongly
    features = lambda points: match.Features(points, np.zeros((len(points), 8)), (600, 800))  # noqa: E731
    matcher = SimpleNamespace(
        features=lambda cut: features(theirs),
        pairs=lambda a, b: np.column_stack([np.arange(40), np.arange(40)]),
    )
    store = SimpleNamespace(matcher=matcher)
    monkeypatch.setattr(cli, "_features", lambda store, photo_id: features(mine))
    monkeypatch.setattr(cli.carrying, "framed", lambda copy, axes: registration.ViewOf(0.0))
    monkeypatch.setattr(cli.match, "perspective", lambda *a: None)
    sphere = SimpleNamespace(id="p", path="p.jpg", axes=np.eye(3))

    found = cli._matched_again("pm:1", None, sphere, store, lambda path: np.zeros(1), (600, 800))

    assert len(found.photo_px) == 32  # the eight wrong ones left out


def test_a_copy_better_by_exactly_the_margin_stands():
    kept = {"camera": {"position": [0, 0, 0]}, "flat": 0.006, "parallax": 0.005, "held": 10}

    assert cli._copy_standing(kept) is not None  # at least 0.1 %: float noise aside


def test_a_copy_whose_panorama_picture_is_missing_is_not_judged_but_tried_again(monkeypatch):
    mine = match.Features(np.zeros((40, 2)), np.zeros((40, 8)), (600, 800))
    monkeypatch.setattr(cli, "_features", lambda store, photo_id: mine)
    sphere = SimpleNamespace(id="p", path="p.jpg", axes=np.eye(3))

    with pytest.raises(FileNotFoundError):  # not "no pairs": nothing kept, done next time
        cli._matched_again("pm:1", None, sphere, None, lambda path: None, (600, 800))


def test_photo_pixels_matched_again_are_given_at_the_placement_s_size(monkeypatch):
    rng = np.random.default_rng(3)
    mine = rng.uniform(0, 600, (40, 2))
    theirs = mine * 1.5 + [40.0, 20.0]
    features = lambda points, size: match.Features(points, np.zeros((len(points), 8)), size)  # noqa: E731
    matcher = SimpleNamespace(
        features=lambda cut: features(theirs, (1280, 1280)),
        pairs=lambda a, b: np.column_stack([np.arange(40), np.arange(40)]),
    )
    monkeypatch.setattr(cli, "_features", lambda store, photo_id: features(mine, (600, 800)))
    monkeypatch.setattr(cli.carrying, "framed", lambda copy, axes: registration.ViewOf(0.0))
    monkeypatch.setattr(cli.match, "perspective", lambda *a: None)
    sphere = SimpleNamespace(id="p", path="p.jpg", axes=np.eye(3))

    found = cli._matched_again(
        "pm:1", None, sphere, SimpleNamespace(matcher=matcher), lambda path: np.zeros(1), (300, 400)
    )

    assert np.allclose(found.photo_px, mine / 2)  # its features were at twice the size


def test_a_copy_is_drawn_standing_while_its_walls_hold_and_where_it_stands_better(
    monkeypatch, tmp_path
):
    stood = {("pm:1", "a"): (0.006, 0.003, 30), ("pm:1", "b"): (0.006, 0.0055, 30)}
    walls = {"top": 40.0}
    _copies(tmp_path, monkeypatch, walls, stood)
    held, carried = _held_and_carried()
    cli._copies_standing(held, carried)  # at night

    drawn = cli._copies_drawn(held, carried)
    walls["top"] = 45.0  # a wall drawn again since: back to the matrix until the next night
    later = cli._copies_drawn(held, carried)

    assert [("standing" in c) for c in drawn["pm:1"]] == [True, False]
    assert drawn["pm:1"][0]["standing"]["position"] == [1.0, 0.5, 0.2]
    assert not any("standing" in c for c in later["pm:1"])
    assert drawn["pm:flat"] == carried["pm:flat"]["copies"]


def test_a_photo_shown_from_a_proposal_said_yes_to_has_no_copy_standing(monkeypatch, tmp_path):
    stood = {("pm:1", p): (0.006, 0.003, 30) for p in "ab"}
    made = _copies(tmp_path, monkeypatch, {"top": 40.0}, stood)
    held, carried = _held_and_carried()
    monkeypatch.setattr(cli, "_matches", lambda: {"pm:1": {"inputs": "i", "tried": 3}})  # unplaced

    cli._copies_standing(held, carried)
    drawn = cli._copies_drawn(held, carried)

    assert made == [] and drawn["pm:1"] == carried["pm:1"]["copies"]


def _rays(n: int) -> match.Rays:
    return match.Rays(
        np.arange(n),
        np.full(n, "p", dtype=object),
        np.zeros((n, 3)),
        np.tile([0.0, 1.0, 0.0], (n, 1)),
        np.full((n, 3), np.nan),
        np.full((n, 6), np.nan),
    )


def test_a_photo_checked_keeps_only_rays_too_few_for_match_alone(monkeypatch):
    camera = match.Camera(1.0, 2.0, 1.5, 90.0, 0.5, 60.0, agree=30, panoramas=2)
    placed = lambda observed, where, trace, *more: match.Found(  # noqa: E731
        camera if len(observed.rays.keys) >= 100 else None, np.zeros((0, 6)), None
    )
    monkeypatch.setattr(cli.match, "place", placed)
    observed = match.Observed(np.zeros((800, 2)), (60, 80), _rays(800), 1)

    cut = cli._degraded(observed, None, np.random.default_rng(0))

    assert 50 <= len(cut.rays.keys) < 100  # halved until match alone no longer places it
    assert set(cut.rays.keys) <= set(range(800))


def test_a_photo_match_still_places_from_a_few_rays_is_left_out_of_the_check(monkeypatch):
    camera = match.Camera(1.0, 2.0, 1.5, 90.0, 0.5, 60.0, agree=30, panoramas=2)
    monkeypatch.setattr(
        cli.match, "place", lambda observed, where, trace, *more: match.Found(camera, None, None)
    )
    observed = match.Observed(np.zeros((40, 2)), (60, 80), _rays(40), 1)

    assert cli._degraded(observed, None, np.random.default_rng(0)) is None


def test_photos_checked_together_stand_in_the_same_block():
    assert cli._check_fold(652060.0, 6861010.0) == cli._check_fold(652190.0, 6861140.0)
    folds = {cli._check_fold(652000.0 + 150 * i, 6861000.0) for i in range(40)}
    assert folds == set(range(cli.CHECK_FOLDS))


def test_the_check_hides_its_photos_and_sets_each_decoy_where_another_stands():
    view = lambda x: match.View(x, 0.0, 30.0, 0.0, 0.5, 60.0)  # noqa: E731
    anchor = lambda i, x: SimpleNamespace(id=i, view=view(x))  # noqa: E731
    work = object.__new__(cli._Chaining)
    work.round_zero = {"h": anchor("h", 0.0), "d": anchor("d", 100.0), "a": anchor("a", 10.0)}
    work.where = {
        i: match.Where(None, around=(x, 0.0)) for i, x in (("h", 0), ("d", 100), ("a", 10))
    }
    work.negatives, work.cut, work._seen = {}, {}, {"a": "as matched"}
    me = lambda i: SimpleNamespace(id=i)  # noqa: E731

    replay = cli._replay(work, {"h": "cut h"}, {"d": ("cut d", "h")})

    assert set(replay.round_zero) == {"a"} and set(work.round_zero) == {"h", "d", "a"}
    assert replay.observed(me("h")) == "cut h" and replay.observed(me("a")) == "as matched"
    assert work.observed(me("a")) == "as matched" and not work.cut
    assert replay.where["d"].around == (0, 0.0)  # the decoy is said to be where h stands
    assert [a.id for a in replay.anchors(me("d"), replay.round_zero)] == ["a"]


@pytest.mark.parametrize(
    ("answered", "wrong", "fooled", "passed"),
    [(9, 0, 0, False), (10, 1, 0, True), (10, 2, 0, False), (20, 0, 1, False)],
)
def test_the_check_passes_on_enough_answers_few_wrong_and_no_decoy_taken(
    answered, wrong, fooled, passed
):
    rows = [{"kind": "vérité", "off": [0.2, 0.1], "wrong": i < wrong} for i in range(answered)]
    rows += [{"kind": "vérité", "why": "sans réponse"}] * 30  # no answer: no evidence either way
    rows += [{"kind": "leurre", "wrong": i < fooled} for i in range(3)] + [{"kind": "leurre"}]

    assert cli._check_passed(rows)[0] is passed


def test_a_lomag_keypoint_beside_a_disk_one_is_that_detail():
    disk = np.array([[0.0, 0.0], [10.0, 10.0]])
    lomag = np.array([[10.5, 10.0], [50.0, 50.0], [1.0, 1.0]])

    assert cli._merged(disk, lomag, 3).tolist() == [1, 4, 0]
    assert cli._merged(np.empty((0, 2)), lomag, 3).tolist() == [3, 4, 5]


class _Kept:
    """A store of features and pairs, as chain.Store answers them."""

    def __init__(self, keypoints: dict[str, list], pairs: dict[str, list]):
        self.keypoints, self.found = keypoints, pairs

    def features(self, photo_id, image, read):
        k = np.array(self.keypoints[photo_id], float).reshape(-1, 2)
        return match.Features(k, np.zeros((len(k), 4), np.float32), (60, 80))

    def pairs(self, a, b, features):
        return np.array(self.found[b[0]], int).reshape(-1, 2)


def _two_matchers(monkeypatch) -> cli._Chaining:
    monkeypatch.setattr(cli, "_image", lambda photo_id: "img")
    work = object.__new__(cli._Chaining)
    work.store = _Kept(
        {"me": [[0, 0], [10, 10]], "L": [[0, 0]] * 4, "D": [[0, 0]] * 4}, {"D": [[0, 3]]}
    )
    work.lomag_store = _Kept(
        {"me": [[10.5, 10], [50, 50], [10.2, 10]], "L": [[1, 1]] * 3, "D": [[2, 2]]},
        {"L": [[0, 0], [1, 2], [2, 1]]},
    )
    work.by_lomag = {"L"}
    work._lomag_keypoints, work._disk_keypoints, work._rows_of = {}, {}, {}
    return work


def test_a_photo_pairs_with_each_anchor_through_the_matcher_that_placed_it(monkeypatch):
    work = _two_matchers(monkeypatch)
    anchor = lambda i: work.anchor(i, None, None, "")  # noqa: E731
    observed = match.Observed(np.zeros((6, 2)), (60, 80), _rays(0))  # 2 DISK, 1 band, 3 LoMa-G

    pairs = work.paired(SimpleNamespace(id="me"), observed)

    assert [r.tolist() for r in pairs(anchor("D"))] == [[0], [3]]
    # LoMa-G rows: its keypoints beside DISK's are theirs (1), the others after them (4);
    # the anchor's after its own DISK keypoints (4 + 0, 4 + 2); one detail once per anchor
    assert [r.tolist() for r in pairs(anchor("L"))] == [[1, 4], [4, 6]]


def test_only_an_anchor_lomag_placed_has_its_lomag_keypoints_after_its_disk_ones(monkeypatch):
    work = _two_matchers(monkeypatch)
    view = match.View(0.0, 0.0, 30.0, 0.0, 0.5, 60.0)

    anchor = work.anchor("me", view, None, "inputs")

    assert anchor.keypoints.tolist() == [[0, 0], [10, 10]]  # paired through DISK: its rows
    lomag_placed = work.anchor("L", view, None, "inputs")  # paired through LoMa-G: after them
    assert lomag_placed.keypoints.tolist() == [[0, 0]] * 4 + [[1, 1]] * 3


def test_a_photo_only_lomag_observed_has_its_rays_on_its_lomag_keypoints_rows(monkeypatch):
    work = _two_matchers(monkeypatch)
    monkeypatch.setattr(cli, "_inputs", lambda scene, photo_id: "now")
    rays = _rays(4)  # on its LoMa-G keypoints 0, 1, 2, and on a band zoomed on a landmark (3)
    lomag_seen = match.Observed(np.zeros((4, 2)), (60, 80), rays, 2)
    work.store.observed = lambda photo_id, inputs: None  # DISK: never observed it
    work.lomag_store.observed = lambda photo_id, inputs: lomag_seen if inputs == "then" else None
    work.scenes, work.cut, work._seen = {"me": None}, {}, {}
    work.situated = {"me": SimpleNamespace(inputs="then:lomag")}

    seen = work.observed(SimpleNamespace(id="me"))

    # its details: 2 DISK keypoints, then its 3 LoMa-G ones (the first and last are DISK's
    # second); the ray on the band left out
    assert len(seen.pixels) == 5 and seen.size == (60, 80) and seen.views == 2
    assert seen.rays.keys.tolist() == [1, 2 + 1, 1]


def test_the_chaining_check_says_for_each_area_whether_it_passes():
    back = {"kind": "vérité", "off": [0.2, 0.1], "wrong": False}
    rows = [back | {"area": "cite"}] * 10 + [back | {"area": "saint-severin"}] * 3

    by_area = cli._by_area(rows)

    assert cli._check_passed(by_area["cite"]).passed
    assert not cli._check_passed(by_area["saint-severin"]).passed


def test_an_area_trusts_the_chaining_where_its_check_passed_for_things_as_they_are(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cli, "CHAIN_CHECK", tmp_path / "check.json")
    monkeypatch.setattr(cli, "CHAINS", tmp_path / "chains.json")
    monkeypatch.setattr(cli, "_chain_setup", lambda: "now")
    monkeypatch.setattr(cli, "_per_area", lambda path, matcher: {})
    monkeypatch.setattr(cli, "_check_current", lambda check, m, name, near_by: False)
    gate = lambda passed: {"passed": passed}  # noqa: E731
    areas = {"cite": gate(True), "saint-severin": gate(False)}
    cli.write_json(cli.CHAIN_CHECK, {"setup": "now", "areas": areas})
    cli.write_json(cli.CHAINS, {"setup": "now", "photos": {}})

    assert cli._checked_in([]) == {"saint-severin": [], "cite": [placed.CHAINED]}
    cli.write_json(cli.CHAINS, {"setup": "before", "photos": {}})
    assert cli._checked_in([]) == {"saint-severin": [], "cite": []}  # chained by other code
    cli.write_json(cli.CHAIN_CHECK, {"setup": "before", "areas": areas})
    assert cli._checked_in([]) == {"saint-severin": [], "cite": []}  # checked with other code


def test_a_chained_placement_stays_while_of_the_photo_now_and_its_anchors_placed(
    tmp_path, monkeypatch
):
    exported = tmp_path / "old.jpg"
    exported.touch()
    said_yes = {"photo_id": "kept-yes", "verdict": "yes", "inputs": "i:from-ab12cd34"}
    (tmp_path / "anchors.jsonl").write_text(json.dumps(said_yes) + "\n")
    row = lambda inputs, *from_: {"situated": _kept(inputs=inputs), "from": list(from_)}  # noqa: E731
    cli.write_json(
        tmp_path / "chains.json",
        {
            "setup": "chaining's code and panoramas then",
            "photos": {
                "same": row("i:from-ab12cd34", "placed"),
                "cropped": row("j:from-ab12cd34", "placed"),
                "child": row("i:from-ab12cd34", "gone"),  # its anchor no longer placed
                "grandchild": row("i:from-ab12cd34", "child"),
                "kept-yes": row("i:from-ab12cd34", "gone"),  # said yes to: holds
                "after": row("i:from-ab12cd34", "same"),  # from one chained before it
                "alone": row("i", "placed"),  # placed without anchors: no points of its own
            },
        },
    )
    for name, value in {
        "CHAINS": tmp_path / "chains.json",
        "ANCHORS": tmp_path / "anchors.jsonl",
        "_scenes_now": lambda: {
            k: "s" for k in ("same", "cropped", "child", "grandchild", "kept-yes", "after", "alone")
        },
        "_inputs": lambda scene, photo_id: "i",  # each photo as it is now
        "_exported": lambda photo_id: exported,
    }.items():
        monkeypatch.setattr(cli, name, value)

    chained = cli._chained(cli._scenes_now(), cli._Judged(), {"placed"})

    assert set(chained["photos"]) == {"same", "kept-yes", "after"}
