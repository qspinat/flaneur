"""Landmarks proposed in photos: predicted from where and when, then looked for."""

import json

import cv2
import numpy as np
import pytest

from flaneur import landmarks, match, proposals

KNOWN = [*landmarks.LANDMARKS, *proposals.BUILDINGS]  # all with references


def test_the_landmarks_looked_for_stand_then_and_near_nearest_first():
    tours = landmarks.LANDMARKS["notre-dame-tours"].at

    found = proposals.candidates(tours, [1850], KNOWN)

    assert found[0] == "notre-dame"  # its towers and spire, as one building
    assert "sainte-chapelle-fleche" not in found  # raised in 1853
    assert "tour-eiffel" not in found  # 4 km away, and not yet built
    assert len(found) == proposals.MOST


def test_only_landmarks_with_references_are_looked_for_none_taking_their_place():
    tours = landmarks.LANDMARKS["notre-dame-tours"].at
    every = proposals.candidates(tours, [1900], KNOWN)

    found = proposals.candidates(tours, [1900], every[1:] + ["tour-saint-jacques"])

    assert found[: len(every) - 1] == every[1:]  # the nearest has none: the next go first
    assert "notre-dame" not in found


def test_a_find_of_a_building_stands_for_its_parts_standing_then():
    assert proposals.parts_of("notre-dame", [1855]) == ("notre-dame-tours",)  # spire: 1859
    assert proposals.parts_of("notre-dame", [1900]) == ("notre-dame-tours", "notre-dame-fleche")
    assert proposals.parts_of("saint-severin-tour", [1855]) == ("saint-severin-tour",)


def test_a_buildings_parts_are_looked_for_in_one_box_around_them_picture_by_picture():
    listed = {
        "notre-dame-fleche": {"a/1.jpg": [(0.4, 0.0, 0.5, 0.6)], "b/2.jpg": [(0.4, 0.0, 0.5, 0.6)]},
        "notre-dame-tours": {"a/1.jpg": [(0.1, 0.1, 0.3, 0.5)]},
        "saint-severin-tour": {"c/3.jpg": [(0.2, 0.2, 0.4, 0.9)]},
    }

    assert proposals.joined(listed) == {
        "saint-severin-tour": {"c/3.jpg": [(0.2, 0.2, 0.4, 0.9)]},
        "notre-dame": {"a/1.jpg": [(0.1, 0.0, 0.5, 0.6)], "b/2.jpg": [(0.4, 0.0, 0.5, 0.6)]},
    }


def test_references_are_read_by_landmark_each_with_its_boxes(tmp_path):
    rows = tmp_path / "refs.jsonl"
    both = {"landmark": "notre-dame-fleche", "box": [0.4, 0.0, 0.5, 0.6]}
    tours = {"landmark": "notre-dame-tours", "box": [0.1, 0.1, 0.3, 0.5]}
    rows.write_text(
        json.dumps({"file": "a/1.jpg", "boxes": [both, tours], **SOURCE})
        + "\n"
        + json.dumps({"file": "b/2.jpg", "boxes": [both], **SOURCE})
        + "\n"
    )

    found = proposals.boxes_of(rows)

    assert found == {
        "notre-dame-fleche": {"a/1.jpg": [(0.4, 0.0, 0.5, 0.6)], "b/2.jpg": [(0.4, 0.0, 0.5, 0.6)]},
        "notre-dame-tours": {"a/1.jpg": [(0.1, 0.1, 0.3, 0.5)]},
    }


SOURCE = {"title": "Paris.jpg", "page": "https://commons.wikimedia.org/x", "licence": "CC0"}


class Sizes:
    """A matcher whose features of a picture are only its size."""

    def features(self, picture: np.ndarray) -> match.Features:
        return match.Features(np.zeros((1, 2)), np.zeros((1, 128)), picture.shape[:2])


def test_a_building_is_looked_for_in_the_pictures_of_its_parts_standing_then(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        cv2.imwrite(str(tmp_path / name / "1.jpg"), np.zeros((100, 200, 3), np.uint8))
    spire = {"landmark": "notre-dame-fleche", "box": [0.5, 0.0, 0.6, 0.5]}
    towers = {"landmark": "notre-dame-tours", "box": [0.0, 0.2, 0.4, 0.8]}
    rows = tmp_path / "refs.jsonl"
    rows.write_text(
        json.dumps({"file": "a/1.jpg", "boxes": [spire, towers], **SOURCE})
        + "\n"
        + json.dumps({"file": "b/1.jpg", "boxes": [spire], **SOURCE})  # its spire alone
        + "\n"
    )
    references = proposals.References(rows, tmp_path, Sizes())

    before = references.of("notre-dame", ["notre-dame-tours"])  # no spire yet
    after = references.of("notre-dame", NOTRE_DAME)

    assert [f.size for f in before] == [(80, 120)]  # a's box around both: towers to spire
    assert sorted(f.size for f in after) == [(50, 20), (80, 120)]


@pytest.mark.parametrize(
    ("box", "source"),
    [
        ({"landmark": "notre-dame-tours", "box": [0.1, 0.1, 0.3, 0.5]}, {}),  # no licence
        ({"landmark": "tour-montparnasse", "box": [0.1, 0.1, 0.3, 0.5]}, SOURCE),  # unknown
        ({"landmark": "notre-dame-tours", "box": [0.3, 0.1, 0.1, 0.5]}, SOURCE),  # right of left
        ({"landmark": "notre-dame-tours", "box": [0.1, 0.1, 0.3, 1.5]}, SOURCE),  # out of it
    ],
)
def test_a_reference_badly_written_says_where(tmp_path, box, source):
    rows = tmp_path / "refs.jsonl"
    rows.write_text("\n" + json.dumps({"file": "a/1.jpg", "boxes": [box], **source}) + "\n")

    with pytest.raises(ValueError, match="refs.jsonl:2"):
        proposals.boxes_of(rows)


def test_a_box_is_cut_from_its_picture_and_shrunk_when_large():
    picture = np.zeros((2000, 3000, 3), np.uint8)
    picture[200:1000, 1500:2100] = 255  # the box's part: 600 wide, 800 high

    part = proposals.cut(picture, (0.5, 0.1, 0.7, 0.5))

    assert part.shape[:2] == (800, 600) and part.min() == 255
    big = proposals.cut(picture, (0.0, 0.0, 1.0, 1.0))
    assert max(big.shape[:2]) == proposals.REFERENCE_SIDE


@pytest.mark.parametrize(
    ("height", "width", "corners"),
    [
        (1200, 1600, [(0, 0), (400, 0), (800, 0)]),  # landscape: one row over its upper part
        (1600, 1200, [(0, 0), (400, 0), (0, 160), (400, 160)]),  # portrait: two rows
        (300, 500, [(0, 0)]),  # smaller than a tile: whole
        (1200, 1201, [(0, 0), (401, 0)]),  # one a pixel short of the last: left out
    ],
)
def test_tiles_cover_the_photos_upper_part(height, width, corners):
    photo = np.zeros((height, width, 3), np.uint8)

    found = proposals.tiles(photo)

    assert [(x, y) for x, y, _ in found] == corners
    covered = np.zeros((height, width), bool)
    for x, y, tile in found:
        covered[y : y + tile.shape[0], x : x + tile.shape[1]] = True
    assert covered[: int(proposals.UPPER * height)].all()


def _seen_by_two_cameras(n, rng):
    """Pixels of n points of a scene in two cameras looking at it from 2 m apart."""
    world = np.column_stack([rng.uniform(-4, 4, n), rng.uniform(-3, 3, n), rng.uniform(8, 20, n)])

    def pixels(at, turn):
        c, s = np.cos(turn), np.sin(turn)
        local = (world - at) @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]).T
        return (500 * local[:, :2] / local[:, 2:] + 400).astype(np.float32)

    return pixels(np.zeros(3), 0.0), pixels(np.array([2.0, 0.0, 0.0]), 0.15)


def test_pairs_find_the_landmark_when_sure_and_agreeing_on_one_geometry():
    rng = np.random.default_rng(5)
    in_photo, in_box = _seen_by_two_cameras(60, rng)
    wrong = rng.uniform(0, 800, (20, 2)).astype(np.float32)  # pairs agreeing with nothing
    none = np.zeros((80, 128), np.float32)  # descriptors: not looked at
    photo = match.Features(np.vstack([in_photo, wrong]), none, (800, 800))
    box = match.Features(
        np.vstack([in_box, rng.uniform(0, 800, (20, 2)).astype(np.float32)]), none, (800, 800)
    )
    pairs = np.column_stack([np.arange(80), np.arange(80)])
    sure = np.full(80, 0.9, np.float32)
    sure[:10] = 0.3  # LightGlue unsure of these

    at, there = proposals.seen(pairs, sure, photo, box)

    good = {tuple(p) for p in in_photo[10:]}
    assert {tuple(p) for p in at} <= good
    assert len(at) >= 0.9 * len(good)
    assert np.allclose(there, in_box[[np.flatnonzero((in_photo == p).all(1))[0] for p in at]] / 800)
    assert len(proposals.seen(pairs, sure, photo, box, len(good) + 1)[0]) == 0


def test_a_landmarks_column_is_told_by_the_pairs_on_the_top_of_its_box():
    in_box = np.array([[0.5, 0.1], [0.5, 0.2], [0.5, 0.6], [0.5, 0.7], [0.5, 0.8]])
    in_photo = np.array([[100, 0], [110, 0], [300, 0], [310, 0], [320, 0]], np.float32)

    assert proposals.column_of(in_photo, in_box) == 105  # its top third: down to 0.33
    below = in_box + [0, 0.3]  # none on the top: all count
    assert proposals.column_of(in_photo, below) == 300


SOUTH = (651973.0, 6861453.0)  # 100 m south of Saint-Séverin's tower: 350-450 m from the rest
NOTRE_DAME = ("notre-dame-tours", "notre-dame-fleche")


def test_a_proposal_agrees_with_a_mark_of_a_same_landmark_as_near_as_the_camera_must_see_it():
    mark = {"landmarks": ["notre-dame-fleche", "sainte-chapelle-fleche"], "column": 0.5}

    def agree(landmarks: tuple[str, ...], column: float) -> bool:
        return proposals.agree(proposals.Proposal(landmarks, column, 30), mark, SOUTH)

    assert agree(("sainte-chapelle-fleche",), 0.52)  # 2.3° off at 90° wide, within 2° + 0.6°
    assert not agree(("sainte-chapelle-fleche",), 0.6)  # 11° off
    assert agree(NOTRE_DAME, 0.5)  # its spire is one of them
    assert not agree(("saint-severin-tour",), 0.5)


def test_a_find_on_twin_towers_agrees_with_a_mark_as_far_off_as_they_stand_from_their_point():
    def agree(landmark: str) -> bool:
        mark = {"landmarks": [landmark], "column": 0.5}
        return proposals.agree(proposals.Proposal((landmark,), 0.54, 30), mark, SOUTH)

    assert agree("notre-dame-tours")  # 4.6° off, 340 m away: its towers stand 18.6 m from it
    assert not agree("notre-dame-fleche")  # a spire, 370 m away: within 2.8° only


def test_a_caption_on_the_landmark_does_not_make_every_column_agree():
    on_it = landmarks.LANDMARKS["notre-dame-tours"].at
    mark = {"landmarks": ["notre-dame-tours"], "column": 0.9}

    assert not proposals.agree(proposals.Proposal(NOTRE_DAME, 0.1, 30), mark, on_it)


def test_proposals_are_scored_against_the_marks_of_landmarks_looked_for():
    marks = [
        {"landmarks": ["saint-severin-tour"], "column": 0.4},
        {"landmarks": ["notre-dame-tours"], "column": 0.7},
        {"landmarks": ["tour-eiffel"], "column": 0.1},  # not looked for: not missed
    ]
    found = [
        proposals.Proposal(("saint-severin-tour",), 0.41, 40),
        proposals.Proposal(("pantheon-dome",), 0.2, 25),
    ]

    score = proposals.score(found, marks, ["saint-severin-tour", "notre-dame-tours"], SOUTH)

    assert score == proposals.Score(right=1, wrong=1, missed=1)


def test_a_second_proposal_on_one_mark_is_wrong():
    marks = [{"landmarks": ["notre-dame-fleche", "sainte-chapelle-fleche"], "column": 0.5}]
    found = [
        proposals.Proposal(NOTRE_DAME, 0.5, 40),
        proposals.Proposal(("sainte-chapelle-fleche",), 0.51, 30),
    ]

    assert proposals.score(found, marks, NOTRE_DAME, SOUTH) == proposals.Score(1, 1, 0)


def test_a_building_found_by_one_part_misses_none_of_its_others():
    marks = [
        {"landmarks": ["notre-dame-tours"], "column": 0.3},
        {"landmarks": ["notre-dame-fleche"], "column": 0.6},
    ]
    found = [proposals.Proposal(NOTRE_DAME, 0.6, 40)]  # on the spire

    assert proposals.score(found, marks, NOTRE_DAME, SOUTH) == proposals.Score(1, 0, 0)


def test_scores_of_several_photos_add_up():
    assert proposals.Score(1, 2, 3) + proposals.Score(4, 0, 1) == proposals.Score(5, 2, 4)


def test_spires_found_in_one_column_make_one_mark_of_either():
    found = [
        proposals.Proposal(("sainte-chapelle-fleche",), 0.51, 30),
        proposals.Proposal(NOTRE_DAME, 0.5, 40),
        proposals.Proposal(("saint-severin-tour",), 0.2, 50),
    ]

    assert proposals.marks_of(found) == [
        {"landmarks": ["saint-severin-tour"], "column": 0.2},
        {"landmarks": ["notre-dame-fleche", "sainte-chapelle-fleche"], "column": 0.5},  # spires
    ]
