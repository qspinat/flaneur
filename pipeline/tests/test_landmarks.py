"""Landmarks: marked in a photo by hand, or seen from a known camera on the check."""

import json

import pytest

from flaneur import landmarks
from flaneur.blocks import Blocks
from flaneur.landmarks import Landmark
from flaneur.match import View


def facing(heading: float) -> View:
    """A camera at the origin, 1.6 m up, seeing 60° across."""
    return View(0.0, 0.0, 1.6, heading, 0.5, 60.0)


def test_the_landmarks_marked_in_a_photo_are_read_with_their_alternatives(tmp_path):
    path = tmp_path / "sights.jsonl"
    rows = [
        {"photo_id": "pm:1", "image": "a1", "landmarks": ["saint-severin-tour"], "column": 0.4},
        {"photo_id": "pm:1", "image": "a1", "landmarks": ["notre-dame-tours"], "column": 0.7},
        {
            "photo_id": "pm:2",
            "image": "b2",
            "landmarks": ["notre-dame-fleche", "sainte-chapelle-fleche"],
            "column": 0.6,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    marked = landmarks.marked(path)

    assert marked["pm:1"].image == "a1"
    assert [s.column for s in marked["pm:1"].sights] == [0.4, 0.7]
    assert marked["pm:2"].sights[0].points == (
        landmarks.LANDMARKS["notre-dame-fleche"].at,
        landmarks.LANDMARKS["sainte-chapelle-fleche"].at,
    )
    assert landmarks.marked(tmp_path / "none.jsonl") == {}


def test_a_landmark_ahead_is_seen_at_its_column():
    tower = {"tower": Landmark((100.0, 300.0), 50.0, "tour")}

    [sight] = landmarks.in_view(facing(10.0), 0.0, Blocks.of([]), 2000, tower)

    assert sight.points == ((100.0, 300.0),)
    assert sight.column == pytest.approx(0.5 + 0.5 * 0.1584 / 0.5774, abs=0.01)  # 8.4° right


def test_a_landmark_behind_the_camera_or_out_of_the_frame_is_not_seen():
    tower = {"tower": Landmark((100.0, 300.0), 50.0, "tour")}

    assert landmarks.in_view(facing(190.0), 0.0, Blocks.of([]), 2000, tower) == ()
    assert landmarks.in_view(facing(60.0), 0.0, Blocks.of([]), 2000, tower) == ()


def test_a_landmark_not_standing_that_year_is_not_seen():
    spire = {"spire": Landmark((100.0, 300.0), 50.0, "flèche", ((1859, 2019), (2024, None)))}

    assert landmarks.in_view(facing(10.0), 0.0, Blocks.of([]), 2021, spire) == ()


@pytest.mark.parametrize(
    ("years", "stands"),
    [
        ([1850, 1855], False),
        ([1855, 1860], True),  # its span reaches 1859
        ([2020, 2023], False),
        ([1900, None], True),
        ([None, None], True),  # undated: it may stand
        ([], True),
    ],
)
def test_a_landmark_stands_in_a_photo_when_their_years_meet(years, stands):
    spire = Landmark((0.0, 0.0), 50.0, "flèche", ((1859, 2019), (2024, None)))

    assert spire.stands(years) is stands


def test_only_landmarks_among_the_buildings_known_are_checked():
    block = Blocks.of([([(0, 0), (500, 0), (500, 500), (0, 500)], 0.0, 10.0)])
    inside = Landmark((100.0, 300.0), 50.0, "dedans")
    beyond = Landmark((900.0, 300.0), 50.0, "au-delà")

    assert landmarks.among(block, {"in": inside, "out": beyond}) == {"in": inside}


def test_a_landmark_hidden_by_a_taller_building_is_not_seen():
    tower = {"tower": Landmark((0.0, 300.0), 50.0, "tour")}
    wall = Blocks.of([([(-30, 20), (30, 20), (30, 30), (-30, 30)], 0.0, 25.0)])  # 25 m, 20 m ahead
    low = Blocks.of([([(-30, 20), (30, 20), (30, 30), (-30, 30)], 0.0, 3.0)])

    assert landmarks.in_view(facing(0.0), 0.0, wall, 2000, tower) == ()
    assert len(landmarks.in_view(facing(0.0), 0.0, low, 2000, tower)) == 1


@pytest.mark.parametrize(
    "row",
    [
        {"photo_id": "pm:1", "image": "a1", "landmarks": ["notre-dame-tour"], "column": 0.4},
        {"photo_id": "pm:1", "image": "a1", "landmarks": ["notre-dame-tours"]},
        {"photo_id": "pm:1", "image": "a1", "landmarks": ["notre-dame-tours"], "column": 1.4},
    ],
)
def test_a_mark_written_wrong_is_reported_with_its_line(tmp_path, row):
    path = tmp_path / "sights.jsonl"
    path.write_text("\n" + json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="sights.jsonl:2"):
        landmarks.marked(path)


@pytest.mark.parametrize(
    "row",
    [
        {"landmarks": ["notre-dame-tours"], "column": 0.4},
        {"photo_id": "pm:1", "image": "a1", "none": True, "landmarks": ["typo"]},
        {"photo_id": "pm:1", "image": "a1", "landmarks": ["notre-dame-tours"], "column": True},
        [1],
    ],
)
def test_a_row_without_its_photo_or_mixing_kinds_is_reported_with_its_line(tmp_path, row):
    path = tmp_path / "sights.jsonl"
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="sights.jsonl:1"):
        landmarks.rows(path)


def test_a_photo_marked_on_two_images_is_reported(tmp_path):
    path = tmp_path / "sights.jsonl"
    rows = [
        {"photo_id": "pm:1", "image": "a1", "none": True},
        {"photo_id": "pm:1", "image": "b2", "landmarks": ["notre-dame-tours"], "column": 0.4},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="sights.jsonl:2"):
        landmarks.rows(path)
