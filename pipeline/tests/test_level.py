"""How a panorama's camera leaned, from the walls' vertical edges in its picture."""

import json

import cv2
import numpy as np
import pytest
from shapely.geometry import Point

from flaneur import level, match
from flaneur.gazetteer import to_degrees
from flaneur.panoramax import Picture


def street_picture(side: int = 1024) -> np.ndarray:
    """A level 360° picture (2:1) of walls all around: vertical edges every 7.5°, from 30°
    below the horizon to 40° above it; sky and street plain."""
    h, w = side, 2 * side
    picture = np.full((h, w, 3), 140, np.uint8)
    columns = np.arange(w)
    stripes = np.where((columns // (w // 48)) % 2 == 0, 60, 210).astype(np.uint8)
    top, bottom = int(h * (0.5 - 40 / 180)), int(h * (0.5 + 30 / 180))
    picture[top:bottom] = stripes[None, :, None]
    return picture


def leaned(picture: np.ndarray, roll: float, pitch: float) -> np.ndarray:
    """The picture a camera at the same spot takes when it leans `roll` degrees to its right
    (its right side dipping) and raises its nose `pitch` degrees."""
    h, w = picture.shape[:2]
    rows, columns = np.mgrid[0:h, 0:w].astype(float)
    bearing = np.radians((columns + 0.5) / w * 360 - 180)  # from the picture's centre
    elevation = np.radians(90 - (rows + 0.5) / h * 180)
    # in the camera's own axes: right, ahead (its centre), up
    own = np.stack(
        [
            np.sin(bearing) * np.cos(elevation),
            np.cos(bearing) * np.cos(elevation),
            np.sin(elevation),
        ],
        axis=-1,
    )
    r, p = np.radians(roll), np.radians(pitch)
    nose_up = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    to_right = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
    level_axes = own @ (nose_up @ to_right).T  # the same directions, in level axes
    x, y, z = np.moveaxis(level_axes, -1, 0)
    u = ((np.degrees(np.arctan2(x, y)) + 180) / 360 * w - 0.5) % w
    v = (90 - np.degrees(np.arcsin(np.clip(z, -1, 1)))) / 180 * h - 0.5
    return cv2.remap(
        picture,
        u.astype(np.float32),
        v.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


@pytest.mark.parametrize(("roll", "pitch"), [(6.0, -3.0), (-2.0, 9.0)])
def test_how_a_panorama_leaned_is_read_from_its_walls_upright_edges(roll, pitch):
    picture = leaned(street_picture(), roll, pitch)

    found = level.tilt(picture)

    assert found == pytest.approx((roll, pitch), abs=0.4)


def test_a_leaning_panorama_is_cut_into_level_views():
    picture = street_picture()
    azimuth, roll, pitch = 30.0, 6.0, -3.0
    leaning = match.Panorama("leaning", 0.0, 0.0, 2.5, azimuth, None, roll, pitch)

    for heading in (azimuth, azimuth + 90, azimuth + 200):
        want = match.perspective(picture, match.camera_axes(azimuth), heading, pitch=0.0)
        got = match.perspective(leaned(picture, roll, pitch), leaning.axes, heading, pitch=0.0)

        assert np.abs(got.astype(float) - want).mean() < 4


def slanted(picture: np.ndarray, lean: float) -> np.ndarray:
    """The picture, its vertical edges leaning `lean` degrees (tops to the right)."""
    h, w = picture.shape[:2]
    rows, columns = np.mgrid[0:h, 0:w]
    return picture[rows, (columns - (rows - h / 2) * np.tan(np.radians(lean))).astype(int) % w]


def test_one_view_whose_edges_truly_lean_does_not_tilt_the_panorama():
    picture = street_picture()
    w = picture.shape[1]
    sector = slice(int(w * (0.5 + 70 / 360)), int(w * (0.5 + 110 / 360)))  # 90° to the right
    picture[:, sector] = slanted(picture, 8)[:, sector]  # there a facade leans 8°

    found = level.tilt(leaned(picture, 6.0, -3.0))

    assert found == pytest.approx((6.0, -3.0), abs=0.4)


def test_views_that_do_not_agree_on_one_lean_give_no_tilt():
    straight = street_picture()
    picture, w = straight.copy(), straight.shape[1]
    for k, lean in enumerate([8, -6, 7, -8, 6, -7, 8, -6, 7, -8, 6, -7]):  # each view its own
        sector = slice(k * w // 12, (k + 1) * w // 12)
        picture[:, sector] = slanted(straight, lean)[:, sector]

    assert level.tilt(picture) is None


def picture_at(id: str, x: float, run: str, rank: int) -> Picture:
    """A 360° picture `x` Lambert-93 metres east along a street, the `rank`th of its capture
    `run`."""
    where = to_degrees(Point(x, 6861000.0))
    return Picture(id, where.x, where.y, 0.0, True, "2025-01-01", "", "", sequence=run, rank=rank)


def test_a_panorama_not_measured_leans_as_those_just_before_and_after_it():
    pictures = [
        picture_at("before", 652000.0, "run", 1),
        picture_at("here", 652004.0, "run", 2),
        picture_at("after", 652008.0, "run", 3),
        picture_at("further", 652009.0, "run", 4),
        picture_at("other run", 652005.0, "other", 2),
        picture_at("back again", 652004.0, "run", 90),  # the same run, passing there later
        picture_at("never measured", 652010.0, "run", 5),
        picture_at("far", 652400.0, "run", 6),
        picture_at("alone", 652400.0, "lonely", 1),
    ]
    tilts = {
        "before": [2.0, -1.0],
        "here": None,
        "after": [3.0, -2.0],
        "further": [4.0, -3.0],
        "other run": [-9.0, 9.0],
        "back again": [-9.0, 9.0],
        "far": None,
    }

    found = level.borrowed(tilts, pictures)

    assert found["here"] == pytest.approx([3.0, -2.0])
    assert found["never measured"] == pytest.approx([3.5, -2.5])
    assert found["before"] == [2.0, -1.0]
    assert found["far"] is None and "alone" not in found


def test_a_panorama_showing_no_walls_has_no_lean_measured():
    assert level.tilt(np.full((512, 1024, 3), 128, np.uint8)) is None


def test_a_lean_beyond_any_seen_is_not_trusted(monkeypatch):
    monkeypatch.setattr(level, "MOST_DEG", 5.0)

    assert level.tilt(leaned(street_picture(), 6.0, -3.0)) is None


def test_panoramas_are_measured_again_once_the_way_of_measuring_changed(tmp_path):
    blank = tmp_path / "blank.jpg"
    cv2.imwrite(str(blank), np.full((512, 1024, 3), 128, np.uint8))
    out = tmp_path / "tilts.json"
    pictures = {"a": blank, "b": blank}

    out.write_text(json.dumps({"setup": level.setup(), "tilts": {"a": [1.0, 2.0], "b": None}}))
    kept = level.measure(pictures, out)
    out.write_text(json.dumps({"setup": "before", "tilts": {"a": [1.0, 2.0], "b": None}}))
    again = level.measure(pictures, out)

    assert kept == {"a": [1.0, 2.0], "b": None}
    assert again == {"a": None, "b": None}


def test_only_the_panoramas_asked_for_are_told(tmp_path):
    out = tmp_path / "tilts.json"
    out.write_text(json.dumps({"setup": level.setup(), "tilts": {"a": [1.0, 2.0], "gone": None}}))

    assert level.measure({"a": tmp_path / "a.jpg"}, out) == {"a": [1.0, 2.0]}


def test_how_much_panoramas_lean_leaves_out_those_not_measured():
    found = level.summary({"a": [3.0, -4.0], "b": None, "c": [0.0, 1.0]})

    assert found == {"panoramas": 3, "measured": 2, "median_deg": 3.0, "p90_deg": 4.6}


def test_tilts_measured_otherwise_are_told_apart(tmp_path):
    out = tmp_path / "tilts.json"
    assert level.current(out)

    out.write_text(json.dumps({"setup": "before", "tilts": {}}))
    assert not level.current(out)

    out.write_text(json.dumps({"setup": level.setup(), "tilts": {}}))
    assert level.current(out)
