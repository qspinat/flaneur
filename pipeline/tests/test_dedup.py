import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageEnhance
from shapely.geometry import Point

from flaneur import dedup


def _street(seed: int) -> Image.Image:
    """A made-up photo: shapes of every size and grey, with corners to match."""
    rng = np.random.default_rng(seed)
    image = Image.new("L", (640, 480), 128)
    draw = ImageDraw.Draw(image)
    for _ in range(400):
        x, y = rng.integers(0, 640), rng.integers(0, 480)
        w, h = rng.integers(4, 60, size=2)
        shape = draw.rectangle if rng.random() < 0.5 else draw.ellipse
        shape((x, y, x + w, y + h), fill=int(rng.integers(0, 256)))
    return image


def _on_mount(print_: Image.Image) -> Image.Image:
    mount = Image.new("L", (print_.width + 200, print_.height + 200), 225)
    mount.paste(print_, (100, 100))
    return mount


def test_prints_of_one_negative_are_grouped_whatever_their_tone_and_mount():
    negative = _street(1)
    prints = {
        "a": negative,
        "b": _on_mount(ImageEnhance.Brightness(negative).enhance(0.8)),
        "other": _street(2),
    }

    found = dedup.groups({k: dedup.print_of(v) for k, v in prints.items()})

    assert found == [{"a", "b"}]


@pytest.mark.parametrize(("apart_m", "grouped"), [(30.0, True), (500.0, False)])
def test_prints_far_apart_in_tone_are_compared_when_their_captions_place_them_near(
    monkeypatch, apart_m, grouped
):
    monkeypatch.setattr(dedup, "CANDIDATE_BITS", -1)  # their fingerprints never close enough
    negative = _street(1)
    prints = {"a": negative, "b": ImageEnhance.Brightness(negative).enhance(0.8)}
    places = {"a": Point(652000.0, 6861000.0), "b": Point(652000.0 + apart_m, 6861000.0)}

    found = dedup.groups(prints, places)

    assert found == ([{"a", "b"}] if grouped else [])


def test_a_frame_shared_by_a_collection_does_not_make_two_photos_one(monkeypatch):
    monkeypatch.setattr(dedup, "CANDIDATE_BITS", dedup.FINGERPRINT_SIDE**2)  # compare all
    frame = _street(3)
    photos = {}
    for name, seed in (("first", 4), ("second", 5)):
        photo = frame.copy()
        photo.paste(_street(seed).crop((60, 45, 580, 435)), (60, 45))  # inside a frame
        photos[name] = photo

    assert dedup.groups(photos) == []


def test_a_negative_is_dated_by_what_all_its_prints_allow():
    assert dedup.years([(1905, 1906), (1885, 1925), (None, None)]) == (1905, 1906)
    assert dedup.years([(1898, 1899), (1900, None)]) == (1898, 1900)  # they contradict
    assert dedup.years([(None, None)]) == (None, None)


def test_a_print_dated_only_by_its_end_year_still_counts():
    assert dedup.years([(None, 1905), (1910, 1912)]) == (1905, 1912)


def test_a_negative_is_named_after_an_openly_licensed_print_first(tmp_path):
    from flaneur import db

    conn = db.connect(tmp_path / "f.sqlite")
    for photo_id, license in (("ak:0", "unverified"), ("ak:1", "CC-BY-4.0")):
        conn.execute(
            "INSERT INTO photos (id, source, source_id, license, credit, raw) "
            f"VALUES ('{photo_id}', 'ak', '{photo_id}', '{license}', 'x', '{{}}')"
        )

    dedup.save(conn, [{"ak:0", "ak:1"}])

    assert {r[0] for r in conn.execute("SELECT negative FROM copies")} == {"ak:1"}
