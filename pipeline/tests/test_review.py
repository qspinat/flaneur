import io
import json

import pytest
from PIL import Image

from flaneur import db, dedup, review
from flaneur.gazetteer import Gazetteer
from flaneur.pilot import TILE_SIZE, _contact_sheet


@pytest.fixture
def pilot(tmp_path):
    """A pilot folder with two placed street views (one matched, one not) and a sheet."""
    conn = db.connect(tmp_path / "f.sqlite")
    for n, match in ((1, 2), (2, None)):
        conn.execute(
            "INSERT INTO photos (id, source, source_id, title, license, credit, raw) "
            f"VALUES ('pm:{n}', 'pm', '{n}', 'Rue {n}', 'CC0', 'x', '{{}}')"
        )
        conn.execute(
            "INSERT INTO placements (photo_id, is_street_view, match, match_panoramax_id, lon, "
            f"lat, heading, confidence) VALUES ('pm:{n}', 1, ?, ?, 2.346, 48.852, 90, 'medium')",
            [match, f"pnx-{match}" if match else None],
        )
        folder = tmp_path / "pilot" / "photos" / f"pm_{n}"
        folder.mkdir(parents=True)
        (folder / "info.json").write_text(json.dumps({"sheet": "rue", "years": [1900, 1900]}))
        Image.new("RGB", (40, 30), "grey").save(folder / "photo.jpg")
    tiles = [
        (Image.new("RGB", TILE_SIZE, colour), {"panoramax_id": f"pnx-{n}"})
        for n, colour in enumerate(("white", "red", "white"), start=1)
    ]
    shown = _contact_sheet(tiles, tmp_path / "pilot" / "sheets" / "rue.jpg")  # tile #2 is red
    (tmp_path / "pilot" / "sheets" / "rue.json").write_text(json.dumps(shown))
    return conn, tmp_path / "pilot", tmp_path / "reviews.jsonl"


def test_queue_puts_unmatched_photos_first_and_skips_decided_ones(pilot):
    conn, pilot_dir, reviews = pilot
    gazetteer = Gazetteer([])

    first = review.queue(conn, pilot_dir, gazetteer, reviews)
    review.record(reviews, {"id": "photo:pm:2", "decision": "confirm", "lon": 2.3, "lat": 48.85})
    after = review.queue(conn, pilot_dir, gazetteer, reviews)

    assert [i.id for i in first] == ["photo:pm:2", "photo:pm:1"]
    assert first[1].today_url == "/tile/rue/2.jpg"
    assert [i.id for i in after] == ["photo:pm:1"]


def test_a_skipped_item_comes_back(pilot):
    conn, pilot_dir, reviews = pilot
    review.record(reviews, {"id": "photo:pm:2", "decision": "skip"})

    assert "photo:pm:2" in [i.id for i in review.queue(conn, pilot_dir, Gazetteer([]), reviews)]


def test_tile_is_the_numbered_picture_of_the_sheet(pilot):
    _, pilot_dir, _ = pilot

    image = Image.open(io.BytesIO(review.tile(pilot_dir, "rue", 2)))

    assert image.size == TILE_SIZE
    red, green, blue = image.getpixel((250, 140))
    assert red > 200 and green < 60 and blue < 60


def test_an_unknown_decision_is_refused(tmp_path):
    with pytest.raises(ValueError):
        review.record(tmp_path / "r.jsonl", {"id": "photo:x", "decision": "maybe"})


def test_files_outside_the_pilot_folder_are_not_served(pilot):
    _, pilot_dir, _ = pilot

    assert review.inside(pilot_dir, "photos/pm_1/info.json") is not None
    assert review.inside(pilot_dir, "../f.sqlite") is None


def test_an_entrance_cannot_be_judged_a_street_view(tmp_path):
    with pytest.raises(ValueError):
        review.record(
            tmp_path / "r.jsonl", {"id": "entrance:pm:1", "decision": "not_a_street_view"}
        )


def test_the_last_decision_on_an_item_is_the_one_that_counts(tmp_path):
    reviews = tmp_path / "r.jsonl"
    here = {"lon": 2.346, "lat": 48.852}
    review.record(reviews, {"id": "photo:a", "decision": "confirm", "heading": 10} | here)
    review.record(reviews, {"id": "photo:a", "decision": "fix", "heading": 40} | here)

    assert review.latest(reviews)["photo:a"]["heading"] == 40


def test_a_position_outside_paris_or_a_bad_heading_is_refused(tmp_path):
    reviews = tmp_path / "r.jsonl"
    for bad in (
        {"lon": None, "lat": 48.85},
        {"lon": 12.0, "lat": 48.85},
        {"lon": 2.34, "lat": 48.85, "heading": 720},
        [],
    ):
        with pytest.raises(ValueError):
            review.record(
                reviews,
                bad if isinstance(bad, list) else {"id": "photo:a", "decision": "fix"} | bad,
            )


def test_the_today_view_is_found_by_its_panoramax_id_when_numbers_change(pilot):
    conn, pilot_dir, reviews = pilot
    shown = json.loads((pilot_dir / "sheets" / "rue.json").read_text())
    for t in shown:  # the sheet was rebuilt: pnx-2 is now number 3
        t["n"] = {"pnx-1": 1, "pnx-2": 3, "pnx-3": 2}[t["panoramax_id"]]
    (pilot_dir / "sheets" / "rue.json").write_text(json.dumps(shown))

    (matched,) = [i for i in review.queue(conn, pilot_dir, Gazetteer([]), reviews) if i.today_url]

    assert matched.today_url == "/tile/rue/3.jpg"


def test_a_tile_number_beyond_the_sheet_is_not_found(pilot):
    _, pilot_dir, _ = pilot

    with pytest.raises(FileNotFoundError):
        review.tile(pilot_dir, "rue", 9)


def test_a_crop_is_kept_as_fractions_and_a_bad_one_is_refused(tmp_path):
    reviews = tmp_path / "r.jsonl"
    here = {"id": "photo:a", "decision": "fix", "lon": 2.346, "lat": 48.852}

    row = review.record(reviews, here | {"crop": [0.05, 0.1, 0.5, 0.9]})

    assert row["crop"] == [0.05, 0.1, 0.5, 0.9]
    for bad in ([0.5, 0.1, 0.2, 0.9], [0, 0, 1.5, 1], [0, 0, 1]):
        with pytest.raises(ValueError):
            review.record(reviews, here | {"crop": bad})


def test_a_negative_printed_twice_is_reviewed_once_on_the_print_it_is_placed_on(pilot):
    conn, pilot_dir, reviews = pilot
    dedup.save(conn, [{"pm:1", "pm:2"}])
    other = {"photo_id": "pm:2", "title": "Rue 2", "caption": None, "years": [1900, 1900]}
    for n, info in ((1, {"other_prints": [other]}), (2, {})):  # as flaneur.pilot writes them
        path = pilot_dir / "photos" / f"pm_{n}" / "info.json"
        path.write_text(json.dumps(json.loads(path.read_text()) | info | {"placed_on": "pm:1"}))

    before = review.queue(conn, pilot_dir, Gazetteer([]), reviews)
    review.record(reviews, {"id": "photo:pm:1", "decision": "confirm", "lon": 2.3, "lat": 48.85})
    after = review.queue(conn, pilot_dir, Gazetteer([]), reviews)

    assert [i.id for i in before] == ["photo:pm:1"]
    assert before[0].photos == ["pm:1", "pm:2"]
    assert [c["title"] for c in before[0].copies] == ["Rue 2"]
    assert after == []


def test_the_print_that_stands_for_a_negative_is_the_one_matched_to_today(pilot):
    conn, _, _ = pilot
    dedup.save(conn, [{"pm:1", "pm:2"}])

    assert dedup.negatives(conn)["pm:2"] == ["pm:1", "pm:2"]  # pm:1 has a matched view


def test_a_negative_is_shown_on_another_print_when_its_own_is_not_placed_as_a_street_view(
    pilot,
):
    conn, pilot_dir, reviews = pilot
    conn.execute("UPDATE placements SET is_street_view = 0 WHERE photo_id = 'pm:1'")
    db.replace(conn, "copies", [{"photo_id": i, "negative": "pm:1"} for i in ("pm:1", "pm:2")])

    assert [i.id for i in review.queue(conn, pilot_dir, Gazetteer([]), reviews)] == ["photo:pm:2"]


def test_a_today_view_matched_on_another_prints_sheet_is_still_found(pilot):
    conn, pilot_dir, reviews = pilot
    conn.execute("UPDATE placements SET match_panoramax_id = 'pnx-2' WHERE photo_id = 'pm:2'")
    path = pilot_dir / "photos" / "pm_2" / "info.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | {"sheet": "autre_rue"}))

    items = review.queue(conn, pilot_dir, Gazetteer([]), reviews)

    assert {i.id: i.today_url for i in items}["photo:pm:2"] == "/tile/rue/2.jpg"


def test_an_entrance_decision_does_not_hide_a_street_view_of_the_same_negative(pilot):
    conn, pilot_dir, reviews = pilot
    dedup.save(conn, [{"pm:1", "pm:9"}])
    review.record(
        reviews,
        {
            "id": "entrance:pm:9",
            "decision": "confirm",
            "lon": 2.3,
            "lat": 48.85,
            "photos": ["pm:9"],
        },
    )

    assert "photo:pm:1" in [i.id for i in review.queue(conn, pilot_dir, Gazetteer([]), reviews)]


def test_a_photo_whose_crop_is_in_doubt_comes_first_with_why(pilot):
    conn, pilot_dir, reviews = pilot
    info = pilot_dir / "photos" / "pm_1" / "info.json"
    info.write_text(
        json.dumps({**json.loads(info.read_text()), "crop_doubts": ["tirage coupé : haut"]})
    )

    items = review.queue(conn, pilot_dir, Gazetteer([]), reviews)

    assert [i.id for i in items] == ["photo:pm:1", "photo:pm:2"]
    assert items[0].crop_doubts == ["tirage coupé : haut"]
    assert items[1].crop_doubts is None


def test_a_seed_is_never_offered_to_review(pilot):
    conn, pilot_dir, reviews = pilot
    conn.execute("UPDATE placements SET confidence = 'seed' WHERE photo_id = 'pm:2'")

    queued = [item.id for item in review.queue(conn, pilot_dir, Gazetteer([]), reviews)]

    assert "photo:pm:1" in queued and "photo:pm:2" not in queued
