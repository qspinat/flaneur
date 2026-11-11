"""The walk's links, on a made-up street running north from 48.85 N, 2.35 E."""

import json

from flaneur import export, landmarks
from flaneur.export import WalkPhoto

NORTH_PER_M = 1 / 111_200  # degrees of latitude per metre
EAST_PER_M = 1 / 73_200  # degrees of longitude per metre, in Paris


def photo(id, north_m=0.0, east_m=0.0, heading=0.0, years=(1900, 1900)):
    return WalkPhoto(
        id=id,
        title=id,
        caption=None,
        years=list(years),
        photographer="x",
        credit="x",
        record_url=None,
        lon=2.35 + east_m * EAST_PER_M,
        lat=48.85 + north_m * NORTH_PER_M,
        heading=heading,
        street=None,
        arrondissement=None,
        image_url="",
        crop=None,
    )


def links(walk):
    export.link(walk)
    return {p.id: {k: v.id if v else None for k, v in p.links.items()} for p in walk}


def test_forward_is_the_nearest_photo_ahead_facing_the_same_way():
    walk = [
        photo("me"),
        photo("near ahead", north_m=40),
        photo("far ahead", north_m=120),
        photo("too close", north_m=4),
        photo("ahead but facing back", north_m=20, heading=180),
        photo("aside", north_m=30, east_m=40),
    ]

    assert links(walk)["me"]["forward"] == "near ahead"
    assert links(walk)["far ahead"]["back"] == "near ahead"


def test_turning_finds_a_photo_from_about_here_facing_sideways():
    walk = [photo("me"), photo("east", east_m=5, heading=90), photo("west", heading=270)]

    found = links(walk)["me"]

    assert (found["right"], found["left"]) == ("east", "west")


def test_eras_are_the_same_view_in_other_years_oldest_first():
    walk = [
        photo("1900"),
        photo("1910", north_m=5, heading=10, years=(1910, 1910)),
        photo("1865", east_m=3, years=(1865, 1868)),
        photo("same year", north_m=2),
        photo("1910 further", north_m=12, years=(1910, 1910)),
    ]
    export.link(walk)

    assert walk[0].eras == ["1865", "1910"]  # one per date: the nearest


def test_angles_wrap_around_north():
    from flaneur.gazetteer import angle

    assert angle(350, 10) == 20
    assert angle(10, 350) == -20


def two_photos(tmp_path):
    """Two placed photos of the pilot, pm:1 and pm:2."""
    from flaneur import db

    conn = db.connect(tmp_path / "f.sqlite")
    for n in (1, 2):
        conn.execute(
            "INSERT INTO photos (id, source, source_id, title, license, credit, image_url, raw) "
            f"VALUES ('pm:{n}', 'pm', '{n}', 'Rue', 'CC0', 'x', 'http://img/{n}', '{{}}')"
        )
        conn.execute(
            "INSERT INTO placements (photo_id, is_street_view, lon, lat, heading, confidence) "
            f"VALUES ('pm:{n}', 1, 2.346, 48.852, 90, 'low')"
        )
        folder = tmp_path / "pilot" / "photos" / f"pm_{n}"
        folder.mkdir(parents=True)
        (folder / "info.json").write_text(
            json.dumps({"years": [1900, 1900], "suggested_crop": [0.1, 0.1, 0.9, 0.9]})
        )
    return conn


def test_a_reviewed_place_and_crop_replace_the_proposed_ones(tmp_path):
    conn = two_photos(tmp_path)
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(
        json.dumps(
            {
                "id": "photo:pm:1",
                "decision": "fix",
                "lon": 2.347,
                "lat": 48.853,
                "heading": 100,
                "crop": [0, 0, 1, 1],
                "at": "2026-10-29T10:00:00",
            }
        )
        + "\n"
        + json.dumps(
            {"id": "photo:pm:2", "decision": "not_a_street_view", "at": "2026-10-29T10:01:00"}
        )
        + "\n"
    )

    (walk,) = export.photos(conn, tmp_path / "pilot", reviews)

    assert (walk.id, walk.lon, walk.heading, walk.reviewed) == ("pm:1", 2.347, 100, True)
    assert walk.crop is None  # the whole photo, kept on purpose


def test_street_names_are_written_as_on_their_plates():
    assert export.plate_name("rue Saint-Julien le Pauvre") == ("Rue", "Saint-Julien-le-Pauvre")
    assert export.plate_name("place Saint-André des Arts") == ("Place", "Saint-André-des-Arts")
    assert export.plate_name("rue des Prêtres Saint-Séverin") == (
        "Rue",
        "des Prêtres-Saint-Séverin",
    )
    assert export.plate_name("rue de la Parcheminerie") == ("Rue", "de la Parcheminerie")
    assert export.plate_name("rue Du Sommerard") == ("Rue", "Du Sommerard")
    assert export.plate_name("place Paul Painlevé") == ("Place", "Paul-Painlevé")


def test_a_photographer_is_named_as_one_writes_a_name():
    assert export.display_name("Atget, Eugène (Jean Eugène Auguste Atget, dit)") == "Eugène Atget"
    assert export.display_name(None) == "Photographe inconnu"


def test_a_place_fixed_far_from_the_proposal_is_a_different_view():
    proposed = {"lon": 2.35, "lat": 48.85, "heading": 0}
    near = {"decision": "fix", "lon": 2.35, "lat": 48.85 + 5 * NORTH_PER_M, "heading": 10}
    far = {"decision": "fix", "lon": 2.35, "lat": 48.85 + 40 * NORTH_PER_M, "heading": 0}
    turned = {"decision": "fix", "lon": 2.35, "lat": 48.85, "heading": 90}

    assert not export._moved(proposed, near)
    assert export._moved(proposed, far)
    assert export._moved(proposed, turned)


def test_a_broken_image_in_the_cache_leaves_the_photo_out_and_is_dropped(tmp_path):
    broken = tmp_path / "cache" / "archive" / "a.jpg"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"<html>not an image</html>")

    written = export._write_one(
        photo("a"), client=None, cache_dir=tmp_path / "cache", out_dir=tmp_path
    )

    assert written is False
    assert not broken.exists()


def test_a_photo_situated_against_today_takes_its_found_place_unless_reviewed(tmp_path):
    conn = two_photos(tmp_path)
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(
        json.dumps(
            {
                "id": "photo:pm:1",
                "decision": "confirm",
                "lon": 2.346,
                "lat": 48.852,
                "heading": 90,
                "at": "2026-10-29T10:00:00",
            }
        )
        + "\n"
    )
    found = {"lon": 2.3465, "lat": 48.8525, "heading": 120.0}

    walk = export.photos(conn, tmp_path / "pilot", reviews, {"pm:1": found, "pm:2": found})

    reviewed, situated = sorted(walk, key=lambda p: p.id)
    assert (reviewed.lon, reviewed.heading) == (2.346, 90)
    assert (situated.lon, situated.lat, situated.heading) == (2.3465, 48.8525, 120.0)
    assert not situated.reviewed


def test_images_of_photos_no_longer_in_the_walk_are_removed(tmp_path):
    me = WalkPhoto(
        "pm:a", "t", None, [None, None], "p", "c", None, 0.0, 0.0, None, None, None, "u", None
    )
    image = export.image_name(me.id)
    kept = [
        image,
        *(export.sized(image, side) for side in export.SIZES),
        f"today/{export.safe_name(me.id)}.webp",
    ]
    for name in [*kept, "photos/gone.webp", "today/gone.webp"]:
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_bytes(b"x")

    export._remove_others(tmp_path, [me])  # its download may have failed: its images stay

    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.webp")) == sorted(kept)


def test_each_size_is_listed_by_the_width_it_has():
    me = WalkPhoto(
        "pm:a", "t", None, [None, None], "p", "c", None, 0.0, 0.0, None, None, None, "u", None
    )
    me.image = export.image_name(me.id)
    me.widths = {320: 213, 960: 640, 1600: 1067}  # a portrait: its widths are 2/3 of its sides
    me.aspect = 1.5

    site = me.for_site()
    images = site["images"]
    assert site["aspect"] == 1.5  # height over width: the site fits it to its frame

    assert images == {
        "213": export.sized(me.image, 320),
        "640": export.sized(me.image, 960),
        "1067": me.image,
    }


def test_the_3d_walk_is_marked_on_the_images_as_written_and_too_far_on_answers(tmp_path):
    def photo(n):
        return WalkPhoto(
            n, "t", None, [None, None], "p", "c", None, 0.0, 0.0, None, None, None, "u", None
        )

    walk = [photo("pm:a"), photo("pm:b"), photo("pm:c")]
    for me in walk:
        (tmp_path / export.image_name(me.id)).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / export.image_name(me.id)).write_bytes(me.id.encode())
    now = landmarks.image_hash(tmp_path / export.image_name("pm:a"))

    export.mark_3d(walk, tmp_path, {"pm:a": now, "pm:b": "an image it had before"}, {"pm:c"})

    assert [(me.in_3d, me.no_3d) for me in walk] == [(True, False), (False, False), (False, True)]


def _reviewed(tmp_path, at):
    reviews = tmp_path / "reviews.jsonl"
    fix = {
        "id": "photo:pm:1",
        "decision": "fix",
        "lon": 2.346,
        "lat": 48.852,
        "heading": 90,
        "at": at,
    }
    reviews.write_text(json.dumps(fix) + "\n")
    return reviews


PLACED = {"pm:1": {"lon": 2.3462, "lat": 48.8523, "heading": 75.0, "at": "2026-11-04T10:00:00"}}


def test_a_place_set_on_the_placing_page_after_the_review_wins_over_it(tmp_path):
    conn = two_photos(tmp_path)

    walk = export.photos(
        conn, tmp_path / "pilot", _reviewed(tmp_path, "2026-10-29T10:00:00"), {}, PLACED
    )

    me = next(p for p in walk if p.id == "pm:1")
    assert (me.lon, me.lat, me.heading, me.reviewed) == (2.3462, 48.8523, 75.0, True)


def test_a_review_made_after_the_placing_page_wins_over_it(tmp_path):
    conn = two_photos(tmp_path)

    walk = export.photos(
        conn, tmp_path / "pilot", _reviewed(tmp_path, "2026-11-05T10:00:00"), {}, PLACED
    )

    me = next(p for p in walk if p.id == "pm:1")
    assert (me.lon, me.heading) == (2.346, 90)


def test_a_photo_moved_on_the_placing_page_loses_the_view_of_today_of_its_old_place(tmp_path):
    conn = two_photos(tmp_path)
    far = {"pm:1": {"lon": 2.3475, "lat": 48.8525, "heading": 270.0, "at": "2026-11-04T10:00:00"}}

    walk = export.photos(conn, tmp_path / "pilot", tmp_path / "none.jsonl", {}, far)

    assert next(p for p in walk if p.id == "pm:1").moved_by_review


def test_a_print_placed_flipped_is_shown_the_right_way_round_and_says_so(tmp_path):
    conn = two_photos(tmp_path)
    placed = {"pm:1": {**PLACED["pm:1"], "flipped": True}}

    walk = export.photos(conn, tmp_path / "pilot", tmp_path / "none.jsonl", {}, placed)

    me = next(p for p in walk if p.id == "pm:1")
    assert me.for_site()["flipped"] is True and "image retournée" in me.for_site()["credit"]


def test_a_cc_by_photo_is_published_with_its_credit_and_an_unverified_one_is_not(tmp_path):
    conn = two_photos(tmp_path)
    conn.execute("UPDATE photos SET license = 'CC-BY-4.0', credit = 'Musée, A 1' WHERE id = 'pm:1'")
    conn.execute("UPDATE photos SET license = 'unverified' WHERE id = 'pm:2'")

    (walk,) = export.photos(conn, tmp_path / "pilot", tmp_path / "reviews.jsonl")

    assert (walk.id, walk.credit) == ("pm:1", "Musée, A 1, CC BY 4.0")


def test_every_licence_that_may_be_published_is_named_under_the_photo():
    from flaneur import db

    assert set(db.OPEN_LICENSES) <= set(export.LICENSES)
