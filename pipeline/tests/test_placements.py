import json

import pytest
from PIL import Image
from shapely.geometry import LineString, Point

from flaneur import db, panoramax, placements
from flaneur.gazetteer import to_metres
from flaneur.pilot import _panorama_views


def picture(n, lon):
    return panoramax.Picture(f"p{n}", lon, 48.852, 90, False, "2024-01-01", "", "")


def test_spread_keeps_pictures_from_one_end_of_the_street_to_the_other():
    street = to_metres(LineString([(2.3400, 48.852), (2.3500, 48.852)])).buffer(5)
    pictures = [picture(n, 2.3400 + n * 0.0001) for n in range(100)]

    chosen = panoramax.spread(pictures, street, 5)

    ids = {p.id for p in chosen}
    assert len(ids) == 5
    assert {"p0", "p99"} <= ids


def test_placements_are_stored_and_compared_with_the_known_camera_position(tmp_path):
    conn = db.connect(tmp_path / "f.sqlite")
    conn.execute(
        "INSERT INTO photos (id, source, source_id, license, credit, camera_lon, camera_lat, raw) "
        "VALUES ('a:1', 'a', '1', 'x', 'x', 2.3460, 48.8520, '{}')"
    )
    folder = tmp_path / "pilot" / "photos" / "a_1"
    folder.mkdir(parents=True)
    (folder / "info.json").write_text(json.dumps({"sheet": "rue_x"}))
    (tmp_path / "pilot" / "sheets").mkdir()
    (tmp_path / "pilot" / "sheets" / "rue_x.json").write_text(
        json.dumps([{"n": 3, "panoramax_id": "pnx-3"}])
    )
    (folder / "placement.json").write_text(
        json.dumps(
            {
                "photo_id": "a:1",
                "is_street_view": True,
                "match": 3,
                "lon": 2.3461,
                "lat": 48.8520,
                "heading": 200,
                "confidence": "high",
                "reason": "même façade",
            }
        )
    )

    placements.save(conn, placements.load(tmp_path / "pilot"))

    row = conn.execute(
        "SELECT is_street_view, match, match_panoramax_id, heading, confidence FROM placements"
    )
    assert tuple(row.fetchone()) == (1, 3, "pnx-3", 200, "high")
    ((confidence, metres),) = placements.distances_to_truth(conn)
    expected = to_metres(Point(2.3461, 48.8520)).distance(to_metres(Point(2.3460, 48.8520)))
    assert confidence == "high" and abs(metres - expected) < 0.1


def test_a_photo_that_is_not_a_street_view_is_not_compared_with_its_known_position(tmp_path):
    conn = db.connect(tmp_path / "f.sqlite")
    conn.execute(
        "INSERT INTO photos (id, source, source_id, license, credit, camera_lon, camera_lat, raw) "
        "VALUES ('a:2', 'a', '2', 'x', 'x', 2.3460, 48.8520, '{}')"
    )

    placements.save(conn, [{"photo_id": "a:2", "is_street_view": False, "lon": 2.31, "lat": 48.86}])

    assert placements.distances_to_truth(conn) == []


def test_a_panorama_is_cut_into_four_views_facing_four_directions():
    # A 360° picture: each quarter of its width has its own colour; the centre faces 200°.
    panorama = Image.new("RGB", (400, 200))
    for i, colour in enumerate(["red", "green", "blue", "white"]):
        panorama.paste(colour, (i * 100, 0, (i + 1) * 100, 200))

    views = _panorama_views(panorama, 200)

    assert [heading for _, heading in views] == [200, 290, 20, 110]
    # the first view is centred on the panorama's centre: half green, half blue
    first = views[0][0]
    assert first.getpixel((10, 10)) == (0, 128, 0) and first.getpixel((90, 10)) == (0, 0, 255)


def test_a_placement_with_an_unknown_confidence_is_refused(tmp_path):
    folder = tmp_path / "photos" / "a_1"
    folder.mkdir(parents=True)
    (folder / "info.json").write_text(json.dumps({"sheet": None}))
    (folder / "placement.json").write_text(
        json.dumps(
            {
                "photo_id": "a:1",
                "is_street_view": True,
                "lon": 2.3,
                "lat": 48.8,
                "confidence": "High",
            }
        )
    )

    with pytest.raises(ValueError, match="confidence"):
        placements.load(tmp_path)


def test_saved_placements_mirror_the_files(tmp_path):
    conn = db.connect(tmp_path / "f.sqlite")
    placements.save(conn, [{"photo_id": "a:1", "is_street_view": False}])

    placements.save(conn, [{"photo_id": "a:2", "is_street_view": False}])

    assert [r[0] for r in conn.execute("SELECT photo_id FROM placements")] == ["a:2"]


def test_spread_gives_a_dense_stretch_one_slot_not_most_of_them():
    street = to_metres(LineString([(2.3400, 48.852), (2.3500, 48.852)])).buffer(5)
    cluster = [picture(n, 2.3400 + n * 0.000001) for n in range(90)]
    rest = [picture(100 + n, 2.3410 + n * 0.0009) for n in range(10)]

    chosen = panoramax.spread(cluster + rest, street, 8)

    assert sum(p.id.startswith("p") and int(p.id[1:]) < 90 for p in chosen) <= 2


def test_derived_tables_with_old_columns_are_rebuilt(tmp_path):
    import sqlite3

    path = tmp_path / "f.sqlite"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE locations (photo_id TEXT PRIMARY KEY, method TEXT NOT NULL)")
    old.commit()
    old.close()

    conn = db.connect(path)

    assert [r["name"] for r in conn.execute("PRAGMA table_info(locations)")] == db.LOCATION_COLUMNS


def test_a_negative_placed_once_gives_its_placement_to_its_other_prints():
    placed = {"photo_id": "a", "is_street_view": True, "lon": 2.34, "lat": 48.85}
    own = {"photo_id": "c", "is_street_view": False}

    found = placements.inherit([placed, own], {"a": ["a", "b"], "c": ["c", "d"]})

    by_photo = {p["photo_id"]: p for p in found}
    assert by_photo["b"] == placed | {"photo_id": "b"}
    assert by_photo["d"]["is_street_view"] is False
    assert len(found) == 4


def _photo(
    conn, pilot_dir, photo_id, license="CC0", camera=None, kind="unknown", door=None, street=None
):
    """A prepared photo of the areas: its record, its reading, where its caption puts it."""
    from flaneur.gazetteer import to_geojson
    from flaneur.pilot import safe_name

    lon, lat = camera or (None, None)
    conn.execute(
        "INSERT INTO photos (id, source, source_id, license, credit, camera_lon, camera_lat, raw) "
        "VALUES (?, 'x', ?, ?, 'x', ?, ?, '{}')",
        (photo_id, photo_id, license, lon, lat),
    )
    conn.execute(
        "INSERT INTO readings (photo_id, method, kind, streets, arrondissements) "
        "VALUES (?, 'rules', ?, '[]', '[4]')",
        (photo_id, kind),
    )
    if door or street:
        db.save_locations(
            conn,
            [
                {
                    "photo_id": photo_id,
                    "status": "address" if door else "street",
                    "label": "x",
                    "street_id": street,
                    "point": to_geojson(door) if door else None,
                    "precision_m": 50,
                }
            ],
        )
    folder = pilot_dir / "photos" / safe_name(photo_id)
    folder.mkdir(parents=True)
    (folder / "info.json").write_text(json.dumps({"photo_id": photo_id}))
    return folder


def test_a_photo_with_no_placement_of_its_own_starts_where_its_source_or_caption_puts_it(tmp_path):
    from shapely.geometry import LineString, Point

    from flaneur import pilot
    from flaneur.gazetteer import to_degrees, to_geojson, to_metres

    conn = db.connect(tmp_path / "f.sqlite")
    cite = pilot.AREAS["cite"]
    door = Point(cite.centroid.x + 20, cite.centroid.y)
    line = LineString(
        [
            (cite.centroid.x - 400, cite.centroid.y + 50),
            (cite.centroid.x + 400, cite.centroid.y + 50),
        ]
    )  # crosses the square
    conn.execute(
        "INSERT INTO streets (id, label, status, arrondissements, geometry) "
        "VALUES ('quai', 'quai aux Fleurs', 'current', '[4]', ?)",
        (to_geojson(line),),
    )
    pilot_dir = tmp_path / "pilot"
    _photo(conn, pilot_dir, "ak:1", camera=(2.3525, 48.8547))
    _photo(conn, pilot_dir, "pm:door", door=door)
    _photo(conn, pilot_dir, "pm:street", street="quai")

    seeds = {s["photo_id"]: s for s in placements.seeds(conn, pilot_dir, pilot.AREA)}

    assert {s["confidence"] for s in seeds.values()} == {placements.SEED}
    assert all(s["is_street_view"] for s in seeds.values())
    assert (seeds["ak:1"]["lon"], seeds["ak:1"]["lat"]) == (2.3525, 48.8547)
    at = to_degrees(door)
    assert (seeds["pm:door"]["lon"], seeds["pm:door"]["lat"]) == pytest.approx((at.x, at.y))
    middle = to_metres(Point(seeds["pm:street"]["lon"], seeds["pm:street"]["lat"]))
    assert middle.within(pilot.AREA) and middle.distance(line) < 0.01


def test_no_start_for_an_unverified_image_an_interior_or_a_photo_placed_by_hand(tmp_path):
    from flaneur import pilot

    conn = db.connect(tmp_path / "f.sqlite")
    pilot_dir = tmp_path / "pilot"
    _photo(conn, pilot_dir, "ak:2", license="unverified", camera=(2.3525, 48.8547))
    _photo(conn, pilot_dir, "pm:inside", kind="interior", door=pilot.AREAS["cite"].centroid)
    placed = _photo(conn, pilot_dir, "pm:placed", door=pilot.AREAS["cite"].centroid)
    (placed / "placement.json").write_text("{}")

    assert placements.seeds(conn, pilot_dir, pilot.AREA) == []
