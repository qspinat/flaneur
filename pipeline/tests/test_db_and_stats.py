from flaneur import db, stats
from flaneur.sources import albert_kahn, paris_musees


def test_harvesting_twice_keeps_one_row_per_photo(tmp_path, load_fixture):
    conn = db.connect(tmp_path / "f.sqlite")
    photo = paris_musees.parse(load_fixture("paris_musees_menilmontant.json"))

    db.upsert(conn, [photo])
    db.upsert(conn, [photo])

    assert conn.execute("SELECT count(*) FROM photos").fetchone()[0] == 1


def test_stats_count_decades_streets_and_camera_positions(tmp_path, load_fixture):
    conn = db.connect(tmp_path / "f.sqlite")
    db.upsert(
        conn,
        [
            paris_musees.parse(load_fixture("paris_musees_menilmontant.json")),
            albert_kahn.parse(load_fixture("albert_kahn_passage_du_caire.json")),
        ],
    )

    by_source = {s.source: s for s in stats.compute(conn)}

    assert by_source["paris_musees"].by_decade == {1910: 1}
    assert by_source["paris_musees"].naming_a_street == 1
    assert by_source["albert_kahn"].with_camera_position == 1


def test_street_words_are_found_but_not_inside_other_words():
    assert stats.names_a_street("Le 86 de la rue de Ménilmontant")
    assert stats.names_a_street("Quai de la Tournelle")
    assert stats.names_a_street("Av. de l'Opéra")
    assert not stats.names_a_street("Portrait de Jeanne Nadaud, chanteuse lyrique")


def test_table_columns_match_the_code(tmp_path):
    conn = db.connect(tmp_path / "f.sqlite")

    def columns(table):
        return [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]

    assert columns("photos") == db.COLUMNS
    assert columns("readings") == db.READING_COLUMNS


def test_harvesting_again_keeps_columns_added_by_later_steps(tmp_path, load_fixture):
    conn = db.connect(tmp_path / "f.sqlite")
    photo = paris_musees.parse(load_fixture("paris_musees_menilmontant.json"))
    db.upsert(conn, [photo])
    conn.execute("ALTER TABLE photos ADD COLUMN reviewed INTEGER")
    conn.execute("UPDATE photos SET reviewed = 1")

    db.upsert(conn, [photo])

    assert conn.execute("SELECT reviewed FROM photos").fetchone()[0] == 1


def test_save_readings_never_replaces_a_fine_reading(tmp_path):
    conn = db.connect(tmp_path / "f.sqlite")
    reading = {
        "photo_id": "p:1",
        "kind": "unknown",
        "streets": "[]",
        "arrondissements": "[]",
        "viewpoint": None,
        "towards": None,
    }
    db.save_readings(conn, [reading | {"method": "fine", "kind": "facade"}])

    db.save_readings(conn, [reading | {"method": "rules"}])

    assert tuple(conn.execute("SELECT method, kind FROM readings").fetchone()) == (
        "fine",
        "facade",
    )


def test_every_table_matches_its_column_list_so_none_is_dropped_on_connect(tmp_path):
    conn = db.connect(tmp_path / "f.sqlite")
    conn.execute("INSERT INTO placements (photo_id, is_street_view) VALUES ('a:1', 0)")
    conn.commit()
    conn.close()

    again = db.connect(tmp_path / "f.sqlite")

    for table, columns in db.DERIVED_TABLES.items():
        assert [r["name"] for r in again.execute(f"PRAGMA table_info({table})")] == columns
    assert again.execute("SELECT count(*) FROM placements").fetchone()[0] == 1
