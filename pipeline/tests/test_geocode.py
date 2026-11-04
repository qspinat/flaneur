"""Gazetteer and geocoding on a tiny Paris: two streets crossing near the Halles."""

import json

import pytest

from flaneur.gazetteer import Gazetteer, key
from flaneur.geocode import locate


def rectangle(lon_min, lat_min, lon_max, lat_max):
    ring = [
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min],
    ]
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}}


# rue de la Lingerie runs north-south, rue Berger east-west; they cross. Quai de la
# Mégisserie is further south and meets neither. Two "avenue Foch" in different
# arrondissements, as in the real data.
def current(record_id, label, arrondissement, *bounds):
    return {
        "id": record_id,
        "typo_min": label,
        "arrdt": f"['{arrondissement:02d}e']",
        "geo_shape": rectangle(*bounds),
    }


def obsolete(record_id, label, note):
    return {"id": record_id, "typo_min": label, "arrdt": "01e", "historique": note}


CURRENT = [
    current("1", "rue de la Lingerie", 1, 2.3460, 48.8600, 2.3462, 48.8620),
    current("2", "rue Berger", 1, 2.3440, 48.8610, 2.3480, 48.8612),
    current("3", "rue du Faubourg Saint-Denis", 10, 2.3550, 48.8700, 2.3560, 48.8800),
    current("4", "quai de la Mégisserie", 1, 2.3420, 48.8570, 2.3480, 48.8572),
    current("5", "avenue Foch", 16, 2.2800, 48.8700, 2.2900, 48.8710),
    current("6", "avenue Foch", 12, 2.4000, 48.8400, 2.4010, 48.8410),
    current("7", "rue L", 18, 2.3400, 48.8930, 2.3402, 48.8932),
]
OBSOLETE = [
    obsolete("10", "rue aux Fers", "Actuellement rue Berger."),
    obsolete("11", "rue de la Poterie", "Supprimée en 1975. Cette rue, ouverte vers 1555, …"),
    obsolete(
        "12",
        "rue des Lavandières",
        "Ou rue des Lavandières Sainte-Opportune. Actuellement partie de la rue de la Lingerie.",
    ),
    obsolete("13", "rue de la Vieille Lingerie", "Voir : Fers (rue aux)."),
    obsolete("14", "rue Berger", "Dénommée quai de la Mégisserie par arrêté du 1er mai 1850."),
    obsolete("15", "quai de la Mégisserie", "Dénommée rue Berger par arrêté du 2 mai 1850."),
]


@pytest.fixture
def gazetteer():
    return Gazetteer.from_records(CURRENT, OBSOLETE)


def reading(streets, viewpoint=None, arrondissements=(1,)):
    return {
        "streets": json.dumps([{"type": t, "name": n, "number": None} for t, n in streets]),
        "arrondissements": json.dumps(list(arrondissements)),
        "viewpoint": viewpoint,
    }


def test_key_ignores_accents_case_articles_and_abbreviations():
    assert key("Rue de l'Élysée") == key("rue Elysee")
    assert key("Bd St-Michel") == key("boulevard Saint Michel")


def test_caption_variants_find_the_street(gazetteer):
    assert gazetteer.lookup("rue Lingerie")[0].label == "rue de la Lingerie"
    assert gazetteer.lookup("faubourg Saint-Denis")[0].label == "rue du Faubourg Saint-Denis"
    assert gazetteer.lookup("rue Berger enneigée")[0].label == "rue Berger"


def test_obsolete_names_are_linked_or_marked_removed(gazetteer):
    (renamed,) = gazetteer.lookup("rue aux Fers")
    (removed,) = gazetteer.lookup("rue de la Poterie")

    assert renamed.status == "renamed" and renamed.current_label == "rue Berger"
    assert renamed.geometry is not None
    assert removed.status == "removed" and removed.geometry is None


def test_viewpoint_on_a_crossing_street_places_the_camera_at_the_corner(gazetteer):
    location = locate(reading([("rue", "de la Lingerie")], viewpoint="rue Berger"), gazetteer)

    assert location.status == "corner"
    assert location.point.geom_type == "Point"
    assert location.precision_m == 30


def test_street_without_viewpoint_is_placed_on_the_street(gazetteer):
    location = locate(reading([("rue", "Berger")]), gazetteer)

    assert location.status == "street" and location.point is None
    # rue Berger is ~290 m long: the camera may be anywhere along it.
    assert 100 < location.precision_m < 200


def test_locate_all_stores_each_street_once(tmp_path, gazetteer):
    from flaneur import db
    from flaneur.geocode import locate_all

    conn = db.connect(tmp_path / "f.sqlite")
    db.save_readings(
        conn,
        [
            {
                "photo_id": f"p:{i}",
                "method": "rules",
                "kind": "unknown",
                **reading([("rue", "Berger")]),
            }
            | {"towards": None}
            for i in range(3)
        ],
    )

    assert locate_all(conn, gazetteer) == {"street": 3}
    assert conn.execute("SELECT count(*) FROM streets").fetchone()[0] == 1


def test_removed_and_unknown_streets(gazetteer):
    assert locate(reading([("rue", "de la Poterie")]), gazetteer).status == "removed"
    assert locate(reading([("rue", "Imaginaire")]), gazetteer).status == "not_found"


def test_note_target_anywhere_partial_and_cross_reference(gazetteer):
    (part,) = gazetteer.lookup("rue des Lavandières")
    (seen,) = gazetteer.lookup("rue de la Vieille Lingerie")

    assert part.status == "renamed" and part.partial
    assert seen.status == "renamed" and seen.current_label == "rue Berger"


def test_partial_street_has_unknown_precision(gazetteer):
    location = locate(reading([("rue", "des Lavandières")]), gazetteer)

    assert location.status == "street" and location.precision_m is None


def test_corner_is_never_a_street_with_itself(gazetteer):
    location = locate(
        reading([("rue", "de la Lingerie")], viewpoint="rue de la Lingerie et rue Berger"),
        gazetteer,
    )

    assert location.status == "corner"
    assert location.label == "rue de la Lingerie × rue Berger"


def test_viewpoint_street_that_does_not_meet_places_the_camera_on_it(gazetteer):
    location = locate(reading([("rue", "Berger")], viewpoint="quai de la Mégisserie"), gazetteer)

    assert location.status == "viewpoint_street"
    # today's quai, not the old name of rue Berger
    assert location.street.label == "quai de la Mégisserie"
    assert location.street.status == "current"


def test_shared_name_without_arrondissement_is_ambiguous(gazetteer):
    assert locate(reading([("avenue", "Foch")], arrondissements=()), gazetteer).status == (
        "ambiguous"
    )
    assert locate(reading([("avenue", "Foch")], arrondissements=(16,)), gazetteer).status == (
        "street"
    )


def test_a_lone_street_type_or_a_shortened_proper_name_never_matches(gazetteer):
    assert gazetteer.lookup("rue de") == []
    assert gazetteer.lookup("rue L") == []
    assert gazetteer.lookup("rue Berger Saint-Honoré") == []


def test_photo_no_longer_a_street_view_loses_its_caption_position(tmp_path, gazetteer):
    from flaneur import db
    from flaneur.geocode import locate_all

    conn = db.connect(tmp_path / "f.sqlite")
    row = {"photo_id": "p:1", "method": "rules", "kind": "unknown", "towards": None}
    db.save_readings(conn, [row | reading([("rue", "Berger")])])
    locate_all(conn, gazetteer)
    db.save_readings(conn, [row | reading([("rue", "Berger")]) | {"kind": "interior"}])

    locate_all(conn, gazetteer)

    assert conn.execute("SELECT count(*) FROM locations").fetchone()[0] == 0


def test_name_shared_by_today_street_and_an_old_name_is_today_street(gazetteer):
    location = locate(reading([("rue", "Berger")]), gazetteer)

    assert location.status == "street"
    assert location.street.status == "current"


def test_prints_of_one_negative_are_placed_from_all_their_captions_together(tmp_path, gazetteer):
    from flaneur import db, dedup
    from flaneur.geocode import locate_all

    conn = db.connect(tmp_path / "f.sqlite")
    # one print names the street, the other only where the camera stood
    db.save_readings(
        conn,
        [
            {"photo_id": "p:1", "method": "rules", "kind": "unknown", "towards": None}
            | reading([("rue", "de la Lingerie")]),
            {"photo_id": "p:2", "method": "rules", "kind": "unknown", "towards": None}
            | reading([("rue", "Imaginaire")], viewpoint="rue Berger"),
            {"photo_id": "p:3", "method": "rules", "kind": "unknown", "towards": None}
            | reading([("rue", "de la Lingerie")]),
        ],
    )
    dedup.save(conn, [{"p:1", "p:2"}])

    assert locate_all(conn, gazetteer) == {"corner": 2, "street": 1}  # p:3 is another photo


def test_a_negative_whose_prints_were_never_read_is_left_alone(tmp_path, gazetteer):
    from flaneur import db, dedup
    from flaneur.geocode import locate_all

    conn = db.connect(tmp_path / "f.sqlite")
    dedup.save(conn, [{"p:1", "p:2"}])

    assert locate_all(conn, gazetteer) == {}
