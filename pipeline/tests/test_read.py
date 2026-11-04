"""Real titles and captions from Paris Musées (Carnavalet) and Albert-Kahn."""

import pytest

from flaneur.read import place_of, read


def labels(reading):
    return [s.label for s in reading.streets]


def test_marville_street_with_viewpoint():
    r = read(
        "Rue de la Poterie, vue prise de la rue de la Lingerie, 1er arrondissement, Paris.",
        "Rue de la Poterie, vue prise de la rue de la Lingerie. Paris (Ier arr.). 1865-1868. "
        "Photographie de Charles Marville (1813-1879). Paris, musée Carnavalet.",
    )

    assert labels(r)[0] == "rue de la Poterie"
    assert r.viewpoint == "rue de la Lingerie"
    assert r.arrondissements == [1]
    assert r.kind == "unknown" and not r.needs_reading


def test_address_with_house_number():
    r = read("Cabaret de l'Epée de bois,  54 rue Quincampoix, 4ème arrondissement, Paris.", None)

    assert labels(r) == ["rue Quincampoix"]
    assert r.streets[0].number == "54"
    assert r.arrondissements == [4]


def test_viewpoint_from_a_monument_and_two_arrondissements():
    r = read(
        "Voitures à cheval et personnes circulant rue Soufflot, vue prise de la grille du "
        "Panthéon, la tour Eiffel au fond, 5ème et 7ème arrondissements, Paris",
        None,
    )

    assert labels(r) == ["rue Soufflot"]
    assert r.viewpoint == "grille du Panthéon"
    assert r.arrondissements == [5, 7]


def test_portraits_interiors_courtyards_and_aerials_are_not_street_views():
    portrait = read("Portrait de Klastz, actrice.", None)
    interior = read(
        'Intérieurs parisiens, "Intérieur monsieur M, Financier, avenue '
        'Elisée-Reclus, Champs-de-Mars", 7ème arrondissement, Paris',
        None,
    )
    courtyard = read("Vue sur cour, 30 rue des Entrepreneurs, 15ème arrondissement, Paris.", None)
    aerial = read(
        "Vue aérienne de Paris : L'île de la Cité, 1er, 4ème, 5ème et 6ème arrondissements, Paris.",
        None,
    )

    assert [r.kind for r in (portrait, interior, courtyard, aerial)] == [
        "portrait",
        "interior",
        "courtyard",
        "aerial",
    ]
    assert not any(r.needs_reading for r in (portrait, interior, courtyard, aerial))


def test_roman_numeral_arrondissement():
    r = read("Ecole polytechnique, rue Descartes", "Paris (Vème arr.). Atget.")

    assert r.arrondissements == [5]
    assert labels(r) == ["rue Descartes"]


def test_unidentified_street_needs_reading():
    r = read(
        "Large rue marquée par une certaine dénivellation et tournant vers la droite.",
        "Rue non identifié. 1926-1936. Photographie anonyme. Paris, musée Carnavalet.",
    )

    assert r.streets == []
    assert r.needs_reading


def test_abbreviated_street_types():
    assert labels(read("Bd Saint-Michel, 5ème arrondissement", None)) == ["bd Saint-Michel"]
    assert labels(read("Av. de l'Opéra, Paris", None)) == ["av. de l'Opéra"]


def test_albert_kahn_caption():
    r = read(
        "France, Paris, Passage du Caire",
        "L' entrée du passage du Caire au n°33 de la rue d'Alexandrie",
    )

    assert labels(r) == ["passage du Caire"]


def test_read_all_never_overwrites_a_fine_reading(tmp_path, load_fixture):
    from flaneur import db
    from flaneur.read import read_all
    from flaneur.sources import paris_musees

    conn = db.connect(tmp_path / "f.sqlite")
    db.upsert(conn, [paris_musees.parse(load_fixture("paris_musees_menilmontant.json"))])
    db.save_readings(
        conn,
        [
            {
                "photo_id": "paris_musees:88259",
                "method": "fine",
                "kind": "facade",
                "streets": "[]",
                "arrondissements": "[20]",
                "viewpoint": None,
                "towards": None,
            }
        ],
    )

    assert read_all(conn) == 0
    assert conn.execute("SELECT method FROM readings").fetchone()[0] == "fine"


def test_vers_with_a_date_is_not_a_direction():
    assert read("Boulevard Saint-Germain. Vers 1900.", None).towards is None
    assert read("Vue aérienne oblique vers le Sud-Est", None).kind == "aerial"
    assert read("Rue Soufflot, vue prise du Panthéon vers le Luxembourg", None).towards == (
        "Luxembourg"
    )


def test_viewpoint_stops_before_the_direction():
    r = read("Rue de la Poterie, vue prise de la rue de la Lingerie vers le Sud-Est", None)

    assert r.viewpoint == "rue de la Lingerie"
    assert r.towards == "Sud-Est"


def test_arrondissement_lists():
    assert read("Rue X, 1er, 4ème, 5ème et 6ème arrondissements", None).arrondissements == [
        1,
        4,
        5,
        6,
    ]
    assert read("Rue Soufflot, 5eme et 7eme arrondissements", None).arrondissements == [5, 7]


def test_typographic_apostrophe_and_saint_abbreviation():
    assert labels(read("Quai d’Orsay, 7e arrondissement", None)) == ["quai d'Orsay"]
    assert labels(read("Rue St. Antoine, Paris", None)) == ["rue Saint-Antoine"]


def test_lowercase_name_after_an_article():
    r = read("Rue de la montagne Sainte-Geneviève, 5ème arrondissement", None)

    assert labels(r) == ["rue de la montagne Sainte-Geneviève"]


def test_inside_of_a_passage_is_still_a_street_view():
    r = read("Interieur de la cour du Dragon, rue du Dragon, 6ème arrondissement", None)

    assert r.kind == "unknown"
    assert "rue du Dragon" in labels(r)


def test_railway_is_not_a_street():
    r = read(
        "Construction du chemin de fer métropolitain municipal de Paris, boulevard de la "
        "Villette, 10ème arrondissement",
        None,
    )

    assert labels(r) == ["boulevard de la Villette"]


def test_church_interiors_courtyards_and_details_are_places_not_street_views():
    titles = {
        "Le déambulatoire, église Saint-Séverin, rue des Prêtres-Saint-Séverin, 5ème": "interior",
        "Vitraux de l'église Saint-Séverin, rue des Prêtres-Saint-Séverin": "interior",
        # a relief kept in a courtyard: you enter the courtyard to see it
        "Bas-relief conservé dans la cour, 3 rue des Prêtres-Saint-Séverin": "courtyard",
        "Ancien charnier, jardin du presbytère de l'église Saint-Séverin": "courtyard",
        "Jardin jouxtant l'église Saint-Séverin, rue des Prêtres-Saint-Séverin": "courtyard",
    }

    assert {title: read(title, None).kind for title in titles} == titles


def test_street_views_that_mention_a_church_or_a_courtyard_stay_street_views():
    for title in [
        "La rue Saint-Julien-le-Pauvre, vue prise de la cour de l'ancien Hotel-Dieu",
        "La porte Saint-Martin, boulevard Saint-Denis, 10ème arrondissement",
        "Porte d'Orléans, travaux",
        "Presbytère de l'église Saint-Séverin, rue des Prêtres-Saint-Séverin",
    ]:
        assert read(title, None).kind == "unknown", title


def test_street_and_monument_names_with_church_or_detail_words_stay_street_views():
    for title in [
        "Rue de la Chapelle, 18ème arrondissement",
        "La colonne de Juillet, place de la Bastille",
        "Rue du Cloître-Saint-Merri, 4ème arrondissement",
        "Passage des Charniers",
        "Rue des Grilles",
        "Porte des Lilas",
        "Les toits de Paris vus de Notre-Dame",
    ]:
        assert read(title, None).kind == "unknown", title


def test_kind_words_after_the_first_comma_still_count():
    assert read("Église Saint-Séverin, intérieur", None).kind == "interior"
    assert read("Hôtel de Sens, cour intérieure", None).kind == "courtyard"
    assert read("Quai de la Tournelle, vue aérienne", None).kind == "aerial"


def test_place_names():
    assert place_of("La chapelle de la Vierge, église Saint-Séverin") == "Église Saint-Séverin"
    assert place_of("Le chœur, église Saint-Séverin et rue des Prêtres") == "Église Saint-Séverin"
    assert place_of("Cour, hôtel d’Aumont, 7 rue de Jouy") == "Hôtel d'Aumont"


@pytest.mark.parametrize(
    ("title", "viewpoint"),
    [
        ("Rue du Petit-Pont, vue prise coin Rue Galande (disparu), Paris", "Rue Galande"),
        (
            "Rue de la Cité, vue prise au coin de la rue des Marmousets",
            "coin de la rue des Marmousets",
        ),
        (
            "La Seine, vue prise au bas du pont Neuf le 30 mai 1903",
            "bas du pont Neuf le 30 mai 1903",
        ),
    ],
)
def test_viewpoint_written_without_de(title, viewpoint):
    assert read(title, None).viewpoint == viewpoint
