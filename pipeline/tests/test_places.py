from shapely.geometry import box

from flaneur.gazetteer import Street, to_metres
from flaneur.places import PlacePhoto, _merge_same_places, entrance
from flaneur.read import place_of


def test_two_courtyards_of_the_same_street_are_two_places():
    merged = _merge_same_places(
        {
            "Cour, 3 rue des Prêtres-Saint-Séverin": [PlacePhoto("a", "courtyard", "")],
            "Cour, 5 rue des Prêtres-Saint-Séverin": [PlacePhoto("b", "courtyard", "")],
        }
    )

    assert len(merged) == 2


def test_a_place_on_the_street_itself_has_no_direction():
    street = Street(
        "1",
        "rue Test",
        "current",
        (5,),
        geometry=to_metres(box(2.3450, 48.8510, 2.3451, 48.8530)),
    )

    assert entrance([(2.34505, 48.8520)], street).heading is None


def test_the_church_named_in_the_title_is_the_place():
    assert place_of("Le déambulatoire, église Saint-Séverin, rue des Prêtres-Saint-Séverin") == (
        "Église Saint-Séverin"
    )
    assert place_of("Ancien charnier, jardin du presbytère de l'église Saint-Séverin, rue") == (
        "Église Saint-Séverin"
    )


def test_a_courtyard_without_a_name_is_named_by_its_address():
    assert place_of("Bas-relief conservé dans la cour, 3 rue des Prêtres-Saint-Séverin, 5ème") == (
        "Cour, 3 rue des Prêtres-Saint-Séverin"
    )


def test_a_title_naming_no_building_has_no_place():
    assert place_of("Pierres tombales découvertes, Paris") is None


def test_names_differing_by_case_or_a_typo_are_one_place():
    def photos(n):
        return [PlacePhoto(f"p:{i}", "interior", "") for i in range(n)]

    merged = _merge_same_places(
        {
            "Église Saint-Séverin": photos(5),
            "Église Saint-Sévrin": photos(1),
            "Hôtel des abbés de Cluny": photos(3),
            "Hôtel des Abbés de Cluny": photos(2),
            "Église Saint-Julien-le-Pauvre": photos(4),
        }
    )

    assert {name: len(p) for name, p in merged.items()} == {
        "Église Saint-Séverin": 6,
        "Hôtel des abbés de Cluny": 5,
        "Église Saint-Julien-le-Pauvre": 4,
    }


def test_the_entrance_is_on_the_street_facing_the_place():
    # A north-south street; the place is 30 m east of it.
    street = Street(
        "1",
        "rue Test",
        "current",
        (5,),
        geometry=to_metres(box(2.3450, 48.8510, 2.3451, 48.8530)),
    )

    door = entrance([(2.3455, 48.8520), (2.3456, 48.8520), (2.3455, 48.8521)], street)

    assert abs(door.lat - 48.8520) < 0.0002
    assert 80 <= door.heading <= 100  # going in means facing east
