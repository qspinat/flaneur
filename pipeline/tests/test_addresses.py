"""House numbers of 1836 and 1888 (SoDUCo): a caption's street and number to a door."""

import io
import zipfile

import pytest
from shapely.geometry import Point

from flaneur import addresses
from flaneur.gazetteer import to_metres

HEADER = (
    ",name_json,source,layer,lat,lon,validtime_start,validtime_end,addendum_json_details,"
    "housenumber,street\n"
)


def row(atlas, number, street, lon, lat):
    return f'0,,{atlas},address,{lat},{lon},1887,1889,"{{}}",{number} ,{street}\n'


@pytest.fixture
def doors():
    return addresses.Addresses(
        [
            ("atlas_municipal_1888", "Rue De La Lingerie", "12 ", 2.3461, 48.8605),
            ("atlas_jacoubet_1836", "Rue De La Lingerie", "12 ", 2.3461, 48.8615),
            ("atlas_jacoubet_1836", "Rue Des Bons Enfans", "3 ", 2.3400, 48.8630),
            # one name, two streets far apart (before the annexation of 1860): no door
            ("atlas_municipal_1888", "Rue Saint Denis", "4 ", 2.3500, 48.8640),
            ("atlas_municipal_1888", "Rue Saint Denis", "4 ", 2.4400, 48.8300),
            # the same door read twice, a few metres apart: one door
            ("atlas_municipal_1888", "Rue Berger", "7 ", 2.3450, 48.86110),
            ("atlas_municipal_1888", "Rue Berger", "7 ", 2.3450, 48.86112),
        ]
    )


@pytest.mark.parametrize(
    ("text", "number"),
    [("12", 12), ("12 bis", 12), ("12-14", 12), ("3ter", 3), (None, None), ("", None)],
)
def test_a_house_number_is_its_first_whole_number(text, number):
    assert addresses.number_of(text) == number


def test_a_door_of_1888_is_found_before_one_of_1836(doors):
    door = doors.lookup("rue de la Lingerie", "12 bis")

    assert door.distance(to_metres(Point(2.3461, 48.8605))) < 1


def test_a_door_only_the_1836_atlas_knows_is_found_there(doors):
    door = doors.lookup("rue des Bons-Enfans", "3")

    assert door.distance(to_metres(Point(2.3400, 48.8630))) < 1


def test_a_caption_s_name_is_tried_as_the_gazetteer_tries_it(doors):
    assert doors.lookup("rue de la Lingerie enneigée", "12") is not None


def test_no_door_without_a_number_or_for_one_the_atlases_lack(doors):
    assert doors.lookup("rue de la Lingerie", None) is None
    assert doors.lookup("rue de la Lingerie", "99") is None
    assert doors.lookup("rue Imaginaire", "12") is None


def test_one_name_for_doors_far_apart_gives_none_and_near_ones_their_middle(doors):
    assert doors.lookup("rue Saint-Denis", "4") is None
    door = doors.lookup("rue Berger", "7")
    assert door.distance(to_metres(Point(2.3450, 48.86111))) < 1


def test_the_atlases_are_read_from_the_deposit_s_two_house_number_files(tmp_path):
    deposit = io.BytesIO()
    with zipfile.ZipFile(deposit, "w") as z:
        folder = "paper_dhq_2024-v.1.0.1/data-analysis/data-geocoder-gazetteer/"
        z.writestr(
            folder + "atlas_municipal_1888_housenumbers.csv",
            HEADER + row("atlas_municipal_1888", 12, "Rue De La Lingerie", 2.3461, 48.8605),
        )
        z.writestr(
            folder + "atlas_jacoubet_1836_housenumbers.csv",
            HEADER + row("atlas_jacoubet_1836", 3, "Rue Des Bons Enfans", 2.34, 48.863),
        )
        z.writestr(folder + "atlas_verniquet_1791_streets_centerpoints.csv", HEADER)

    path = tmp_path / "deposit.zip"
    path.write_bytes(deposit.getvalue())
    found = addresses.Addresses.from_deposit(path)

    assert found.lookup("rue de la Lingerie", "12") is not None
    assert found.lookup("rue des Bons Enfans", "3") is not None
