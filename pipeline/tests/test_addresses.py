"""House numbers of 1836 and 1888 (SoDUCo) and of today (BAN): a caption's street and number
to a door."""

import gzip
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
    found = addresses.Addresses(addresses.deposit_doors(path))

    assert found.lookup("rue de la Lingerie", "12") is not None
    assert found.lookup("rue des Bons Enfans", "3") is not None


@pytest.fixture
def then_and_now():
    return addresses.Addresses(
        [
            ("atlas_municipal_1888", "Rue Berger", "7 ", 2.3450, 48.8611),
            (addresses.TODAY, "Rue Berger", "7", 2.3456, 48.8611),  # renumbered since: 44 m on
            (addresses.TODAY, "Rue Berger", "40", 2.3470, 48.8611),  # only today's map has it
        ]
    )


@pytest.mark.parametrize(("year", "lon"), [(1900, 2.3456), (1870, 2.3450), (None, 2.3450)])
def test_today_s_numbers_come_first_for_photos_after_the_renumberings(then_and_now, year, lon):
    door = then_and_now.lookup("rue Berger", "7", year)

    assert door.distance(to_metres(Point(lon, 48.8611))) < 1


def test_today_s_numbers_place_what_the_atlases_lack_whatever_the_year(then_and_now):
    assert then_and_now.lookup("rue Berger", "40", 1870) is not None


def test_today_s_numbers_are_read_from_the_national_address_base(tmp_path):
    path = tmp_path / "ban-75-2026-11-09.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write("id;numero;rep;nom_voie;lon;lat\n")
        out.write("75102_1;12;;Rue de la Lingerie;2.3461;48.8605\n")

    found = addresses.Addresses(addresses.ban_doors(path))

    assert found.lookup("rue de la Lingerie", "12 bis", 1920) is not None  # a caption's bis


def test_today_s_bis_and_ter_doors_are_not_taken_for_the_plain_number(tmp_path):
    path = tmp_path / "ban-75-2026-11-09.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write("id;numero;rep;nom_voie;lon;lat\n")
        out.write("75102_1;12;;Rue de la Lingerie;2.3461;48.8605\n")
        out.write("75102_2;12;bis;Rue de la Lingerie;2.3466;48.8605\n")  # 37 m on

    found = addresses.Addresses(addresses.ban_doors(path))

    assert (
        found.lookup("rue de la Lingerie", "12", 1920).distance(to_metres(Point(2.3461, 48.8605)))
        < 1
    )


def test_a_door_of_today_far_from_the_atlas_s_is_another_street_s_by_the_same_name(doors):
    renamed = addresses.Addresses(
        [
            ("atlas_municipal_1888", "Rue Berger", "7 ", 2.3450, 48.8611),
            (addresses.TODAY, "Rue Berger", "7", 2.3700, 48.8400),  # 3 km off: today's namesake
        ]
    )

    door = renamed.lookup("rue Berger", "7", 1920)

    assert door.distance(to_metres(Point(2.3450, 48.8611))) < 1


def test_each_source_s_door_is_offered_in_turn(then_and_now):
    found = list(then_and_now.doors("rue Berger", "7", 1900))

    assert [source for source, _ in found] == [addresses.TODAY, "atlas_municipal_1888"]


@pytest.mark.parametrize(
    ("content", "valid"),
    [
        (b"<html>maintenance</html>", False),
        (gzip.compress(b"id;numero;rep;nom_voie;lon;lat\n75102_1;1;;Rue X;2.3;48.8\n"), True),
        (gzip.compress(b"id;num;voie\n1;1;x\n"), False),  # columns renamed
    ],
)
def test_a_download_of_today_s_numbers_is_kept_only_if_it_reads(tmp_path, content, valid):
    path = tmp_path / "ban.csv.gz"
    path.write_bytes(content)

    assert addresses.readable_ban(path) is valid
