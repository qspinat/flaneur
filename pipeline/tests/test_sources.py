import re

import httpx
import pytest

from flaneur.sources import albert_kahn, paris_musees


def test_paris_musees_record_becomes_a_cc0_photo(load_fixture):
    photo = paris_musees.parse(load_fixture("paris_musees_menilmontant.json"))

    assert photo.id == "paris_musees:88259"
    assert photo.license == "CC0"
    assert photo.credit == "Paris Musées / Musée Carnavalet, Histoire de Paris"
    assert photo.photographer == "Anonyme"
    assert (photo.year_start, photo.year_end) == (1916, 1916)
    assert "rue de Ménilmontant, 20ème arrondissement" in photo.caption
    assert photo.image_url.endswith(".jpg")


def test_paris_musees_record_without_a_free_image_is_skipped(load_fixture):
    record = load_fixture("paris_musees_menilmontant.json")
    for visual in record["fieldVisuels"]:
        visual["entity"]["fieldImageLibre"] = False

    assert paris_musees.parse(record) is None


def test_paris_musees_null_entities_from_the_api_are_dropped(tmp_path):
    def server(request: httpx.Request) -> httpx.Response:
        if "offset" not in request.read().decode():
            return httpx.Response(200, json={"data": {"nodeQuery": {"count": 3}}})
        entities = [{"nid": 1}, None, {"nid": 3}]
        return httpx.Response(200, json={"data": {"nodeQuery": {"entities": entities}}})

    client = httpx.Client(transport=httpx.MockTransport(server))

    records = list(paris_musees.fetch(client, "token", tmp_path))

    assert [r["nid"] for r in records] == [1, 3]


def test_albert_kahn_record_keeps_its_camera_position_and_unverified_image(load_fixture):
    photo = albert_kahn.parse(load_fixture("albert_kahn_passage_du_caire.json"), {})

    assert photo.id == "albert_kahn:IF36953"
    assert (photo.camera_lat, photo.camera_lon) == (48.868362, 2.349735)
    assert (photo.year_start, photo.year_end) == (1914, 1914)
    assert photo.caption == "L' entrée du passage du Caire au n°33 de la rue d'Alexandrie"
    assert photo.license == "unverified"


CC0 = "Librement réutilisable (CC-00) - Domaine Public"
CC_BY = "Librement réutilisable (CC-BY-4.0)"


def _licensed(record: dict, label: str | None, transferable: str | None = None):
    row = {
        "identifiant_fakir": record["identifiant_fakir"],
        "licence_reutilisation_1": label,
        "droits_cessibles": transferable,
    }
    return albert_kahn.parse(record, {record["identifiant_fakir"]: row})


def test_albert_kahn_image_takes_the_licence_the_museum_gives_it(load_fixture):
    record = load_fixture("albert_kahn_passage_du_caire.json")

    assert _licensed(record, CC0).license == "CC0"
    assert _licensed(record, CC_BY).license == "CC-BY-4.0"


def test_albert_kahn_credit_names_the_museum_and_the_inventory_number(load_fixture):
    photo = _licensed(load_fixture("albert_kahn_passage_du_caire.json"), CC_BY)

    assert (
        photo.credit == "Département des Hauts-de-Seine, musée départemental Albert-Kahn, A 7 479"
    )


@pytest.mark.parametrize(
    ("on_record", "label", "on_licence"),
    [
        (None, CC0, "Non"),  # not transferable, said on the licence row
        ("Non", CC0, None),  # said on the Paris record
        (None, "Droits réservés", None),  # another licence
    ],
)
def test_albert_kahn_image_stays_unverified_without_a_transferable_open_licence(
    load_fixture, on_record, label, on_licence
):
    record = {**load_fixture("albert_kahn_passage_du_caire.json"), "droits_cessibles": on_record}

    assert _licensed(record, label, transferable=on_licence).license == "unverified"


def _fake_api(total: int, broken: set[int] = frozenset()) -> httpx.Client:
    """A Paris Musées API with `total` records numbered from 0; `broken` ones answer 500."""

    def server(request: httpx.Request) -> httpx.Response:
        query = request.read().decode()
        if "offset" not in query:
            return httpx.Response(200, json={"data": {"nodeQuery": {"count": total}}})
        offset, limit = (int(x) for x in re.search(r"offset: (\d+), limit: (\d+)", query).groups())
        if any(offset <= b < offset + limit for b in broken):
            return httpx.Response(500)
        entities = [{"nid": n} for n in range(offset, min(offset + limit, total))]
        return httpx.Response(200, json={"data": {"nodeQuery": {"entities": entities}}})

    return httpx.Client(transport=httpx.MockTransport(server))


def test_paris_musees_page_with_a_broken_record_loses_only_that_record():
    records, skipped = paris_musees._fetch_range(_fake_api(8, broken={5}), "token", 0, 8)

    assert [r["nid"] for r in records] == [0, 1, 2, 3, 4, 6, 7]
    assert skipped == 1


def test_paris_musees_pages_are_cached_only_when_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(paris_musees, "PAGE_SIZE", 4)
    client = _fake_api(10, broken={5})

    nids = [r["nid"] for r in paris_musees.fetch(client, "token", tmp_path)]

    assert nids == [0, 1, 2, 3, 4, 6, 7, 8, 9]
    # page 0 is complete; page 4 lost a record; page 8 is the short last page
    assert sorted(f.name for f in tmp_path.iterdir()) == ["000000.json"]


def test_paris_musees_harvest_fails_when_the_server_is_down(tmp_path):
    client = _fake_api(4, broken={0, 1, 2, 3})

    with pytest.raises(RuntimeError, match="every record"):
        list(paris_musees.fetch(client, "token", tmp_path))


def test_paris_musees_record_with_null_lists_is_skipped_not_fatal(load_fixture):
    record = load_fixture("paris_musees_menilmontant.json")
    record["fieldOeuvreAuteurs"] = None
    assert paris_musees.parse(record).photographer is None

    record["fieldVisuels"] = None
    assert paris_musees.parse(record) is None


def test_a_pinned_file_is_downloaded_once_and_kept_only_if_its_hash_matches(tmp_path):
    import hashlib

    from flaneur.sources import pinned_file

    good = hashlib.sha256(b"atlas").hexdigest()
    asked = []

    def download(part):
        asked.append(part)
        part.write_bytes(b"atlas")

    path = tmp_path / "f.zip"
    assert pinned_file(path, good, download) == path
    assert pinned_file(path, good, download) == path and len(asked) == 1

    with pytest.raises(RuntimeError, match="https://example.org/g.zip"):
        pinned_file(tmp_path / "g.zip", "0" * 64, download, "https://example.org/g.zip")
    assert not (tmp_path / "g.zip").exists() and not list(tmp_path.glob("*.part"))

    with pytest.raises(RuntimeError, match="SHA-256"):
        pinned_file(path, "0" * 64, download)  # kept, then changed: refused
