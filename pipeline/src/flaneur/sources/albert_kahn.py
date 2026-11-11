"""Albert-Kahn museum, Archives de la Planète – Paris: autochromes 1909–1930s.

The dataset gives the camera position of each photo, so these photos are the ground truth
for measuring how well we locate photos. Metadata: Licence Ouverte (Etalab). Each image's
licence is the one the museum gives it in the whole "Archives de la Planète" dataset: public
domain (CC0) or CC BY 4.0; an image it marks as not transferable, or does not list, stays
"unverified" and off the public site.
"""

import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import httpx

from flaneur.db import Photo
from flaneur.sources import cached_json

EXPORT_URL = (
    "https://opendata.hauts-de-seine.fr/api/explore/v2.1/catalog/datasets/"
    "archives-de-la-planete-paris/exports/json"
)
# The whole collection, where each image's reuse licence is given (the Paris extract has none).
LICENCES_URL = (
    "https://opendata.hauts-de-seine.fr/api/explore/v2.1/catalog/datasets/"
    "archives-de-la-planete/exports/json"
    "?select=identifiant_fakir,licence_reutilisation_1,droits_cessibles"
)
LICENCES = {
    "Librement réutilisable (CC-00) - Domaine Public": "CC0",
    "Librement réutilisable (CC-BY-4.0)": "CC-BY-4.0",
}
CREDIT = "Département des Hauts-de-Seine, musée départemental Albert-Kahn"
YEAR = re.compile(r"\b(1[89]\d\d)\b")


def _cached_export(client: httpx.Client, url: str, path: Path) -> list[dict]:
    """The rows of an export, downloaded once and kept on disk."""

    def download() -> list[dict]:
        response = client.get(url)
        response.raise_for_status()
        return response.json()

    return cached_json(path, download)


def fetch(client: httpx.Client, cache_dir: Path) -> list[dict]:
    """Return every record of the Paris extract."""
    return _cached_export(client, EXPORT_URL, cache_dir / "export.json")


def fetch_licences(client: httpx.Client, cache_dir: Path) -> dict[str, str]:
    """Each image's licence in the whole collection, by its id (see licences_by_id)."""
    return licences_by_id(_cached_export(client, LICENCES_URL, cache_dir / "licences.json"))


def _refused(transferable: str | None) -> bool:
    """Whether the museum says the rights are not transferable ("Non", however spelt): any
    word there refuses, so an unexpected one keeps the image off the site."""
    return bool((transferable or "").strip())


def licences_by_id(rows: Iterable[dict]) -> dict[str, str]:
    """ "CC0" or "CC-BY-4.0" for each image the museum gives one of them, else "unverified";
    an image listed twice keeps its most restrictive row."""
    found: dict[str, str] = {}
    for row in rows:
        licence = (
            "unverified"
            if _refused(row.get("droits_cessibles"))
            else LICENCES.get(row.get("licence_reutilisation_1"), "unverified")
        )
        image = row["identifiant_fakir"]
        found[image] = licence if found.get(image, licence) == licence else "unverified"
    return found


def harvest(client: httpx.Client, cache_dir: Path) -> Iterator[Photo]:
    licences = fetch_licences(client, cache_dir)
    return (parse(record, licences) for record in fetch(client, cache_dir))


def licence_of(record: dict, licences: Mapping[str, str]) -> str:
    """The image's licence from `licences` (licences_by_id); "unverified" when it is not listed
    or the Paris extract's own record says its rights are not transferable."""
    if _refused(record.get("droits_cessibles")):
        return "unverified"
    return licences.get(record["identifiant_fakir"], "unverified")


def parse(record: dict, licences: Mapping[str, str]) -> Photo:
    date_text = record.get("date_de_prise_de_vue")
    years = [int(y) for y in YEAR.findall(date_text or "")]
    geo = record.get("geo_point") or {}
    photo = record.get("photo") or {}
    caption = record.get("legende_revisee") or record.get("legende_originale_titre")
    return Photo(
        source="albert_kahn",
        source_id=record["identifiant_fakir"],
        record_url=None,  # the dataset links to no record page
        title=record.get("legende_originale_titre"),
        caption=caption,
        photographer=", ".join(record.get("operateur") or []) or None,
        date_text=date_text,
        year_start=min(years) if years else None,
        year_end=max(years) if years else None,
        image_url=photo.get("url"),
        license=licence_of(record, licences),
        credit=", ".join(filter(None, (CREDIT, record.get("numero_d_inventaire")))),
        camera_lat=geo.get("lat"),
        camera_lon=geo.get("lon"),
        raw=record,
    )
