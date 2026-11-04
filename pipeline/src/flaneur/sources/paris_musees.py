"""Paris Musées collections (Musée Carnavalet): photographs whose image is CC0.

API: Drupal GraphQL at apicollections.parismusees.paris.fr, token in the `auth-token` header.
"""

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import httpx

from flaneur.db import Photo
from flaneur.sources import write_json

API_URL = "https://apicollections.parismusees.paris.fr/graphql"
CARNAVALET = "12"
PAGE_SIZE = 200
CREDIT = "Paris Musées / Musée Carnavalet, Histoire de Paris"

log = logging.getLogger(__name__)

FILTER = f"""{{conditions: [
  {{field: "type", value: "oeuvre"}},
  {{field: "field_musee", value: "{CARNAVALET}"}},
  {{field: "field_visuels.entity.field_image_libre", value: "1"}},
  {{field: "field_oeuvre_types_objet.entity.name", value: "Photographie"}}
]}}"""

COUNT_QUERY = f"{{ nodeQuery(limit: 1, filter: {FILTER}) {{ count }} }}"

PAGE_QUERY = f"""{{ nodeQuery(offset: %d, limit: %d, sort: {{field: "nid"}}, filter: {FILTER}) {{
  entities {{ ... on NodeOeuvre {{
    nid title absolutePath
    fieldDateProduction {{ startYear endYear century }}
    fieldOeuvreAuteurs {{ entity {{ fieldAuteurAuteur {{ entity {{ entityLabel }} }} }} }}
    fieldOeuvreGeolocalisation {{ lat lng }}
    fieldVisuels {{ entity {{ publicUrl fieldImageLibre fieldLegende }} }}
  }} }} }} }}"""


def _post(client: httpx.Client, token: str, query: str) -> dict:
    response = client.post(API_URL, headers={"auth-token": token}, json={"query": query})
    response.raise_for_status()
    body = response.json()
    if body.get("errors"):
        raise RuntimeError(f"Paris Musées API error: {body['errors'][:2]}")
    return body["data"]["nodeQuery"]


def _fetch_range(
    client: httpx.Client, token: str, offset: int, limit: int
) -> tuple[list[dict], int]:
    """Fetch records [offset, offset + limit); return them and how many had to be skipped.

    The API answers 500 for a whole page when one record in it is broken: the range is then
    split in halves until the broken record is alone, and that record is skipped.
    """
    try:
        return _post(client, token, PAGE_QUERY % (offset, limit))["entities"], 0
    except httpx.HTTPStatusError as error:
        if error.response.status_code != 500:
            raise
        if limit == 1:
            log.warning("Paris Musées: record at offset %d skipped (server error)", offset)
            return [], 1
        half = limit // 2
        first, first_skipped = _fetch_range(client, token, offset, half)
        rest, rest_skipped = _fetch_range(client, token, offset + half, limit - half)
        return first + rest, first_skipped + rest_skipped


def fetch(client: httpx.Client, token: str, cache_dir: Path) -> Iterator[dict]:
    """Yield every raw record, page by page; complete pages are kept on disk and not fetched
    again. A page that is short (the last one) or lost records to server errors is not kept,
    so the next harvest fetches it again.

    The API returns some entities of a page as null (records not published): dropped here.
    """
    total = _post(client, token, COUNT_QUERY)["count"]
    for offset in range(0, total, PAGE_SIZE):
        page_file = cache_dir / f"{offset:06d}.json"
        if page_file.exists():
            page = json.loads(page_file.read_text())
        else:
            limit = min(PAGE_SIZE, total - offset)
            page, skipped = _fetch_range(client, token, offset, limit)
            if skipped == limit:
                raise RuntimeError(f"Paris Musées: every record from offset {offset} failed")
            if not skipped and len(page) == PAGE_SIZE:
                write_json(page_file, page)
        yield from (record for record in page if record)


def harvest(client: httpx.Client, cache_dir: Path) -> Iterator[Photo]:
    token = os.environ.get("PARIS_MUSEES_TOKEN")
    if not token:
        raise RuntimeError("PARIS_MUSEES_TOKEN manquant (pipeline/.env)")
    return (photo for record in fetch(client, token, cache_dir) if (photo := parse(record)))


def parse(record: dict) -> Photo | None:
    """Turn one raw record into a Photo, or None when it has no free image."""
    visuals = [v["entity"] for v in record["fieldVisuels"] or [] if v["entity"]]
    visual = next((v for v in visuals if v["fieldImageLibre"]), None)
    if visual is None:
        return None
    dates = record["fieldDateProduction"] or {}
    authors = [
        person["entity"]["entityLabel"]
        for a in record["fieldOeuvreAuteurs"] or []
        if a["entity"] and (person := a["entity"]["fieldAuteurAuteur"]) and person["entity"]
    ]
    geo = record.get("fieldOeuvreGeolocalisation") or {}
    return Photo(
        source="paris_musees",
        source_id=str(record["nid"]),
        record_url=record["absolutePath"],
        title=record["title"],
        caption=visual["fieldLegende"],
        photographer=", ".join(authors) or None,
        date_text=dates.get("century"),
        year_start=dates.get("startYear"),
        year_end=dates.get("endYear") or dates.get("startYear"),
        image_url=visual["publicUrl"],
        license="CC0",
        credit=CREDIT,
        camera_lat=geo.get("lat"),
        camera_lon=geo.get("lng"),
        raw=record,
    )
