"""Fine placements: where each photo of the pilot was taken and which way it faced, decided by
looking at the photo and at today's pictures around its street (see pilot.py), one
`placement.json` per photo folder:

    {"photo_id": "...", "is_street_view": true, "match": 7, "lon": 2.34, "lat": 48.85,
     "heading": 210, "confidence": "high" | "medium" | "low", "reason": "..."}

`match` is the number of the contact-sheet picture showing the same place, or null.
They are kept apart from `locations`, which holds what the caption alone tells.
"""

import json
import sqlite3
from pathlib import Path

from shapely.geometry import MultiPoint, Point
from shapely.geometry.base import BaseGeometry

from flaneur import db, dedup, read
from flaneur.gazetteer import from_geojson, to_degrees, to_metres
from flaneur.pilot import safe_name, sheet_of

CONFIDENCES = ("high", "medium", "low")
# A placement no one made: where a photo's camera is first looked for (see seeds).
SEED = "seed"


def load(pilot_dir: Path) -> list[dict]:
    """Every placement.json, checked, with the Panoramax id of the matched picture added (the
    number alone points to a sheet that may be rebuilt)."""
    placements = []
    for path in sorted((pilot_dir / "photos").glob("*/placement.json")):
        placement = json.loads(path.read_text())
        if not isinstance(placement.get("is_street_view"), bool):
            raise ValueError(f"{path}: is_street_view must be true or false")
        if placement.get("lon") is not None and placement.get("confidence") not in CONFIDENCES:
            raise ValueError(f"{path}: confidence must be one of {CONFIDENCES}")
        sheet = sheet_of(path.parent)
        tiles = json.loads((pilot_dir / "sheets" / f"{sheet}.json").read_text()) if sheet else []
        matched = next((t for t in tiles if t["n"] == placement.get("match")), None)
        placement["match_panoramax_id"] = matched["panoramax_id"] if matched else None
        placements.append(placement)
    return placements


def inherit(placements: list[dict], negatives: dict[str, list[str]]) -> list[dict]:
    """The placements, plus one for every print of a placed negative that was not placed on
    its own (a negative is placed once, see pilot.prepare): the same as its placed print."""
    by_photo = {p["photo_id"]: p for p in placements}
    inherited = []
    for prints in {tuple(p) for p in negatives.values()}:
        placed = next((by_photo[i] for i in prints if i in by_photo), None)
        if placed:
            inherited += [{**placed, "photo_id": i} for i in prints if i not in by_photo]
    return placements + inherited


def save(conn: sqlite3.Connection, placements: list[dict]) -> int:
    """Replace the stored placements with these: the table mirrors the placement files."""
    return db.replace(
        conn, "placements", ({c: p.get(c) for c in db.PLACEMENT_COLUMNS} for p in placements)
    )


def distances_to_truth(conn: sqlite3.Connection) -> list[tuple[str, float]]:
    """For placed street views whose camera position is known: (confidence, metres)."""
    rows = conn.execute(
        "SELECT pl.confidence, pl.lon, pl.lat, p.camera_lon, p.camera_lat "
        "FROM placements pl JOIN photos p ON p.id = pl.photo_id "
        "WHERE pl.is_street_view AND pl.lon IS NOT NULL AND p.camera_lat IS NOT NULL"
    ).fetchall()
    return [
        (
            r["confidence"],
            to_metres(Point(r["lon"], r["lat"])).distance(
                to_metres(Point(r["camera_lon"], r["camera_lat"]))
            ),
        )
        for r in rows
    ]


def publishable(
    conn: sqlite3.Connection, pilot_dir: Path, negatives: dict[str, list[str]]
) -> list[sqlite3.Row]:
    """The pilot's placed street views that may be published, one per negative (see
    dedup.one_per_negative), with their photo's text and street; those with no matched
    picture of today first (the least certain)."""
    rows = [
        r
        for r in conn.execute(
            "SELECT pl.*, p.title, p.caption, p.photographer, p.credit, p.license, "
            "p.record_url, p.image_url, s.label AS street, s.arrondissements "
            "FROM placements pl JOIN photos p ON p.id = pl.photo_id "
            "LEFT JOIN locations l ON l.photo_id = pl.photo_id "
            "LEFT JOIN streets s ON s.id = l.street_id "
            "WHERE pl.is_street_view AND pl.lon IS NOT NULL "
            f"AND p.license IN {db.OPEN_LICENSES_SQL} "
            "ORDER BY pl.match IS NOT NULL, pl.photo_id"
        )
        if (pilot_dir / "photos" / safe_name(r["photo_id"]) / "info.json").exists()
    ]
    return dedup.one_per_negative(rows, negatives)


def area(conn: sqlite3.Connection, margin_m: float) -> BaseGeometry:
    """The box around the placed street views, widened by `margin_m` (Lambert-93)."""
    placed = conn.execute(
        "SELECT lon, lat FROM placements WHERE is_street_view AND lon IS NOT NULL"
    ).fetchall()
    points = to_metres(MultiPoint([(r["lon"], r["lat"]) for r in placed]))
    return points.envelope.buffer(margin_m).envelope


def seeds(conn: sqlite3.Connection, pilot_dir: Path, area: BaseGeometry) -> list[dict]:
    """A start for each prepared photo with no placement of its own and an open licence: the
    museum's camera position, else the door or corner its caption names, else the middle of its
    street within `area` (Lambert-93); none for what the caption reading says is not a street
    view. A seed is never shown on the walk until a camera found from it is confirmed
    (export.build)."""
    started = []
    for r in conn.execute(
        "SELECT p.id, p.camera_lon, p.camera_lat, l.point, s.geometry, g.kind "
        "FROM photos p LEFT JOIN locations l ON l.photo_id = p.id "
        "LEFT JOIN streets s ON s.id = l.street_id LEFT JOIN readings g ON g.photo_id = p.id "
        f"WHERE p.license IN {db.OPEN_LICENSES_SQL}"
    ):
        folder = pilot_dir / "photos" / safe_name(r["id"])
        if (
            not (folder / "info.json").exists()
            or (folder / "placement.json").exists()
            or r["kind"] in read.NOT_A_STREET_VIEW
        ):
            continue
        if r["camera_lon"] is not None:
            lon, lat = r["camera_lon"], r["camera_lat"]
        else:
            start = _start(r["point"], r["geometry"], area)
            if start is None:
                continue
            lon, lat = start
        started.append(
            {
                "photo_id": r["id"],
                "is_street_view": True,
                "match": None,
                "match_panoramax_id": None,
                "lon": lon,
                "lat": lat,
                "heading": 0.0,
                "confidence": SEED,
                "reason": "amorce : la place que la source ou la légende donne",
            }
        )
    return started


def _start(point: str | None, street: str | None, area: BaseGeometry) -> tuple[float, float] | None:
    """Where a caption puts a photo, in degrees: its point, else the middle of its street's
    stretch within `area` (none outside it)."""
    if point:
        at = from_geojson(point)
    elif street:
        stretch = from_geojson(street).intersection(area)
        if stretch.is_empty:
            return None
        if hasattr(stretch, "geoms"):  # in pieces: the longest
            stretch = max(stretch.geoms, key=lambda piece: piece.length)
        at = stretch.interpolate(0.5, normalized=True) if stretch.length else stretch.centroid
    else:
        return None
    at = to_degrees(at)
    return at.x, at.y
