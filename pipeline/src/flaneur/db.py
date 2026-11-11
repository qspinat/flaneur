"""Pipeline state: one SQLite file holding every harvested photo."""

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id TEXT PRIMARY KEY,          -- "<source>:<source_id>"
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    record_url TEXT,              -- the record's page at the source
    title TEXT,
    caption TEXT,
    photographer TEXT,
    date_text TEXT,
    year_start INTEGER,
    year_end INTEGER,
    image_url TEXT,
    license TEXT NOT NULL,        -- "CC0", "CC-BY-4.0", ... or "unverified"
    credit TEXT NOT NULL,
    camera_lat REAL,              -- camera position, when the source already knows it
    camera_lon REAL,
    raw TEXT NOT NULL             -- the source record, as JSON
);

CREATE TABLE IF NOT EXISTS readings (
    photo_id TEXT PRIMARY KEY REFERENCES photos(id),
    method TEXT NOT NULL,         -- "rules" or "fine"
    kind TEXT NOT NULL,           -- street view or not: see flaneur.read
    streets TEXT NOT NULL,        -- JSON list of {type, name, number}
    arrondissements TEXT NOT NULL,-- JSON list of ints
    viewpoint TEXT,               -- where the camera stood, as written
    towards TEXT                  -- what it looked at, as written
);

CREATE TABLE IF NOT EXISTS streets (
    id TEXT PRIMARY KEY,          -- see flaneur.gazetteer.Street.id
    label TEXT NOT NULL,
    status TEXT NOT NULL,         -- "current", "renamed" or "removed"
    current_label TEXT,
    arrondissements TEXT NOT NULL,-- JSON list of ints
    note TEXT,                    -- the City's historical note
    geometry TEXT                 -- GeoJSON in degrees; none for a removed street
);

CREATE TABLE IF NOT EXISTS locations (
    photo_id TEXT PRIMARY KEY REFERENCES photos(id),
    status TEXT NOT NULL,         -- what the caption tells: see flaneur.geocode.Location
    label TEXT,                   -- the street (or corner) that was found
    street_id TEXT REFERENCES streets(id),
    point TEXT,                   -- GeoJSON point in degrees, when the camera point is known
    precision_m REAL              -- how far the camera may be from the street or the point
);

CREATE TABLE IF NOT EXISTS placements (
    photo_id TEXT PRIMARY KEY REFERENCES photos(id),
    is_street_view INTEGER NOT NULL,
    match INTEGER,                -- number of the today picture showing the same place
    match_panoramax_id TEXT,      -- that picture's Panoramax id (sheet numbers can change)
    lon REAL,                     -- where the camera stood
    lat REAL,
    heading REAL,                 -- compass direction it faced (0 = north, 90 = east)
    confidence TEXT,              -- "high", "medium" or "low"
    reason TEXT
);

CREATE TABLE IF NOT EXISTS panoramas (
    id TEXT PRIMARY KEY,          -- Panoramax id of a 360° picture
    lon REAL NOT NULL,
    lat REAL NOT NULL,
    azimuth REAL,                 -- compass direction of the panorama's centre
    taken TEXT,
    sequence TEXT,
    rank INTEGER,
    license TEXT,
    producer TEXT,
    sd_url TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS copies (
    photo_id TEXT PRIMARY KEY REFERENCES photos(id),
    negative TEXT NOT NULL        -- the print that stands for the negative (see flaneur.dedup)
);
"""


@dataclass(frozen=True)
class Photo:
    source: str
    source_id: str
    record_url: str | None
    title: str | None
    caption: str | None
    photographer: str | None
    date_text: str | None
    year_start: int | None
    year_end: int | None
    image_url: str | None
    license: str
    credit: str
    camera_lat: float | None
    camera_lon: float | None
    raw: dict

    @property
    def id(self) -> str:
        return f"{self.source}:{self.source_id}"


COLUMNS = ["id"] + [f.name for f in fields(Photo)]
# The licences under which a photo may be shown on the site; any other ("unverified") never is.
OPEN_LICENSES = ("CC0", "CC-BY-4.0")
OPEN_LICENSES_SQL = "(" + ", ".join(f"'{name}'" for name in OPEN_LICENSES) + ")"


def connect(path: Path) -> sqlite3.Connection:
    """Open the pipeline state. A table that is entirely rebuilt by a pipeline step (see
    DERIVED_TABLES) is dropped when its columns are not the ones the code expects: the step
    rebuilds it. Other tables hold work that cannot be redone and are never dropped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    for table, columns in DERIVED_TABLES.items():
        found = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
        if found and found != columns:
            conn.execute(f"DROP TABLE {table}")
    conn.executescript(SCHEMA)
    return conn


def upsert(conn: sqlite3.Connection, photos: Iterable[Photo]) -> int:
    """Insert photos, or update the source columns of photos already there (columns that
    later steps add to the table are left alone). Returns how many photos were written."""
    rows = (
        [json.dumps(p.raw, ensure_ascii=False) if c == "raw" else getattr(p, c) for c in COLUMNS]
        for p in photos
    )
    updates = ", ".join(f"{c} = excluded.{c}" for c in COLUMNS if c != "id")
    with conn:
        cursor = conn.executemany(
            f"INSERT INTO photos ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})"
            f" ON CONFLICT(id) DO UPDATE SET {updates}",
            rows,
        )
    return cursor.rowcount


READING_COLUMNS = [
    "photo_id",
    "method",
    "kind",
    "streets",
    "arrondissements",
    "viewpoint",
    "towards",
]
LOCATION_COLUMNS = ["photo_id", "status", "label", "street_id", "point", "precision_m"]
PLACEMENT_COLUMNS = [
    "photo_id",
    "is_street_view",
    "match",
    "match_panoramax_id",
    "lon",
    "lat",
    "heading",
    "confidence",
    "reason",
]
COPY_COLUMNS = ["photo_id", "negative"]
PANORAMA_COLUMNS = [
    "id",
    "lon",
    "lat",
    "azimuth",
    "taken",
    "sequence",
    "rank",
    "license",
    "producer",
    "sd_url",
]
STREET_COLUMNS = ["id", "label", "status", "current_label", "arrondissements", "note", "geometry"]


# Tables rebuilt from scratch by `flaneur geocode`, `placements`, `dedup` and `panoramas`.
DERIVED_TABLES = {
    "locations": LOCATION_COLUMNS,
    "streets": STREET_COLUMNS,
    "placements": PLACEMENT_COLUMNS,
    "copies": COPY_COLUMNS,
    "panoramas": PANORAMA_COLUMNS,
}


def _upsert(
    conn: sqlite3.Connection,
    table: str,
    columns: list[str],
    rows: Iterable[dict],
    keep_existing_when: str | None = None,
) -> int:
    """Insert or update rows (dicts keyed by `columns`; the first column is the key), inside
    the caller's transaction. An existing row is kept as is when `keep_existing_when` (SQL)
    is true."""
    updates = ", ".join(f"{c} = excluded.{c}" for c in columns[1:])
    condition = f" WHERE NOT ({keep_existing_when})" if keep_existing_when else ""
    return conn.executemany(
        f"INSERT INTO {table} ({', '.join(columns)})"
        f" VALUES ({', '.join(f':{c}' for c in columns)})"
        f" ON CONFLICT({columns[0]}) DO UPDATE SET {updates}{condition}",
        rows,
    ).rowcount


def _save(
    conn: sqlite3.Connection,
    table: str,
    columns: list[str],
    rows: Iterable[dict],
    keep_existing_when: str | None = None,
) -> int:
    with conn:
        return _upsert(conn, table, columns, rows, keep_existing_when)


def save_readings(conn: sqlite3.Connection, readings: Iterable[dict]) -> int:
    """A fine reading is never replaced by a reading from the rules."""
    return _save(
        conn,
        "readings",
        READING_COLUMNS,
        readings,
        keep_existing_when="readings.method = 'fine' AND excluded.method = 'rules'",
    )


def save_locations(conn: sqlite3.Connection, locations: Iterable[dict]) -> int:
    return _save(conn, "locations", LOCATION_COLUMNS, locations)


def save_placements(conn: sqlite3.Connection, placements: Iterable[dict]) -> int:
    return _save(conn, "placements", PLACEMENT_COLUMNS, placements)


def save_streets(conn: sqlite3.Connection, streets: Iterable[dict]) -> int:
    return _save(conn, "streets", STREET_COLUMNS, streets)


def replace(conn: sqlite3.Connection, table: str, rows: Iterable[dict]) -> int:
    """Replace every row of a table rebuilt from scratch (see DERIVED_TABLES)."""
    with conn:  # one transaction: a failed insert leaves the old rows in place
        conn.execute(f"DELETE FROM {table}")
        return _upsert(conn, table, DERIVED_TABLES[table], rows)
