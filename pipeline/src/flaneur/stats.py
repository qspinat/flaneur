"""Numbers each phase gate needs, computed from the pipeline state."""

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass

from flaneur.read import ABBREVIATIONS, STREET_TYPES

# Words that name a street-like place: the same list the caption reader uses.
STREET_WORDS = re.compile(rf"\b(?:{STREET_TYPES})\b|\b(?:{ABBREVIATIONS})", re.IGNORECASE)


@dataclass(frozen=True)
class SourceStats:
    source: str
    photos: int
    by_decade: dict[int | None, int]
    naming_a_street: int
    with_camera_position: int
    by_license: dict[str, int]


def names_a_street(text: str | None) -> bool:
    return bool(text and STREET_WORDS.search(text))


def _by_decade(years: list[int | None]) -> dict[int | None, int]:
    """Photo count per decade, oldest first, unknown dates last."""
    counts = Counter(year // 10 * 10 if year else None for year in years)
    return dict(sorted(counts.items(), key=lambda item: (item[0] is None, item[0] or 0)))


def compute(conn: sqlite3.Connection) -> list[SourceStats]:
    by_source: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in conn.execute(
        "SELECT source, title, caption, year_start, camera_lat, license FROM photos"
    ):
        by_source[row["source"]].append(row)
    return [
        SourceStats(
            source=source,
            photos=len(rows),
            by_decade=_by_decade([r["year_start"] for r in rows]),
            naming_a_street=sum(
                names_a_street(r["title"]) or names_a_street(r["caption"]) for r in rows
            ),
            with_camera_position=sum(r["camera_lat"] is not None for r in rows),
            by_license=dict(Counter(r["license"] for r in rows)),
        )
        for source, rows in sorted(by_source.items())
    ]
