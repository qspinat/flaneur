"""Places you can enter: a church, a courtyard, a garden. Photos of a place's inside
(read.PLACE_KINDS) are grouped by the place their title names (read.place_of), and each
place gets an entrance, so that the walk can show an "Entrer" button at its door.
"""

import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

from shapely.geometry import Point
from shapely.ops import nearest_points

from flaneur.gazetteer import Gazetteer, Street, bearing, key, to_degrees, to_metres
from flaneur.geocode import choose
from flaneur.read import PLACE_KINDS, StreetMention, place_of

# Two names this similar (after normalisation) are the same place: catalogue typos such as
# "Saint-Sévrin" for "Saint-Séverin".
SAME_PLACE = 0.95
# Below this distance between the door and the place, the direction to the place is noise.
MIN_DOOR_TO_PLACE_M = 3


@dataclass(frozen=True)
class PlacePhoto:
    photo_id: str
    kind: str  # one of read.PLACE_KINDS
    title: str


@dataclass(frozen=True)
class Entrance:
    lon: float
    lat: float
    heading: float | None  # the direction you face when going in; None when unknown
    street: str


def group(conn: sqlite3.Connection, photo_ids: list[str]) -> dict[str | None, list[PlacePhoto]]:
    """The given photos of a place's inside, by place (None: the title names no place)."""
    kinds = sorted(PLACE_KINDS)
    rows = conn.execute(
        f"SELECT p.id, p.title, r.kind FROM photos p JOIN readings r ON r.photo_id = p.id "
        f"WHERE r.kind IN ({', '.join('?' * len(kinds))}) "
        "AND p.id IN (SELECT value FROM json_each(?))",  # any number of ids, one parameter
        [*kinds, json.dumps(photo_ids)],
    ).fetchall()
    places: dict[str | None, list[PlacePhoto]] = defaultdict(list)
    for row in rows:
        title = row["title"] or ""
        places[place_of(title)].append(PlacePhoto(row["id"], row["kind"], title))
    return _merge_same_places(places)


def _merge_same_places(
    places: dict[str | None, list[PlacePhoto]],
) -> dict[str | None, list[PlacePhoto]]:
    """Merge buildings whose names differ only by case, accents or a typo; the name with the
    most photos is kept. Courtyards named by their address are merged only when equal: "3 rue
    X" and "5 rue X" are two courtyards."""
    merged: dict[str | None, list[PlacePhoto]] = {}
    if None in places:
        merged[None] = list(places[None])
    for name, photos in sorted(
        ((n, p) for n, p in places.items() if n is not None), key=lambda kv: -len(kv[1])
    ):
        same = next(
            (
                kept
                for kept in merged
                if kept is not None
                and (
                    key(kept) == key(name)
                    or not name.startswith("Cour, ")
                    and SequenceMatcher(None, key(kept), key(name)).ratio() >= SAME_PLACE
                )
            ),
            name,
        )
        merged.setdefault(same, []).extend(photos)
    return merged


def entrance(points: list[tuple[float, float]], street: Street) -> Entrance:
    """Where to go in: the point of `street` closest to the place (the median of the points,
    in degrees, where its photos were taken), facing the place."""
    lon = statistics.median(p[0] for p in points)
    lat = statistics.median(p[1] for p in points)
    centre = to_metres(Point(lon, lat))
    door = nearest_points(street.geometry, centre)[0]
    # A place placed on the street itself gives no direction: left for the review.
    heading = round(bearing(door, centre)) if door.distance(centre) >= MIN_DOOR_TO_PLACE_M else None
    door_degrees = to_degrees(door)
    return Entrance(door_degrees.x, door_degrees.y, heading, street.label)


def entrance_of(
    conn: sqlite3.Connection, photos: list[PlacePhoto], gazetteer: Gazetteer
) -> Entrance | None:
    """The entrance of a place, from its photos' placements and the street their captions
    name most; None when they are not placed or name no known street."""
    ids = json.dumps([p.photo_id for p in photos])
    points = [
        (r["lon"], r["lat"])
        for r in conn.execute(
            "SELECT lon, lat FROM placements WHERE lon IS NOT NULL "
            "AND photo_id IN (SELECT value FROM json_each(?))",
            [ids],
        )
    ]
    if not points:
        return None
    named = Counter(
        (StreetMention(**mention).label, r["arrondissements"])
        for r in conn.execute(
            "SELECT streets, arrondissements FROM readings "
            "WHERE photo_id IN (SELECT value FROM json_each(?))",
            [ids],
        )
        for mention in json.loads(r["streets"])
    )
    for (label, arrondissements), _ in named.most_common():
        if street := choose(gazetteer.lookup(label, json.loads(arrondissements))):
            return entrance(points, street)
    return None
