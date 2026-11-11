"""Place each photo from its reading: the street it names, the door of the number it gives on
that street (the atlases of 1836 and 1888, see flaneur.addresses), or the corner where the
camera stood when the caption says "vue prise de <another street>".

A photo of a street that has since been removed gets the status "removed": its place will
come from the old maps (Jacoubet), not from today's streets.
"""

import json
import re
import sqlite3
import statistics
from collections import Counter
from dataclasses import dataclass

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from flaneur import db, dedup
from flaneur.addresses import Addresses
from flaneur.gazetteer import Gazetteer, Street, from_geojson, long_axis, to_geojson, to_metres
from flaneur.read import NOT_A_STREET_VIEW, StreetMention, find_streets

# Two streets closer than this are taken to meet: the camera stood where they meet.
CORNER_TOLERANCE_M = 15
# A door (flaneur.addresses): the true camera 24 m from it in the median, 86 % within this,
# for the atlases' (Albert-Kahn, 71 photos); 17 m and 95 % for today's (77 photos). The number
# is on the facade, the photographer across the street or back from it.
ADDRESS_M = 50
# A door further than this from every street of its name the gazetteer knows is another
# street's (a namesake elsewhere, or renamed or renumbered since).
DOOR_FROM_STREET_M = 50


@dataclass(frozen=True)
class Location:
    # "corner": the camera stood where the photo's street meets the "vue prise de" street;
    # "address": before the door of the number the caption gives (an atlas, or today's map);
    # "viewpoint_street": on the "vue prise de" street, which does not meet the photo's;
    # "street": somewhere along the photo's street; "ambiguous": the name fits several
    # streets and nothing tells which; "removed": the street is gone (old maps will place
    # it); "not_found": no street of the caption is known.
    status: str
    label: str | None = None
    street: Street | None = None
    point: Point | None = None  # Lambert-93, when the camera point is known
    precision_m: float | None = None  # None when unknown


def _first_found(labels: list[str], arrondissements: list[int], gazetteer: Gazetteer):
    """The first named street the gazetteer knows, with its candidates."""
    for label in labels:
        if found := gazetteer.lookup(label, arrondissements):
            return label, found
    return None, []


def _half_length(geometry: BaseGeometry) -> float:
    """Half the long side of the street's footprint: how far along it the camera may be."""
    start, end = long_axis(geometry)
    return Point(start).distance(Point(end)) / 2


def _along(street: Street) -> float | None:
    """Precision of "somewhere along this street": unknown when the old street was only part
    of today's (its geometry is the whole of today's street)."""
    return None if street.partial else _half_length(street.geometry)


def choose(found: list[Street]) -> Street | None:
    """The street a name designates among the gazetteer's candidates, or None when several
    places fit. A name shared by today's street and an old name of another street ("pont
    Neuf" was also an old name of the pont Saint-Michel) is taken as today's street.
    TODO: compare the photo's date with the renaming date in the City's note."""
    placed = [s for s in found if s.geometry is not None]
    candidates = [s for s in placed if s.status == "current"] or placed
    return candidates[0] if len({id(s.geometry) for s in candidates}) == 1 else None


def _named_too(
    found: list[Street], others: list[str], arrondissements: list[int], gazetteer: Gazetteer
) -> Street | None:
    """Among the streets an old name designates, the one the caption names too (`others`):
    "rue des Sept-Voies, actuelle rue Valette" (two streets bore the name, one the Valette)."""
    for other in others:
        if named := choose(gazetteer.lookup(other, arrondissements)):
            if same := next((s for s in found if s.geometry is named.geometry), None):
                return same
    return None


def _seen_from(
    found: list[Street], viewpoint: str | None, arrondissements: list[int], gazetteer: Gazetteer
) -> Street | None:
    """Among the streets an old name designates, the only one that meets the street the
    camera stood in ("rue des Sept-Voies, vue prise de la rue Saint-Hilaire")."""
    if not viewpoint:
        return None
    own = {s.record_id for s in found}
    stood = [  # streets the viewpoint names unambiguously, not the photo's own name again
        street.geometry
        for mention in find_streets(viewpoint)
        if (street := choose(gazetteer.lookup(mention.label, arrondissements)))
        and street.record_id not in own
    ]
    meeting = {
        id(s.geometry): s
        for s in found
        if s.geometry is not None
        and any(s.geometry.distance(g) <= CORNER_TOLERANCE_M for g in stood)
    }
    return next(iter(meeting.values())) if len(meeting) == 1 else None


NO_DOORS = Addresses()


def _door(
    mentions: list[StreetMention],
    arrondissements: list[int],
    year: int | None,
    gazetteer: Gazetteer,
    addresses: Addresses,
) -> Location | None:
    """Before the door of the first numbered street of the caption whose number is known (for
    a photo of `year`: see Addresses.lookup), on that
    street as the gazetteer knows it: the one nearest the door, within DOOR_FROM_STREET_M of
    it (a street gone, or unknown today, cannot tell: its door is taken)."""
    for mention in mentions:
        found = gazetteer.lookup(mention.label, arrondissements)
        placed = [s for s in found if s.geometry is not None]
        for source, door in addresses.doors(mention.label, mention.number, year):
            if placed:
                street = min(placed, key=lambda s: s.geometry.distance(door))
                if street.geometry.distance(door) > DOOR_FROM_STREET_M:
                    continue  # another street's: the next source's door may be this one's
            else:
                street = found[0] if found else None
            label = f"{mention.number} {mention.label} ({source})"  # where the door comes from
            return Location("address", label, street, door, precision_m=ADDRESS_M)
    return None


def locate(reading: dict, gazetteer: Gazetteer, addresses: Addresses = NO_DOORS) -> Location:
    """Where a photo was taken, by its reading: the most precise of what its streets and its
    numbered doors give (see STATUS_RANK)."""
    mentions = [StreetMention(**s) for s in json.loads(reading["streets"])]
    arrondissements = json.loads(reading["arrondissements"])
    door = _door(mentions, arrondissements, reading["year"], gazetteer, addresses)
    labels = [m.label for m in mentions]
    found = [_by_street(labels, arrondissements, reading["viewpoint"], gazetteer), door]
    if found[0].status == "not_found" and door is None and reading["title"]:
        found += _by_title(reading, arrondissements, gazetteer, addresses)
    return min(filter(None, found), key=_rank)


# Where a title says where the view was taken from ("X vus du Y", "prise de la rue Y", "depuis
# Y"): the camera stood there, not on what comes before; and where it turns to what the photo
# looks toward or stands near: not where it was taken.
FROM = re.compile(
    r"\b(?:vue?s?|prise?s?)\s+(?:du|des|de\s+la|de\s+l['’]|d['’]|de)\s*|\bdepuis\s", re.IGNORECASE
)
TOWARDS = re.compile(
    r"\b(?:vers|en\s+direction\s+(?:du|des|de)|près\s+(?:du|des|de))\b|\b(?:en\s+direction|près)\s+d['’]",
    re.IGNORECASE,
)


def _by_title(
    reading, arrondissements: list[int], gazetteer: Gazetteer, addresses: Addresses
) -> list[Location | None]:
    """Where a photo was taken by the street its title names (see Gazetteer.named_in), when
    its reading found none the gazetteer knows ("Le Pont-Neuf": the type inside the name):
    the place it was taken from when the title says so; along the first street that lies in
    an arrondissement the caption gives (a name can be a shorter one's, elsewhere), or before
    its door. A name in lowercase ("la cour du château de Vitry") is a street only in a
    caption that gives its arrondissement."""
    parts = FROM.split(reading["title"], maxsplit=1)
    seen = TOWARDS.split(parts[-1], maxsplit=1)[0]
    for mention in gazetteer.named_in(seen):
        if not arrondissements and not any(c.isupper() for c in mention.name):
            continue
        found = gazetteer.lookup(mention.label, arrondissements)
        if arrondissements and not any(
            set(s.arrondissements) & set(arrondissements) for s in found
        ):
            continue
        return [
            _by_street([mention.label], arrondissements, reading["viewpoint"], gazetteer),
            _door([mention], arrondissements, reading["year"], gazetteer, addresses),
        ]
    return []


def _by_street(
    streets: list[str], arrondissements: list[int], viewpoint: str | None, gazetteer: Gazetteer
) -> Location:
    """Where a photo was taken by its streets alone (their names): their corner, the
    viewpoint's street, or somewhere along its own."""
    label, found = _first_found(streets, arrondissements, gazetteer)
    if not found:
        return Location("not_found")
    if all(s.geometry is None for s in found):
        return Location("removed", label, found[0])
    mentioned = [s for s in streets if s != label]
    street = (
        choose(found)
        or _named_too(found, mentioned, arrondissements, gazetteer)
        or _seen_from(found, viewpoint, arrondissements, gazetteer)
    )
    if street is None:
        return Location("ambiguous", label)
    if viewpoint:
        # Every street named in the viewpoint, except the photo's own street.
        crossing = [
            (mention.label, other)
            for mention in find_streets(viewpoint)
            if (other := choose(gazetteer.lookup(mention.label, arrondissements)))
            and other.geometry is not street.geometry
        ]
        for other_label, other in crossing:
            if street.geometry.distance(other.geometry) <= CORNER_TOLERANCE_M:
                meeting = street.geometry.buffer(CORNER_TOLERANCE_M).intersection(
                    other.geometry.buffer(CORNER_TOLERANCE_M)
                )
                return Location(
                    "corner",
                    f"{label} × {other_label}",
                    street,
                    meeting.centroid,
                    precision_m=2 * CORNER_TOLERANCE_M,
                )
        if crossing:
            other_label, other = crossing[0]
            return Location("viewpoint_street", other_label, other, precision_m=_along(other))
    return Location("street", label, street, precision_m=_along(street))


# From the most precise kind of place to the least.
STATUS_RANK = (  # a caption saying where the camera stood beats a door of the street shown
    "corner",
    "viewpoint_street",
    "address",
    "street",
    "removed",
    "ambiguous",
    "not_found",
)


def _rank(location: Location) -> tuple[int, float]:
    precision = location.precision_m if location.precision_m is not None else float("inf")
    return STATUS_RANK.index(location.status), precision


def locate_negative(
    readings: list[dict], gazetteer: Gazetteer, addresses: Addresses = NO_DOORS
) -> Location:
    """Where a negative printed several times was taken, from all its prints' captions read
    together: one print may name the street, another where the camera stood ("vue prise de
    la rue Galande"), so each print's streets are tried with every print's viewpoint."""
    viewpoints = {r["viewpoint"] for r in readings}
    return min(
        (locate({**r, "viewpoint": v}, gazetteer, addresses) for r in readings for v in viewpoints),
        key=_rank,
    )


def _street_row(street: Street) -> dict:
    return {
        "id": street.id,
        "label": street.label,
        "status": street.status,
        "current_label": street.current_label,
        "arrondissements": json.dumps(list(street.arrondissements)),
        "note": street.note,
        "geometry": to_geojson(street.geometry) if street.geometry is not None else None,
    }


# each reading with its photo's first year (which house numbers to look in first)
_READINGS = (
    "SELECT r.*, p.year_start AS year, p.title FROM readings r "
    "LEFT JOIN photos p ON p.id = r.photo_id"
)


def locate_all(
    conn: sqlite3.Connection, gazetteer: Gazetteer, addresses: Addresses = NO_DOORS
) -> Counter[str]:
    """Locate every photo that may be a street view. Returns how many got each status."""
    excluded = ", ".join("?" * len(NOT_A_STREET_VIEW))
    readings = conn.execute(
        f"{_READINGS} WHERE r.kind NOT IN ({excluded})",
        sorted(NOT_A_STREET_VIEW),
    ).fetchall()
    locations = {r["photo_id"]: locate(r, gazetteer, addresses) for r in readings}
    # A print of a negative printed several times (see flaneur.dedup) takes the place all
    # its prints' captions give together when it is of a more precise kind than its own (a
    # corner rather than a street). Two streets of the same kind are not compared: a shorter
    # street is not a truer one.
    read = {
        r["photo_id"]: dict(r)
        for r in conn.execute(f"{_READINGS} WHERE r.photo_id IN (SELECT photo_id FROM copies)")
    }
    for negative in {tuple(p) for p in dedup.negatives(conn).values()}:
        prints = [read[i] for i in negative if i in read]
        if not prints:
            continue
        best = locate_negative(prints, gazetteer, addresses)
        for r in prints:
            own = locations.get(r["photo_id"])
            if own and _rank(best)[0] < _rank(own)[0]:
                locations[r["photo_id"]] = best
    # Each street is stored once, however many photos show it.
    streets = {loc.street.id: loc.street for loc in locations.values() if loc.street}
    db.save_streets(conn, (_street_row(s) for s in streets.values()))
    with conn:  # photos no longer read as street views lose their position
        conn.execute(
            f"DELETE FROM locations WHERE photo_id NOT IN "
            f"(SELECT photo_id FROM readings WHERE kind NOT IN ({excluded}))",
            sorted(NOT_A_STREET_VIEW),
        )
    db.save_locations(
        conn,
        (
            {
                "photo_id": photo_id,
                "status": loc.status,
                "label": loc.label,
                "street_id": loc.street.id if loc.street else None,
                "point": to_geojson(loc.point) if loc.point else None,
                "precision_m": loc.precision_m,
            }
            for photo_id, loc in locations.items()
        ),
    )
    return Counter(loc.status for loc in locations.values())


def distances_to_truth(conn: sqlite3.Connection) -> dict[str, list[float]]:
    """For photos whose camera position is known, by status: the distance in metres from
    that position to where we placed them (the point, else the street; 0 on the street)."""
    rows = conn.execute(
        "SELECT p.camera_lon, p.camera_lat, l.status, COALESCE(l.point, s.geometry) AS placed "
        "FROM photos p JOIN locations l ON l.photo_id = p.id "
        "LEFT JOIN streets s ON s.id = l.street_id "
        "WHERE p.camera_lat IS NOT NULL AND COALESCE(l.point, s.geometry) IS NOT NULL"
    ).fetchall()
    distances: dict[str, list[float]] = {}
    for r in rows:
        truth = to_metres(Point(r["camera_lon"], r["camera_lat"]))
        placed = from_geojson(r["placed"])
        distances.setdefault(r["status"], []).append(placed.distance(truth))
    return distances


def summary(distances: list[float]) -> dict[str, float]:
    return {
        "n": len(distances),
        "within_50m": sum(d <= 50 for d in distances) / len(distances),
        "median_m": statistics.median(distances),
    }
