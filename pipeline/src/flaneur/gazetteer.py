"""Street-name gazetteer of Paris: a street name, as written in an old caption, to a geometry.

Built from the City of Paris open data (ODbL):
- today's streets, with their polygons;
- obsolete names ("dénominations caduques"), linked to today's street when the City's notes
  say where the name went ("Actuellement rue de Condé", "Dénommée rue X par arrêté…"), or
  to another obsolete name ("Voir : Baudais (porte)"); names of streets that were removed
  ("Supprimée…") are kept, without geometry, so that the caller knows the street existed
  and is gone.
"""

import json
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

import httpx
import shapely
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry

from flaneur.read import ABBREVIATIONS, STREET_TYPES
from flaneur.sources import cached_json

DATASETS = {
    "current": "denominations-emprises-voies-actuelles",
    "obsolete": "denominations-des-voies-caduques",
}
EXPORT_URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/{}/exports/json"

# Geometries are kept in Lambert-93, the French projection, so that distances are in metres.
_TO_METRES = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True).transform
_TO_DEGREES = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform


def to_metres(geometry: BaseGeometry) -> BaseGeometry:
    return shapely.transform(geometry, _TO_METRES, interleaved=False)


def to_degrees(geometry: BaseGeometry) -> BaseGeometry:
    return shapely.transform(geometry, _TO_DEGREES, interleaved=False)


def bearing(start: Point, end: Point) -> float:
    """Compass direction from one point to another (Lambert-93 metres): 0 north, 90 east."""
    return math.degrees(math.atan2(end.x - start.x, end.y - start.y)) % 360


def angle(a: float, b: float) -> float:
    """The signed turn from compass direction a to b, in (−180, 180]: positive to the right."""
    return -((a - b + 180) % 360 - 180)


def to_geojson(geometry: BaseGeometry) -> str:
    """Lambert-93 geometry to GeoJSON in degrees, for storage."""
    return json.dumps(to_degrees(geometry).__geo_interface__)


def from_geojson(text: str) -> BaseGeometry:
    """Stored GeoJSON in degrees back to a Lambert-93 geometry."""
    return to_metres(shape(json.loads(text)))


def long_axis(geometry: BaseGeometry) -> tuple[tuple[float, float], tuple[float, float]]:
    """End points of the long side of the geometry's smallest enclosing rectangle: the
    direction a street runs in, and its length."""
    rectangle = geometry.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon":  # a degenerate footprint: a line or a point
        coords = list(rectangle.coords)
        return coords[0], coords[-1]
    corners = list(rectangle.exterior.coords)
    return max(
        zip(corners, corners[1:], strict=False),
        key=lambda edge: Point(edge[0]).distance(Point(edge[1])),
    )


ABBREVIATED = {"st": "saint", "ste": "sainte", "bd": "boulevard", "av": "avenue"}
# Articles that captions often drop or add: "rue Richelieu" for "rue de Richelieu".
PARTICLES = {"de", "du", "des", "la", "le", "les", "l", "d"}
# Lowercase words a caption may add after the name ("jardin du Luxembourg enneigé"); a
# capitalised word is part of the name and is never dropped.
MAX_TRAILING_WORDS = 2
# Where what a caption says of the place starts, after its name: "boulevard Malesherbes entre
# l'église de la Madeleine", "quai X le 14 juillet 1888", "rue Y pendant la crue". Not "ou":
# "rue Saint-Martin ou Saint-Denis" does not say which.
DESCRIBED = re.compile(
    r"\s+(?:entre|depuis|pendant|lors|avec|devant|derrière|(?:le|la|les|en|du)\s+\d|\d{3,4}\b)",
    re.IGNORECASE,
)
# The longest street name looked for in a title, in words (see Gazetteer.named_in); names
# with these words are not looked for there: a title's "Grande Galerie", "cour", "jardin" or
# "marché" is a room, a yard, a garden or a market far more often than a street of that name.
MOST_WORDS = 7
NOT_IN_TITLES = {"galerie", "cour", "jardin", "marche"}

_TYPE = rf"(?i:{STREET_TYPES}|{ABBREVIATIONS})"
# Where an obsolete name went, anywhere in the City's note: "Actuellement rue de Condé",
# "Dénommée avenue du Général Leclerc par arrêté…", "Devenue partie de la rue de Sèvres".
TARGET = re.compile(
    r"\b(?:Actuellement|Dénommée?|Devenue?)\s+(?:une\s+)?(?P<part>partie\s+(?:de\s+la\s+|du\s+|"
    r"des\s+|de\s+l'|de\s+))?(?:la\s+|le\s+|l')?"
    rf"(?P<label>{_TYPE}\s[^,.;]+?)(?=\s+par\b|\s+et\b|[,.;]|$)"
)
# A cross-reference to another obsolete name, written "name (type)": "Voir : Baudais (porte)".
SEE = re.compile(r"\bVoir\s*:\s*(?P<name>[^(.;]+?)\s*\((?P<type>[^)]+)\)")
REMOVED = re.compile(r"^Supprimée?\b")


def key(label: str) -> str:
    """Normalised form of a street name, for matching: accents, case, hyphens, apostrophes,
    articles and the usual abbreviations do not matter."""
    text = unicodedata.normalize("NFKD", label)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    words = re.sub(r"[-'’.]", " ", text).split()
    return " ".join(ABBREVIATED.get(w, w) for w in words if w not in PARTICLES)


def _usable(k: str) -> bool:
    """A key names a street only with a word besides the type ("rue L" alone would match
    every caption reduced to "rue de …")."""
    return len(k.split()) >= 2


def keys(label: str) -> Iterator[str]:
    """The keys a caption's street name is looked up by, most literal first (see _variants),
    those naming a street only (see _usable)."""
    return (k for variant in _variants(label) if _usable(k := key(variant)))


def lambert(lon, lat):
    """Longitudes and latitudes (WGS 84, arrays) in metres, Lambert-93."""
    return _TO_METRES(lon, lat)


def _variants(label: str) -> list[str]:
    """Forms of a caption's street name to try, most literal first: the name itself (the
    part before " - "), "faubourg X" as "rue du Faubourg X", then the name without up to
    MAX_TRAILING_WORDS trailing lowercase words."""
    label = label.split(" - ")[0].strip()
    variants = [label]
    if (cut := DESCRIBED.split(label, maxsplit=1)[0].strip()) != label:
        variants.append(cut)
        label = cut
    if label.lower().startswith("faubourg "):
        variants.append(f"rue du {label}")
    words = label.split()
    for _ in range(MAX_TRAILING_WORDS):
        if len(words) <= 2 or not words[-1][:1].islower() or words[-1] in PARTICLES:
            break
        words = words[:-1]
        variants.append(" ".join(words))
    return variants


@dataclass(frozen=True)
class Street:
    record_id: str  # the City's record id: two records can share a name
    label: str  # as the City writes it
    status: str  # "current", "renamed" (an obsolete name of a current street) or "removed"
    arrondissements: tuple[int, ...]
    geometry: BaseGeometry | None = field(default=None, compare=False)  # Lambert-93, metres
    current_label: str | None = None  # for "renamed": today's name
    # For "renamed": the old street is only part of today's (its geometry is the whole of
    # today's street, so it says nothing about where along it the old one ran).
    partial: bool = False
    # The City's historical note; for a removed street it says where it ran ("commençait
    # rue X, finissait rue Y"), which is how it will be found on the old maps.
    note: str | None = None

    @property
    def id(self) -> str:
        return f"{self.status}:{self.record_id}"


def _arrondissements(value) -> tuple[int, ...]:
    return tuple(sorted({int(n) for n in re.findall(r"\d{1,2}", str(value or ""))}))


class Gazetteer:
    def __init__(self, streets: list[Street]):
        self.by_key: dict[str, list[Street]] = defaultdict(list)
        for street in streets:
            self.add(street)

    def add(self, street: Street) -> None:
        if _usable(k := key(street.label)):
            self.by_key[k].append(street)

    def lookup(self, label: str, arrondissements: list[int] | None = None) -> list[Street]:
        """Streets that may carry this name; narrowed to the given arrondissements when that
        leaves at least one."""
        found = next((hit for k in keys(label) if (hit := self.by_key.get(k))), [])
        if arrondissements:
            inside = [s for s in found if set(s.arrondissements) & set(arrondissements)]
            found = inside or found
        return found

    def named_in(self, text: str) -> str | None:
        """The longest street name of the gazetteer written in `text`, words whole (as its
        key): "Le Pont-Neuf, 1er arrondissement" names the pont Neuf."""
        parts = [key(part).split() for part in re.split(r"[^\w\s'’-]", text)]  # never across
        for n in range(MOST_WORDS, 1, -1):
            for words in parts:
                for i in range(len(words) - n + 1):
                    k = " ".join(words[i : i + n])
                    if k in self.by_key and not NOT_IN_TITLES & set(words[i : i + n]):
                        return k
        return None

    @classmethod
    def from_records(cls, current: list[dict], obsolete: list[dict]) -> "Gazetteer":
        gazetteer = cls(
            [
                Street(
                    record_id=str(r.get("id")),
                    label=r["typo_min"],
                    status="current",
                    arrondissements=_arrondissements(r["arrdt"]),
                    geometry=to_metres(shape(r["geo_shape"]["geometry"])),
                    note=r.get("historique"),
                )
                for r in current
                if r.get("typo_min") and r.get("geo_shape")
            ]
        )
        # First pass: names that went to a current street, and removed streets. Targets are
        # looked up among current streets only, before anything obsolete is added.
        resolved: list[Street] = []
        cross_references: list[tuple[Street, str]] = []
        for r in obsolete:
            if not r.get("typo_min"):
                continue
            note = (r.get("historique") or "").strip()
            street = Street(
                str(r.get("id")),
                r["typo_min"],
                "removed",
                _arrondissements(r.get("arrdt")),
                note=note,
            )
            target = TARGET.search(note)
            candidates = (
                gazetteer.lookup(target["label"], list(street.arrondissements)) if target else []
            )
            matches = [s for s in candidates if s.status == "current"]
            if matches:
                resolved.append(
                    replace(
                        street,
                        status="renamed",
                        arrondissements=street.arrondissements or matches[0].arrondissements,
                        geometry=matches[0].geometry,
                        current_label=matches[0].label,
                        # "partie de la rue X", or "rue X et rue Y": only part of rue X.
                        partial=bool(target["part"])
                        or note[target.end() :].lstrip().startswith("et "),
                    )
                )
            elif REMOVED.match(note):
                resolved.append(street)
            elif see := SEE.search(note):
                cross_references.append((street, f"{see['type']} {see['name']}"))
        for street in resolved:
            gazetteer.add(street)
        # Second pass: "Voir : X (rue)" takes what X became.
        for street, label in cross_references:
            if seen := gazetteer.lookup(label, list(street.arrondissements)):
                target = seen[0]
                gazetteer.add(
                    replace(
                        target,
                        record_id=street.record_id,
                        label=street.label,
                        status="renamed" if target.status == "current" else target.status,
                        current_label=target.current_label
                        or (target.label if target.status == "current" else None),
                        arrondissements=street.arrondissements or target.arrondissements,
                        note=street.note,
                    )
                )
        return gazetteer

    @classmethod
    def load(cls, client: httpx.Client, cache_dir: Path) -> "Gazetteer":
        current, obsolete = (
            cached_json(cache_dir / f"{dataset}.json", partial(_download, client, dataset))
            for dataset in DATASETS.values()
        )
        return cls.from_records(current, obsolete)


def _download(client: httpx.Client, dataset: str) -> list[dict]:
    response = client.get(EXPORT_URL.format(dataset))
    response.raise_for_status()
    return response.json()
