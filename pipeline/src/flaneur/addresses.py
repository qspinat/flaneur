"""Paris house numbers of 1836 (Jacoubet's atlas) and 1888 (the Atlas municipal), vectorised by
the SoDUCo project from the Bibliothèque de l'Hôtel de Ville's scans, under CC BY-SA 4.0
(Zenodo, doi 10.5281/zenodo.16994482), and today's (the Base Adresse Nationale, Licence
Ouverte 2.0). A caption naming a number on a street is placed at that door (how near the
camera stands: geocode.ADDRESS_M); a photo from before the renumberings looks in the atlases
first, a later one in today's numbers. Positions derived from the atlases are shared under
their licence; both are credited on the site."""

import csv
import gzip
import io
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import date
from itertools import chain
from operator import itemgetter
from pathlib import Path

import httpx
import numpy as np
from shapely.geometry import Point

from flaneur.gazetteer import key, keys, lambert
from flaneur.sources import pinned_file

DEPOSIT_URL = "https://zenodo.org/records/16994482/files/paper_dhq_2024-v.1.0.1.zip?download=1"
DEPOSIT_SHA256 = "dec49bbe3961bfaa662bf9ea7dfd5dea0dfd73b60e4134596bf681ed1a5b2f3b"
ATLASES = ("atlas_municipal_1888", "atlas_jacoubet_1836")  # the older after the newer
TODAY = "ban"  # today's numbers (Base Adresse Nationale)
BAN_URL = "https://adresse.data.gouv.fr/data/ban/adresses/latest/csv/adresses-75.csv.gz"
# Photos from this year on look in today's numbers first: measured on Albert-Kahn (1908–1931),
# today's door is 17 m from the true camera in the median against 24 m for the atlases; doors
# of 1888 and today further apart than 50 m are twice as many on photos before 1890.
RENUMBERED_BY = 1890
_TO_BAN = itemgetter("nom_voie", "numero", "lon", "lat")
BAN_COLUMNS = {"nom_voie", "numero", "rep", "lon", "lat"}
# today's door this far from the atlases' for the same name and number is another street, which
# took the name since (renumberings move a door along its own street, rarely this far)
OTHER_STREET_M = 300
SAME_DOOR_M = 30  # one name's doors read for a number further apart than this: two streets
_NUMBER = re.compile(r"\d+")
_FIELDS = itemgetter("source", "street", "housenumber", "lon", "lat")


def number_of(text: str | None) -> int | None:
    """A house number's first whole number ("12 bis", "12-14": 12)."""
    found = _NUMBER.search(text or "")
    return int(found.group()) if found else None


# a door as read: its source (an atlas, or TODAY), street as written, number, longitude, latitude
Door = tuple[str, str, str, str | float, str | float]


class Addresses:
    """The doors of the atlases and of today's map, by source, street and number."""

    def __init__(self, doors: Iterable[Door] = ()):
        doors = [(n, d) for d in doors if (n := number_of(d[2])) is not None]
        x, y = lambert(
            np.array([float(d[3]) for _, d in doors]), np.array([float(d[4]) for _, d in doors])
        )
        street_key = {name: key(name) for name in {d[1] for _, d in doors}}
        seen = defaultdict(list)
        for (n, (source, street, *_)), xy in zip(doors, np.column_stack([x, y]), strict=True):
            seen[source, street_key[street], n].append(xy)
        self._doors = {}
        for door, points in seen.items():
            if len(points) == 1:
                self._doors[door] = Point(points[0])
                continue
            points = np.array(points)
            middle = points.mean(axis=0)
            if np.hypot(*(points - middle).T).max() <= SAME_DOOR_M:  # else one name, two streets
                self._doors[door] = Point(middle)

    @classmethod
    def load(cls, client: httpx.Client, folder: Path) -> "Addresses":
        """The atlases' doors, from the deposit downloaded once into `folder` (see
        sources.pinned_file), and today's, from the newest Base Adresse Nationale file for
        Paris kept there, else one downloaded now (named by its day: it changes every day,
        so it cannot be pinned; delete it to take a newer one)."""
        deposit = pinned_file(
            folder / "soduco_paper_dhq_2024-v1.0.1.zip",
            DEPOSIT_SHA256,
            lambda part: _download(client, DEPOSIT_URL, part),
            DEPOSIT_URL,
        )
        ban = max(folder.glob("ban-75-*.csv.gz"), default=None)
        if ban is None:
            ban = folder / f"ban-75-{date.today().isoformat()}.csv.gz"
            part = ban.with_name(f"{ban.name}.part")
            _download(client, BAN_URL, part)
            if not readable_ban(part):
                part.unlink()
                raise RuntimeError(f"{BAN_URL} : fichier illisible, non gardé")
            part.replace(ban)
        return cls(chain(deposit_doors(deposit), ban_doors(ban)))

    def doors(
        self, label: str, number: str | None, year: int | None = None
    ) -> Iterator[tuple[str, Point]]:
        """The doors (source, Lambert-93) of `number` on the street a caption names (`label`),
        best first: for a photo from RENUMBERED_BY on, today's, then the 1888 atlas's, then the
        1836 one's; before (or undated), the atlases' first, today's last. Today's door further
        than OTHER_STREET_M from the atlases' is another street that took the name since: left
        out."""
        if (n := number_of(number)) is None:
            return
        names = list(keys(label))
        found = {}
        for source in (*ATLASES, TODAY):
            if door := next((d for k in names if (d := self._doors.get((source, k, n)))), None):
                found[source] = door
        old = next((found[a] for a in ATLASES if a in found), None)
        if TODAY in found and old is not None and found[TODAY].distance(old) > OTHER_STREET_M:
            del found[TODAY]
        order = (TODAY, *ATLASES) if year and year >= RENUMBERED_BY else (*ATLASES, TODAY)
        yield from ((source, found[source]) for source in order if source in found)

    def lookup(self, label: str, number: str | None, year: int | None = None) -> Point | None:
        """The best of `doors`, if any."""
        return next((door for _, door in self.doors(label, number, year)), None)


def deposit_doors(deposit: Path) -> list[Door]:
    """The doors of the SoDUCo deposit (its zip): the atlases' house-number files."""
    wanted = tuple(f"/{atlas}_housenumbers.csv" for atlas in ATLASES)
    doors = []
    with zipfile.ZipFile(deposit) as z:
        for name in (n for n in z.namelist() if n.endswith(wanted)):
            with z.open(name) as f:
                doors += map(_FIELDS, csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
    return doors


def ban_doors(path: Path) -> list[Door]:
    """Today's doors, from the Base Adresse Nationale's file for Paris: plain numbers only (a
    "12 bis" is its own door, often metres from the 12)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [(TODAY, *_TO_BAN(r)) for r in csv.DictReader(f, delimiter=";") if not r["rep"]]


def readable_ban(path: Path) -> bool:
    """Whether a downloaded file is the Base Adresse Nationale's: gzip, CSV, its columns."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            header = next(csv.reader(f, delimiter=";"))
    except (OSError, EOFError, UnicodeDecodeError, StopIteration, csv.Error):
        return False
    return BAN_COLUMNS <= set(header)


def _download(client: httpx.Client, url: str, path: Path) -> None:
    with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        with path.open("wb") as out:
            for chunk in response.iter_bytes():
                out.write(chunk)
