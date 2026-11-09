"""Paris house numbers of 1836 (Jacoubet's atlas) and 1888 (the Atlas municipal), vectorised by
the SoDUCo project from the Bibliothèque de l'Hôtel de Ville's scans, under CC BY-SA 4.0
(Zenodo, doi 10.5281/zenodo.16994482). A caption naming a number on a street is placed at
that door (how near the camera stands: geocode.ADDRESS_M). Positions derived from them are
shared under the same licence, credited on the site."""

import csv
import io
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from operator import itemgetter
from pathlib import Path

import httpx
import numpy as np
from shapely.geometry import Point

from flaneur.gazetteer import key, keys, lambert
from flaneur.sources import pinned_file

DEPOSIT_URL = "https://zenodo.org/records/16994482/files/paper_dhq_2024-v.1.0.1.zip?download=1"
DEPOSIT_SHA256 = "dec49bbe3961bfaa662bf9ea7dfd5dea0dfd73b60e4134596bf681ed1a5b2f3b"
ATLASES = ("atlas_municipal_1888", "atlas_jacoubet_1836")  # tried in this order
SAME_DOOR_M = 30  # one name's doors read for a number further apart than this: two streets
_NUMBER = re.compile(r"\d+")
_FIELDS = itemgetter("source", "street", "housenumber", "lon", "lat")


def number_of(text: str | None) -> int | None:
    """A house number's first whole number ("12 bis", "12-14": 12)."""
    found = _NUMBER.search(text or "")
    return int(found.group()) if found else None


class Addresses:
    """The doors of the atlases: (atlas, street as written, number, longitude, latitude)."""

    def __init__(self, doors: Iterable[tuple[str, str, str, float, float]] = ()):
        doors = [d for d in doors if number_of(d[2]) is not None]
        x, y = lambert(
            np.array([float(d[3]) for d in doors]), np.array([float(d[4]) for d in doors])
        )
        street_key = {name: key(name) for name in {d[1] for d in doors}}
        seen = defaultdict(list)
        for (atlas, street, number, _, _), xy in zip(doors, np.column_stack([x, y]), strict=True):
            seen[atlas, street_key[street], number_of(number)].append(xy)
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
    def from_deposit(cls, deposit: Path) -> "Addresses":
        """The doors of the SoDUCo deposit (its zip): the atlases' house-number files."""
        wanted = tuple(f"/{atlas}_housenumbers.csv" for atlas in ATLASES)
        doors = []
        with zipfile.ZipFile(deposit) as z:
            for name in (n for n in z.namelist() if n.endswith(wanted)):
                with z.open(name) as f:
                    doors += map(_FIELDS, csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
        return cls(doors)

    @classmethod
    def load(cls, client: httpx.Client, folder: Path) -> "Addresses":
        """The doors, from the deposit downloaded once into `folder` (see sources.pinned_file)."""

        def download(part: Path) -> None:
            with client.stream("GET", DEPOSIT_URL, follow_redirects=True) as response:
                response.raise_for_status()
                with part.open("wb") as out:
                    for chunk in response.iter_bytes():
                        out.write(chunk)

        deposit = folder / "soduco_paper_dhq_2024-v1.0.1.zip"
        return cls.from_deposit(pinned_file(deposit, DEPOSIT_SHA256, download, DEPOSIT_URL))

    def lookup(self, label: str, number: str | None) -> Point | None:
        """The door (Lambert-93) of `number` on the street a caption names (`label`), as the
        1888 atlas has it, else the 1836 one; none if neither knows it."""
        if (n := number_of(number)) is None:
            return None
        names = list(keys(label))
        return next(
            (door for atlas in ATLASES for k in names if (door := self._doors.get((atlas, k, n)))),
            None,
        )
