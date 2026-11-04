"""Landmarks of Paris, standing above the roofs: where they stand, how high, and in which
years. A photo showing one is marked by hand (data/overrides/sights.jsonl: the landmark and
its column); a camera found for it must see it there. The known-answer check marks its views
itself."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from flaneur.blocks import Blocks
from flaneur.match import OWN_M, Sight, View, project

Period = tuple[int | None, int | None]  # the first and the last year it stands (None: open)
ALWAYS: tuple[Period, ...] = ((None, None),)


@dataclass(frozen=True)
class Landmark:
    at: tuple[float, float]  # metres Lambert-93
    top: float  # metres NGF
    name: str  # as the user reads it
    standing: tuple[Period, ...] = ALWAYS
    spread: float = 0.0  # metres from its point to each of its tops (twin: half their spacing)

    def stands(self, years: Sequence[int | None]) -> bool:
        """Whether it stands at some time in a photo's `years` (all of them, when unknown)."""
        known = [year for year in years if year is not None]
        if not known:
            return True
        first, last = min(known), max(known)
        return any(
            (since is None or since <= last) and (until is None or first <= until)
            for since, until in self.standing
        )


# Their tops in IGN's LiDAR HD surface model (MNS, flown 2021-2022), at the highest point of
# the part named; twin towers or spires: midway between them, their spread measured there too
# (each top 0.5 m near). The dates are when the part was finished (Wikipedia, in French); a
# photo may show it rising a few years earlier.
LANDMARKS = {
    "notre-dame-tours": Landmark((652242.0, 6861664.0), 105.4, "Notre-Dame (tours)", spread=18.6),
    # not in the LiDAR (flown without it): 96 m over the crossing; Viollet-le-Duc's, 1859,
    # burnt in April 2019, again from 2024
    "notre-dame-fleche": Landmark(
        (652300.0, 6861625.0), 129.8, "Notre-Dame (flèche)", ((1859, 2019), (2024, None))
    ),
    "sainte-chapelle-fleche": Landmark(
        (651944.0, 6861903.0), 94.9, "Sainte-Chapelle (flèche)", ((1853, None),)
    ),
    "saint-severin-tour": Landmark((651973.0, 6861553.0), 80.1, "Saint-Séverin (tour)"),
    "pantheon-dome": Landmark((652015.0, 6860880.5), 149.6, "Panthéon (dôme)"),
    "saint-etienne-du-mont-tour": Landmark(
        (652134.5, 6860929.5), 111.6, "Saint-Étienne-du-Mont (tour)"
    ),
    "tour-clovis": Landmark((652146.7, 6860877.4), 101.3, "Tour Clovis (lycée Henri-IV)"),
    "sorbonne-dome": Landmark((651801.5, 6861132.5), 98.5, "Sorbonne (dôme de la chapelle)"),
    "val-de-grace-dome": Landmark((651706.8, 6860257.7), 115.8, "Val-de-Grâce (dôme)"),
    "tour-saint-jacques": Landmark((652230.0, 6862192.5), 90.8, "Tour Saint-Jacques"),
    "saint-germain-des-pres-tour": Landmark(
        (651125.5, 6861766.0), 91.9, "Saint-Germain-des-Prés (tour)"
    ),
    # its north tower 107.0 m, its south one, never finished, 103.5 m
    "saint-sulpice-tours": Landmark(
        (651138.5, 6861417.8), 105.2, "Saint-Sulpice (tours)", spread=20.0
    ),
    "invalides-dome": Landmark((649557.6, 6861884.8), 132.4, "Invalides (dôme)"),
    "institut-coupole": Landmark((651351.6, 6862116.6), 68.2, "Institut de France (coupole)"),
    "tour-de-l-horloge": Landmark(
        (652023.1, 6861996.8), 70.9, "Tour de l'Horloge (Palais de justice)"
    ),
    "tribunal-de-commerce-dome": Landmark(  # built 1860-1865
        (652061.6, 6861942.0), 80.8, "Tribunal de commerce (dôme)", ((1865, None),)
    ),
    "saint-germain-l-auxerrois-beffroi": Landmark(  # built 1858-1863
        (651648.3, 6862399.8), 73.5, "Beffroi de Saint-Germain-l'Auxerrois", ((1863, None),)
    ),
    "saint-eustache": Landmark((651918.2, 6862814.4), 87.1, "Saint-Eustache (façade)"),
    "saint-gervais-tour": Landmark((652660.0, 6861915.0), 92.3, "Saint-Gervais (tour)"),
    # rebuilt 1874-1882 after the fire of May 1871, its old façade and belfry reproduced
    "hotel-de-ville": Landmark(
        (652459.6, 6862028.4), 92.1, "Hôtel de Ville (campanile)", ((None, 1871), (1882, None))
    ),
    "saint-paul-saint-louis-dome": Landmark(
        (653140.0, 6861799.2), 95.4, "Saint-Paul-Saint-Louis (dôme)"
    ),
    "colonne-de-juillet": Landmark((653708.6, 6861643.2), 82.5, "Colonne de Juillet"),
    "saint-nicolas-des-champs-tour": Landmark(
        (652592.7, 6863023.4), 69.6, "Saint-Nicolas-des-Champs (tour)"
    ),
    "porte-saint-denis": Landmark((652525.9, 6863498.8), 61.8, "Porte Saint-Denis"),
    "porte-saint-martin": Landmark((652727.8, 6863430.5), 53.2, "Porte Saint-Martin"),
    "saint-vincent-de-paul-tours": Landmark(
        (652464.7, 6864497.4), 96.4, "Saint-Vincent-de-Paul (tours)", spread=13.7
    ),
    # its nave opened in 1891; the dome, then the bell tower (1912) followed
    "sacre-coeur-dome": Landmark(
        (651824.4, 6865385.2), 208.0, "Sacré-Cœur (dôme)", ((1891, None),)
    ),
    "trinite-tour": Landmark(  # built 1861-1867
        (650974.0, 6864306.2), 105.5, "La Trinité (tour)", ((1867, None),)
    ),
    "saint-augustin-dome": Landmark(  # built 1860-1871
        (650035.6, 6864256.4), 112.1, "Saint-Augustin (dôme)", ((1871, None),)
    ),
    "opera": Landmark(  # its façade unveiled in 1867, opened in 1875
        (650978.8, 6863771.6), 101.8, "Opéra Garnier (faîte)", ((1867, None),)
    ),
    "madeleine": Landmark((650435.4, 6863522.8), 71.4, "La Madeleine (fronton)"),
    # not in the LiDAR (too thin): 44.3 m; pulled down in May 1871, raised again 1873-1875
    "colonne-vendome": Landmark(
        (650809.4, 6863257.9), 78.3, "Colonne Vendôme", ((None, 1871), (1875, None))
    ),
    "sainte-clotilde-fleches": Landmark(
        (650065.4, 6862280.2), 99.4, "Sainte-Clotilde (flèches)", ((1857, None),), spread=8.7
    ),
    "arc-de-triomphe": Landmark((648292.7, 6863982.1), 108.4, "Arc de Triomphe"),
    "tour-eiffel": Landmark(  # its top as the LiDAR saw it (292 m above the ground)
        (648238.0, 6862258.4), 326.1, "Tour Eiffel", ((1889, None),)
    ),
}
# the spires a photo often cannot tell apart: marked as either
SPIRES = ("notre-dame-fleche", "sainte-chapelle-fleche")
# a building's parts, one landmark each (marked and checked at its own point), looked for as
# one in photos (from afar, Notre-Dame's transept, between them, looks like its towers)
BUILDINGS = {"notre-dame": ("notre-dame-tours", "notre-dame-fleche")}
EDGE = 0.05  # a landmark nearer the edge of a view than this share of it is not marked
SIDE = 1000  # pixels across a view, to find its landmarks' columns


def image_hash(path: Path) -> str:
    """An image's content, hashed: what a photo's marks were made on (its image as exported,
    cropped: cropped otherwise, a mark's column is no longer where the landmark is)."""
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


def rows(path: Path) -> list[dict]:
    """The rows of the marks file (data/overrides/sights.jsonl), none when it is missing:
    one per mark, each `photo_id`, `image` (see image_hash) and either a landmark of the
    catalogue (`landmarks`: several when the photo cannot tell which), another by its name
    (`other`: not for the matcher yet), or `none` (the photo shows no landmark); `checked`
    when the user saved it (marks proposed otherwise)."""
    if not path.exists():
        return []
    found, images = [], {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or not all(
                isinstance(row.get(key), str) and row[key] for key in ("photo_id", "image")
            ):
                raise ValueError("une ligne a une photo et une image")
            if row.get("none"):
                if row["none"] is not True or not row.keys() <= NONE_KEYS:
                    raise ValueError("« aucun repère » ne porte rien d'autre")
            else:
                mark(row)
            if images.setdefault(row["photo_id"], row["image"]) != row["image"]:
                raise ValueError(f"{row['photo_id']} marquée sur deux images")
        except (ValueError, TypeError) as e:
            raise ValueError(f"{path}:{n} : repère mal écrit ({e})") from e
        found.append(row)
    return found


def _fraction(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and 0 <= value <= 1


def mark(row: dict) -> dict:
    """A mark, checked: its landmark (or landmarks), or another's name; its column and, if
    marked, its top (fractions of the width, of the height)."""
    if not isinstance(row, dict):
        raise ValueError("un repère est un objet")
    names, other = row.get("landmarks"), row.get("other")
    if names is not None:
        if not isinstance(names, list) or not names or not all(n in LANDMARKS for n in names):
            raise ValueError(f"repères inconnus : {names}")
        found = {"landmarks": list(names)}
    elif isinstance(other, str) and other.strip():
        found = {"other": other.strip()}
    else:
        raise ValueError("un repère a un nom du catalogue ou un autre nom")
    if not _fraction(row.get("column")):
        raise ValueError("la colonne est une fraction de la largeur")
    found["column"] = round(row["column"], 4)
    if row.get("top") is not None:
        if not _fraction(row["top"]):
            raise ValueError("le sommet est une fraction de la hauteur")
        found["top"] = round(row["top"], 4)
    return found


def among(blocks: Blocks, landmarks: dict[str, Landmark] = LANDMARKS) -> dict[str, Landmark]:
    """The landmarks standing among `blocks` (inside their extent): from a camera among them
    too, whether one is hidden can be told; beyond, the buildings before it are not known."""
    corners = blocks.edges[:, :4].reshape(-1, 2)
    if not len(corners):
        return {}
    low, high = corners.min(axis=0), corners.max(axis=0)
    return {
        key: landmark
        for key, landmark in landmarks.items()
        if (low <= landmark.at).all() and (landmark.at <= high).all()
    }


NONE_KEYS = {"photo_id", "image", "none", "checked"}  # a photo showing no landmark: no more


class Marks(NamedTuple):
    image: str  # what they were marked on (see image_hash)
    sights: tuple[Sight, ...]


def marked(path: Path) -> dict[str, Marks]:
    """The landmarks of the catalogue marked in each photo, by its id."""
    images, found = {}, defaultdict(list)
    for row in rows(path):  # checked: one image per photo
        if "landmarks" not in row:
            continue
        images[row["photo_id"]] = row["image"]
        named = [LANDMARKS[name] for name in row["landmarks"]]
        found[row["photo_id"]].append(
            Sight(
                tuple(lm.at for lm in named),
                row["column"],
                tuple(lm.top for lm in named),
                tuple(lm.spread for lm in named),
            )
        )
    return {photo: Marks(images[photo], tuple(sights)) for photo, sights in found.items()}


def in_view(
    view: View,
    origin_z: float,
    blocks: Blocks,
    year: int,
    landmarks: dict[str, Landmark] = LANDMARKS,
) -> tuple[Sight, ...]:
    """The landmarks standing in `year` that a level camera with a square view sees (its top
    inside the view, nothing standing before it), as they would be marked in its photo
    (heights above `origin_z`, metres NGF)."""
    standing = [landmark for landmark in landmarks.values() if landmark.stands([year])]
    if not standing:
        return ()
    ats = np.array([landmark.at for landmark in standing])
    tops = np.column_stack([ats, [landmark.top - origin_z for landmark in standing]])
    spreads = np.array([landmark.spread for landmark in standing])
    u, v = project(view, tops, (SIDE, SIDE)).T  # NaN behind the camera
    inside = (u >= EDGE * SIDE) & (u <= (1 - EDGE) * SIDE) & (v >= 0)
    origin = np.array([view.x, view.y, view.z])
    towards = tops[inside] - origin
    hit, _ = blocks.hit(origin, towards / np.linalg.norm(towards, axis=1, keepdims=True))
    away = np.hypot(*(ats[inside] - origin[:2]).T)
    before = np.hypot(*(hit[:, :2] - origin[:2]).T)  # NaN: nothing met
    clear = ~(before < away - OWN_M)
    return tuple(
        Sight((tuple(at),), float((column + 0.5) / SIDE), (float(top) + origin_z,), (spread,))
        for at, column, top, spread in zip(
            ats[inside][clear],
            u[inside][clear],
            tops[inside][clear][:, 2],
            spreads[inside][clear],
            strict=True,
        )
    )
