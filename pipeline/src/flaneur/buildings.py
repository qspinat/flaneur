"""Today's buildings (IGN BD TOPO, Etalab 2.0): footprints and heights, the surfaces the 3D
walk projects panoramas and old photos onto."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import shapely
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry

from flaneur.sources import cached_json

MARGIN_M = 120  # around the photos: what their cameras can see
WFS_URL = "https://data.geopf.fr/wfs/ows"
LAYER = "BDTOPO_V3:batiment"
PAGE = 1000
DEFAULT_HEIGHT_M = 15.0  # a Paris house of four or five floors, when BD TOPO has no height
SIMPLIFY_M = 0.3
INSIDE_M = 1.0  # a spot this far inside a footprint is in the building (drawn to a metre or so)


@dataclass(frozen=True)
class Prism:
    footprint: list[tuple[float, float]]  # outer ring, metres east and north of the origin
    ground: float | None  # altitude of its lowest ground point (metres, NGF), when known
    height: float
    appeared: int | None  # year the building is known from (BD TOPO "date d'apparition")


def fetch(client: httpx.Client, area: BaseGeometry, cache_dir: Path) -> list[dict]:
    """Every building inside `area` (Lambert-93), as GeoJSON features in Lambert-93, read from
    the WFS page by page; each page is kept on disk and not asked for again."""
    x0, y0, x1, y1 = (round(v) for v in area.bounds)
    features: list[dict] = []

    def get(start: int) -> dict:
        response = client.get(
            WFS_URL,
            params={
                "SERVICE": "WFS",
                "VERSION": "2.0.0",
                "REQUEST": "GetFeature",
                "TYPENAMES": LAYER,
                "SRSNAME": "EPSG:2154",
                "BBOX": f"{x0},{y0},{x1},{y1},urn:ogc:def:crs:EPSG::2154",
                "COUNT": PAGE,
                "STARTINDEX": start,
                "SORTBY": "cleabs",  # pages stay the same from one request to the next
                "outputFormat": "application/json",
            },
        )
        response.raise_for_status()
        return response.json()

    while True:
        start = len(features)
        page = cache_dir / f"{x0}_{y0}_{x1}_{y1}_{start}.json"
        found = cached_json(page, lambda start=start: get(start))["features"]
        features += found
        if len(found) < PAGE:
            return features


def outlines(feature: dict) -> list[Polygon]:
    """A building's outline (Lambert-93), simplified: one polygon per part."""
    found = shape(feature["geometry"]).buffer(0).simplify(SIMPLIFY_M)
    return [g for g in getattr(found, "geoms", [found]) if not g.is_empty]


def appeared(feature: dict) -> int | None:
    """The year BD TOPO knows the building from, if it does."""
    date = feature["properties"].get("date_d_apparition")
    return int(date[:4]) if date else None


def standing(features: list[dict], year: int) -> list[dict]:
    """The buildings standing in `year`: BD TOPO's date of appearance is no later (an undated
    building is counted as standing)."""
    return [f for f in features if (known := appeared(f)) is None or known <= year]


def outside(features: list[dict]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Whether points are outside all these buildings, many at once (a spot less than INSIDE_M
    inside a footprint counts as outside: footprints are drawn to a metre or so)."""
    covered = shapely.union_all([part.buffer(-INSIDE_M) for f in features for part in outlines(f)])
    shapely.prepare(covered)
    return lambda xs, ys: ~shapely.contains_xy(covered, xs, ys)


def prisms(feature: dict, origin: tuple[float, float]) -> list[Prism]:
    p = feature["properties"]
    return [
        Prism(
            footprint=[
                (round(x - origin[0], 2), round(y - origin[1], 2))
                for x, y, *_ in part.exterior.coords[:-1]
            ],
            ground=p.get("altitude_minimale_sol"),
            height=p["hauteur"] or DEFAULT_HEIGHT_M,
            appeared=appeared(feature),
        )
        for part in outlines(feature)
    ]


def summary(features: list[dict]) -> dict[str, int]:
    return {
        "buildings": len(features),
        "no_height": sum(not f["properties"].get("hauteur") for f in features),
        "no_ground": sum(f["properties"].get("altitude_minimale_sol") is None for f in features),
    }
