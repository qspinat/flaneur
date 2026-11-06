"""The 360° panoramas of an area: where one stands in the 3D walk, and the ways on to the
next ones."""

import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import httpx
import numpy as np
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

from flaneur import db, panoramax
from flaneur.gazetteer import angle, bearing, to_degrees
from flaneur.panoramax import Picture

TILE_DEG = 0.0007  # about 50 × 75 m: under the search's limit of 1000 pictures in Paris
TILE_LIMIT = 1000
TILES_AT_ONCE = 6
LINK_M = (2, 15)
SECTOR_DEG = 45
DOUBTFUL_DEG = 30


PANORAMA_HEIGHT_M = 2.5  # a panorama's camera above the street: a car's roof, a backpack


def harvest(client: httpx.Client, area: BaseGeometry) -> tuple[list[Picture], int]:
    """The 360° pictures inside `area` (Lambert-93), searched tile by tile, and how many tiles
    came back full (some pictures may be missing there)."""
    lon_min, lat_min, lon_max, lat_max = to_degrees(area).bounds
    tiles = [
        (lon, lat, lon + TILE_DEG, lat + TILE_DEG)
        for lon, lat in product(
            np.arange(lon_min, lon_max, TILE_DEG), np.arange(lat_min, lat_max, TILE_DEG)
        )
    ]
    with ThreadPoolExecutor(TILES_AT_ONCE) as pool:
        answers = list(pool.map(lambda t: panoramax.search(client, t, TILE_LIMIT), tiles))
    found = {p.id: p for a in answers for p in a if p.panoramic and p.point.within(area)}
    return list(found.values()), sum(len(a) >= TILE_LIMIT for a in answers)


def travel_bearing(pictures: list[Picture]) -> dict[str, float]:
    """The direction each picture's camera was travelling: from the picture before it in its
    sequence to the one after (or to/from itself at either end)."""
    by_sequence = defaultdict(list)
    for p in pictures:
        if p.sequence is not None and p.rank is not None:
            by_sequence[p.sequence].append(p)
    travel = {}
    for run in by_sequence.values():
        run.sort(key=lambda p: p.rank)
        for i, p in enumerate(run):
            before, after = run[max(i - 1, 0)], run[min(i + 1, len(run) - 1)]
            if before is not after:
                travel[p.id] = bearing(before.point, after.point)
    return travel


def heading_doubtful(azimuth: float, travel: float, tolerance: float = DOUBTFUL_DEG) -> bool:
    """A camera mounted on a car, a bike or a backpack faces the way it travels, or backwards:
    any other azimuth is likely wrong."""
    turn = abs(angle(travel, azimuth))
    return min(turn, 180 - turn) > tolerance


def links(pictures: list[Picture]) -> dict[str, list[tuple[str, float, float]]]:
    """For each picture, the nearest other one in each 45° sector around it, between 2 and
    15 m away: (id, bearing, distance)."""
    points = [p.point for p in pictures]
    pairs = STRtree(points).query(points, predicate="dwithin", distance=LINK_M[1])
    near = defaultdict(list)
    for i, j in pairs.T:
        near[i].append(pictures[j])
    found = {}
    for i, here in enumerate(pictures):
        nearest: dict[int, tuple[str, float, float]] = {}
        for there in near[i]:
            distance = here.point.distance(there.point)
            if not LINK_M[0] <= distance <= LINK_M[1]:
                continue
            direction = bearing(here.point, there.point)
            sector = int(((direction + SECTOR_DEG / 2) % 360) // SECTOR_DEG)
            if sector not in nearest or distance < nearest[sector][2]:
                nearest[sector] = (there.id, direction, distance)
        found[here.id] = sorted(nearest.values(), key=lambda link: link[1])
    return found


def save(conn: sqlite3.Connection, pictures: list[Picture]) -> int:
    return db.replace(
        conn, "panoramas", ({c: getattr(p, c) for c in db.PANORAMA_COLUMNS} for p in pictures)
    )


def load(conn: sqlite3.Connection) -> list[Picture]:
    return [
        Picture(**dict(row), panoramic=True, thumb_url="")
        for row in conn.execute(f"SELECT {', '.join(db.PANORAMA_COLUMNS)} FROM panoramas")
    ]


def summary(pictures: list[Picture]) -> dict[str, float]:
    """How many panoramas, how many with a doubtful heading, the median link length."""
    travel = travel_bearing(pictures)
    lengths = sorted(d for ways in links(pictures).values() for _, _, d in ways)
    return {
        "panoramas": len(pictures),
        "doubtful": sum(
            heading_doubtful(p.azimuth, travel[p.id])
            for p in pictures
            if p.azimuth is not None and p.id in travel
        ),
        "median_link_m": lengths[len(lengths) // 2] if lengths else 0.0,
    }
