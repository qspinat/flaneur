"""The facades an old photo is projected onto in the 3D walk, where they stood: the block
edges of today (BD TOPO) and of the Vasserot atlas (1810–1836, ALPAGE, ODbL), whichever
stood nearer, for a street widened since was narrower."""

import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import shapefile
import shapely
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from flaneur.gazetteer import angle, bearing

_FROM_VASSEROT = Transformer.from_crs("EPSG:27591", "EPSG:2154", always_xy=True).transform
_TO_VASSEROT = Transformer.from_crs("EPSG:2154", "EPSG:27591", always_xy=True).transform
MERGE_M = 0.3  # buildings closer than this form one block
EDGE_MIN_M = 1.0
ALONG_DEG = 35  # a photo turned further than this from its street faces a facade
SIDE_SAMPLES_M = (5, 10, 15, 20, 25, 30)  # where the street's width is measured, ahead
FACADE_MIN_M = 3
PARALLEL_DEG = 20
STOOD_ON_M = 1.5
MIN_SIDE_M = 2.5  # room the camera had on either side
SIDE_REACH_M = 30
DEFAULT_SIDE_M = 6
DEPTH_REACH_M = 150
MIN_DEPTH_M = 15
DEFAULT_DEPTH_M = 80
FACADE_REACH_M = 60
HEIGHT_M = 22  # until the photo tells better
INSET_M = 0.8
FACADE_M = 25


@dataclass(frozen=True)
class Wall:
    line: tuple[tuple[float, float], tuple[float, float]]  # Lambert-93 metres
    height: float


def load_vasserot(path: Path, area: BaseGeometry) -> list[Polygon]:
    """The Vasserot buildings inside `area` (Lambert-93), in Lambert-93."""
    x0, y0, x1, y1 = shapely.transform(area, _TO_VASSEROT, interleaved=False).bounds
    found = []
    for record in shapefile.Reader(path).iterShapes(bbox=(x0, y0, x1, y1)):
        outline = shapely.transform(
            shape(record.__geo_interface__), _FROM_VASSEROT, interleaved=False
        )
        if outline.intersects(area):
            found.append(outline.buffer(0))
    return found


def edges(buildings: list[Polygon]) -> list[LineString]:
    """The outer edges of the blocks the buildings form (not the walls between houses)."""
    merged = unary_union([p.buffer(MERGE_M, join_style="mitre") for p in buildings])
    blocks = merged.buffer(-MERGE_M, join_style="mitre")
    found = []
    for block in getattr(blocks, "geoms", [blocks]):
        ring = orient(block).exterior.coords  # anticlockwise: the outside is on the right
        found += [LineString(pair) for pair in pairwise(ring)]
    return [e for e in found if e.length >= EDGE_MIN_M]


def facade_bearings(block_edges: list[LineString], here: Point) -> list[tuple[float, float]]:
    """The directions of the block edges within 25 m, both ways, with their length."""
    found = []
    for edge in block_edges:
        if edge.distance(here) <= FACADE_M:
            start, end = (Point(c) for c in edge.coords)
            b = bearing(start, end)
            found += [(b, edge.length), ((b + 180) % 360, edge.length)]
    return found


def _hit(
    tree: STRtree,
    edges: list[LineString],
    start: Point,
    bearing_deg: float,
    reach: float,
    along: float | None = None,
) -> float | None:
    """How far a ray from `start` towards `bearing_deg` runs before meeting an edge; with
    `along`, only an edge at least 3 m long running within 20° of that direction (a
    facade of the street, not a corner or a scrap of outline)."""
    r = math.radians(bearing_deg)
    ray = LineString([start, (start.x + reach * math.sin(r), start.y + reach * math.cos(r))])
    hits = [
        start.distance(ray.intersection(edges[i]))
        for i in tree.query(ray, "intersects")
        if along is None or _runs_along(edges[i], along)
    ]
    # an outline right where the photographer stood was gone when the photo was taken
    return min((h for h in hits if h >= STOOD_ON_M), default=None)


def _runs_along(edge: LineString, direction: float) -> bool:
    start, end = (Point(c) for c in edge.coords)
    turn = abs(angle(direction, bearing(start, end)))
    return edge.length >= FACADE_MIN_M and min(turn, 180 - turn) <= PARALLEL_DEG


def _side(tree, edges, here: Point, axis: float, side: float) -> float:
    """The distance to the street's facade on one side (−90 left, +90 right): the median of
    what is found at several points ahead."""
    found = [
        _hit(tree, edges, Point(_towards(here, axis, t)), axis + side, SIDE_REACH_M, along=axis)
        for t in SIDE_SAMPLES_M
    ]
    found = sorted(d for d in found if d is not None)
    return found[len(found) // 2] if found else DEFAULT_SIDE_M


def _towards(point: Point, bearing_deg: float, metres: float) -> tuple[float, float]:
    r = math.radians(bearing_deg)
    return (
        round(point.x + metres * math.sin(r), 2),
        round(point.y + metres * math.cos(r), 2),
    )


def facades(
    here: Point, heading: float, street: float | None, fov: float, edges: list[LineString]
) -> list[Wall]:
    """The facades a photo is projected onto, as they stood (the block edges of old and of
    today, whichever stood nearer). Looking along a street (or when the street is unknown):
    its left facade, the one closing it, its right facade, from where the photographer stood.
    Facing a facade: that facade."""
    tree = STRtree(edges)
    if street is not None and abs(angle(heading, street)) > ALONG_DEG:
        found = (_hit(tree, edges, here, heading, FACADE_REACH_M) or DEFAULT_DEPTH_M) - INSET_M
        half = found * math.tan(math.radians(fov) / 2) * 1.5
        centre = Point(_towards(here, heading, found))
        return [Wall((_towards(centre, street, -half), _towards(centre, street, half)), HEIGHT_M)]
    axis = heading if street is None else street
    # each a little in front of the facade found, so that the facade never hides it
    to_left = max(_side(tree, edges, here, axis, -90) - INSET_M, MIN_SIDE_M)
    to_right = max(_side(tree, edges, here, axis, 90) - INSET_M, MIN_SIDE_M)
    depth = max(_hit(tree, edges, here, axis, DEPTH_REACH_M) or DEFAULT_DEPTH_M, MIN_DEPTH_M)
    depth -= INSET_M
    near_left = Point(_towards(here, axis - 90, to_left))
    near_right = Point(_towards(here, axis + 90, to_right))
    far_left, far_right = _towards(near_left, axis, depth), _towards(near_right, axis, depth)
    return [
        Wall((near_left.coords[0], far_left), HEIGHT_M),
        Wall((far_left, far_right), HEIGHT_M),
        Wall((far_right, near_right.coords[0]), HEIGHT_M),
    ]
