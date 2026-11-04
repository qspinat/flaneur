"""The 3D walk's scene (scene.json): today's panoramas and buildings, the ground, and the old
photos with their cameras and old walls. Positions are metres east and north of the scene's
origin; heights are metres above the origin's ground."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon, shape
from shapely.geometry.base import BaseGeometry

from flaneur import buildings, panoramas
from flaneur.blocks import MOST_PAIRS
from flaneur.export import WalkPhoto, image_name, panoramax_credit
from flaneur.gazetteer import to_metres
from flaneur.panoramax import Picture

GROUND_STEP_M = 10
NEAREST_BASES = 8
PANORAMA_HEIGHT_M = 2.5
CORE_INSET_M = 3  # a camera on a car's roof or a backpack


@dataclass(frozen=True)
class Ground:
    """Heights of the street on a square grid (metres NGF), from the ground altitudes of the
    nearest buildings: BD TOPO gives one for each building, and nothing between them."""

    x0: float
    y0: float
    step: float
    heights: np.ndarray  # rows south to north, columns west to east

    @classmethod
    def from_buildings(
        cls, features: list[dict], area: BaseGeometry, step: float = GROUND_STEP_M
    ) -> "Ground":
        bases = np.array(
            [
                (
                    *shape(f["geometry"]).centroid.coords[:1][0][:2],
                    f["properties"]["altitude_minimale_sol"],
                )
                for f in features
                if f["properties"].get("altitude_minimale_sol") is not None
            ]
        )
        if not len(bases):
            raise ValueError("no building with a ground altitude: no street heights")
        x0, y0, x1, y1 = area.bounds
        xs, ys = np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step)
        nodes = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

        def heights_of(part: np.ndarray) -> np.ndarray:
            distance = np.hypot(*(part[:, None, :] - bases[None, :, :2]).transpose(2, 0, 1))
            k = min(NEAREST_BASES, len(bases))
            nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
            weight = 1 / (np.take_along_axis(distance, nearest, axis=1) + 1) ** 2
            return (weight * bases[nearest, 2]).sum(1) / weight.sum(1)

        # nodes × buildings, a batch at a time: a district at once would take gigabytes
        batch = max(1, MOST_PAIRS // len(bases))
        heights = np.concatenate(
            [heights_of(nodes[k : k + batch]) for k in range(0, len(nodes), batch)]
        )
        return cls(x0, y0, step, heights.reshape(len(ys), len(xs)))

    def at(self, point: Point) -> float:
        """The ground's height at a point (bilinear between grid nodes)."""
        return float(self.at_many(np.array([point.x]), np.array([point.y]))[0])

    def at_many(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """The ground's height at many points (bilinear between grid nodes)."""
        rows, columns = self.heights.shape
        fx = np.clip((xs - self.x0) / self.step, 0, columns - 1.001)
        fy = np.clip((ys - self.y0) / self.step, 0, rows - 1.001)
        i, j, dx, dy = fy.astype(int), fx.astype(int), fx % 1, fy % 1
        h = self.heights
        return (
            h[i, j] * (1 - dx) * (1 - dy)
            + h[i, j + 1] * dx * (1 - dy)
            + h[i + 1, j] * (1 - dx) * dy
            + h[i + 1, j + 1] * dx * dy
        )

    def for_site(self, origin: tuple[float, float, float]) -> dict:
        return {
            "x0": round(self.x0 - origin[0], 2),
            "y0": round(self.y0 - origin[1], 2),
            "step": self.step,
            "columns": self.heights.shape[1],
            "heights": [round(float(h - origin[2]), 2) for h in self.heights.ravel()],
        }


def buildings_of(features: list[dict], origin: tuple[float, float, float]) -> list[dict]:
    """Today's buildings as prisms standing on their own ground."""
    return [
        {
            "footprint": prism.footprint,
            "base": round((prism.ground if prism.ground is not None else origin[2]) - origin[2], 2),
            "height": prism.height,
        }
        for f in features
        for prism in buildings.prisms(f, origin[:2])
    ]


def cores_of(features: list[dict], origin: tuple[float, float, float]) -> list[dict]:
    """Today's buildings shrunk by 3 m on every side: what hides an old photo behind it. A
    facade of old standing a few metres behind today's (the estimate of where it stood is
    rarely closer than that) still shows."""
    found = []
    for building in buildings_of(features, origin):
        core = Polygon(building["footprint"]).buffer(-CORE_INSET_M, join_style="mitre")
        for part in getattr(core, "geoms", [core]):
            if part.is_empty:
                continue
            found.append(
                {
                    "footprint": [(round(x, 2), round(y, 2)) for x, y in part.exterior.coords[:-1]],
                    "base": building["base"],
                    "height": building["height"],
                }
            )
    return found


def _local(x: float, y: float, origin) -> tuple[float, float]:
    return round(x - origin[0], 2), round(y - origin[1], 2)


def panorama_z(ground: Ground, point: Point, origin_z: float) -> float:
    """The height of a panorama's camera, above the origin's ground."""
    return ground.at(point) - origin_z + PANORAMA_HEIGHT_M


@dataclass(frozen=True)
class CameraHeights:
    """A panorama camera's height at any (x, y), above `origin_z` (sent to worker processes)."""

    ground: Ground
    origin_z: float

    def __call__(self, x: float, y: float) -> float:
        return panorama_z(self.ground, Point(x, y), self.origin_z)

    def street(self, x: float, y: float) -> float:
        """The street's height there."""
        return float(self.streets(np.array([x]), np.array([y]))[0])

    def streets(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """The street's height at many points."""
        return self.ground.at_many(xs, ys) - self.origin_z


def panoramas_of(pictures: list[Picture], ground: Ground, origin) -> list[dict]:
    ways = panoramas.links(pictures)
    found = []
    for p in pictures:
        point = p.point
        x, y = _local(point.x, point.y, origin)
        found.append(
            {
                "id": p.id,
                "x": x,
                "y": y,
                "z": round(panorama_z(ground, point, origin[2]), 2),
                "lon": round(p.lon, 7),  # exact, for the map
                "lat": round(p.lat, 7),
                "azimuth": round(p.azimuth or 0, 2),
                "taken": p.taken,
                "credit": panoramax_credit(p),
                "url": f"https://api.panoramax.xyz/#focus=pic&pic={p.id}",
                "image": p.sd_url,
                "links": [[i, round(b, 1), round(d, 1)] for i, b, d in ways[p.id]],
            }
        )
    return found


def walk_of(walk: list[WalkPhoto], origin) -> list[dict]:
    """The photo walk's photos, where the 3D walk needs them: in the scene (the way back to
    the photos, the street's plate) and on the map (the plan)."""
    found = []
    for me in walk:
        here = to_metres(Point(me.lon, me.lat))
        x, y = _local(here.x, here.y, origin)
        found.append(
            {
                "id": me.id,
                "x": x,
                "y": y,
                "lon": me.lon,
                "lat": me.lat,
                "heading": me.heading,
                "street": list(me.street) if me.street else None,
                "arrondissement": me.arrondissement,
            }
        )
    return found


def photos_of(
    proposals: list[dict],
    walk: list[WalkPhoto],
    data_dir: Path,
    ground: Ground,
    origin,
) -> list[dict]:
    by_id = {me.id: me for me in walk}
    found = []
    for proposal in proposals:  # each confirmed by the user, inside a building of today or not
        me = by_id.get(proposal["photo_id"])
        if me is None:
            continue
        with Image.open(data_dir / image_name(me.id)) as image:
            width, height = image.size
        pose = proposal["pose"]
        here = Point(pose["x"], pose["y"])
        x, y = _local(pose["x"], pose["y"], origin)
        found.append(
            {
                "id": me.id,
                "title": me.title,
                "years": me.years,
                "credit": f"{me.photographer}. {me.credit}",
                "image": image_name(me.id),
                "aspect": round(height / width, 4),
                "pose": {
                    **pose,
                    "x": x,
                    "y": y,
                    "z": round(ground.at(here) - origin[2] + pose["z"], 2),
                },
                "walls": [
                    {
                        "line": [_local(*end, origin) for end in wall["line"]],
                        "height": wall["height"],
                    }
                    for wall in proposal["walls"]
                ],
                "base": round(ground.at(here) - origin[2], 2),  # the street where the facades stand
            }
        )
    return found


def build(
    pictures: list[Picture],
    features: list[dict],
    proposals: list[dict],
    walk: list[WalkPhoto],
    area: BaseGeometry,
    data_dir: Path,
) -> dict:
    ground = Ground.from_buildings(features, area)
    centre = area.centroid
    origin = (
        round(centre.x, -1),
        round(centre.y, -1),
        round(float(np.median(ground.heights)), 1),
    )
    return {
        "origin": list(origin),
        "walk": walk_of(walk, origin),
        "ground": ground.for_site(origin),
        "buildings": buildings_of(features, origin),
        "cores": cores_of(features, origin),
        "panoramas": panoramas_of(pictures, ground, origin),
        "photos": photos_of(
            proposals,
            walk,
            data_dir,
            ground,
            origin,
        ),
    }


def build3d(
    pictures: list[Picture],
    shown: dict[str, dict],
    copies: dict[str, list[dict]],
    walk: list[WalkPhoto],
) -> tuple[dict, set[str]]:
    """The 3D walk on today's panoramas: every panorama (levelled by the viewer from its
    azimuth, roll and pitch; the next ones along the street, by their index; its credit, by
    its index in `credits`), and each photo placed in one (`shown`: the answers that hold, on
    the image the photo has now), where it shows: its home, then its `copies` carried to the
    neighbours. And the panoramas that hold a photo, to copy onto the site."""
    by_id = {me.id: me for me in walk}
    known = {p.id for p in pictures}  # a photo or a copy on a panorama gone is left out
    photos, holding = [], set()
    for photo_id, row in shown.items():
        me = by_id.get(photo_id)
        if me is None or row["registration"]["panorama"] not in known:
            continue
        home = {"panorama": row["registration"]["panorama"], "registration": row["registration"]}
        where = [{**home, "strength": 1.0}] + [
            {"panorama": c["registration"]["panorama"], **c}
            for c in copies.get(photo_id, [])
            if c["registration"]["panorama"] in known
        ]
        holding |= {w["panorama"] for w in where}
        credit = f"{me.photographer}. {me.credit}" + ("; image retournée" if row["flipped"] else "")
        photos.append(
            {
                "id": me.id,
                "title": me.title,
                "years": me.years,
                "credit": credit,
                "image": image_name(me.id),
                "flipped": row["flipped"],
                "shown": where,
            }
        )
    ways = panoramas.links(pictures)
    index = {p.id: n for n, p in enumerate(pictures)}
    credits: dict[str, int] = {}
    spheres = [
        {
            "id": p.id,
            "lon": round(p.lon, 7),
            "lat": round(p.lat, 7),
            "azimuth": round(p.azimuth or 0, 2),
            "roll": round(p.roll, 2),
            "pitch": round(p.pitch, 2),
            "taken": p.taken,
            "credit": credits.setdefault(panoramax_credit(p), len(credits)),
            "image": f"panoramas/{p.id}.jpg" if p.id in holding else p.sd_url,
            "links": [[index[i], round(b, 1), round(d, 1)] for i, b, d in ways[p.id]],
        }
        for p in pictures
    ]
    return {"credits": list(credits), "panoramas": spheres, "photos": photos}, holding
