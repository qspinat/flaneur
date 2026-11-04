"""The 3D walk's scene (scene.json, see build3d): today's panoramas, and the old photos placed
in them. And the ground and buildings of today that matching stands panoramas on (Ground,
CameraHeights, buildings_of): metres east and north of an origin, heights above its ground."""

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry

from flaneur import buildings, panoramas
from flaneur.blocks import MOST_PAIRS
from flaneur.export import WalkPhoto, image_name, panoramax_credit
from flaneur.panoramax import Picture

GROUND_STEP_M = 10
NEAREST_BASES = 8
PANORAMA_HEIGHT_M = 2.5  # a camera on a car's roof or a backpack


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
