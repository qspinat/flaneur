"""Panoramax: today's street-level pictures (open licences: Etalab 2.0 for the IGN instance,
CC BY-SA for OpenStreetMap France), used to see what an old photo's street looks like now."""

from dataclasses import dataclass
from functools import cached_property

import httpx
from PIL import Image
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from flaneur.gazetteer import long_axis, to_degrees, to_metres

SEARCH_URL = "https://api.panoramax.xyz/api/search"
# A picture counts as "on the street" when taken within this distance of its footprint.
ON_STREET_M = 8


@dataclass(frozen=True)
class Picture:
    id: str
    lon: float
    lat: float
    azimuth: float | None  # compass direction of the picture's centre
    panoramic: bool  # 360°: every direction is in the picture
    taken: str
    thumb_url: str
    sd_url: str  # 2048 px; for a 360° picture, the whole panorama (equirectangular)
    license: str | None = None  # "etalab-2.0", "CC-BY-SA-4.0"
    producer: str | None = None  # who took it, for the credit
    sequence: str | None = None  # the series it was taken in, one after the other
    rank: int | None = None  # its place in that series
    roll: float = 0.0  # degrees its camera leaned to its right (a 360° picture: see level)
    pitch: float = 0.0  # degrees it raised its nose

    @cached_property
    def point(self) -> Point:
        """Where it was taken, in Lambert-93 metres."""
        return to_metres(Point(self.lon, self.lat))


def _parse(feature: dict) -> Picture:
    p = feature["properties"]
    lon, lat = feature["geometry"]["coordinates"][:2]
    return Picture(
        id=feature["id"],
        lon=lon,
        lat=lat,
        azimuth=p.get("view:azimuth"),
        panoramic=(p.get("pers:interior_orientation") or {}).get("field_of_view") == 360,
        taken=(p.get("datetime") or "")[:10],
        thumb_url=feature["assets"]["thumb"]["href"],
        sd_url=feature["assets"]["sd"]["href"],
        license=p.get("license"),
        producer=next(
            (q["name"] for q in reversed(feature.get("providers", [])) if "producer" in q["roles"]),
            None,
        ),
        sequence=feature.get("collection"),
        rank=p.get("geovisio:rank_in_collection"),
    )


def search(client: httpx.Client, bounds: tuple[float, ...], limit: int) -> list[Picture]:
    """Pictures inside a box in degrees (lon_min, lat_min, lon_max, lat_max), at most
    `limit`: the search gives no next page."""
    response = client.get(SEARCH_URL, params={"bbox": ",".join(map(str, bounds)), "limit": limit})
    response.raise_for_status()
    return [_parse(f) for f in response.json()["features"]]


def near(client: httpx.Client, area: BaseGeometry, limit: int = 2000) -> list[Picture]:
    """Pictures taken inside `area` (Lambert-93), at most `limit`."""
    pictures = search(client, to_degrees(area).bounds, limit)
    return [p for p in pictures if p.point.within(area)]


def by_id(client: httpx.Client, picture_id: str) -> Picture:
    """One picture, by its id (IndexError when Panoramax no longer has it)."""
    response = client.get(SEARCH_URL, params={"ids": picture_id})
    response.raise_for_status()
    return _parse(response.json()["features"][0])


def spread(pictures: list[Picture], along: BaseGeometry, count: int) -> list[Picture]:
    """At most `count` pictures spread evenly along the long axis of `along`, so that a contact
    sheet covers the whole street: for evenly spaced positions from one end of the pictures
    to the other, the picture nearest to each (a cluster of pictures in one stretch takes
    one slot, not most of them)."""
    if len(pictures) <= count:
        return pictures
    (x0, y0), (x1, y1) = long_axis(along)

    def position(p: Picture) -> float:  # projection onto the axis
        return (p.point.x - x0) * (x1 - x0) + (p.point.y - y0) * (y1 - y0)

    remaining = sorted(pictures, key=position)
    first, last = position(remaining[0]), position(remaining[-1])
    chosen = []
    for i in range(count):
        target = first + (last - first) * (i / (count - 1) if count > 1 else 0.5)
        nearest = min(remaining, key=lambda p: abs(position(p) - target))
        remaining.remove(nearest)
        chosen.append(nearest)
    return chosen


def view(panorama: Image.Image, azimuth: float, heading: float) -> Image.Image:
    """The 90° view facing `heading`, cut from a 360° picture. The panorama is
    equirectangular: its full width is 360°, its centre faces `azimuth`."""
    width, height = panorama.size
    top, bottom = int(height * 0.28), int(height * 0.62)  # around the horizon, upper side
    # Side by side twice, so that a view across the panorama's edge is one crop.
    doubled = Image.new("RGB", (width * 2, height))
    doubled.paste(panorama, (0, 0))
    doubled.paste(panorama, (width, 0))
    centre = width * (0.5 + ((heading - azimuth) % 360) / 360)
    left = int(centre - width / 8) % width
    return doubled.crop((left, top, left + width // 4, bottom))
