"""The pilot area: where the walk is built first (the densest area measured, around
Saint-Séverin, 5th arrondissement), and the material to place its photos precisely.

For each photo, `prepare` writes a folder with (a contact sheet missing from sheets/ is
rebuilt from a new Panoramax search, so its numbers can change: a placement keeps the
Panoramax id of the picture it matched, see placements.py):
- photo.jpg: the archive photo, at most MAX_SIDE pixels;
- info.json: caption, date, the street found from the caption, and the contact sheet to use;
and for each street, once, a contact sheet of today's Panoramax pictures along it
(sheets/<street>.jpg, numbered tiles) with sheets/<street>.json (position and direction of
each numbered picture).
"""

import io
import json
import re
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PIL import Image, ImageDraw
from shapely.geometry import Point, box

from flaneur import crop, db, dedup, panoramax
from flaneur.gazetteer import from_geojson, to_metres
from flaneur.sources import write_json

# 500 m square around the densest point measured (48.8518 N, 2.3459 E).
CENTRE = to_metres(Point(2.3459, 48.8518))
AREA = box(CENTRE.x - 250, CENTRE.y - 250, CENTRE.x + 250, CENTRE.y + 250)
# A street belongs to the pilot when at least this share of its footprint is inside.
INSIDE_SHARE = 0.5
# Photos with a camera position from the caption: around it, not along the whole street.
AROUND_POINT_M = 40
MAX_SIDE = 1024
SHEET_COLUMNS = 5
PLACED = ("street", "corner", "address", "viewpoint_street")


def select(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Photos of the pilot area: openly licensed photos, and every Albert-Kahn photo (their
    camera position is known: they measure how well photos are placed; those whose image
    licence is unverified are only used here, never published)."""
    rows = conn.execute(
        f"""SELECT p.id, p.source, p.title, p.caption, p.year_start, p.year_end, p.image_url,
                   p.license, l.status, l.label, l.point, s.id AS street_id,
                   s.label AS street_label, s.geometry
            FROM locations l JOIN photos p ON p.id = l.photo_id
            LEFT JOIN streets s ON s.id = l.street_id
            WHERE l.status IN ({", ".join("?" * len(PLACED))})
              AND (l.point IS NOT NULL OR s.id IS NOT NULL)
              AND (p.license IN {db.OPEN_LICENSES_SQL} OR p.source = 'albert_kahn')""",
        PLACED,
    ).fetchall()
    street_inside: dict[str, bool] = {}  # many photos share a street: decided once per street
    selected = []
    for row in rows:
        if row["point"]:
            inside = from_geojson(row["point"]).within(AREA)
        else:
            if row["street_id"] not in street_inside:
                street = from_geojson(row["geometry"])
                street_inside[row["street_id"]] = (
                    street.intersection(AREA).area >= INSIDE_SHARE * street.area
                )
            inside = street_inside[row["street_id"]]
        if inside:
            selected.append(row)
    return selected


def safe_name(name: str) -> str:
    """A name usable as a file or folder name."""
    return re.sub(r"[^\w.-]+", "_", name)


# Regular pictures taken on the street, and 360° pictures taken around it: a view from a
# bridge, a quay or a square is often what the old photographer saw.
SHEET_FLAT = 8
SHEET_PANORAMAS = 3
PANORAMAS_WITHIN_M = 60
TILE_SIZE = (500, 281)
# One tile of a contact sheet: the image, and what it shows (place, direction, date).
Tile = tuple[Image.Image, dict]


# Images are downloaded this many at a time.
DOWNLOADS_AT_ONCE = 8


def get_image(client: httpx.Client, url: str) -> Image.Image:
    response = client.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def save_image(image: Image.Image, path: Path, quality: int) -> None:
    """Write atomically: an interrupted write never leaves a truncated image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp" + path.suffix)
    image.save(tmp, quality=quality)
    tmp.replace(path)


def cached_image(client: httpx.Client, url: str, path: Path) -> Image.Image:
    """The image at `url`, downloaded once into `path` (written atomically)."""
    if not path.exists():
        response = client.get(url)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        # a temporary file of its own: two threads may fetch the same picture at once
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as f:
            f.write(response.content)
        Path(f.name).replace(path)
    return Image.open(path)


def _save_photo(client: httpx.Client, url: str, path: Path) -> None:
    """The archive photo, at most MAX_SIDE pixels."""
    if path.exists():
        return
    image = get_image(client, url)
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    save_image(image, path, quality=85)


def _panorama_views(panorama: Image.Image, azimuth: float) -> list[tuple[Image.Image, float]]:
    """Four 90° views cut from a 360° picture, with the compass direction each one faces."""
    headings = [(azimuth + 90 * turn) % 360 for turn in range(4)]
    return [(panoramax.view(panorama, azimuth, h), h) for h in headings]


def _tiles(client: httpx.Client, pictures: list[panoramax.Picture]) -> list[Tile]:
    """Images for the contact sheet, with what each one shows: a regular picture is one tile,
    a 360° picture four (one per direction)."""

    def tiles_of(p: panoramax.Picture) -> list[Tile]:
        about = {"panoramax_id": p.id, "lon": p.lon, "lat": p.lat, "taken": p.taken}
        try:
            return tiles_from(p, about)
        except httpx.HTTPError:  # a picture removed from Panoramax: the sheet goes without it
            return []

    def tiles_from(p: panoramax.Picture, about: dict) -> list[Tile]:
        if p.panoramic and p.azimuth is not None:
            views = _panorama_views(get_image(client, p.sd_url), p.azimuth)
            return [(view, about | {"azimuth": h, "panoramic": True}) for view, h in views]
        about |= {"azimuth": p.azimuth, "panoramic": False}
        return [(get_image(client, p.thumb_url), about)]

    with ThreadPoolExecutor(DOWNLOADS_AT_ONCE) as pool:
        return [tile for tiles in pool.map(tiles_of, pictures) for tile in tiles]


def tile_box(n: int) -> tuple[int, int, int, int]:
    """Where picture number `n` (from 1) sits on a contact sheet: left, top, right, bottom."""
    width, height = TILE_SIZE
    x, y = ((n - 1) % SHEET_COLUMNS) * width, ((n - 1) // SHEET_COLUMNS) * height
    return x, y, x + width, y + height


def sheet_of(photo_folder: Path) -> str | None:
    """The contact sheet of a prepared photo, from its info.json."""
    return json.loads((photo_folder / "info.json").read_text()).get("sheet")


def _contact_sheet(tiles: list[Tile], path: Path) -> list[dict]:
    """Numbered tiles, SHEET_COLUMNS per row; returns what each number shows."""
    width, height = TILE_SIZE
    rows = (len(tiles) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    sheet = Image.new("RGB", (SHEET_COLUMNS * width, rows * height), "white")
    draw = ImageDraw.Draw(sheet)
    for n, (image, _) in enumerate(tiles, start=1):
        x, y, _, _ = tile_box(n)
        sheet.paste(image.resize((width, height)), (x, y))
        draw.rectangle((x, y, x + 44, y + 30), fill="black")
        draw.text((x + 8, y + 6), f"#{n}", fill="yellow", font_size=20)
    save_image(sheet, path, quality=80)
    return [{"n": n} | about for n, (_, about) in enumerate(tiles, start=1)]


def _sheet_for(client: httpx.Client, row: sqlite3.Row, sheets_dir: Path) -> str | None:
    """Name of the contact sheet for this photo (built once per street or per point), or None
    when Panoramax has no picture there (remembered, so that it is not searched again)."""
    if row["point"]:
        place = from_geojson(row["point"])
        name = safe_name(f"point_{place.x:.0f}_{place.y:.0f}")
        along = on_street = place.buffer(AROUND_POINT_M)
    else:
        name = safe_name(row["street_id"])
        along = from_geojson(row["geometry"])
        on_street = along.buffer(panoramax.ON_STREET_M)
    shown = sheets_dir / f"{name}.json"
    if shown.exists():
        return name if json.loads(shown.read_text()) else None
    nearby = panoramax.near(client, along.buffer(PANORAMAS_WITHIN_M))
    flat = [p for p in nearby if not p.panoramic and p.point.within(on_street)]
    panoramas = [p for p in nearby if p.panoramic]
    pictures = panoramax.spread(flat, along, SHEET_FLAT) + panoramax.spread(
        panoramas, along, SHEET_PANORAMAS
    )
    tiles = _tiles(client, pictures)
    shown_tiles = _contact_sheet(tiles, sheets_dir / f"{name}.jpg") if tiles else []
    write_json(shown, shown_tiles)
    return name if shown_tiles else None


def prepare(conn: sqlite3.Connection, client: httpx.Client, out_dir: Path) -> dict[str, int]:
    """Write the material of every pilot photo; images already there are not fetched again,
    info.json is rewritten so that it always matches the current state."""
    counts = {"photos": 0, "without_today_pictures": 0}
    selected = select(conn)
    # A negative printed several times is placed once, on the first of its prints selected
    # here (the one that stands for it, when selected), reading every print's caption.
    negatives = dedup.negatives(conn)
    chosen = {row["id"] for row in selected}
    texts = {
        r["id"]: {
            "photo_id": r["id"],
            "title": r["title"],
            "caption": r["caption"],
            "years": [r["year_start"], r["year_end"]],
        }
        for r in conn.execute("SELECT p.* FROM copies c JOIN photos p ON p.id = c.photo_id")
    }
    for row in selected:
        folder = out_dir / "photos" / safe_name(row["id"])
        sheet = _sheet_for(client, row, out_dir / "sheets")
        counts["photos"] += 1
        counts["without_today_pictures"] += sheet is None
        _save_photo(client, row["image_url"], folder / "photo.jpg")
        info = {
            **_cropped(folder),
            "photo_id": row["id"],
            "license": row["license"],
            "title": row["title"],
            "caption": row["caption"],
            "years": [row["year_start"], row["year_end"]],
            "found_from_caption": {"status": row["status"], "label": row["label"]},
            "street": row["street_label"],
            "sheet": sheet,
        }
        if prints := negatives.get(row["id"]):
            info.update(
                years=list(dedup.years([tuple(texts[i]["years"]) for i in prints])),
                other_prints=[texts[i] for i in prints if i != row["id"]],
                placed_on=next(i for i in prints if i in chosen),
            )
        write_json(folder / "info.json", info)
    return counts


def _cropped(folder: Path) -> dict:
    """The print on its mount in a pilot photo's scan (see flaneur.crop), and why that crop
    may need a hand, as info.json keeps them."""
    with Image.open(folder / "photo.jpg") as scan:
        found, doubts = crop.examine(scan)
    return {
        "suggested_crop": [round(v, 4) for v in found] if found else None,
        "crop_doubts": doubts,
    }


def crop_again(folder: Path) -> tuple[bool, list[str]]:
    """Find again a pilot photo's crop and its doubts (after flaneur.crop changed): whether its
    info.json changed, and the doubts."""
    path = folder / "info.json"
    info = json.loads(path.read_text())
    cropped = _cropped(folder)
    if all(info.get(key) == value for key, value in cropped.items()):
        return False, cropped["crop_doubts"]
    write_json(path, {**info, **cropped})
    return True, cropped["crop_doubts"]
