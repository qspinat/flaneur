"""The review tool: a local page where the user confirms or fixes, one key per item, where
each photo of the pilot was taken and which way it faced, and where each place is entered.

Decisions are appended to data/overrides/reviews.jsonl (versioned with the code), so that
no hand work is lost when the pipeline runs again. Only for local use: it serves the pilot
folder and listens on localhost.
"""

import io
import json
import sqlite3
from dataclasses import asdict, dataclass
from http.server import HTTPServer
from pathlib import Path

from PIL import Image

from flaneur import db, dedup, localpage, placements, places
from flaneur.gazetteer import Gazetteer
from flaneur.pilot import safe_name, tile_box
from flaneur.sources import append_row, rows, seconds_taken

PAGE = Path(__file__).with_name("review_page.html")
DECISIONS = ("confirm", "fix", "not_a_street_view", "skip")
# An entrance is a door: it can be confirmed, moved or skipped, not judged a street view.
ENTRANCE_DECISIONS = ("confirm", "fix", "skip")


@dataclass(frozen=True)
class Item:
    id: str  # "photo:<photo_id>", or "entrance:<photo_id of the place's first photo>"
    kind: str  # "photo" or "entrance"
    title: str
    caption: str | None
    years: list[int | None] | None
    photo_url: str | None  # the archive photo
    full_url: str | None  # the same photo at the source's full size, for zooming in
    today_url: str | None  # the matched picture of today, cut from its contact sheet
    lon: float
    lat: float
    heading: float | None
    confidence: str | None
    reason: str | None
    # for an entrance, the place's photos; for a photo printed several times, all its prints
    photos: list[str] | None = None
    copies: list[dict] | None = None  # the other prints' titles, captions and years
    suggested_crop: list[float] | None = None  # the print on its mount, found automatically
    crop_doubts: list[str] | None = None  # why that crop may need a hand (see crop.examine)


def latest(reviews_path: Path, key: str = "id") -> dict[str, dict]:
    """The last decision taken on each item (an item reviewed again keeps its last word)."""
    return {row[key]: row for row in rows(reviews_path)}


def decisions_by_photo(reviews_path: Path, negatives: dict[str, list[str]]) -> dict[str, dict]:
    """The decision that holds for each photo, by id: the last one taken on any print of its
    negative (a skipped photo has none)."""
    found: dict[str, dict] = {}
    for row in latest(reviews_path).values():
        if not row["id"].startswith("photo:") or row["decision"] == "skip":
            continue
        decided = row["id"].removeprefix("photo:")
        for print_ in [decided, *(row.get("photos") or [])]:
            for i in negatives.get(print_, [print_]):
                if i not in found or row["at"] > found[i]["at"]:
                    found[i] = row
    return found


# An entrance reviewed earlier is the same place when it shares at least this share of its
# photos (a place's grouping can change a little as rules or photos change).
SAME_ENTRANCE = 0.5


def _today_tile(
    pilot_dir: Path, sheet: str | None, panoramax_id: str | None
) -> tuple[str, int] | None:
    """The matched picture's contact sheet and number, found by its Panoramax id (numbers
    change when a sheet is rebuilt): on the photo's own sheet, else on any other (a placement
    inherited from another print of the negative was matched on that print's sheet)."""
    if not panoramax_id:
        return None
    own = pilot_dir / "sheets" / f"{safe_name(sheet)}.json" if sheet else None
    others = sorted((pilot_dir / "sheets").glob("*.json"))
    for path in ([own] if own in others else []) + others:
        for tile in json.loads(path.read_text()):
            if tile["panoramax_id"] == panoramax_id:
                return path.stem, tile["n"]
    return None


def queue(
    conn: sqlite3.Connection, pilot_dir: Path, gazetteer: Gazetteer, reviews_path: Path
) -> list[Item]:
    """What is left to review: placed street views of photos that may be published (those
    whose crop is in doubt first, then those with no matched picture of today: the least
    certain), then place entrances. A negative printed several times is shown once: a
    decision holds for all its prints."""
    decisions = [row for row in latest(reviews_path).values() if row["decision"] != "skip"]
    reviewed_places = [
        set(row.get("photos") or []) for row in decisions if row["id"].startswith("entrance:")
    ]
    negatives = dedup.negatives(conn)
    done = decisions_by_photo(reviews_path, negatives)
    # a negative is shown once, with the other prints' captions
    items = []
    for r in placements.publishable(conn, pilot_dir, negatives):
        folder = pilot_dir / "photos" / safe_name(r["photo_id"])
        if r["photo_id"] in done:
            continue
        info = json.loads((folder / "info.json").read_text())
        others = info.get("other_prints") or []
        today = _today_tile(pilot_dir, info["sheet"], r["match_panoramax_id"])
        items.append(
            Item(
                id=f"photo:{r['photo_id']}",
                kind="photo",
                title=r["title"],
                caption=r["caption"],
                years=info["years"],
                photo_url=f"/pilot/photos/{folder.name}/photo.jpg",
                full_url=r["image_url"],
                today_url=f"/tile/{today[0]}/{today[1]}.jpg" if today else None,
                lon=r["lon"],
                lat=r["lat"],
                heading=r["heading"],
                confidence=r["confidence"],
                reason=r["reason"],
                suggested_crop=info.get("suggested_crop"),
                crop_doubts=info.get("crop_doubts") or None,
                photos=[r["photo_id"], *(o["photo_id"] for o in others)] if others else None,
                copies=others or None,
            )
        )
    items.sort(key=lambda item: not item.crop_doubts)  # a crop to draw first, else as listed
    publishable = [
        r["photo_id"]
        for r in conn.execute(
            "SELECT pl.photo_id FROM placements pl JOIN photos p ON p.id = pl.photo_id "
            f"WHERE p.license IN {db.OPEN_LICENSES_SQL}"
        )
    ]
    for name, photos in places.group(conn, publishable).items():
        ids = {p.photo_id for p in photos}
        if name is None or any(
            len(ids & seen) >= SAME_ENTRANCE * len(ids) for seen in reviewed_places
        ):
            continue
        if door := places.entrance_of(conn, photos, gazetteer):
            # an interior shows best what is behind the door
            behind = next((p for p in photos if p.kind == "interior"), photos[0])
            items.append(
                Item(
                    id=f"entrance:{min(ids)}",
                    kind="entrance",
                    title=f"Entrée : {name}",
                    caption=f"{len(photos)} photos à l'intérieur ; rue : {door.street}",
                    years=None,
                    photo_url=f"/pilot/photos/{safe_name(behind.photo_id)}/photo.jpg",
                    full_url=None,
                    today_url=None,
                    lon=door.lon,
                    lat=door.lat,
                    heading=door.heading,
                    confidence=None,
                    reason="Point de la rue le plus proche du lieu, face au lieu.",
                    photos=sorted(ids),
                )
            )
    return items


def tile(pilot_dir: Path, sheet: str, n: int) -> bytes:
    """Picture number `n` of a contact sheet, as JPEG. FileNotFoundError when the sheet or
    the number does not exist."""
    shown = json.loads((pilot_dir / "sheets" / f"{safe_name(sheet)}.json").read_text())
    if not 1 <= n <= len(shown):
        raise FileNotFoundError(f"no picture {n} on sheet {sheet}")
    image = Image.open(pilot_dir / "sheets" / f"{safe_name(sheet)}.jpg")
    out = io.BytesIO()
    image.crop(tile_box(n)).save(out, "JPEG", quality=85)
    return out.getvalue()


# Where a camera may stand: a box around Paris, in degrees.
PARIS = {"lon": (2.2, 2.5), "lat": (48.8, 48.95)}


def record(reviews_path: Path, decision: dict) -> dict:
    """Append one decision, checked; returns what was written."""
    if not isinstance(decision, dict) or "id" not in decision:
        raise ValueError("a decision is an object with an id")
    if decision.get("decision") in ("confirm", "fix"):
        for axis, (low, high) in PARIS.items():
            value = decision.get(axis)
            if not isinstance(value, int | float) or not low <= value <= high:
                raise ValueError(f"{axis} must be a number inside Paris")
        heading = decision.get("heading")
        if heading is not None and (not isinstance(heading, int | float) or not 0 <= heading < 360):
            raise ValueError("heading must be between 0 and 360, or null")
    crop = decision.get("crop")
    if crop is not None and not (
        isinstance(crop, list)
        and len(crop) == 4
        and all(isinstance(v, int | float) and 0 <= v <= 1 for v in crop)
        and crop[0] < crop[2]
        and crop[1] < crop[3]
    ):
        raise ValueError("crop must be [left, top, right, bottom], fractions of the image")
    allowed = (
        ENTRANCE_DECISIONS if str(decision.get("id", "")).startswith("entrance:") else DECISIONS
    )
    if decision.get("decision") not in allowed:
        raise ValueError(f"decision must be one of {allowed}")
    row = {
        "id": str(decision["id"]),
        "decision": decision["decision"],
        "lon": decision.get("lon"),
        "lat": decision.get("lat"),
        "heading": decision.get("heading"),
        "seconds": None if decision.get("seconds") is None else seconds_taken(decision["seconds"]),
        "photos": decision.get("photos"),
        # [left, top, right, bottom] as fractions of the image, or None: of the item's own
        # print only (prints of one negative differ in trimming and mount)
        "crop": crop,
    }
    return append_row(reviews_path, row)


def inside(root: Path, relative: str) -> Path | None:
    """`root/relative` when it stays inside `root`, else None."""
    path = (root / relative).resolve()
    return path if path.is_relative_to(root.resolve()) and path.is_file() else None


def serve(
    conn: sqlite3.Connection, pilot_dir: Path, gazetteer: Gazetteer, reviews_path: Path, port: int
) -> HTTPServer:
    class Handler(localpage.Handler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, localpage.page(PAGE), "text/html; charset=utf-8")
            elif path == "/api/queue":
                items = queue(conn, pilot_dir, gazetteer, reviews_path)
                self._json([asdict(i) for i in items])
            elif path.startswith("/tile/"):
                try:
                    sheet, n = path.removeprefix("/tile/").removesuffix(".jpg").rsplit("/", 1)
                    self._send(200, tile(pilot_dir, sheet, int(n)), "image/jpeg")
                except (ValueError, FileNotFoundError):
                    self._send(404, b"not found", "text/plain")
            elif path.startswith("/pilot/") and (
                file := inside(pilot_dir, path.removeprefix("/pilot/"))
            ):
                self._send(200, file.read_bytes(), "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/decision":
                self._send(404, b"not found", "text/plain")
                return
            try:
                self._json(record(reviews_path, self._body()))
            except (ValueError, KeyError, TypeError) as error:
                self._send(400, str(error).encode(), "text/plain")

    # One request at a time: the SQLite connection belongs to this thread.
    return localpage.server(Handler, port)
