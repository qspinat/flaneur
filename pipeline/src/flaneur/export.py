"""The walk's data: every placed street view of the pilot that may be published, one per
negative, with the user's review applied, the links to step from photo to photo, and a
view of the same place today. Written as one JSON file and resized images, for the site.

Links (thresholds fixed in the design before looking at the result):
- forward: the nearest photo 8–250 m ahead (within ±25° of where I look), facing within
  ±45° of my direction; back: likewise behind me;
- left / right: the nearest photo turned about 90° (±45°): from the same spot when there is
  one, else from a cross street within 60 m (the design's "or the nearest cross-street
  photo");
- eras: photos from the same spot (< 20 m) facing about the same way (±30°), of other
  years, the nearest of each date.
"""

import json
import re
import sqlite3
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from PIL import Image
from shapely.geometry import Point

from flaneur import crop, dedup, panoramax, placements, review
from flaneur.gazetteer import angle, bearing, to_metres
from flaneur.pilot import DOWNLOADS_AT_ONCE, cached_image, safe_name, save_image
from flaneur.sources import cached_json, write_json

DIRECTIONS = ("forward", "back", "left", "right")
FORWARD_M = (8, 250)
AHEAD_DEG = 25
SAME_WAY_DEG = 45
TURN_M = 60
TURN_DEG = (45, 135)
ERA_M = 20
ERA_DEG = 30
PHOTO_SIDE = 1600
SMALL_SIDE = 960  # for a phone's screen: about a third of the bytes
THUMB_SIDE = 320  # for a small print: a way on, a photograph chosen on the plan
SIZES = (THUMB_SIDE, SMALL_SIDE)  # the smaller sizes, beside the photograph's own
TODAY_SIDE = 1200
# How a licence is named under a photo.
LICENSES = {
    "CC0": "domaine public (CC0)",
    "CC-BY-4.0": "CC BY 4.0",
    "etalab-2.0": "Licence Ouverte 2.0",
    "CC-BY-SA-4.0": "CC BY-SA 4.0",
}
# Licences that ask for any change to the image to be said (CC BY 4.0, section 3(a)(1)(B)).
CHANGES_SAID = {"CC-BY-4.0"}


def panoramax_credit(picture: panoramax.Picture) -> str:
    """The credit of a Panoramax picture as shown: who to credit, then its licence."""
    return f"{panoramax.credit_of(picture)}, {LICENSES.get(picture.license, picture.license)}"


def https_or_none(url: str | None) -> str | None:
    """A link out of the site, kept only when it is an https address (it is put on a page)."""
    return url if url and url.startswith("https://") else None


@dataclass
class Link:
    id: str
    distance_m: int


@dataclass
class Today:
    panoramax_id: str
    heading: float
    panoramic: bool
    taken: str
    image: str | None = None
    credit: str | None = None  # who took it, and under which licence
    url: str | None = None  # the picture in Panoramax's own viewer


@dataclass
class WalkPhoto:
    id: str
    title: str
    caption: str | None
    years: list[int | None]
    photographer: str  # as one writes a name: "Eugène Atget"
    credit: str  # the source and the licence, as shown under the photo
    record_url: str | None
    lon: float
    lat: float
    heading: float | None
    street: tuple[str, str] | None  # as on its plate: ("Rue", "Saint-Julien-le-Pauvre")
    arrondissement: int | None
    image_url: str  # the source's full-size image (not exported)
    crop: list[float] | None  # (not exported: applied to the image)
    reviewed: bool = False  # its place was confirmed or fixed by hand (not exported)
    moved_by_review: bool = False  # fixed by hand far from the proposed place or direction
    in_3d: bool = False  # shown in the 3D walk (placed in its panorama by the user)
    flipped: bool = False  # the print turned the right way round (it was mirrored)
    no_3d: bool = False  # no panorama sees it (answered so on the placing page)
    # only a start (placements.SEED): matched, not shown until its camera is confirmed
    seeded: bool = False
    image: str | None = None  # path of the exported image, relative to the data folder
    # each size's width as written, by its side
    widths: dict[int, int] = field(default_factory=dict)
    aspect: float | None = None  # height over width, as exported (the site fits it to its frame)
    today: Today | None = None
    links: dict[str, Link | None] = field(default_factory=dict)
    eras: list[str] = field(default_factory=list)

    def for_site(self) -> dict:
        """What the site reads (walk.json)."""
        return {
            "id": self.id,
            "title": self.title,
            "caption": self.caption,
            "years": self.years,
            "photographer": self.photographer,
            "credit": self.credit,
            "record_url": self.record_url,
            "lon": self.lon,
            "lat": self.lat,
            "heading": self.heading,
            "street": self.street,
            "arrondissement": self.arrondissement,
            # each size, by its width in pixels (the site builds its srcset from them): a side
            # bounds the longer edge, so a portrait is narrower than its side
            "images": self.image
            and {
                **{str(self.widths.get(side, side)): sized(self.image, side) for side in SIZES},
                str(self.widths.get(PHOTO_SIDE, PHOTO_SIDE)): self.image,
            },
            "aspect": self.aspect,
            "today": self.today
            and {
                "image": self.today.image,
                "taken": self.today.taken,
                "credit": self.today.credit,
                "url": self.today.url,
            },
            "links": {
                k: v and {"id": v.id, "distance_m": v.distance_m} for k, v in self.links.items()
            },
            "eras": self.eras,
            "in_3d": self.in_3d,
            "no_3d": self.no_3d,
            "flipped": self.flipped,
        }


ARTICLE = re.compile(r"^(?:de la |de l'|des |du |de |d')", re.IGNORECASE)


def plate_name(label: str) -> tuple[str, str]:
    """A street's kind and name as Paris writes them on its plates: the words of a name
    joined by hyphens ("rue Saint-Julien le Pauvre" → "Rue", "Saint-Julien-le-Pauvre";
    "place Paul Painlevé" → "Place", "Paul-Painlevé"), a person's particle kept ("Du
    Sommerard")."""
    kind, _, name = label.partition(" ")
    article = ARTICLE.match(name)
    article_text = article.group(0) if article and name[0].islower() else ""
    rest = name[len(article_text) :]
    if " " in rest and rest.split(" ")[0] not in ("Du", "De", "Des", "La", "Le"):
        rest = "-".join(rest.split(" "))
    return f"{kind[:1].upper()}{kind[1:]}", f"{article_text}{rest}"


def display_name(notice: str | None) -> str:
    """A photographer as one writes a name, from the museum's notice: "Atget, Eugène (Jean
    Eugène Auguste Atget, dit)" → "Eugène Atget"."""
    if not notice:
        return "Photographe inconnu"
    name = re.sub(r"\s*\(.*\)\s*", "", notice).strip()
    last, _, first = name.partition(",")
    return f"{first.strip()} {last.strip()}" if first else last.strip()


def photos(
    conn: sqlite3.Connection,
    pilot_dir: Path,
    reviews_path: Path,
    situated: dict[str, dict] | None = None,
    placed: dict[str, dict] | None = None,
) -> list[WalkPhoto]:
    """The walk's photos, with the last decision of the review applied: a photo judged not a
    street view is left out, a confirmed or fixed one takes the reviewed place and direction;
    the others take the place where their camera was found against today's panoramas
    (`situated`: lon, lat, heading; see flaneur.match), else the proposed placement. A place
    set on the placing page (`placed`: lon, lat, heading, at, by print; on any print of the
    negative) wins over a review made before it."""
    situated, placed = situated or {}, placed or {}
    negatives = dedup.negatives(conn)
    decisions = review.decisions_by_photo(reviews_path, negatives)
    walk = []
    for r in placements.publishable(conn, pilot_dir, negatives, seeded=True):
        info = json.loads(
            (pilot_dir / "photos" / safe_name(r["photo_id"]) / "info.json").read_text()
        )
        decided = decisions.get(r["photo_id"])
        if decided and decided["decision"] == "not_a_street_view":
            continue
        reviewed = bool(decided and decided["decision"] in ("confirm", "fix"))
        prints = {r["photo_id"], *negatives.get(r["photo_id"], ())}
        set_here = max(
            (placed[i] for i in prints if i in placed), key=lambda p: p["at"], default=None
        )
        if set_here and (not reviewed or set_here["at"] >= decided["at"]):
            by_hand = set_here  # the placing page's, the latest
        else:
            by_hand = decided if reviewed else None
        found = None if by_hand else situated.get(r["photo_id"])
        place = by_hand or found or r
        seeded = place is r and r["confidence"] == placements.SEED
        flipped = bool(set_here and by_hand is set_here and set_here.get("flipped"))
        # the crop drawn or accepted by hand on this very print, else the one proposed
        own = decided if decided and decided["id"] == f"photo:{r['photo_id']}" else {}
        frame = own.get("crop") or info.get("suggested_crop")
        cropped = frame not in (None, [0, 0, 1, 1])
        walk.append(
            WalkPhoto(
                id=r["photo_id"],
                title=r["title"],
                caption=r["caption"],
                years=info["years"],
                photographer=display_name(r["photographer"]),
                credit=f"{r['credit']}, {LICENSES.get(r['license'], r['license'])}"
                + ("; image retournée" if flipped else "")
                + ("; image recadrée" if cropped and r["license"] in CHANGES_SAID else ""),
                record_url=https_or_none(r["record_url"]),
                lon=place["lon"],
                lat=place["lat"],
                heading=place["heading"],
                street=plate_name(r["street"]) if r["street"] else None,
                arrondissement=(json.loads(r["arrondissements"] or "[]") or [None])[0],
                image_url=r["image_url"],
                crop=frame if cropped else None,
                reviewed=by_hand is not None,
                moved_by_review=bool(by_hand) and _moved(r, by_hand),
                flipped=flipped,
                seeded=seeded,
            )
        )
    return walk


# A fix this far from the proposed place, or turned this much, is a different view.
MOVED_M = 15
TURNED_DEG = 30


def _moved(proposed, decided: dict) -> bool:
    """Whether a place set by hand (a review's fix, or the placing page's) moved the photo."""
    if decided.get("decision", "fix") != "fix":  # the placing page sets a place: a fix
        return False
    here = to_metres(Point(proposed["lon"], proposed["lat"]))
    there = to_metres(Point(decided["lon"], decided["lat"]))
    turned = (
        proposed["heading"] is not None
        and decided["heading"] is not None
        and abs(angle(proposed["heading"], decided["heading"])) > TURNED_DEG
    )
    return here.distance(there) > MOVED_M or turned


def link(walk: list[WalkPhoto]) -> None:
    """Fill each photo's links and eras (see the module's docstring)."""
    points = {p.id: to_metres(Point(p.lon, p.lat)) for p in walk}
    for me in walk:
        best: dict[str, tuple[float, WalkPhoto]] = {}
        eras: dict[tuple, tuple[float, WalkPhoto]] = {}  # the nearest of each date

        def keep(found: dict, key, distance: float, other: WalkPhoto) -> None:
            if key not in found or distance < found[key][0]:
                found[key] = (distance, other)

        for other in walk if me.heading is not None else []:
            if other is me or other.heading is None:
                continue
            distance = points[me.id].distance(points[other.id])
            turned = angle(me.heading, other.heading)
            where = angle(me.heading, bearing(points[me.id], points[other.id]))
            if FORWARD_M[0] <= distance <= FORWARD_M[1] and abs(turned) <= SAME_WAY_DEG:
                if abs(where) <= AHEAD_DEG:
                    keep(best, "forward", distance, other)
                elif abs(where) >= 180 - AHEAD_DEG:
                    keep(best, "back", distance, other)
            if distance <= TURN_M and TURN_DEG[0] <= abs(turned) <= TURN_DEG[1]:
                keep(best, "right" if turned > 0 else "left", distance, other)
            if distance < ERA_M and abs(turned) <= ERA_DEG and other.years != me.years:
                keep(eras, tuple(other.years), distance, other)
        me.links = {
            k: Link(best[k][1].id, round(best[k][0])) if k in best else None for k in DIRECTIONS
        }
        me.eras = [
            p.id for _, p in sorted(eras.values(), key=lambda e: (e[1].years[0] or 0, e[1].id))
        ]


def _tiles(pilot_dir: Path) -> dict[str, dict]:
    """Every picture of today shown on the pilot's contact sheets, by Panoramax id."""
    return {
        tile["panoramax_id"]: tile
        for path in sorted((pilot_dir / "sheets").glob("*.json"))
        for tile in json.loads(path.read_text())
    }


def choose_today(me: WalkPhoto, matched: str | None, tiles: dict[str, dict]) -> Today | None:
    """The picture of today showing the same place: only the one matched when placing (the
    nearest picture facing the same way was measured too often wrong: another street, the
    inside of a church), else None."""
    tile = tiles.get(matched) if matched else None
    if tile is None or me.heading is None:
        return None
    return Today(
        panoramax_id=tile["panoramax_id"],
        heading=me.heading if tile["panoramic"] else tile["azimuth"],
        panoramic=tile["panoramic"],
        taken=tile["taken"],
    )


def build(
    conn: sqlite3.Connection,
    pilot_dir: Path,
    walk: list[WalkPhoto],
) -> list[WalkPhoto]:
    """The walk (its photos where each is best known: see photos), linked, each photo with its
    view of today chosen (images not yet written); a seed not confirmed is left out."""
    walk = [me for me in walk if not me.seeded]
    link(walk)
    tiles = _tiles(pilot_dir)
    matched = dict(conn.execute("SELECT photo_id, match_panoramax_id FROM placements"))
    for me in walk:
        # a place fixed by hand no longer shows what the matched picture showed: unchecked
        if not me.moved_by_review:
            me.today = choose_today(me, matched.get(me.id), tiles)
    return walk


def _save(image, side: int, path: Path) -> int:
    """Save `image` no larger than `side` on its longer edge; returns its width."""
    image = image.convert("RGB")
    image.thumbnail((side, side))
    save_image(image, path, quality=80)
    return image.width


def _today_image(client: httpx.Client, today: Today, cache_dir: Path, out: Path) -> None:
    """Fetch the picture of today and its credit; a 360° picture is cut facing the photo."""
    meta = cache_dir / f"{today.panoramax_id}.json"  # credit and address, asked once
    picture = panoramax.Picture(
        **cached_json(meta, lambda: asdict(panoramax.by_id(client, today.panoramax_id)))
    )
    image = cached_image(client, picture.sd_url, cache_dir / f"{today.panoramax_id}.jpg")
    if today.panoramic:
        image = panoramax.view(image.convert("RGB"), picture.azimuth or 0, today.heading)
    _save(image, TODAY_SIDE, out)
    today.credit = panoramax_credit(picture)
    today.url = (
        f"https://api.panoramax.xyz/#focus=pic&pic={today.panoramax_id}"
        f"&xyz={today.heading:.0f}/0/30"
    )


def image_name(photo_id: str) -> str:
    """Where a photo's image is exported, relative to the data folder."""
    return f"photos/{safe_name(photo_id)}.webp"


def sized(image: str, side: int) -> str:
    """A smaller version of an exported image: "photos/x.webp", 960 → "photos/x-960.webp"."""
    return image.removesuffix(".webp") + f"-{side}.webp"


def _framed(me: WalkPhoto, client: httpx.Client, cache_dir: Path) -> Image.Image:
    """A photo's image as shown, cropped; OSError when it is not an image, or a truncated one,
    then dropped from the cache to be fetched again next time."""
    cached = cache_dir / "archive" / f"{safe_name(me.id)}.jpg"
    try:
        image = cached_image(client, me.image_url, cached)
        image.draft("RGB", (PHOTO_SIDE * 2, PHOTO_SIDE * 2))  # decode no larger than needed
        return crop.apply(image, me.crop) if me.crop else image
    except OSError:
        cached.unlink(missing_ok=True)
        raise


def _write_one(me: WalkPhoto, client: httpx.Client, cache_dir: Path, out_dir: Path) -> bool:
    """Write one photo's image and its view of today; False when the photo cannot be had
    (its view of today is dropped alone when only that fails)."""
    name = safe_name(me.id)
    try:
        framed = _framed(me, client, cache_dir)
        me.image = image_name(me.id)
        me.aspect = round(framed.height / framed.width, 4)
        me.widths = {PHOTO_SIDE: _save(framed, PHOTO_SIDE, out_dir / me.image)}
        for side in SIZES:
            me.widths[side] = _save(framed, side, out_dir / sized(me.image, side))
    except httpx.HTTPError:
        return False
    except OSError:  # not an image, or a truncated one (see _framed)
        return False
    if me.today:
        try:
            me.today.image = f"today/{name}.webp"
            _today_image(client, me.today, cache_dir / "today", out_dir / me.today.image)
        except (httpx.HTTPError, IndexError, KeyError, OSError):
            me.today = None
    return True


def write_seeds(
    seeds: list[WalkPhoto], client: httpx.Client, cache_dir: Path, out_dir: Path
) -> int:
    """Write the images of photos only seeded, apart from the site's: the one size matching
    reads, once, never published. Returns how many are there."""
    there = 0
    for me in seeds:
        path = out_dir / image_name(me.id)
        if not path.exists():
            try:
                _save(_framed(me, client, cache_dir), PHOTO_SIDE, path)
            except (httpx.HTTPError, OSError):
                continue
        there += 1
    return there


def mark_3d(
    walk: list[WalkPhoto], out_dir: Path, in_3d: dict[str, str], no_3d: Collection[str]
) -> None:
    """Which photos the 3D walk shows: those placed in their panorama on the image written
    now (`in_3d`: the hash of the image each was placed on), and which no panorama sees."""
    from flaneur.landmarks import image_hash  # here: landmarks loads the matcher's module

    for me in walk:
        placed_on = in_3d.get(me.id)
        written = out_dir / image_name(me.id)
        me.in_3d = bool(placed_on) and written.exists() and image_hash(written) == placed_on
        me.no_3d = me.id in no_3d


def write(
    walk: list[WalkPhoto],
    client: httpx.Client,
    cache_dir: Path,
    out_dir: Path,
    in_3d: dict[str, str] | None = None,
    no_3d: Collection[str] = (),
) -> list[WalkPhoto]:
    """Write the images (cropped and resized) and walk.json into `out_dir`; downloads are
    kept in `cache_dir`; the 3D walk marked on the images written (see mark_3d). Returns the
    photos written: a failed download leaves one out, and the links to it."""
    with ThreadPoolExecutor(DOWNLOADS_AT_ONCE) as pool:
        written = list(pool.map(lambda p: _write_one(p, client, cache_dir, out_dir), walk))
    exported = [p for p, ok in zip(walk, written, strict=True) if ok]
    ids = {p.id for p in exported}
    for me in exported:
        me.links = {k: v if v and v.id in ids else None for k, v in me.links.items()}
        me.eras = [i for i in me.eras if i in ids]
    mark_3d(exported, out_dir, in_3d or {}, no_3d)
    write_json(out_dir / "walk.json", {"photos": [p.for_site() for p in exported]})
    _remove_others(out_dir, walk)
    return exported


def _remove_others(out_dir: Path, walk: list[WalkPhoto]) -> None:
    """Remove the images of photos no longer in the walk (left out by the review, merged as
    copies of one negative…), so that what is published is what the pages use. A photo still
    in the walk keeps its images even when its download failed this time: matching reads them."""
    kept = set()
    for me in walk:
        image = image_name(me.id)
        kept |= {image, *(sized(image, side) for side in SIZES), f"today/{safe_name(me.id)}.webp"}
    for folder in ("photos", "today"):
        for path in (out_dir / folder).glob("*"):
            if f"{folder}/{path.name}" not in kept:
                path.unlink()


def summary(walk: list[WalkPhoto]) -> dict[str, float]:
    """The share of photos with each kind of link (the design's gate: forward ≥ 70%)."""
    n = len(walk) or 1
    return {
        "photos": len(walk),
        "reviewed": sum(p.reviewed for p in walk),
        **{k: sum(1 for p in walk if p.links.get(k)) / n for k in DIRECTIONS},
        "eras": sum(1 for p in walk if p.eras) / n,
        "today": sum(1 for p in walk if p.today) / n,
    }
