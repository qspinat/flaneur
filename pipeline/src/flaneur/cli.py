"""Ligne de commande du pipeline : `uv run flaneur <étape>`."""

import functools
import hashlib
import io
import json
import shutil
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import asdict, replace
from datetime import date
from enum import StrEnum
from http.server import HTTPServer
from pathlib import Path
from typing import Annotated, NamedTuple

import cv2
import httpx
import numpy as np
import typer
from dotenv import load_dotenv
from PIL import Image
from shapely.geometry import Point

from flaneur import (
    buildings,
    carrying,
    chain,
    confirming,
    db,
    dedup,
    depth,
    export,
    fakes,
    geocode,
    landmarks,
    level,
    marking,
    match,
    outlines,
    panoramas,
    panoramax,
    pilot,
    placements,
    placing,
    placings,
    proposals,
    proposing,
    queueing,
    reconstruct,
    register,
    registration,
    review,
    scene,
    stats,
)
from flaneur.blocks import Blocks
from flaneur.gazetteer import Gazetteer, angle, to_degrees, to_metres
from flaneur.panoramax import Picture
from flaneur.read import read_all
from flaneur.sources import albert_kahn, paris_musees, write_json

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
DB_PATH = DATA / "flaneur.sqlite"
WEB_DATA = ROOT / "web" / "public" / "data"  # the walk, as the site reads it (not versioned)
REVIEWS = DATA / "overrides" / "reviews.jsonl"
SIGHTS = DATA / "overrides" / "sights.jsonl"  # the landmarks marked in photos, by hand
CORRECTIONS = DATA / "pilot" / "panorama_corrections.json"  # on the buildings (register): unused
KEPT_DEPTHS = 24  # views' details with their depth the placing page keeps (1.4 MB each)
FIXES = DATA / "pilot" / "panorama_fixes.json"  # panoramas placed with each other (reconstruct)
PLACINGS = DATA / "overrides" / "placings.jsonl"  # the user's answers on the placing page
TILTS = DATA / "pilot" / "panorama_tilts.json"  # how each one's camera leaned (level)
PANORAMAS = DATA / "raw" / "panoramas"  # their pictures

SOURCES = {"paris_musees": paris_musees, "albert_kahn": albert_kahn}
Source = StrEnum("Source", list(SOURCES))

app = typer.Typer(help="Pipeline Flâneur.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Récolte, lecture, placement et export des photos d'archive de Paris."""
    load_dotenv(ROOT / "pipeline" / ".env")


@app.command()
def harvest(source: Source) -> None:
    """Récolte une source dans la base (les réponses brutes restent en cache dans data/raw)."""
    with httpx.Client(timeout=120) as client:
        photos = SOURCES[source.value].harvest(client, DATA / "raw" / source.value)
        with closing(db.connect(DB_PATH)) as conn:
            count = db.upsert(conn, photos)
    typer.echo(f"{source.value} : {count} photos enregistrées")


@app.command("read")
def read_captions() -> None:
    """Lit titres et légendes (règles) : genre de photo, rues, arrondissement, point de vue."""
    with closing(db.connect(DB_PATH)) as conn:
        count = read_all(conn)
    typer.echo(f"{count} photos lues")


@app.command("geocode")
def geocode_photos() -> None:
    """Place chaque photo sur sa rue (ou au coin d'où elle a été prise) ; puis compare avec
    les positions de prise de vue connues (Albert-Kahn)."""
    with httpx.Client(timeout=300) as client:
        gazetteer = Gazetteer.load(client, DATA / "raw" / "gazetteer")
    with closing(db.connect(DB_PATH)) as conn:
        counts = geocode.locate_all(conn, gazetteer)
        distances = geocode.distances_to_truth(conn)
    typer.echo(f"placées : {dict(counts)}")
    for status, found in sorted(distances.items()):
        s = geocode.summary(found)
        typer.echo(
            f"vérité terrain, {status} : {s['n']} photos ; la vraie position de prise de vue est"
            f" à moins de 50 m de ce qui a été trouvé pour {s['within_50m']:.0%}, médiane"
            f" {s['median_m']:.0f} m"
        )


@app.command("pilot")
def prepare_pilot() -> None:
    """Prépare le quartier pilote : photo, légende, et planche des vues actuelles de sa rue."""
    with (
        closing(db.connect(DB_PATH)) as conn,
        httpx.Client(timeout=120, follow_redirects=True) as client,
    ):
        counts = pilot.prepare(conn, client, DATA / "pilot")
    typer.echo(f"quartier pilote : {counts}")


@app.command("crops")
def crop_photos() -> None:
    """Retrouve le tirage sur son carton dans chaque photo du pilote, et dit lesquels
    recadrer à la main (plusieurs tirages, carton dans le cadre, tirage coupé)."""
    folders = sorted(  # a photo not yet downloaded has no scan to crop
        p.parent
        for p in (DATA / "pilot" / "photos").glob("*/info.json")
        if (p.parent / "photo.jpg").exists()
    )
    found = [pilot.crop_again(folder) for folder in folders]
    changed = sum(changed for changed, _ in found)
    doubted = sum(bool(doubts) for _, doubts in found)
    typer.echo(
        f"{changed} cadrages changés sur {len(folders)} ; {doubted} à recadrer à la main "
        "(en tête de la relecture)"
    )


@app.command("dedup")
def find_copies() -> None:
    """Regroupe les tirages d'un même négatif (le musée en garde souvent deux ou trois).
    Après `geocode` : les tirages que leurs légendes placent près l'un de l'autre sont
    comparés ; relancer `geocode` ensuite donne à chaque négatif le lieu de ses tirages."""
    with closing(db.connect(DB_PATH)) as conn:
        found = dedup.find(DATA / "pilot" / "photos", dedup.places(conn))
        dedup.save(conn, found)
    typer.echo(f"{len(found)} négatifs tirés plusieurs fois, {sum(map(len, found))} tirages")


@app.command("placements")
def import_placements() -> None:
    """Enregistre les placements fins du quartier pilote et les compare aux positions connues."""
    with closing(db.connect(DB_PATH)) as conn:
        found = placements.inherit(placements.load(DATA / "pilot"), dedup.negatives(conn))
        placements.save(conn, found)
        distances = placements.distances_to_truth(conn)
    typer.echo(f"{len(found)} placements enregistrés")
    for confidence in placements.CONFIDENCES:
        d = [m for c, m in distances if c == confidence]
        if d:
            s = geocode.summary(d)
            typer.echo(
                f"vérité terrain, confiance {confidence} : {s['n']} photos, "
                f"{s['within_50m']:.0%} à moins de 50 m, médiane {s['median_m']:.0f} m"
            )


def _placing_answers(
    path: Path = PLACINGS,
) -> tuple[dict[str, dict], dict[str, dict], set[str]]:
    """From the placing page's answers: where each photo answered stands (lon, lat, heading),
    the answers placing a photo in its panorama (on the image it had then), and the photos no
    panorama sees."""
    held = placings.holding(path)
    places = {
        i: {k: r[k] for k in ("lon", "lat", "heading", "at", "flipped")} for i, r in held.items()
    }
    in_3d = {i: r for i, r in held.items() if r["answer"] in placings.PLACED}
    return places, in_3d, {i for i, r in held.items() if r["answer"] == "far"}


def _walk(conn, places: dict[str, dict] | None = None) -> list[export.WalkPhoto]:
    """The walk's photos where each is best known now: the placing page (`places`, else read),
    the review, a match confirmed, the caption."""
    places = _placing_answers()[0] if places is None else places
    return export.photos(conn, DATA / "pilot", REVIEWS, _places(_confirmed()), places)


def _on_image_now(placed: dict[str, dict]) -> dict[str, dict]:
    """The answers placing a photo in its panorama made on the image it has now."""
    return {i: r for i, r in placed.items() if _exported(i).exists() and r["inputs"] == _image(i)}


def _carried(placed: dict[str, dict]) -> dict[str, dict]:
    """The copies carried to the neighbouring panoramas (`flaneur porter`) still made from
    the answer that holds (`placed`)."""
    kept = json.loads(CARRIED.read_text()) if CARRIED.exists() else {}
    return {i: c for i, c in kept.items() if i in placed and c["made"] == _answered_as(placed[i])}


@app.command("export")
def export_walk(images: bool = True) -> None:
    """Écrit la balade (photos relues appliquées, liens, vue d'aujourd'hui) pour le site."""
    places, in_3d, no_3d = _placing_answers()
    with closing(db.connect(DB_PATH)) as conn:
        walk = export.build(conn, DATA / "pilot", _walk(conn, places))
    elsewhere = len(in_3d) - len(_on_image_now(in_3d))
    if elsewhere:  # their image changed since: the placing page asks for them again
        typer.echo(f"{elsewhere} photos placées sur une image qu'elles n'ont plus : à revoir")
    if images:
        placed_on = {i: r["inputs"] for i, r in in_3d.items()}
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            walk = export.write(walk, client, DATA / "raw" / "images", WEB_DATA, placed_on, no_3d)
        typer.echo(f"écrit dans {WEB_DATA}")
    s = export.summary(walk)
    typer.echo(
        f"{s['photos']} photos dont {s['reviewed']} relues ; avec un lien en avant "
        f"{s['forward']:.0%}, en arrière {s['back']:.0%}, à gauche {s['left']:.0%}, "
        f"à droite {s['right']:.0%} ; autres époques {s['eras']:.0%} ; vue d'aujourd'hui "
        f"{s['today']:.0%}"
    )


@app.command("panoramas")
def find_panoramas(margin_m: float = 50) -> None:
    """Récolte les panoramas 360° de Panoramax autour des photos placées (balade en 3D)."""
    with closing(db.connect(DB_PATH)) as conn:
        with httpx.Client(timeout=60) as client:
            found, full = panoramas.harvest(client, placements.area(conn, margin_m))
        panoramas.save(conn, found)
    s = panoramas.summary(found)
    typer.echo(
        f"{s['panoramas']} panoramas ; cap douteux : {s['doubtful']} "
        f"({s['doubtful'] / max(s['panoramas'], 1):.0%}) ; liens, distance médiane "
        f"{s['median_link_m']:.1f} m"
        + (f" ; {full} cases pleines (panoramas peut-être manquants)" if full else "")
    )


def _buildings(area) -> list[dict]:
    """Today's buildings in `area` (downloaded once)."""
    with httpx.Client(timeout=120) as client:
        return buildings.fetch(client, area, DATA / "raw" / "buildings")


@app.command("buildings")
def find_buildings() -> None:
    """Télécharge les bâtiments d'aujourd'hui (BD TOPO) autour des photos placées."""
    with closing(db.connect(DB_PATH)) as conn:
        s = buildings.summary(_buildings(placements.area(conn, buildings.MARGIN_M)))
    typer.echo(
        f"{s['buildings']} bâtiments ; sans hauteur : {s['no_height']} ; sans altitude du sol : "
        f"{s['no_ground']}"
    )


def _world(features: list[dict], area) -> tuple[scene.CameraHeights, Blocks]:
    """The frame panoramas are lined up and photos situated in: metres Lambert-93, heights
    above the median ground; the panoramas' camera heights and today's buildings in it."""
    heights = _heights(features, area)
    return heights, _blocks(features, heights)


def _heights(features: list[dict], area) -> scene.CameraHeights:
    ground = scene.Ground.from_buildings(features, area)
    return scene.CameraHeights(ground, float(np.median(ground.heights)))


def _blocks(features: list[dict], heights: scene.CameraHeights) -> Blocks:
    origin = (0.0, 0.0, heights.origin_z)
    return Blocks.of(
        [(b["footprint"], b["base"], b["height"]) for b in scene.buildings_of(features, origin)]
    )


def _panoramas(pictures: list[Picture], heights: scene.CameraHeights) -> list[match.Panorama]:
    """The lined-up panoramas whose picture is at hand, as old photos are matched against."""
    return [
        match.Panorama(
            p.id,
            p.point.x,
            p.point.y,
            heights(p.point.x, p.point.y),
            p.azimuth or 0.0,
            path,
            p.roll,
            p.pitch,
        )
        for p in pictures
        if (path := PANORAMAS / f"{p.id}.jpg").exists()
    ]


def _lined_up(pictures: list[Picture]) -> list[Picture]:
    """The panoramas where they were taken, and levelled (see _levelled), when their tilts
    were measured as tilts are now."""
    if not level.current(TILTS):
        raise typer.BadParameter(
            "les inclinaisons des panoramas ont été mesurées autrement : lancer `flaneur level`"
        )
    return _levelled(pictures)


def _levelled(pictures: list[Picture]) -> list[Picture]:
    """The panoramas where they were taken (see _placed), and levelled: those not measured
    leaning as their run's neighbours, else taken as level (see level.borrowed)."""
    placed = _placed(pictures)
    tilts = level.borrowed(level.kept(TILTS), placed)
    return [
        replace(p, roll=tilt[0], pitch=tilt[1]) if (tilt := tilts.get(p.id)) else p for p in placed
    ]


def _placed(pictures: list[Picture]) -> list[Picture]:
    """The panoramas where they were taken: placed with each other (reconstruct), else moved
    as their run's nearest panoramas so placed (reconstruct.borrowed), else as recorded (not
    by their own roofs and corners: see register)."""
    fixes = register.load(FIXES)
    found = {i: tuple(fixes[i]["fix"]) for i in _fixed_by_colmap(pictures, fixes)}
    return register.applied(pictures, reconstruct.borrowed(found, pictures))


class _Depths:
    """Today's details with their depth around panoramas, per view (see depth.around): made
    with the matcher (`lock` held while it runs; asked twice at once, made once), kept for
    KEPT_DEPTHS views; one that cannot be made is asked again next time."""

    def __init__(
        self,
        spheres: list[match.Panorama],
        fixed: set[str],
        views: Callable[[], match.Views],
        lock: threading.Lock,
    ):
        self.by_id = {p.id: p for p in spheres}
        self.spheres, self.fixed, self.views, self.lock = spheres, fixed, views, lock
        self.made: OrderedDict[tuple[str, int], depth.Depths] = OrderedDict()  # 1.4 MB each

    def toward(self, panorama: str, k: int) -> depth.Depths | None:
        if (panorama, k) in self.made:
            self.made.move_to_end((panorama, k))  # used lately: kept longest
        else:
            with self.lock:  # the matcher, one thing at a time
                if (panorama, k) not in self.made:
                    home = self.by_id[panorama]
                    found = depth.of_panorama(home, self.spheres, self.fixed, self.views(), [k])
                    if found is None:
                        return None
                    self.made[panorama, k] = found
                    while len(self.made) > KEPT_DEPTHS:
                        self.made.popitem(last=False)
        return self.made.get((panorama, k))

    def around(self, panorama: str, heading: float) -> depth.Depths | None:
        if panorama not in self.by_id:
            return None
        return depth.around(heading, lambda k: self.toward(panorama, k))


def _fixed_by_colmap(pictures: list[Picture], fixes: dict) -> set[str]:
    """The panoramas COLMAP placed itself (`fixes`: register.load(FIXES)), as they are recorded
    now (the others borrow a fix from their run, or stay as recorded: see _placed)."""
    return {p.id for p in pictures if register.still(fixes.get(p.id), p)}


@app.command("register")
def register_panoramas() -> None:
    """Recale chaque panorama sur les bâtiments d'aujourd'hui par sa ligne des toits
    (position et cap enregistrés par le téléphone : souvent faux de plusieurs mètres). Plus
    utilisé pour situer les photos : moins juste que COLMAP et que le GPS (voir register)."""
    with closing(db.connect(DB_PATH)) as conn:
        area = placements.area(conn, buildings.MARGIN_M)
        features = _buildings(area)
        pictures = panoramas.load(conn)
    heights, blocks = _world(features, area)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        done = register.correct(
            pictures,
            heights,
            blocks,
            client,
            PANORAMAS,
            CORRECTIONS,
        )
    s = register.summary(done, pictures)
    typer.echo(
        f"{s['panoramas']} panoramas ; recalés : "
        f"{s['kept']} (déplacement médian {s['median_shift_m']:.1f} m, rotation "
        f"{s['median_turn_deg']:.1f}° ; toits sur des contours de force {s['fit_before']:.2f} → "
        f"{s['fit_after']:.2f}) ; écrit dans {CORRECTIONS}"
    )


@app.command("reconstruct")
def reconstruct_panoramas() -> None:
    """Place les panoramas les uns par rapport aux autres (COLMAP, par fenêtres, portes et
    corniches en commun), puis sur le plan par l'ensemble de leurs GPS : tuile par tuile."""
    with closing(db.connect(DB_PATH)) as conn:
        pictures = panoramas.load(conn)
    fixes, tried = reconstruct.run(pictures, PANORAMAS, DATA / "raw" / "reconstruction", FIXES)
    s = reconstruct.summary(fixes)
    typer.echo(
        f"{s['placed']} panoramas placés sur {tried} utilisables ({len(pictures)} en tout) ; "
        f"déplacement médian "
        f"{s['median_shift_m']:.1f} m, rotation médiane {s['median_turn_deg']:.1f}° ; "
        f"écrit dans {FIXES}"
    )


@app.command("level")
def level_panoramas() -> None:
    """Mesure de combien chaque panorama penche (roulis et tangage de son appareil), par les
    arêtes verticales des murs qu'il montre : ses vues sont ensuite coupées d'aplomb."""
    with closing(db.connect(DB_PATH)) as conn:
        pictures = panoramas.load(conn)
    spheres = reconstruct.usable(pictures, PANORAMAS)
    measured = level.measure({p.id: PANORAMAS / f"{p.id}.jpg" for p in spheres}, TILTS)
    s = level.summary(measured)
    lent = level.borrowed(measured, _placed(pictures))
    lent = sum(1 for i, t in lent.items() if t is not None and measured.get(i) is None)
    typer.echo(
        f"{s['measured']} panoramas mesurés sur {s['panoramas']}, {lent} autres penchant comme "
        f"leurs voisins de prise de vue (le reste, pris d'aplomb) ; penchent de "
        f"{s['median_deg']:.1f}° (médiane), {s['p90_deg']:.1f}° pour 1 sur 10 ; écrit dans {TILTS}"
    )


MATCHES = DATA / "pilot" / "matches.json"  # old photos situated against today (match)
MATCH_TRACE = DATA / "pilot" / "match_trace.json"  # how far each photo got, and why not further
CHECK_TRACE = DATA / "pilot" / "match_check_trace.json"  # the same, for the known-answer check
MATCH_CHECK = DATA / "pilot" / "match_check.json"  # the known-answer check's last result
CHAINS = DATA / "pilot" / "chains.json"  # old photos placed from old photos (chain)
CHAIN_CHECK = DATA / "pilot" / "chain_check.json"  # the chaining's known-answer check
ANCHORS = DATA / "overrides" / "anchors.jsonl"  # the user's yes or no to photos placed
MATCH_RADIUS_M = 60  # a placement from a caption can be 25 m off
MATCH_PANORAMAS = 16  # ranked by a quick first pass, the best views matched in full
EARLIEST_YEAR = 1850  # an undated photo: only the buildings standing by then rule a camera out
# The gate before matching real photos: views of panoramas, taken for old photos, found
# within this of their true camera (median).
CHECK_M = 1.0
CHECK_DEG = 2.0
CHECK_SHARE = 0.7  # and at least this share of them situated
CHECK_KEPT = (0.25, 0.35)  # the share of the true view a band keeps
FAR_M = 200  # a decoy: a view of a street this far away
CHECK_FOV = 60  # its views' width (and height: they are square)
MARKED_TO = 0.015  # a landmark's column, marked by hand: off by this (a share of the width)
RIGHT_M, RIGHT_DEG = 3.0, 5.0  # a camera found farther off than this is a wrong one
WRONG_SHARE = 0.05  # of all the views, bands and decoys: at most this share wrong


@functools.cache  # the panoramas do not move while a command runs
def _setup() -> str:
    """What a match now depends on (where the panoramas whose picture is at hand stand, as
    lined up and levelled, and the code that finds it): a match or a check found from another
    setup is stale."""
    with closing(db.connect(DB_PATH)) as conn:
        levelled = _levelled(panoramas.load(conn))
    where = sorted(
        (p.id, round(p.lon, 7), round(p.lat, 7), round(p.azimuth or 0.0, 2), p.roll, p.pitch)
        for p in levelled
        if (PANORAMAS / f"{p.id}.jpg").exists()
    )
    digest = hashlib.sha1(repr(where).encode())
    for path in (
        Path(match.__file__),
        Path(buildings.__file__),
        Path(landmarks.__file__),  # the check's own landmarks
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _exported(photo_id: str) -> Path:
    """A photo's image as the walk shows it and the matcher reads it (cropped)."""
    return WEB_DATA / export.image_name(photo_id)


@functools.cache  # for one run: an export while it runs would not be seen
def _image(photo_id: str) -> str:
    """A photo's image as exported (cropped), hashed: its content (an export rewrites it
    unchanged)."""
    return landmarks.image_hash(_exported(photo_id))


@functools.cache
def _sights() -> dict[str, tuple[match.Sight, ...]]:
    """The landmarks marked in each photo, those marked on the image it has now (cropped
    otherwise since, a mark's column is no longer where the landmark is)."""
    kept = {}
    for photo_id, marks in landmarks.marked(SIGHTS).items():
        if _exported(photo_id).exists() and marks.image == _image(photo_id):
            kept[photo_id] = marks.sights
        else:
            typer.echo(f"{photo_id} : repères marqués sur un autre cadrage, laissés de côté")
    return kept


def _inputs(photo_id: str, setup: str) -> str:
    """What a photo's match is found from: the panoramas' setup, its image as exported
    (cropped) and the landmarks marked in it."""
    inputs = f"{setup}:{_image(photo_id)}"
    if sights := _sights().get(photo_id):
        inputs += f":{hashlib.sha1(repr(sights).encode()).hexdigest()[:8]}"
    return inputs


def _matches() -> dict[str, dict | None]:
    return register.load(MATCHES)


def _situated() -> dict[str, match.Situated]:
    """The old photos whose camera was found against today's panoramas, from what they are
    now (a match found from other panoramas or another crop of the photo is left out), and
    not turned down by the user."""
    setup, refused = _setup(), _refused()
    found = {}
    for photo_id, m in _matches().items():
        if m and "x" in m and _exported(photo_id).exists():
            if m["inputs"] == _inputs(photo_id, setup) and (photo_id, m["inputs"]) not in refused:
                found[photo_id] = match.Situated(**m)
    return found


def _confirmed() -> dict[str, match.Situated]:
    """The situated photos the user said yes to as they are placed now: the photo walk takes
    their place and the 3D walk shows them (unless their place was set by hand since: a
    reviewed place wins, see export.photos). A placement not judged, or judged on other inputs,
    waits for a yes (a camera can collapse onto a panorama's spot and still look right).
    Photos placed by chaining are not among them: chaining is set aside."""
    verdicts = _verdicts()
    return {
        photo_id: m
        for photo_id, m in _situated().items()
        if (v := verdicts.get(photo_id)) and v["verdict"] == "yes" and v["inputs"] == m.inputs
    }


def _refused() -> set[tuple[str, str]]:
    """The placements the user did not take as they are (no, or near: the right buildings,
    the camera off): the photo, and what it was placed from."""
    return {(k, v["inputs"]) for k, v in _verdicts().items() if v["verdict"] != "yes"}


def _turned_down() -> set[str]:
    """The photos the user's last word on is not a yes: chaining leaves them out, as close
    placements of them come back from the same details through a point or two of another
    photo (they wait for their camera to be set right, or a new match to be judged)."""
    return {k for k, _ in _refused()}


def _places(situated: dict[str, match.Situated]) -> dict[str, dict]:
    """Where the situated photos stand, for the photo walk."""
    places = {}
    for photo_id, m in situated.items():
        lon, lat = to_degrees(Point(m.x, m.y)).coords[0]
        places[photo_id] = {"lon": lon, "lat": lat, "heading": m.heading}
    return places


class Then:
    """The buildings standing in a photo's year: where its camera could stand (outside them),
    and their walls, whose outlines the photo should show (each found when first asked for)."""

    def __init__(self, standing: list[dict], heights: scene.CameraHeights):
        self._standing, self._heights = standing, heights

    @functools.cached_property
    def free(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return buildings.outside(self._standing)

    @functools.cached_property
    def blocks(self) -> Blocks:
        return _blocks(self._standing, self._heights)


def _matching() -> tuple[scene.CameraHeights, list[match.Panorama], Callable[[int], Then]]:
    """The panoramas old photos are matched against, as lined up, and the buildings standing
    in a given year (today's: this year's)."""
    with closing(db.connect(DB_PATH)) as conn:
        area = placements.area(conn, buildings.MARGIN_M)
        features = _buildings(area)
        pictures = _lined_up(panoramas.load(conn))
    heights = _heights(features, area)
    dates = sorted({d for f in features if (d := buildings.appeared(f)) is not None})

    @functools.lru_cache(maxsize=4)  # each holds the area's walls: a few at a time
    def standing_since(latest: int) -> Then:
        return Then(buildings.standing(features, latest), heights)

    def then(year: int) -> Then:  # years with the same buildings standing share them
        return standing_since(max((d for d in dates if d <= year), default=0))

    return heights, _panoramas(pictures, heights), then


def _views(kept: int = 64) -> match.Views:
    """The panoramas' views' features (`kept` panoramas' in memory, about 13 MB each)."""
    with httpx.Client(timeout=300) as client:
        models = match.fetch_models(client, DATA / "raw" / "models")
    return match.Views(match.Matcher(models), DATA / "raw" / "view_features", kept)


def _nearest(
    near_by: list[match.Panorama], x: float, y: float, radius: float = np.inf, but: str = ""
) -> list[match.Panorama]:
    """The panoramas within `radius` of (x, y), nearest first (`but`: one left out)."""
    distance = {p.id: np.hypot(p.x - x, p.y - y) for p in near_by}
    return sorted(
        (p for p in near_by if distance[p.id] <= radius and p.id != but),
        key=lambda p: distance[p.id],
    )


SHEET_COLUMN = 300  # pixels per picture on the check sheets
REBUILT_FROM = 3  # the rebuilt view takes each spot from the best of these panoramas
UNSEEN_COLOUR = (180, 0, 180)  # hatching (BGR), where no panorama sees: not a colour of Paris


def _hatching(size: tuple[int, int], every: int = 8) -> np.ndarray:
    """Diagonal stripes over a picture of `size`."""
    rows, columns = np.mgrid[0 : size[0], 0 : size[1]]
    return (rows + columns) % every < 2


def _to_width(image: np.ndarray, width: int) -> np.ndarray:
    return cv2.resize(
        image, (width, round(width * image.shape[0] / image.shape[1])), interpolation=cv2.INTER_AREA
    )


@app.command("match")
def match_photos(
    photo: Annotated[list[str] | None, typer.Option(help="seulement ces photos")] = None,
    again: Annotated[bool, typer.Option(help="refaire les photos déjà faites")] = False,
    force: Annotated[bool, typer.Option(help="sans le contrôle à réponse connue")] = False,
) -> None:
    """Situe les photos en les comparant aux panoramas d'aujourd'hui autour d'elles : position,
    direction, champ et horizon, là où des façades d'alors sont encore debout."""
    setup = _setup()
    check = register.load(MATCH_CHECK)
    if not force and not (check.get("passed") and check.get("setup") == setup):
        raise typer.BadParameter(
            "le contrôle à réponse connue n'est pas passé pour ces panoramas : "
            "lancer `flaneur match-check` d'abord (ou --force)"
        )
    with closing(db.connect(DB_PATH)) as conn:
        # searched around the caption's place (or the review's), never a previous match's
        walk = export.photos(conn, DATA / "pilot", REVIEWS)
    heights, near_by, then = _matching()
    blocks = then(date.today().year).blocks
    views = _views()
    store = chain.Store(DATA / "raw", views.matcher)
    found = _matches()
    chosen = [
        me
        for me in walk
        if _exported(me.id).exists()
        and (not photo or me.id in photo)
        and (
            again
            or photo
            or me.id not in found
            or (found[me.id] or {}).get("inputs") != _inputs(me.id, setup)
        )
    ]

    traces, run = register.load(MATCH_TRACE), {}
    for me in sorted(chosen, key=_visit):
        here = _place_of(me)
        near = _nearest(near_by, *here, MATCH_RADIUS_M)[:MATCH_PANORAMAS]
        image = cv2.imread(str(_exported(me.id)))
        now = then(_then_year(me))
        where = _where(me, heights, now)
        trace = match.Trace()
        mine = store.features(me.id, _image(me.id), lambda image=image: match.shrunk(image))
        observed = match.observe(image, near, blocks, views, where, trace, mine, heights.origin_z)
        store.save_observed(me.id, _inputs(me.id, setup), observed)
        camera, walls, points = match.place(observed, where, trace)
        traces[me.id] = run[me.id] = {**_traced(trace), "place": list(here)}  # caption's
        if camera is None:  # kept, with what it was tried from: not tried again for nothing
            found[me.id] = {"inputs": _inputs(me.id, setup)}
            typer.echo(f"{me.id} : rien de sûr")
        else:
            moved = float(np.hypot(camera.x - here[0], camera.y - here[1]))
            outline = outlines.contrast(_grey(image), camera, now.blocks)
            situated = match.Situated.of(
                camera,
                street=heights.street(camera.x, camera.y),
                walls=walls,
                moved_m=moved,
                inputs=_inputs(me.id, setup),
                outline=outline,
            )
            found[me.id] = asdict(situated)
            store.save_points(me.id, _inputs(me.id, setup), points)
            typer.echo(
                f"{me.id} : situé ({camera.agree} points, {camera.panoramas} panoramas, "
                f"déplacé de {moved:.1f} m, contours {outline:.2f})"
            )
        write_json(MATCHES, found)
    write_json(MATCH_TRACE, traces)  # kept for every photo: its last run
    typer.echo(f"où ces photos s'arrêtent : {_funnel(run.values())}")
    ids = {me.id for me in walk}
    typer.echo(
        f"{sum(1 for k in _situated() if k in ids)} photos situées sur {len(walk)} ; "
        f"écrit dans {MATCHES}"
    )


def _then_year(me: export.WalkPhoto) -> int:
    """A photo's first year (undated: the earliest)."""
    return min((y for y in me.years if y), default=EARLIEST_YEAR)


def _visit(me: export.WalkPhoto) -> tuple[int, float]:
    """An order of photos in bands 60 m wide, back and forth: neighbours share panoramas."""
    x, y = _place_of(me)
    band = int(y // MATCH_RADIUS_M)
    return band, x if band % 2 == 0 else -x


def _place_of(me: export.WalkPhoto) -> tuple[float, float]:
    """Where a photo's caption (or the review) puts it, metres Lambert-93."""
    here = to_metres(Point(me.lon, me.lat))
    return here.x, here.y


def _where(me: export.WalkPhoto, heights: scene.CameraHeights, now: "Then") -> match.Where:
    """Where a photo's camera could stand: around its caption's place, outside the buildings
    `now` standing, seeing the landmarks marked in it."""
    return match.Where(heights.streets, now.free, _place_of(me), _sights().get(me.id, ()))


def _verdicts() -> dict[str, dict]:
    """The user's last word on each photo placed (ANCHORS: yes, near or no)."""
    return review.latest(ANCHORS, key="photo_id")


def _chain_setup() -> str:
    """What chaining now depends on: what matching does, and chaining's own code."""
    digest = hashlib.sha1(Path(chain.__file__).read_bytes()).hexdigest()[:8]
    return f"{_setup()}:{digest}"


def _placed_from(inputs: str, parents: list[chain.Anchor]) -> str:
    """What a chained photo was placed from: its own inputs, and its anchors' (a verdict on
    it holds while they do); without any, what it was placed from against today."""
    if not parents:
        return inputs
    from_ = repr(sorted((a.id, a.inputs) for a in parents))
    return f"{inputs}:{hashlib.sha1(from_.encode()).hexdigest()[:8]}"


class _Chaining:
    """What chaining needs: the walk's photos and where each could stand, the store of what
    each was matched with, the copies of each negative, and the photos placed against today
    (round 0) that the user did not turn down."""

    def __init__(self) -> None:
        self.setup, self.key = _setup(), _chain_setup()
        with closing(db.connect(DB_PATH)) as conn:
            walk = export.photos(conn, DATA / "pilot", REVIEWS)
            self.negatives = dedup.negatives(conn)
        self.heights, _, then = _matching()
        # by year: the buildings of a year are built once (a few kept at a time)
        self.walk = sorted((me for me in walk if _exported(me.id).exists()), key=_then_year)
        self.where = {me.id: _where(me, self.heights, then(_then_year(me))) for me in self.walk}
        self.then = then
        self.store = chain.Store(DATA / "raw", _views().matcher)
        self.by_id = {me.id: me for me in self.walk}
        self.refused = set(_set_aside())  # judged wrong by eye: not placed so again
        self.turned_down = _turned_down()  # by the user: not chained at all
        self.round_zero: dict[str, chain.Anchor] = {}
        aside = _set_aside()  # by eye: neither an anchor nor a truth to check against
        for photo_id, m in _situated().items():
            if photo_id in self.by_id and (photo_id, m.inputs) not in aside:
                if (points := self.store.points(photo_id, m.inputs)) is not None:
                    view = m.view(self.heights.street(m.x, m.y))
                    self.round_zero[photo_id] = self.anchor(photo_id, view, points, m.inputs)

    def features(self, photo_id: str) -> match.Features:
        return _features(self.store, photo_id)

    def anchor(
        self, photo_id: str, view: match.View, points: match.Points, inputs: str
    ) -> chain.Anchor:
        features = self.features(photo_id)  # its keypoints only: descriptors stay in the store
        return chain.Anchor(photo_id, view, features.keypoints, features.size, points, inputs)

    def anchors(self, me: export.WalkPhoto, placed: dict[str, chain.Anchor]) -> list[chain.Anchor]:
        """The old photos placed near a photo's caption place, its own negative's left out."""
        x, y = self.where[me.id].around
        copies = set(self.negatives.get(me.id, ()))
        return [
            a
            for a in placed.values()
            if a.id != me.id
            and a.id not in copies
            and np.hypot(a.view.x - x, a.view.y - y) <= chain.CHAIN_M
        ]

    def observed(self, me: export.WalkPhoto) -> match.Observed | None:
        return self.store.observed(me.id, _inputs(me.id, self.setup))

    def place(
        self,
        me: export.WalkPhoto,
        observed: match.Observed,
        anchors: list[chain.Anchor],
        trace: match.Trace,
        where: match.Where | None = None,
    ) -> match.Found:
        """A photo's camera from what it was observed with and from its anchors, standing
        where it could (`where`: elsewhere, for the check's decoys)."""

        def pairs(anchor: chain.Anchor) -> tuple[np.ndarray, np.ndarray]:
            rows = self.store.pairs(
                (me.id, _image(me.id)), (anchor.id, _image(anchor.id)), self.features
            )
            return rows[:, 0], rows[:, 1]

        rays, known = chain.from_anchors(anchors, pairs)
        where = self.where[me.id] if where is None else where
        return match.place(observed, where, trace, rays, known)


def _bare(observed: match.Observed) -> match.Observed:
    """An observation with its rays in today's panoramas left out."""
    return replace(observed, rays=match.Rays.of([]))


EYE_CHECKS = DATA / "pilot" / "eye_checks.jsonl"  # placements looked at while the user was away
CHAIN_PASSES = 5  # rounds run again without the chained photo most in doubt, at most this often


def _set_aside() -> dict[tuple[str, str], str]:
    """The placements judged wrong by eye while the user was away (EYE_CHECKS, not the
    user's verdicts: rows of `photo_id`, `inputs`, `eye` right, doubtful or wrong, `why`):
    the photo and what it was placed from, and why. The last word on each placement holds."""
    last = {}
    if EYE_CHECKS.exists():
        for line in EYE_CHECKS.read_text().splitlines():
            if line.strip() and "inputs" in (row := json.loads(line)):
                last[row["photo_id"], row["inputs"]] = row
    return {k: row.get("why", "") for k, row in last.items() if row.get("eye") == "wrong"}


@app.command("chain")
def chain_photos(
    force: Annotated[bool, typer.Option(help="sans le contrôle à réponse connue")] = False,
) -> None:
    """Situe, tour après tour, les photos que les panoramas seuls n'ont pas situées, à partir
    des vieilles photos déjà situées près d'elles (leurs ancres) : au tour k, celles des
    tours précédents. Seules les caméras sûres sont gardées ; chacune garde son tour et ses
    ancres. Puis chaque photo est re-située avec les points des autres (pas de celles situées
    depuis elle) : la photo enchaînée qui bouge le plus est écartée et les tours refaits sans
    elle ; celles du tour 0 qui bougent sont signalées, à juger. Écrit chains.json."""
    work = _Chaining()
    check = register.load(CHAIN_CHECK)
    if not force and not (check.get("passed") and check.get("setup") == work.key):
        raise typer.BadParameter(
            "le contrôle de l'enchaînement n'est pas passé : lancer `flaneur chain-check` "
            "d'abord (ou --force)"
        )
    typer.echo(f"tour 0 : {len(work.round_zero)} photos situées contre les panoramas")
    set_aside: dict[str, str] = {}
    for _ in range(CHAIN_PASSES):
        results, placed = _rounds(work, set(set_aside))
        doubts = _inconsistent(work, placed, results)
        chained = {k: why for k, why in doubts.items() if k in results}
        if not chained:
            break
        worst = max(chained, key=lambda k: doubts[k].moved)
        typer.echo(f"{worst} : en désaccord ({chained[worst]}), écartée ; tours refaits sans elle")
        set_aside[worst] = str(chained[worst])
    else:  # still in doubt after the last pass: none of them is kept
        set_aside |= {k: str(why) for k, why in chained.items()}
        results, placed = _rounds(work, set(set_aside))
    for photo_id, why in doubts.items():
        if photo_id in work.round_zero:
            typer.echo(f"{photo_id} : en désaccord avec les autres ({why})")
    write_json(
        CHAINS,
        {
            "setup": work.key,
            "photos": results,
            "doubts": {k: str(why) for k, why in doubts.items() if k in work.round_zero},
            "set_aside": set_aside,
        },
    )
    typer.echo(
        f"{len(results)} photos situées par enchaînement ; {len(set_aside)} écartées ; "
        f"écrit dans {CHAINS}"
    )


def _rounds(work: _Chaining, left_out: set[str]) -> tuple[dict[str, dict], dict[str, chain.Anchor]]:
    """The rounds of chaining from the photos placed against today (round 0), photos
    `left_out` never placed: each photo chained (a Situated's fields, its round and
    anchors), and all those placed."""
    placed, results = dict(work.round_zero), {}
    tried: dict[str, set[str]] = {}  # the anchors each photo was last tried with
    for k in range(1, chain.MOST_ROUNDS + 1):
        new, why = {}, []
        for me in work.walk:
            if me.id in placed or me.id in left_out or me.id in work.turned_down:
                continue
            anchors = work.anchors(me, placed)
            ids = {a.id for a in anchors}
            if not anchors or ids == tried.get(me.id):  # nothing new: the same verdict
                continue
            tried[me.id] = ids
            if (observed := work.observed(me)) is None:
                continue
            trace = match.Trace()
            found = work.place(me, observed, anchors, trace)
            if found.camera is not None:
                parents = [a for a in anchors if a.id in chain.provenance(found.points)]
                inputs = _placed_from(_inputs(me.id, work.setup), parents)
                if (me.id, inputs) in work.refused:
                    trace.verdict = "refusée"
                else:
                    new[me.id] = (found, inputs)
            why.append({"verdict": trace.verdict})
        typer.echo(f"tour {k} : {len(new)} photos situées ; les autres : {_funnel(why)}")
        for photo_id, (found, inputs) in new.items():
            camera, me = found.camera, work.by_id[photo_id]
            x, y = work.where[photo_id].around
            situated = match.Situated.of(
                camera,
                street=work.heights.street(camera.x, camera.y),
                walls=found.walls,
                moved_m=float(np.hypot(camera.x - x, camera.y - y)),
                inputs=inputs,
                outline=outlines.contrast(
                    _grey(cv2.imread(str(_exported(photo_id)))),
                    camera,
                    work.then(_then_year(me)).blocks,
                ),
            )
            parents = chain.provenance(found.points)
            results[photo_id] = {"situated": asdict(situated), "round": k, "from": parents}
            typer.echo(f"{photo_id} : situé au tour {k} ({camera.agree} points), depuis {parents}")
            placed[photo_id] = work.anchor(photo_id, camera, found.points, inputs)
        if not new:
            break
    return results, placed


def _moved(camera: match.View, view: match.View) -> tuple[float, float]:
    """How far a camera found stands from another, and how far it turns from it."""
    return (
        float(np.hypot(camera.x - view.x, camera.y - view.y)),
        abs(float(angle(camera.heading, view.heading))),
    )


class _Doubt(NamedTuple):
    moved: float  # metres; infinite when no longer placed
    why: str

    def __str__(self) -> str:
        return self.why


def _inconsistent(
    work: _Chaining, placed: dict[str, chain.Anchor], results: dict[str, dict]
) -> dict[str, _Doubt]:
    """The photos placed that, solved again with the others' points as well as their own
    (not those of the photos chained from them, which hold theirs), move more than CHECK_M
    or CHECK_DEG, or are no longer placed (one with no other near it is not solved again:
    nothing has changed for it)."""
    doubts = {}
    for photo_id, anchor in placed.items():
        me = work.by_id[photo_id]
        mine = _chained_from(results, photo_id)
        others = [a for a in work.anchors(me, placed) if a.id not in mine]
        if not others or (observed := work.observed(me)) is None:
            continue
        trace = match.Trace()
        camera = work.place(me, observed, others, trace).camera
        if camera is None:
            doubts[photo_id] = _Doubt(float("inf"), f"plus située : {trace.verdict}")
        elif (off := _moved(camera, anchor.view))[0] > CHECK_M or off[1] > CHECK_DEG:
            doubts[photo_id] = _Doubt(off[0], f"bouge de {off[0]:.1f} m, {off[1]:.1f}°")
    return doubts


def _chained_from(results: dict[str, dict], photo_id: str) -> set[str]:
    """The photos chained, directly or not, from `photo_id` (results in round order)."""
    found = {photo_id}
    for k, row in results.items():
        if found & set(row["from"]):
            found.add(k)
    return found - {photo_id}


@app.command("chain-check")
def chain_check(seed: int = 0) -> None:
    """Contrôle à réponse connue de l'enchaînement. Chaque photo du tour 0 est re-située à
    partir des seules autres vieilles photos (sans ses propres appariements aux panoramas) et
    comparée à sa caméra ; puis une photo prise à plus de FAR_M, donnée pour être là, ne doit
    pas être située. La porte (aucune fausse, à CHECK_M et CHECK_DEG : plus stricte que pour
    une photo seule, une erreur se propageant) est gardée : `chain` ne part que si elle passe."""
    work = _Chaining()
    rng = np.random.default_rng(seed)
    places = {me.id: np.array(work.where[me.id].around) for me in work.walk}
    rows = []
    for photo_id, truth in sorted(work.round_zero.items()):
        me = work.by_id[photo_id]
        others = work.anchors(me, work.round_zero)
        if not others or (observed := work.observed(me)) is None:
            continue
        trace = match.Trace()
        found = work.place(me, _bare(observed), others, trace)
        row = {"photo": photo_id, "kind": "seule", "anchors": len(others), "verdict": trace.verdict}
        if found.camera is not None:
            metres, degrees = _moved(found.camera, truth.view)
            row["off"] = [round(metres, 2), round(degrees, 1)]
            row["wrong"] = metres > CHECK_M or degrees > CHECK_DEG
        rows.append(row)
        far = [o for o, at in places.items() if np.hypot(*(at - places[photo_id])) >= FAR_M]
        decoy = work.by_id[far[rng.integers(len(far))]] if far else None
        if decoy is not None and (seen := work.observed(decoy)) is not None:
            trace = match.Trace()
            where = replace(work.where[photo_id], sights=())
            fooled = work.place(decoy, _bare(seen), others, trace, where).camera is not None
            rows.append(
                {
                    "photo": decoy.id,
                    "kind": "leurre",
                    "as": photo_id,
                    "wrong": fooled,
                    "verdict": trace.verdict,
                }
            )
    wrong = sum(1 for r in rows if r.get("wrong"))
    alone = [r for r in rows if r["kind"] == "seule"]
    placed_ = [r for r in alone if "off" in r]
    passed = wrong == 0
    write_json(CHAIN_CHECK, {"setup": work.key, "passed": passed, "rows": rows})
    typer.echo(
        f"re-situées par les seules autres : {len(placed_)} sur {len(alone)} "
        f"(dont fausses : {sum(1 for r in placed_ if r['wrong'])}) ; "
        f"leurres situés : {sum(1 for r in rows if r['kind'] == 'leurre' and r['wrong'])} "
        f"sur {len(rows) - len(alone)} ; porte {'passée' if passed else 'NON passée'}"
    )
    for r in placed_:
        typer.echo(f"  {r['photo']} : {r['off'][0]} m, {r['off'][1]}° ({r['anchors']} ancres)")


@app.command("match-check")
def match_check(tries: int = 20, seed: int = 0) -> None:
    """Contrôle à réponse connue, sur `tries` panoramas. Une vue découpée dans l'un, prise
    pour une vieille photo, est située contre les autres et comparée à sa vraie caméra ; puis
    la même vue dont il ne reste qu'une bande (le reste : une autre rue), vieillie ; puis une
    vue vieillie d'une autre rue, qui ne doit pas être située. La porte (assez de vues
    situées, à moins de CHECK_M et CHECK_DEG en médiane, presque aucune fausse) est gardée :
    `match` ne part que si elle passe."""
    heights, near_by, then = _matching()
    year = date.today().year
    today = then(year)
    shown = functools.partial(  # only landmarks whose being hidden can be told
        landmarks.in_view,
        origin_z=heights.origin_z,
        blocks=today.blocks,
        year=year,
        landmarks=landmarks.among(today.blocks),
    )
    views = _views()
    rng = np.random.default_rng(seed)
    marking = np.random.default_rng([seed, 1])  # its own: the fakes stay the same
    # only panoramas COLMAP placed: another's place is borrowed from its run, or its phone's
    with closing(db.connect(DB_PATH)) as conn:
        fixed = _fixed_by_colmap(panoramas.load(conn), register.load(FIXES))
    placed = [p for p in near_by if p.id in fixed]
    tries = min(tries, len(placed))
    if not tries:
        raise typer.BadParameter("aucun panorama placé par COLMAP : rien à contrôler")
    found: dict[str, list[Off | None]] = {"vue": [], "bande": [], "leurre": []}
    traces: dict[str, list[dict]] = {kind: [] for kind in found}
    for k in rng.choice(len(placed), tries, replace=False):
        me = placed[k]
        heading = float(rng.uniform(0, 360))
        far = [p for p in near_by if np.hypot(p.x - me.x, p.y - me.y) > FAR_M]
        if not far:
            raise typer.BadParameter(f"aucun panorama à plus de {FAR_M} m : pas de leurre")
        true = _cut_view(me, heading)
        there, facing = far[rng.integers(len(far))], float(rng.uniform(0, 360))
        elsewhere = _cut_view(there, facing)
        keep = rng.uniform(*CHECK_KEPT)
        span = (0.0, keep) if rng.random() < 0.5 else (1 - keep, 1.0)  # one side or the other
        band = fakes.aged(fakes.mixed(true, elsewhere, span), rng)
        others = _nearest(near_by, me.x, me.y, but=me.id)[:MATCH_PANORAMAS]
        # the landmarks each fake shows, marked as by hand (as near): the band's are the true
        # view's inside it (the rest stands for houses rebuilt since, not for another place)
        seen = _marked_by_hand(_check_view(me, heading), shown, marking)
        seen_there = _marked_by_hand(_check_view(there, facing), shown, marking)
        sights = (seen, tuple(s for s in seen if span[0] <= s.column <= span[1]), seen_there)
        line = []
        pictures = (true, band, fakes.aged(elsewhere, rng))
        for kind, fake, marked in zip(found, pictures, sights, strict=True):
            where = match.Where(heights.streets, today.free, (me.x, me.y), marked)
            trace = match.Trace()
            camera, _, _ = match.locate(
                fake, others, today.blocks, views, where, trace, heights.origin_z
            )
            traces[kind].append(_traced(trace, None if kind == "leurre" else (me, heading)))
            if camera is None:
                found[kind].append(None)
                line.append(f"{kind} rien")
                continue
            off = Off(
                *_off(me, heading, camera.x, camera.y, camera.heading),
                outlines.contrast(_grey(fake), camera, today.blocks),
            )
            found[kind].append(off)
            line.append(f"{kind} {off.metres:.1f} m {off.degrees:.1f}° contours {off.outline:.2f}")
        typer.echo(f"{me.id[:8]} : " + ", ".join(line))

    def right(off: Off | None) -> bool:
        return off is not None and _right(off.metres, off.degrees)

    attempts = len(found) * tries
    wrong = [
        off
        for kind, offs in found.items()
        for off in offs
        if off is not None and (kind == "leurre" or not right(off))
    ]
    situated = [off for off in found["vue"] if off is not None]
    metres = register.median([off.metres for off in situated])
    degrees = register.median([off.degrees for off in situated])
    bands = sum(right(off) for off in found["bande"])
    wanting = [
        need
        for met, need in [
            (len(situated) >= CHECK_SHARE * tries, f"au moins {CHECK_SHARE:.0%} situées"),
            (metres < CHECK_M, f"moins de {CHECK_M} m"),
            (degrees < CHECK_DEG, f"moins de {CHECK_DEG}°"),
            (len(wrong) <= WRONG_SHARE * attempts, f"au plus {WRONG_SHARE:.0%} de fausses"),
        ]
        if not met
    ]
    passed = not wanting
    write_json(CHECK_TRACE, traces)
    for kind, rows in traces.items():
        typer.echo(f"{kind} : {_funnel(rows)}")
        lost = [r for r in rows if r["verdict"] != "située"]
        if kind == "leurre" or not lost:
            continue
        # the right camera of a photo lost, and what turned it down (a rival: it passed)
        thrown = Counter(
            next(c["why"] or r["verdict"] for c in r["candidates"] if c["right"])
            for r in lost
            if any(c["right"] for c in r["candidates"])
        )
        typer.echo(
            f"  {kind} non situées : {len(lost)}, dont la juste caméra écartée : "
            f"{thrown.total()} ({', '.join(f'{why} {n}' for why, n in thrown.most_common())})"
        )
    write_json(
        MATCH_CHECK,
        {
            "setup": _setup(),
            "situated": len(situated),
            "tries": tries,
            "median_m": metres,
            "median_deg": degrees,
            "bands_right": bands,
            "wrong": len(wrong),
            "passed": passed,
            # how much better the outlines fit the cameras found right, and the wrong ones
            "outlines": {
                "right": sorted(
                    round(off.outline, 3)
                    for kind, offs in found.items()
                    for off in offs
                    if kind != "leurre" and right(off)
                ),
                "wrong": sorted(round(off.outline, 3) for off in wrong),
            },
        },
    )
    typer.echo(
        f"vues : {len(situated)} situées sur {tries}, écart médian {metres:.1f} m et "
        f"{degrees:.1f}° ; bandes vieillies justes : {bands} sur {tries} ; fausses : "
        f"{len(wrong)} sur {attempts} : "
        + ("porte passée" if passed else f"porte NON passée (il faut {', '.join(wanting)})")
    )


def _traced(trace: match.Trace, truth: tuple[match.Panorama, float] | None = None) -> dict:
    """A photo's trace, as kept on disk; with its true camera (a panorama and the heading the
    view was cut at), how far each camera checked is from it, and whether it is right."""
    row = asdict(trace)
    for candidate in row["candidates"] if truth is not None else []:
        metres, degrees = _off(*truth, candidate["x"], candidate["y"], candidate["heading"])
        candidate.update(
            off_m=round(metres, 2), off_deg=round(degrees, 2), right=_right(metres, degrees)
        )
    return row


def _off(
    panorama: match.Panorama, heading: float, x: float, y: float, found: float
) -> tuple[float, float]:
    """How far a camera found at `x`, `y`, facing `found`, is from the true one: a panorama,
    with the heading a view of it was cut at (metres, degrees)."""
    return float(np.hypot(x - panorama.x, y - panorama.y)), abs(angle(heading, found))


def _right(metres: float, degrees: float) -> bool:
    """Whether a camera this far from the true one counts as right."""
    return bool(metres <= RIGHT_M and degrees <= RIGHT_DEG)


def _funnel(rows: Iterable[dict]) -> str:
    """How many photos stopped at each step, from their traces (as `_traced` keeps them)."""
    verdicts = Counter(row["verdict"] for row in rows)
    return " ; ".join(f"{verdict} {n}" for verdict, n in verdicts.most_common())


class Off(NamedTuple):
    """How far a camera found in the known-answer check is from the true one, and how much
    better the outlines fit it than nearby cameras."""

    metres: float
    degrees: float
    outline: float


def _grey(photo: np.ndarray) -> np.ndarray:
    """A photo (BGR) in grey, at the size it is matched at."""
    return cv2.cvtColor(match.shrunk(photo), cv2.COLOR_BGR2GRAY)


def _marked_by_hand(
    view: match.View,
    shown: Callable[[match.View], tuple[match.Sight, ...]],
    rng: np.random.Generator,
) -> tuple[match.Sight, ...]:
    """The landmarks a view of the check shows, each marked as near as by hand."""
    return tuple(replace(s, column=s.column + rng.normal(0, MARKED_TO)) for s in shown(view))


def _check_view(panorama: match.Panorama, heading: float) -> match.View:
    """The camera of a view the known-answer check cuts (see _cut_view)."""
    return match.View(panorama.x, panorama.y, panorama.z, heading, 0.5, CHECK_FOV)


def _cut_view(panorama: match.Panorama, heading: float) -> np.ndarray:
    """A level view of a panorama, as the known-answer check takes for an old photo."""
    picture = cv2.imread(str(panorama.path))
    return match.perspective(picture, panorama.axes, heading, fov=CHECK_FOV, side=800, pitch=0)


def _sheet_row(
    photo_id: str,
    m: match.Situated,
    heights: scene.CameraHeights,
    blocks: Blocks,
    near_by: list[match.Panorama],
    read: Callable[[Path], np.ndarray | None],
    width: int = SHEET_COLUMN,
) -> np.ndarray | None:
    """A photo situated, then the true view of a panorama nearby framed as it, today's street
    rebuilt from its camera (hatched: what no panorama sees), and the photo laid over it;
    each `width` wide."""
    old = cv2.imread(str(_exported(photo_id)))
    if old is None:
        return None
    old = _to_width(old, width)
    size = old.shape[:2]
    view = m.view(heights.street(m.x, m.y))
    around = [
        (p, picture)
        for p in _nearest(near_by, m.x, m.y)[:REBUILT_FROM]
        if (picture := read(p.path)) is not None
    ]
    if not around:
        return None
    rebuilt, source = match.today_from(view, size, around, blocks, heights.streets)
    unseen = source == match.UNSEEN
    # the true view: from the panorama that gave the rebuilt view the most
    most = (
        int(np.bincount(source[source >= 0], minlength=len(around)).argmax())
        if (source >= 0).any()
        else 0
    )
    panorama, picture = around[most]
    real = match.real_view(view, size, panorama, picture)
    overlay = cv2.addWeighted(old, 0.55, rebuilt, 0.45, 0)
    overlay[unseen] = old[unseen]  # where nothing of today is known: the old photo alone
    rebuilt[unseen] = 128  # mid-grey between the stripes: not the sky, not a shadow
    rebuilt[unseen & _hatching(size)] = UNSEEN_COLOUR
    row = np.hstack([old, real, rebuilt, overlay])
    away = np.hypot(panorama.x - m.x, panorama.y - m.y)
    for column, text in enumerate(
        [
            f"{photo_id}  {m.agree} pts, {m.panoramas} sources, moved {m.moved_m:.0f} m",
            f"real view, from {away:.0f} m away",
            "rebuilt from the camera found",
            "overlay",
        ]
    ):
        for colour, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):  # outlined
            cv2.putText(
                row,
                text,
                (column * width + 6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                thickness,
            )
    return row


@app.command("match-sheet")
def match_sheet(per_page: int = 10) -> None:
    """Planches de contrôle, une ligne par photo située : la photo ; la vraie vue d'un panorama
    voisin, cadrée comme elle ; la rue d'aujourd'hui recomposée depuis sa caméra retrouvée
    (hachures : ce qu'aucun panorama ne voit) ; la photo posée dessus."""
    heights, near_by, then = _matching()
    blocks = then(date.today().year).blocks
    if not near_by:
        raise typer.BadParameter("aucun panorama téléchargé")
    read = functools.lru_cache(maxsize=REBUILT_FROM)(lambda path: cv2.imread(str(path)))
    rows = [
        row
        for photo_id, m in sorted(_situated().items())
        if (row := _sheet_row(photo_id, m, heights, blocks, near_by, read)) is not None
    ]
    for stale in (DATA / "pilot").glob("match_sheet_*.jpg"):  # an earlier run's pages
        stale.unlink()
    for page in range(0, len(rows), per_page):
        out = DATA / "pilot" / f"match_sheet_{page // per_page + 1}.jpg"
        cv2.imwrite(
            str(out), np.vstack(rows[page : page + per_page]), [cv2.IMWRITE_JPEG_QUALITY, 82]
        )
    typer.echo(f"{len(rows)} photos situées ; planches dans {DATA / 'pilot'}/match_sheet_*.jpg")


def _chained() -> dict:
    """What the last chaining found (chains.json), if from the setup as it is now: the photos
    it `photos` (each its `situated`, a Situated's fields; `round`; the photos it was placed
    `from`), and the `doubts` (why each)."""
    kept = register.load(CHAINS)
    return kept if kept.get("setup") == _chain_setup() else {}


@app.command("confirmer")
def confirm_placed(port: int = 8768, open_browser: bool = True) -> None:
    """Ouvre la page de confirmation (locale) : oui ou non à chaque photo située, contre les
    panoramas ou par enchaînement, vue à côté de la rue d'aujourd'hui refaite depuis sa
    caméra. Un « non » la retire ; l'enchaînement suivant re-situe sans elle les photos
    qu'elle avait aidé à situer."""
    heights, near_by, then = _matching()
    blocks = then(date.today().year).blocks
    with closing(db.connect(DB_PATH)) as conn:
        titles = {me.id: me.title for me in export.photos(conn, DATA / "pilot", REVIEWS)}
    situated = {k: m for k, m in _situated().items() if k in titles}
    rounds = {k: {"round": 0, "from": []} for k in situated}
    chained = _chained()
    for k, row in chained.get("photos", {}).items():
        if k in titles:
            situated[k] = match.Situated(**row["situated"])
            rounds[k] = {"round": row["round"], "from": row["from"]}
    for k, why in chained.get("doubts", {}).items():
        if k in rounds:
            rounds[k]["doubt"] = why
    for (k, inputs), why in _set_aside().items():
        if k in rounds and situated[k].inputs == inputs:
            rounds[k]["doubt"] = f"jugée fausse à l'œil : {why}"
    placed = [
        {
            "id": k,
            "title": titles[k],
            "inputs": m.inputs,
            "camera": {"x": round(m.x, 2), "y": round(m.y, 2), "heading": round(m.heading, 2)},
            "agree": m.agree,
            "moved_m": m.moved_m,
            **rounds[k],
        }
        for k, m in situated.items()
    ]
    read = functools.lru_cache(maxsize=16)(lambda path: cv2.imread(str(path)))

    @functools.lru_cache(maxsize=64)
    def sheet(photo_id: str) -> bytes:
        row = _sheet_row(photo_id, situated[photo_id], heights, blocks, near_by, read, width=480)
        if row is None:
            return b""
        return cv2.imencode(".jpg", row, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    _serve_page(
        lambda: confirming.serve(placed, ANCHORS, _verdicts, sheet, port),
        f"photos situées ({len(placed)})",
        port,
        open_browser,
    )


PROPOSALS = DATA / "pilot" / "proposals_3d.json"  # each photo proposed in its nearest panorama
PROPOSING_M = placing.NEAR_M  # a panorama farther from the photo's place is not tried


def _spheres() -> list[match.Panorama]:
    """The panoramas where they were taken, levelled, for cutting views (no height needed)."""
    with closing(db.connect(DB_PATH)) as conn:
        pictures = _lined_up(panoramas.load(conn))
    return _panoramas(pictures, lambda x, y: 0.0)


@app.command("proposer")
def propose_in_panoramas(sheets: int = 20) -> None:
    """Propose chaque photo de la balade dans le panorama le plus proche de sa place (à moins de
    40 m), et mesure : combien en ont une, en combien de secondes ; des planches (photo, vue
    d'aujourd'hui, la photo posée dedans) dans data/pilot/proposals_3d/ pour les regarder."""
    with closing(db.connect(DB_PATH)) as conn:  # where each photo is best known now
        walk = _walk(conn)
    spheres = _spheres()
    views = _views(kept=4)  # each photo is tried in one panorama, photos near one another in turn
    store = chain.Store(DATA / "raw", views.matcher)  # the photos' features, as matching reads them
    out = PROPOSALS.with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):  # a sheet is this run's, or none
        old.unlink()
    found, times = {}, []
    try:
        for me in walk:
            here = to_metres(Point(me.lon, me.lat))
            near = _nearest(spheres, here.x, here.y, PROPOSING_M)
            image = cv2.imread(str(_exported(me.id)))
            if not near or image is None:
                continue
            home = near[0]
            photo = store.features(me.id, _image(me.id), lambda image=image: match.shrunk(image))
            cached = views.of(home)
            started = time.monotonic()  # the proposal alone, whatever was cached before
            proposal = proposing.propose(
                photo, home.id, home.axes, cached, me.heading, views.matcher
            )
            times.append(time.monotonic() - started)
            found[me.id] = {
                "panorama": home.id,
                "inputs": _image(me.id),  # the image it was proposed on
                "metres": round(float(np.hypot(home.x - here.x, home.y - here.y)), 1),
                "registration": proposal and registration.to_json(proposal),
            }
            if proposal and sheets > 0:
                sheets -= 1
                _proposal_sheet(out / f"{pilot.safe_name(me.id)}.jpg", image, home, proposal)
    finally:  # what was found is kept, even when a photo stops the run
        write_json(PROPOSALS, found)
    made = sum(v["registration"] is not None for v in found.values())
    timing = f"{np.median(times):.2f} s par photo (médiane)" if times else "rien de mesuré"
    typer.echo(
        f"{len(found)} photos à moins de {PROPOSING_M} m d'un panorama ; une proposition pour "
        f"{made} ; {timing} ; planches dans {out}"
    )


def _proposal_sheet(
    out: Path, photo: np.ndarray, home: match.Panorama, proposal: registration.Registration
) -> None:
    """The photo, today's view it was proposed in, and the photo laid in that view."""
    picture, view = cv2.imread(str(home.path)), proposal.view
    if picture is None:
        return
    cut = match.perspective(picture, home.axes, view.heading, view.fov, proposing.SIDE, view.pitch)
    laid = registration.composite(cut, photo, proposal, view, home.axes)
    row = [_to_width(image, SHEET_COLUMN) for image in (photo, cut, laid)]
    height = max(image.shape[0] for image in row)
    cv2.imwrite(
        str(out), np.hstack([np.pad(i, ((0, height - i.shape[0]), (0, 0), (0, 0))) for i in row])
    )


CARRIED = DATA / "pilot" / "carried.json"  # each photo placed, on its neighbouring panoramas


def _features(store: chain.Store, photo_id: str, extract: bool = True) -> match.Features | None:
    """A photo's features as matching reads them (see chain.Store.features); without
    `extract`, only if kept."""
    path = _exported(photo_id)
    read = (lambda: match.shrunk(cv2.imread(str(path)))) if extract else None
    return store.features(photo_id, _image(photo_id), read)


def _answered_as(row: dict) -> str:
    """What a copy or a proposal was made from: the answer's registration on its image, the way
    round it was shown."""
    made = json.dumps([row["registration"], row["inputs"], row["flipped"]], sort_keys=True)
    return hashlib.sha1(made.encode()).hexdigest()[:16]


def _uncompared(
    walk: list[export.WalkPhoto],
    images: dict[str, str],
    copies: dict[str, list[str]],
    radius_m: float = MATCH_RADIUS_M,
) -> list[tuple[str, str]]:
    """The pairs of the walk's photos `flaneur voisins` compares that were never compared
    (`images`: each photo's image's hash; `copies`: the prints of each negative)."""
    kept = {path.name for path in (DATA / "raw").glob("old_pairs_*/*.npy")}  # any precision
    return carrying.to_compare(
        {me.id: _place_of(me) for me in walk},
        radius_m,
        copies,
        lambda a, b: chain.pair_name((a, images[a]), (b, images[b])) in kept,
    )


@app.command("voisins")
def compare_neighbours(radius_m: float = MATCH_RADIUS_M) -> None:
    """Compare une fois chaque paire de photos anciennes à moins de `radius-m` l'une de l'autre
    (à leur meilleure place connue, jamais deux tirages d'un négatif) : une photo placée
    propose alors aussitôt toutes celles qui partagent assez de détails avec elle, et la file
    met d'abord celles qui en portent le plus. Chaque paire est gardée aussitôt comparée ; à
    relancer quand des photos ont été placées (elles ont bougé), le placeur le signale."""
    with closing(db.connect(DB_PATH)) as conn:
        walk = [me for me in _walk(conn) if _exported(me.id).exists()]
        copies = dedup.negatives(conn)
    images = {me.id: _image(me.id) for me in walk}
    raw = DATA / "raw"
    todo = _uncompared(walk, images, copies, radius_m)
    typer.echo(f"{len(todo)} paires à comparer")
    if todo:
        store = chain.Store(raw, _views(kept=1).matcher)
        started = time.monotonic()
        for n, (a, b) in enumerate(todo, 1):
            store.pairs((a, images[a]), (b, images[b]), lambda i: _features(store, i))
            if n % 500 == 0 or n == len(todo):
                typer.echo(f"{n} / {len(todo)} ({(time.monotonic() - started) / n:.3f} s la paire)")
    graph = chain.old_neighbours(raw, images, queueing.MIN_SHARED)
    typer.echo(
        f"{len(graph)} photos sur {len(images)} partagent au moins {queueing.MIN_SHARED} "
        f"détails avec une autre ({sum(map(len, graph.values())) // 2} paires)"
    )


@app.command("porter")
def carry_placed() -> None:
    """Porte chaque photo placée dans son panorama sur les panoramas voisins qui voient les
    mêmes façades, de proche en proche le long de la rue, s'estompant au loin
    (data/pilot/carried.json) ; seules les photos répondues depuis la dernière fois sont
    refaites."""
    held = _on_image_now(_placing_answers()[1])  # those placed on another image: not exported
    carried = _carried(held)
    todo = [i for i in held if i not in carried]
    if todo:
        with closing(db.connect(DB_PATH)) as conn:
            pictures = _lined_up(panoramas.load(conn))
        spheres = {p.id: p for p in _panoramas(pictures, lambda x, y: 0.0)}
        street = panoramas.links(pictures)  # the next ones along the street, every way
        views = _views(kept=8)
        for photo_id in todo:
            r = placings.registration_of(held[photo_id])
            home = spheres.get(r.panorama)
            copies = [] if home is None else _to_neighbours(r, home, spheres, street, views)
            carried[photo_id] = {"made": _answered_as(held[photo_id]), "copies": copies}
            write_json(CARRIED, carried)  # kept photo after photo: a stop loses one at most
    write_json(CARRIED, carried)
    counts = [len(c["copies"]) for c in carried.values()]
    typer.echo(
        f"{len(carried)} photos placées ({len(todo)} refaites) portées sur {sum(counts)} "
        f"panoramas voisins ({np.median(counts) if counts else 0:.0f} par photo, médiane)"
    )


def _copy_panoramas(holding: set[str], folder: Path) -> None:
    """The panoramas holding a photo, copied onto the site (again when changed); the others
    copied before, removed."""
    folder.mkdir(parents=True, exist_ok=True)
    for panorama in holding:
        source, copy = PANORAMAS / f"{panorama}.jpg", folder / f"{panorama}.jpg"
        if not copy.exists() or copy.stat().st_mtime < source.stat().st_mtime:
            shutil.copy2(source, copy)
    for old in folder.glob("*.jpg"):
        if old.stem not in holding:
            old.unlink()


def _to_neighbours(
    r: registration.Registration,
    home: match.Panorama,
    spheres: dict[str, match.Panorama],
    street: dict[str, list[tuple[str, float, float]]],
    views: match.Views,
) -> list[dict]:
    """A photo placed on `home`, carried outward along the street to the panoramas that see
    the same details (the home view's under the photo, matched in their views around its
    direction, the nearest first)."""
    n = proposing.around(r.view.heading)[1]  # the kept view it was placed in, or the nearest
    home_view = registration.ViewOf(heading=float(match.VIEW_HEADINGS[n]))
    seen = views.of(home)
    if not seen:
        return []
    kept = seen[n]
    try:
        inverse = np.linalg.inv(registration.in_view(r, home_view, home.axes))
    except np.linalg.LinAlgError:
        return []
    if not len(kept.keypoints):  # a view with no details (sky): nothing to carry it by
        return []
    under = cv2.perspectiveTransform(
        (kept.keypoints / proposing.SIDE)[None].astype(float), inverse
    )[0]
    inside = ((under >= 0) & (under <= 1)).all(axis=1)  # on the photo, warped into the view
    patch = match.Features(kept.keypoints[inside], kept.descriptors[inside], kept.size)
    left, nearest, right = proposing.around(home_view.heading)  # the nearest tried first

    def carry(panorama: str) -> registration.Registration | None:
        sphere = spheres[panorama]
        theirs = views.of(sphere)
        for m in (nearest, left, right) if theirs else ():
            pairs = views.matcher.pairs(patch, theirs[m])
            copy = carrying.carried_today(
                r,
                home_view,
                home.axes,
                patch.keypoints[pairs[:, 0]],
                theirs[m].keypoints[pairs[:, 1]],
                proposing.SIDE,
                panorama,
                registration.ViewOf(heading=float(match.VIEW_HEADINGS[m])),
                sphere.axes,
            )
            if copy is not None:
                return copy
        return None

    def links(panorama: str) -> list[tuple[str, float]]:
        return [(q, metres) for q, _, metres in street.get(panorama, []) if q in spheres]

    def from_home(panorama: str) -> float:
        p = spheres[panorama]
        return float(np.hypot(p.x - home.x, p.y - home.y))

    return [
        {"registration": registration.to_json(copy), "strength": carrying.strength(from_home(q))}
        for q, copy in carrying.outward(home.id, links, from_home, carry)
    ]


@app.command("parallaxe")
def compare_parallax(answers: Path = PLACINGS) -> None:
    """Compare, sur les photos posées par points (leurs paires gardées), les deux recalages
    refaits : à plat (l'appareil au panorama) et en parallaxe (l'appareil où se tenait le
    photographe, cherché depuis là où son recalage était parti). Pour chacune, l'erreur médiane
    des paires avec profondeur, chacune refaite sans elle (part de la diagonale de la photo ;
    un recalage refusé compte comme le pire), et l'écart entre la place trouvée et le départ.
    Avec le modèle (profondeur des panoramas voisins)."""
    rows = [r for r in placings.holding(answers).values() if r.get("pairs") and r["registration"]]
    if not rows:
        typer.echo("aucune réponse par points avec ses paires")
        return
    spheres = _spheres()
    with closing(db.connect(DB_PATH)) as conn:
        fixed = _fixed_by_colmap(panoramas.load(conn), register.load(FIXES))
    views = _views(kept=8)  # a panorama and its neighbours
    made = _Depths(spheres, fixed, lambda: views, threading.Lock())
    found, skipped = [], 0
    for row in sorted(rows, key=lambda r: r["registration"]["panorama"]):  # its views kept
        panorama, image = row["registration"]["panorama"], _exported(row["photo_id"])
        try:
            if not image.exists() or row.get("inputs") != _image(row["photo_id"]):
                raise ValueError("son image a changé depuis (recadrée) : à refaire")
            depths = made.around(panorama, row["pairs"]["view"]["heading"])
            if depths is None:
                raise ValueError("pas de panorama voisin")
            with Image.open(image) as photo:
                size = photo.size[::-1]  # height, width
            one = placing.compared(row, made.by_id[panorama], size, depths)
        except Exception as error:  # the matcher, a fit refused: this photo, not the others
            typer.echo(f"{row['photo_id']} : laissée de côté ({error})")
            skipped += 1
            continue
        found.append(one)
        stood = "refusée" if one["moved_m"] is None else f"à {one['moved_m']:.1f} m du départ"
        typer.echo(
            f"{one['photo_id']} (gardée {one['fit']}) : {one['with_depth']}/{one['pairs']} "
            f"paires avec profondeur, à plat {one['flat']:.1%}, en parallaxe "
            f"{one['parallax']:.1%} ; la caméra {stood} {one['why']}".rstrip()
        )
    if found:
        flat, deep = (np.median([f[k] for f in found]) for k in ("flat", "parallax"))
        typer.echo(
            f"{len(found)} photos ({skipped} laissées de côté) : médiane à plat {flat:.1%} "
            f"({flat * 1000:.0f} px sur 1000 de diagonale), en parallaxe {deep:.1%} "
            f"({deep * 1000:.0f} px) ; parallaxe meilleure sur "
            f"{sum(f['parallax'] < f['flat'] for f in found)}, refusée sur "
            f"{sum(f['moved_m'] is None for f in found)}"
        )


@app.command("placer")
def place_photos(
    port: int = 8772, open_browser: bool = True, live: bool = True, answers: Path = PLACINGS
) -> None:
    """Ouvre la page pour placer les photos (locale) : sur le plan, puis dans leur panorama (la
    proposition vérifiée, ou des points cliqués par paires), ou « trop loin ». `--no-live` :
    seulement les propositions déjà calculées (`flaneur proposer`) ou portées depuis une photo
    placée, sans charger le modèle ; `--answers` : un autre fichier de réponses (pour essayer
    la page)."""
    with closing(db.connect(DB_PATH)) as conn:  # where each photo is best known now
        walk = _walk(conn, _placing_answers(answers)[0])
        copies = dedup.negatives(conn)
        fixed = _fixed_by_colmap(panoramas.load(conn), register.load(FIXES))  # trusted first
    photos = {
        me.id: {
            "title": me.title,
            "years": me.years,
            "lon": me.lon,
            "lat": me.lat,
            "heading": me.heading,
            "image": _exported(me.id),
            "inputs": _image(me.id),  # answers say which image they were made on
        }
        for me in walk
        if _exported(me.id).exists()
    }
    spheres = _spheres()
    by_id = {p.id: p for p in spheres}
    computed = json.loads(PROPOSALS.read_text()) if PROPOSALS.exists() else {}
    store = chain.Store(DATA / "raw", None)  # the photos' features: a matcher once one is loaded
    raw = DATA / "raw"
    images = {i: p["inputs"] for i, p in photos.items()}
    graph = chain.old_neighbours(raw, images, queueing.MIN_SHARED)
    left = _uncompared([me for me in walk if me.id in photos], images, copies)
    if left:  # their proposals and their place in the queue wait on them
        typer.echo(
            f"{len(left)} paires de photos proches jamais comparées : `flaneur voisins` pour "
            "que chaque photo placée propose aussitôt toutes ses voisines"
        )

    @functools.cache
    def models() -> match.Views:  # loaded when first needed
        views = _views(kept=8)  # a panorama and its neighbours, whose depth is made together
        store.matcher = views.matcher
        return views

    made: dict[tuple[str, str], dict[str, dict]] = {}  # by source and the answer it holds

    def from_source(source: str, row: dict, *rest) -> dict[str, dict]:  # kept while it holds
        key = (source, _answered_as(row))
        if key not in made:
            made[key] = carrying.from_source(source, row, *rest)
        return made[key]

    @functools.lru_cache(maxsize=1)
    def carried_at(answered: tuple[float, int]) -> dict[str, dict]:  # made again on new answers
        return carrying.proposals(
            placings.holding(answers),
            graph,
            lambda a, b: chain.kept_pairs(raw, (a, images[a]), (b, images[b])),
            lambda i: _features(store, i, extract=False) if i in photos else None,
            lambda panorama: by_id[panorama].axes if panorama in by_id else None,
            images,
            from_source,
        )

    def carried_now() -> dict[str, dict]:
        stat = answers.stat() if answers.exists() else None
        return carried_at((stat.st_mtime, stat.st_size) if stat else (0.0, 0))

    def proposal(photo_id: str, panorama: str) -> dict | None:
        """Computed by `flaneur proposer` for this panorama, else (live) made now (those carried
        from a photo placed come first: see placing.serve)."""
        known = computed.get(photo_id)
        if known and (known["panorama"], known.get("inputs")) == (panorama, _image(photo_id)):
            found = placings.registration_of(known)
        elif live and panorama in by_id and photo_id in photos:
            views = models()
            home = by_id[panorama]
            found = proposing.propose(
                _features(store, photo_id),
                panorama,
                home.axes,
                views.of(home),
                photos[photo_id]["heading"],
                views.matcher,
            )
        else:
            found = None
        return found and {"registration": found, "carried_from": None}

    revisit = {i: "revoir" for i, v in _verdicts().items() if v["verdict"] == "yes"}  # a camera

    def ahead() -> dict[str, str]:  # earlier yes verdicts, then photos carried from placed ones
        return {**revisit, **{i: "portée" for i in carried_now() if i not in revisit}}

    def full(panorama: str) -> np.ndarray | None:
        """The panorama's picture at most placing.FULL_WIDTH px around, from its original
        (each instance of Panoramax keeps it its own way: asked), decoded straight at the
        smallest scale at least that large; none offline or once no longer on Panoramax."""
        try:
            url = panoramax.by_id(client, panorama).hd_url
            data = client.get(url).raise_for_status().content if url else b""
            with Image.open(io.BytesIO(data)) as original:
                original.draft("RGB", (placing.FULL_WIDTH, placing.FULL_WIDTH // 2))
                rgb = np.asarray(original.convert("RGB"))
        # offline, gone from Panoramax, an answer not as expected, no picture or one too large
        # in what came: the page shows the panorama's own picture
        except (
            httpx.HTTPError,
            IndexError,
            KeyError,
            ValueError,
            OSError,
            Image.DecompressionBombError,
        ):
            return None
        if rgb.shape[1] != 2 * rgb.shape[0]:  # not all around: never stretched into it
            return None
        if rgb.shape[1] > placing.FULL_WIDTH:
            size = (placing.FULL_WIDTH, placing.FULL_WIDTH // 2)
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(rgb[:, :, ::-1])  # as OpenCV reads pictures

    matching = threading.Lock()  # the matcher: proposals, and depth when it must be made
    depths_made = _Depths(spheres, fixed, models, matching)

    def depths(panorama: str, heading: float) -> depth.Depths | None:
        """The depth of today's details around a panorama the way a view faces, kept per view
        as the view turns (see _Depths); live only."""
        if not live:
            raise ValueError("pas de parallaxe sans le modèle : relancer sans --no-live")
        return depths_made.around(panorama, heading)

    sources = placing.Sources(
        photos,
        spheres,
        answers,
        proposal,
        ahead,
        graph,
        carried_now,
        full=full,
        depths=depths,
        matching=matching,
    )
    with httpx.Client(timeout=10, follow_redirects=True) as client:
        _serve_page(
            lambda: placing.serve(sources, port),
            f"placer ({len(photos)} photos)",
            port,
            open_browser,
        )


@app.command("scene")
def write_scene() -> None:
    """Écrit la scène de la balade en 3D : les panoramas d'aujourd'hui, et les photos que
    l'utilisateur a placées dans l'un d'eux, avec leurs copies portées sur les voisins
    (`flaneur porter`) ; copie sur le site les panoramas qui portent une photo."""
    places, in_3d, _ = _placing_answers()
    with closing(db.connect(DB_PATH)) as conn:
        walk = _walk(conn, places)
        pictures = _lined_up(panoramas.load(conn))  # refused when tilts were measured otherwise
    site = json.loads((WEB_DATA / "walk.json").read_text())
    marked = {p["id"] for p in site["photos"] if p["in_3d"]}  # by the export, on its images
    shown = {i: r for i, r in in_3d.items() if i in marked}
    copies = {i: c["copies"] for i, c in _carried(shown).items()}
    built, holding = scene.build3d(
        [p for p in pictures if (PANORAMAS / f"{p.id}.jpg").exists()], shown, copies, walk
    )
    _copy_panoramas(holding, WEB_DATA / "panoramas")
    in_scene = {p["id"] for p in built["photos"]}
    if in_scene != marked:  # a photo whose panorama is gone: no "En 3D" leading nowhere
        for p in site["photos"]:
            p["in_3d"] = p["id"] in in_scene
        write_json(WEB_DATA / "walk.json", site)
    write_json(WEB_DATA / "scene.json", built)
    weight = sum(f.stat().st_size for f in WEB_DATA.rglob("*") if f.is_file()) / 2**20
    typer.echo(
        f"{len(built['panoramas'])} panoramas, {len(built['photos'])} photos, "
        f"{len(holding)} panoramas copiés ; données du site : {weight:.0f} Mo ; "
        f"écrit dans {WEB_DATA / 'scene.json'}"
    )


@app.command("review")
def review_placements(port: int = 8765, open_browser: bool = True) -> None:
    """Ouvre la page de relecture (locale) : valider ou corriger chaque placement."""
    with httpx.Client(timeout=300) as client:
        gazetteer = Gazetteer.load(client, DATA / "raw" / "gazetteer")
    with closing(db.connect(DB_PATH)) as conn:
        _serve_page(
            lambda: review.serve(conn, DATA / "pilot", gazetteer, REVIEWS, port),
            "relecture",
            port,
            open_browser,
        )


@app.command("reperes")
def mark_landmarks(port: int = 8766, open_browser: bool = True) -> None:
    """Ouvre la page des repères (locale) : marquer sur chaque photo les monuments qu'elle
    montre (lequel, sa colonne, son sommet)."""
    with closing(db.connect(DB_PATH)) as conn:
        walk = export.photos(conn, DATA / "pilot", REVIEWS)
    photos = [
        {
            "id": me.id,
            "title": me.title,
            "years": me.years,
            "place": [(here := to_metres(Point(me.lon, me.lat))).x, here.y],
        }
        for me in walk
        if _exported(me.id).exists()
    ]
    _serve_page(
        lambda: marking.serve(photos, SIGHTS, _exported, port),
        f"repères ({len(photos)} photos)",
        port,
        open_browser,
    )


# photos looked at for landmarks: their image's hash, and the references' as used (looked at
# again when either changes)
PROPOSED = DATA / "raw" / "proposed.json"
REFERENCES = DATA / "overrides" / "landmark_refs.jsonl"  # today's pictures of the landmarks,
REFERENCE_PICTURES = DATA / "raw" / "landmark_refs"  # boxed by the user (see proposals)


@app.command("reperes-proposer")
def propose_landmarks(
    write: Annotated[bool, typer.Option(help="écrire les propositions (sinon : mesurer)")] = False,
) -> None:
    """Cherche les repères dans les photos : ceux debout à leur date près de leur lieu,
    comparés à leurs photos d'aujourd'hui (data/overrides/landmark_refs.jsonl). Sans
    --write : mesure la justesse sur les photos que l'utilisateur a marquées (ou dites sans
    repère). Avec --write : propose des repères, « À vérifier », pour les photos qu'il n'a
    pas encore vues (une fois par image)."""
    if not REFERENCES.exists():
        raise typer.BadParameter(f"pas de références des repères ({REFERENCES})")
    matcher = _views().matcher
    references = proposals.References(REFERENCES, REFERENCE_PICTURES, matcher)
    if references.missing:
        typer.echo(f"{len(references.missing)} références absentes du disque : ignorées")
    with closing(db.connect(DB_PATH)) as conn:
        walk = export.photos(conn, DATA / "pilot", REVIEWS)
    rows = marking.by_photo(SIGHTS)
    looked = register.load(PROPOSED)
    # the references as used (the file, and the pictures missing), the method and buildings
    used = "\n".join([proposals.METHOD, repr(landmarks.BUILDINGS), *sorted(references.missing)])
    used = used.encode()
    used += REFERENCES.read_bytes()
    against = hashlib.sha1(used).hexdigest()[:16]
    if write:
        chosen = [
            me
            for me in walk
            if _exported(me.id).exists()
            and not (me.id in rows and marking.checked(rows[me.id]))
            and looked.get(me.id) != f"{_image(me.id)} {against}"
        ]
    else:
        chosen = [
            me
            for me in walk
            if me.id in rows
            and marking.checked(rows[me.id])
            and _exported(me.id).exists()
            and all(r["image"] == _image(me.id) for r in rows[me.id])  # marked on it as it is
        ]
    total = proposals.Score(0, 0, 0)
    for me in sorted(chosen, key=_visit):
        place = _place_of(me)
        found = proposals.look(
            cv2.imread(str(_exported(me.id))), place, me.years, matcher, references
        )
        said = [("/".join(f.landmarks), round(f.column, 3), f.found) for f in found]
        if write:
            written = marking.propose(SIGHTS, me.id, _image(me.id), proposals.marks_of(found))
            looked[me.id] = f"{_image(me.id)} {against}"
            write_json(PROPOSED, looked)  # after each photo: a run stopped loses none
            typer.echo(f"{me.id} : {said}" + ("" if written else " (vue entre-temps : rien écrit)"))
            continue
        marks = [r for r in rows[me.id] if "landmarks" in r]
        looked_for = proposals.looked_for(place, me.years, references.boxes)
        one = proposals.score(found, marks, [p for _, parts in looked_for for p in parts], place)
        total += one
        typer.echo(
            f"{me.id} : proposés {said} ; marqués {[(m['landmarks'], m['column']) for m in marks]}"
        )
    if not write:
        made = total.right + total.wrong
        typer.echo(
            f"sur {len(chosen)} photos marquées : {total.right} justes, {total.wrong} fausses "
            f"(justesse {total.right / made if made else float('nan'):.0%}) ; "
            f"{total.missed} repères marqués non trouvés"
        )


def _serve_page(start: Callable[[], HTTPServer], label: str, port: int, open_browser: bool) -> None:
    """Serve a local page until Ctrl-C."""
    import webbrowser

    try:
        server = start()
    except OSError as error:
        raise typer.BadParameter(f"port {port} indisponible ({error}) : essayer --port") from error
    url = f"http://127.0.0.1:{port}/"
    typer.echo(f"{label} : {url} (Ctrl-C pour arrêter)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


@app.command("stats")
def show_stats() -> None:
    """Affiche les chiffres dont les étapes ont besoin."""
    with closing(db.connect(DB_PATH)) as conn:
        all_stats = stats.compute(conn)
    for s in all_stats:
        typer.echo(f"\n== {s.source} : {s.photos} photos")
        typer.echo(f"   nomment une rue : {s.naming_a_street} ({s.naming_a_street / s.photos:.0%})")
        typer.echo(f"   position de prise de vue connue : {s.with_camera_position}")
        typer.echo(f"   licences : {s.by_license}")
        decades = ", ".join(f"{d or '?'}: {n}" for d, n in s.by_decade.items())
        typer.echo(f"   décennies : {decades}")
