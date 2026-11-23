"""Ligne de commande du pipeline : `uv run flaneur <étape>`."""

import fcntl
import functools
import hashlib
import inspect
import io
import itertools
import json
import math
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence, Set
from contextlib import closing, contextmanager
from copy import copy as shallow
from dataclasses import asdict, fields, replace
from datetime import date, datetime, timedelta
from enum import StrEnum
from http.server import HTTPServer
from pathlib import Path
from types import ModuleType
from typing import Annotated, NamedTuple

import cv2
import httpx
import numpy as np
import typer
from dotenv import load_dotenv
from PIL import Image
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from flaneur import (
    agreement,
    bridging,
    buildings,
    carrying,
    chain,
    confirming,
    db,
    dedup,
    depth,
    depthmaps,
    export,
    facades,
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
    placed,
    placements,
    placing,
    placings,
    plates,
    poses,
    proposals,
    proposing,
    queueing,
    reconstruct,
    register,
    registration,
    review,
    scene,
    segment,
    stats,
    survey,
)
from flaneur.addresses import Addresses
from flaneur.blocks import Blocks
from flaneur.gazetteer import Gazetteer, angle, to_degrees, to_metres
from flaneur.panoramas import PANORAMA_HEIGHT_M
from flaneur.panoramax import Picture
from flaneur.read import read_all
from flaneur.sources import albert_kahn, paris_musees, write_atomic, write_json
from flaneur.sources import rows as jsonl_rows

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
DB_PATH = DATA / "flaneur.sqlite"
WEB_DATA = ROOT / "web" / "public" / "data"  # the walk, as the site reads it (not versioned)
# the images of photos only seeded (placements.SEED): matched, never published until confirmed
SEEDED = DATA / "pilot" / "seeded"
REVIEWS = DATA / "overrides" / "reviews.jsonl"
SIGHTS = DATA / "overrides" / "sights.jsonl"  # the landmarks marked in photos, by hand
CORRECTIONS = DATA / "pilot" / "panorama_corrections.json"  # on the buildings (register): unused
SURVEYED = DATA / "pilot" / "panorama_survey.json"  # volunteers' panoramas set right on the survey
POSES = DATA / "pilot" / "panorama_poses.json"  # panoramas set right on two surveys (see poses)
VIEW_STRUCTURE = (  # each panorama's views' pixel classes (see segment), as views are cut now
    DATA / "raw" / "view_structure" / f"{match.VIEW_SIDE}_{match.VIEW_FOV}_{match.VIEW_PITCH}"
)
PAIR_MATCHES = (  # each pair's raw matches, made once (matcher), as views are cut now
    DATA / "raw" / "pair_matches" / f"{match.VIEW_SIDE}_{match.VIEW_FOV}_{match.VIEW_PITCH}"
)
MERGED_M = 0.3  # BD TOPO's buildings this close are one block (their facades: see _street_facades)
KEPT_SCENES = 16  # panoramas' walls the placing page keeps
WALLS_FITTED = DATA / "raw" / "walls_fitted"  # each panorama's walls fitted, kept (see _Scenes)
WALLS_VERSION = 1  # raised by hand when what the walls are fitted by changes beyond the files
# hashed in _walls_code
KEPT_PAIRS = KEPT_SCENES * depth.MOST_CANDIDATES  # their pairs set right (about 250 kB each)
FIXES = DATA / "pilot" / "panorama_fixes.json"  # panoramas placed with each other (reconstruct)
PLACINGS = DATA / "overrides" / "placings.jsonl"  # the user's answers on the placing page
WALLS = DATA / "overrides" / "walls.jsonl"  # today's walls drawn and removed by hand
TILTS = DATA / "pilot" / "panorama_tilts.json"  # how each one's camera leaned (level)
AGREED = DATA / "pilot" / "panorama_agreed.json"  # each turned to agree with its neighbours
# (accorder)
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
    """Place chaque photo sur sa rue, devant la porte du numéro que donne sa légende (atlas de
    1836 et 1888, SoDUCo, CC BY-SA 4.0 ; numéros d'aujourd'hui, Base Adresse Nationale) ou au
    coin d'où elle a été prise ; puis compare avec
    les positions de prise de vue connues (Albert-Kahn)."""
    with httpx.Client(timeout=300) as client:
        gazetteer = Gazetteer.load(client, DATA / "raw" / "gazetteer")
        doors = Addresses.load(client, DATA / "raw" / "addresses")
    with closing(db.connect(DB_PATH)) as conn:
        counts = geocode.locate_all(conn, gazetteer, doors)
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
        placed = {p["photo_id"] for p in found}  # its own placement, or its negative's
        seeded = placements.seeds(conn, DATA / "pilot", pilot.AREA, placed)
        placements.save(conn, found + seeded)
        distances = placements.distances_to_truth(conn)
    typer.echo(f"{len(found)} placements enregistrés, et {len(seeded)} amorces (à situer)")
    for confidence in placements.CONFIDENCES:
        d = [m for c, m in distances if c == confidence]
        if d:
            s = geocode.summary(d)
            typer.echo(
                f"vérité terrain, confiance {confidence} : {s['n']} photos, "
                f"{s['within_50m']:.0%} à moins de 50 m, médiane {s['median_m']:.0f} m"
            )


def _standing_now(corrected: bool = True) -> dict[str, tuple[float, float, float]]:
    """Where the panoramas stand (see placings.where), from their poses alone (no picture, no
    tilt needed): answers follow them."""
    with closing(db.connect(DB_PATH)) as conn:
        return _standing(_placed(panoramas.load(conn), corrected))


def _standing(pictures: list[Picture]) -> dict[str, tuple[float, float, float]]:
    """Where these panoramas stand (see placings.where), by id."""
    return {p.id: poses.at(p.point.x, p.point.y, p.azimuth or 0.0) for p in pictures}


def _poised(spheres: Iterable[match.Panorama]) -> dict[str, tuple[float, ...]]:
    """How these panoramas stand and lean, by id: where (see poses.at), then roll and pitch to
    the hundredth (a copy carried is made in their levelled views)."""
    return {
        p.id: (*poses.at(p.x, p.y, p.azimuth), round(p.roll, 2), round(p.pitch, 2)) for p in spheres
    }


def _placing_answers(
    path: Path | None = None, corrected: bool = True
) -> tuple[dict[str, dict], dict[str, dict], set[str]]:
    """From the placing page's answers: where each photo answered stands (lon, lat, heading),
    the answers placing a photo in its panorama (on the image it had then), and the photos no
    panorama sees."""
    held = placings.holding(path or PLACINGS)
    if any(r.get("panorama_at") for r in held.values()):  # each following its panorama
        now = _standing_now(corrected)
        held = {i: placings.followed(r, now) for i, r in held.items()}
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


def _carried(placed: dict[str, dict], now: dict[str, tuple]) -> dict[str, dict]:
    """The copies carried to the neighbouring panoramas (`flaneur porter`) still made from
    the answer that holds (`placed`), with every panorama it was carried from or tried on
    standing and leaning as it did (`now`: see _poised, poses.stands)."""
    kept = json.loads(CARRIED.read_text()) if CARRIED.exists() else {}
    return {
        i: c
        for i, c in kept.items()
        if i in placed
        and c["made"] == _answered_as(placed[i])
        and "at" in c
        and all(poses.stands(at, now.get(p)) for p, at in c["at"].items())
    }


def _placed_on(answering: tuple) -> dict[str, str]:
    """The image each photo is placed in 3D on (its hash), for the export to check against the
    image it writes (see export.mark_3d): the list's (see _shown_in_3d), and every answer of
    the placing page's own, for a photo whose image the export is about to write."""
    answered = {i: r["inputs"] for i, r in answering[1].items()}
    return answered | {i: r["inputs"] for i, r in _shown_in_3d(answering).items()}


@app.command("export")
def export_walk(images: bool = True) -> None:
    """Écrit la balade (photos relues appliquées, liens, vue d'aujourd'hui) pour le site."""
    answering = places, in_3d, no_3d = _placing_answers()
    with closing(db.connect(DB_PATH)) as conn:
        every = _walk(conn, places)
        walk = export.build(conn, DATA / "pilot", every)
    elsewhere = len(in_3d) - len(_on_image_now(in_3d))
    if elsewhere:  # their image changed since: the placing page asks for them again
        typer.echo(f"{elsewhere} photos placées sur une image qu'elles n'ont plus : à revoir")
    if images:
        placed_on = _placed_on(answering)
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            walk = export.write(walk, client, DATA / "raw" / "images", WEB_DATA, placed_on, no_3d)
            _image.cache_clear()  # images written for the first time: what is placed on them
            _sights.cache_clear()
            if (now := _placed_on(answering)) != placed_on:
                export.mark_3d(walk, WEB_DATA, now, no_3d)
                write_json(WEB_DATA / "walk.json", {"photos": [p.for_site() for p in walk]})
            seeds = [me for me in every if me.seeded]  # for matching only, kept apart
            kept = export.write_seeds(seeds, client, DATA / "raw" / "images", SEEDED)
        typer.echo(f"{kept} images d'amorces gardées à part ({SEEDED})")
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


def _lined_up(
    pictures: list[Picture], corrected: bool = True, agreed: bool = True
) -> list[Picture]:
    """The panoramas where they were taken, and levelled (see _levelled), when their tilts
    were measured as tilts are now."""
    if not level.current(TILTS):
        raise typer.BadParameter(
            "les inclinaisons des panoramas ont été mesurées autrement : lancer `flaneur level`"
        )
    return _levelled(pictures, corrected, agreed)


def _levelled(
    pictures: list[Picture], corrected: bool = True, agreed: bool = True
) -> list[Picture]:
    """The panoramas where they were taken (see _placed), and levelled: those not measured
    leaning as their run's neighbours, else taken as level (see level.borrowed); then,
    `corrected` and `agreed`, turned to agree with their neighbours as `flaneur accorder` found
    (agreement.APPLIED of it), each while it faces and leans as it was found against. accorder
    reads them as levelled before (`agreed` False): what it wrote never feeds what it computes.
    A panorama turned so has caches of its own (_levelled_name)."""
    placed = _placed(pictures, corrected)
    tilts = level.borrowed(level.kept(TILTS), placed)
    levelled = [
        replace(p, roll=tilt[0], pitch=tilt[1]) if (tilt := tilts.get(p.id)) else p for p in placed
    ]
    if not (corrected and agreed) or agreement.APPLIED == "none":
        return levelled
    return agreement.applied(levelled, agreement.load(AGREED))


def _placed(pictures: list[Picture], corrected: bool = True) -> list[Picture]:
    """The panoramas where they were taken: placed with each other (reconstruct), else moved
    as their run's nearest panoramas so placed (reconstruct.borrowed), else as recorded (not
    by their own roofs and corners: see register); then, `corrected`, set right on the map as
    `flaneur recaler` found (poses.APPLIED of it) — which judges them as COLMAP left them. A
    panorama turned so has caches of its own (_levelled_name: its views are other pixels): the
    corrected world and recaler's each make theirs (masks, pairs) when asked."""
    fixes = register.load(FIXES)
    found = {i: tuple(fixes[i]["fix"]) for i in _fixed_by_colmap(pictures, fixes)}
    placed = register.applied(pictures, reconstruct.borrowed(found, pictures))
    if not corrected:
        return placed
    placed = register.applied(placed, _survey_fixes(placed))
    if poses.APPLIED == "none":
        return placed
    answered = _answered_by_survey(placed)
    verdicts = {i: v for i, v in poses.load(POSES).items() if i not in answered}
    where = _standing([p for p in placed if p.id in verdicts])
    return register.applied(placed, poses.to_apply(verdicts, where, poses.APPLIED))


def _survey_fixes(pictures: list[Picture]) -> dict[str, tuple[float, float, float]]:
    """The volunteers' panoramas set right on the survey (`flaneur releve`) still standing where
    they were, as register.applied takes them (survey.APPLIED of it)."""
    if survey.APPLIED == "none":
        return {}
    rows = survey.load(SURVEYED)
    where = _standing([p for p in pictures if p.id in rows])
    standing = {i: f for i, (f, placed) in rows.items() if poses.stands(placed, where.get(i))}
    return survey.fixes(standing, survey.APPLIED)


def _answered_by_survey(pictures: list[Picture]) -> set[str]:
    """The panoramas the survey speaks for (applied): its own, and those it set right or found
    right — BD TOPO's corrections are not theirs."""
    if survey.APPLIED == "none":
        return set()
    return set(survey.load(SURVEYED)) | {p.id for p in pictures if p.producer == survey.PRODUCER}


class _NotRead(Exception):
    """A pair whose pictures could not be read: nothing kept, asked again next time."""


class _Paired(NamedTuple):
    """A neighbour set right against a panorama, and the details they share placed."""

    pair: depth.Pair
    directions: np.ndarray  # from the panorama, float32
    metres: np.ndarray  # how far along each


class _Scenes:
    """Today's walls around panoramas, for the placing page's parallax (see facades): fitted from
    the details a panorama shares with its neighbours (those 2-15 m off, or those chosen), each
    pair set right from its own matches (see depth); BD TOPO's facades near it proposed
    (`segments`); the walls drawn and removed by hand (`walls_path`) laid over them when asked.
    A pair's matches are made with the matcher (`lock` held while it runs) and kept on disk
    (`matches`), their details KEPT_PAIRS of them; walls fitted are kept, KEPT_SCENES. A pair
    is set right from all its matches, its details are those both panoramas see on a building,
    and walls' tops where the panorama sees building give way to sky (`seen`: what rays from a
    panorama see, see segment.seen)."""

    def __init__(
        self,
        spheres: list[match.Panorama],
        fixed: set[str],
        views: Callable[[], match.Views],
        lock: threading.Lock,
        segments: Callable[[float, float], list[facades.Segment]],
        walls_path: Path,
        matches: Path = PAIR_MATCHES,
        seen: Callable[[match.Panorama, np.ndarray], np.ndarray] | None = None,
        *,
        kept: Path | None = None,
        made: frozenset[str] | None = None,
        masks: segment.Segmenter | None = None,
    ):
        self.by_id = {p.id: p for p in spheres}
        self.now = placings.where(spheres)  # drawn walls follow their panorama
        self.spheres, self.fixed, self.views, self.lock = spheres, fixed, views, lock
        self.segments, self.walls_path, self.matches = segments, walls_path, matches
        self.seen = seen
        self.kept = kept  # the walls fitted from the pairs on disk alone, kept there (see stamp)
        self.made = made  # the pairs' files on disk, listed once (none: looked at each time)
        self.masks = masks  # whose segmentation is made (see _masked)
        self._segmented: dict[str, bool] = {}
        self.paired: OrderedDict[tuple[str, str], _Paired | None] = OrderedDict()
        self.fitted: OrderedDict[tuple, facades.Scene | None] = OrderedDict()
        self.guard = threading.Lock()  # the kept ones, read and changed one thread at a time
        self.pairing = threading.Lock()  # a pair set right once (asked twice: once)
        self.fitting = threading.Lock()  # walls fitted one panorama at a time (asked twice: once)

    def _kept(self, kept: OrderedDict, key, most: int, make: Callable, lock: threading.Lock):
        with self.guard:
            if key in kept:
                kept.move_to_end(key)  # used lately: kept longest
                return kept[key]
        with lock:
            with self.guard:
                if key in kept:  # made meanwhile, by the request that held the lock
                    return kept[key]
            found = make()
            with self.guard:
                kept[key] = found
                while len(kept) > most:
                    kept.popitem(last=False)
            return found

    def pair(self, home: match.Panorama, other: match.Panorama) -> _Paired | None:
        """A neighbour set right against a panorama, and the details they share placed; None
        when it cannot be set right."""

        def made() -> _Paired | None:
            rays_a, rays_b = self._rays(home, other)
            if not len(rays_a):  # a picture not read: asked again next time
                raise _NotRead
            a, b = placing._at(home), placing._at(other)
            pair = depth.set_right(a, b, rays_a, rays_b)
            if pair is None:
                return None
            if self.seen is not None:  # trees, cars, people: no walls
                on = (self.seen(home, rays_a) == segment.BUILT) & (
                    self.seen(other, rays_b) == segment.BUILT
                )
                rays_a, rays_b = rays_a[on], rays_b[on]
            directions, metres = depth.placed(a, pair, rays_a, rays_b)
            return _Paired(pair, directions.astype(np.float32), metres.astype(np.float32))

        try:
            return self._kept(self.paired, (home.id, other.id), KEPT_PAIRS, made, self.pairing)
        except _NotRead:
            return None

    def pair_of(self, panorama: str, neighbour: str) -> depth.Pair | None:
        """A neighbour set right against a panorama, by their ids; None if it cannot be."""
        home, other = self.by_id.get(panorama), self.by_id.get(neighbour)
        found = None if home is None or other is None else self.pair(home, other)
        return found.pair if found else None

    def aimed(
        self, panorama: str, neighbours: Iterable[str], heading: float, fov: float
    ) -> list[str]:
        """The neighbours (of the walls shown) giving most in a view facing `heading`, `fov`
        wide, first (see depth.aimed): the one shown for second clicks."""
        home = self.by_id[panorama]
        found = {i: self.pair(home, self.by_id[i]) for i in neighbours if i in self.by_id}
        return depth.aimed({i: f.directions for i, f in found.items() if f}, heading, fov)

    def _rays(self, home: match.Panorama, other: match.Panorama) -> tuple[np.ndarray, np.ndarray]:
        """The pair's matched rays (see depth.matched), from disk once made: they are the same
        while the two panoramas are levelled as they are (in the name, as their views': see
        match.Views). Kept as float32, and used so when just made: the same walls either way.
        A pair is matched once: kept the other way round, it is read so (as good: measured
        16/11, a neighbour held out)."""
        if stored := self._stored(home, other):
            path, back = stored
            with np.load(path) as found:
                a, b = found["a"].astype(float), found["b"].astype(float)
            return (b, a) if back else (a, b)
        path = self._path(home, other)
        with self.lock:
            rays_a, rays_b = (
                r.astype(np.float32) for r in depth.matched(home, other, self.views())
            )
        if len(rays_a):  # a picture not read: not kept
            path.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(path, lambda out: np.savez(out, a=rays_a, b=rays_b))
        return rays_a.astype(float), rays_b.astype(float)

    def make(self, home: match.Panorama) -> facades.Scene | None:
        """A panorama matched with those of its neighbours (see _asked) it is not matched with
        yet, either way round, and its walls fitted from the pairs on disk, kept (see scene,
        `made_only`): nothing new, nothing paired or fitted again."""
        for other in self._asked(home, None, made_only=False):
            if not self._made(home, other):
                self.pair(home, other)
        return self.scene(home.id, None, made_only=True)

    def has_pairs(self, home: match.Panorama) -> bool:
        """Whether a panorama is matched with a neighbour already, both segmented when that
        is asked (see scene, `made_only`)."""
        return bool(self._asked(home))

    def scene(
        self,
        panorama: str,
        chosen: Iterable[str] | None,
        made_only: bool = False,
        but: frozenset[str] = frozenset(),
    ) -> facades.Scene | None:
        """The walls around a panorama, from the neighbours `chosen` (else those 2-15 m off:
        see depth.candidates; `made_only`, those matched with it already: the matcher never
        woken), with those drawn and removed by hand; none for a panorama not known. `but`:
        panoramas left out, as neighbours and as where walls were drawn. With `masks`, only
        the segmented fit walls (none if the panorama is not: trees are taken for facades).
        Fitted from the pairs on disk alone, by geometry, they are kept in `kept` (see stamp)."""
        home = self.by_id.get(panorama)
        if home is None:
            return None
        chosen = None if chosen is None else tuple(sorted(set(chosen)))  # (): none, by hand
        asked = self._asked(home, chosen, made_only)
        left_out = but & {p.id for p in asked}  # only these change the fit
        asked = [p for p in asked if p.id not in left_out]
        if self.kept is not None and made_only and chosen is None and not left_out:
            fitted = self._kept(
                self.fitted,
                (panorama, tuple(p.id for p in asked)),  # pairs on disk: always read
                KEPT_SCENES,
                lambda: self._read_or_fit(home, asked),
                self.fitting,
            )
        else:
            found = {p.id: f for p in asked if (f := self.pair(home, p))}
            neighbours = tuple(sorted(found))
            fitted = self._kept(
                self.fitted,
                (panorama, chosen, neighbours),
                KEPT_SCENES,
                lambda: self._fit(home, chosen, [found[i] for i in neighbours], neighbours),
                self.fitting,
            )
        drawn = facades.saved_near(self.walls_path, home.x, home.y, now=self.now, but=but)
        walls = facades.with_drawn(list(fitted.walls), *drawn, fitted.ground)
        return replace(fitted, walls=tuple(walls))

    def _asked(
        self,
        home: match.Panorama,
        chosen: tuple[str, ...] | None = None,
        made_only: bool = True,
    ) -> list[match.Panorama]:
        """The neighbours a panorama's walls are fitted from (see scene)."""
        asked = (
            depth.candidates(home, self.spheres, self.fixed)
            if chosen is None
            else [self.by_id[i] for i in chosen if i in self.by_id and i != home.id]
        )
        asked = [p for p in asked if self._masked(p)]
        if not self._masked(home):
            return []
        return [p for p in asked if self._made(home, p)] if made_only else asked

    def details(
        self, home: match.Panorama, neighbours: list[match.Panorama], made_only: bool = False
    ) -> np.ndarray:
        """The details a panorama shares with `neighbours`, placed (n × 3); `made_only`, from
        the pairs already kept on disk only (the matcher never woken)."""
        if made_only:
            neighbours = [p for p in neighbours if self._made(home, p)]
        return _details_of(home, [f for p in neighbours if (f := self.pair(home, p))])

    def _made(self, home: match.Panorama, other: match.Panorama) -> bool:
        """Whether a pair is matched already, either way round: as listed at the start (`made`),
        else on disk."""
        listed = None if self.made is None else (lambda path: path.name in self.made)
        return self._stored(home, other, listed or Path.exists) is not None

    def _stored(
        self,
        home: match.Panorama,
        other: match.Panorama,
        there: Callable[[Path], bool] = Path.exists,
    ) -> tuple[Path, bool] | None:
        """Where a pair is kept (`there`: on disk), either way round, and whether the other way
        (its rays then read swapped: see _rays); none if not matched yet."""
        for path, back in ((self._path(home, other), False), (self._path(other, home), True)):
            if there(path):
                return path, back
        return None

    def _masked(self, panorama: match.Panorama) -> bool:
        """Whether a panorama is segmented (all are, without `masks`), asked once a run."""
        if self.masks is None:
            return True
        if panorama.id not in self._segmented:
            self._segmented[panorama.id] = self.masks.kept(panorama)
        return self._segmented[panorama.id]

    def stamp(self, home: match.Panorama) -> str:
        """What a panorama's walls fitted from the pairs on disk are made from: its neighbours
        matched with it (see _asked), where each stands from it (its height too: the frame of
        heights moves as the area grows), BD TOPO's facades the fit starts from, and the
        code."""
        panoramas = [
            (_levelled_name(p), *(round(v, 2) for v in (p.x - home.x, p.y - home.y, p.z - home.z)))
            for p in [home, *self._asked(home)]
        ]
        facades_near = [
            [round(c, 2) for end in s for c in end] for s in self.segments(home.x, home.y)
        ]
        made = [WALLS_VERSION, _walls_code(), round(home.x, 2), round(home.y, 2)]
        return _digest([made, panoramas, facades_near])

    def _read_or_fit(self, home: match.Panorama, asked: list[match.Panorama]) -> facades.Scene:
        """A panorama's walls fitted from the pairs on disk: as kept if made from what it is
        matched with now (see stamp), else fitted and kept. Heights are kept from the
        panorama's: the frame of heights moves as the area grows."""
        path, stamp = self.kept / f"{_levelled_name(home)}.json", self.stamp(home)
        try:
            row = json.loads(path.read_text())
        except FileNotFoundError:
            row = None
        if row is not None and row["stamp"] == stamp:
            walls = tuple(
                facades.Wall((ax, ay), (bx, by), _above(home.z, top), made)
                for ax, ay, bx, by, top, made in row["walls"]
            )
            ground, origin = home.z + row["ground"], tuple(placing._at(home))
            neighbours = tuple(row["neighbours"])
            return facades.Scene(home.id, origin, ground, walls, neighbours, row["details"])
        found = {p.id: f for p in asked if (f := self.pair(home, p))}
        neighbours = tuple(sorted(found))
        fitted = self._fit(home, None, [found[i] for i in neighbours], neighbours)
        kept = {
            "stamp": stamp,
            "ground": fitted.ground - home.z,
            "walls": [[*w.a, *w.b, _above(-home.z, w.top), w.made] for w in fitted.walls],
            "neighbours": list(neighbours),
            "details": fitted.details,
        }
        write_json(path, kept)
        return fitted

    def _path(self, home: match.Panorama, other: match.Panorama) -> Path:
        return self.matches / f"{_levelled_name(home)}__{_levelled_name(other)}.npz"

    def _fit(
        self, home: match.Panorama, chosen: tuple | None, found: list[_Paired], neighbours: tuple
    ) -> facades.Scene:
        origin = placing._at(home)
        details = _details_of(home, found)
        ground = home.z - PANORAMA_HEIGHT_M
        fitted = facades.fit(details, ground, self.segments(home.x, home.y), origin=origin)
        if self.seen is not None:
            fitted = facades.topped(fitted, origin, ground, functools.partial(self.seen, home))
        return facades.Scene(
            home.id, tuple(origin), ground, tuple(fitted), neighbours, len(details), chosen
        )


def _details_of(home: match.Panorama, found: list[_Paired]) -> np.ndarray:
    """The details pairs placed, from the panorama (n × 3; none: an empty array)."""
    origin = placing._at(home)
    points = [origin + f.directions * f.metres[:, None] for f in found]
    return np.vstack(points) if points else np.empty((0, 3))


def _seeing(
    make: bool, segmenter: segment.Segmenter | None = None
) -> Callable[[match.Panorama, np.ndarray], np.ndarray]:
    """What rays from a panorama see in its views (see segment.seen), read in VIEW_STRUCTURE;
    made the first time with the model (a lock of its own: photos are proposed meanwhile), or,
    without `make`, only read (`flaneur segmenter` makes them). The model failing (not
    fetched, short of memory) or a panorama not segmented: a building everywhere (the walls
    made as before), which is said once for each; the model failing is not asked again.
    `segmenter`: one shared with the depth's (see depthmaps.fills), read once for both."""
    segmenter = segmenter or segment.Segmenter(
        VIEW_STRUCTURE, threading.Lock(), _levelled_name, make
    )
    said: set[str] = set()
    broken: list[Exception] = []

    def seen(panorama: match.Panorama, rays: np.ndarray) -> np.ndarray:
        kinds = None
        try:
            if not broken:
                kinds = segmenter.kinds(panorama)
        except Exception as error:  # the walls rather than none
            if not isinstance(error, segment.NotSegmented):
                broken.append(error)
            if str(error) not in said:
                said.add(str(error))
                why = (
                    f"panorama {error} pas segmenté (`flaneur segmenter`)"
                    if isinstance(error, segment.NotSegmented)
                    else f"segmentation impossible : {error}"
                )
                typer.echo(f"murs sans segmentation : {why}", err=True)
        return segment.seen(kinds, rays)

    return seen


def _levelled_name(p: match.Panorama) -> str:
    """A panorama's name as levelled now (its views, and so its matches, change with it)."""
    return f"{p.id}_{p.azimuth:.1f}_{p.roll:.2f}_{p.pitch:.2f}"


def _street_facades(features: list[dict]) -> Callable[[float, float], list[facades.Segment]]:
    """BD TOPO's street facades on the plan (Lambert-93), those reaching within facades.NEAR_M
    of a spot: candidates for the walls fitted there. Each block's buildings merged (gaps under
    MERGED_M closed), its outer outline kept: the walls between buildings, at their backs and
    round courtyards are not seen from the street."""
    parts = [o for f in features for o in buildings.outlines(f)]
    merged = unary_union([p.buffer(MERGED_M) for p in parts]).buffer(-MERGED_M)
    rings = [g.exterior.coords for g in getattr(merged, "geoms", [merged]) if not g.is_empty]
    edges = Blocks.of([([(x, y) for x, y, *_ in ring][:-1], 0.0, 0.0) for ring in rings])
    return lambda x, y: [
        ((e[0], e[1]), (e[2], e[3])) for e in edges.near(x, y, facades.NEAR_M).edges
    ]


def _placing_world(
    corrected: bool = True, agreed: bool = True
) -> tuple[list[match.Panorama], Callable[[float, float], list]]:
    """The panoramas where they were taken, levelled (see _levelled), at their camera's height
    over the street (parallax needs it on a slope), and BD TOPO's street facades near a spot."""
    with closing(db.connect(DB_PATH)) as conn:
        area = placements.area(conn, buildings.MARGIN_M)
        features = _buildings(area)
        pictures = _lined_up(panoramas.load(conn), corrected, agreed)
    return _panoramas(pictures, _heights(features, area)), _street_facades(features)


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
# Cameras matching turned down although many points agreed with them, once proposed to the
# user (`presque`, 11–12 November; dropped by its rule: 37 % right, 50 % wanted): the file is
# kept as answered (a yes is a placement, a near comes first on the placing page).
PROPOSED_CAMERAS = DATA / "pilot" / "proposed_cameras.json"
PROPOSAL = "proposed"  # a proposed camera's "matcher": PROPOSAL-<where it stood, hashed>
ANCHORS = DATA / "overrides" / "anchors.jsonl"  # the user's yes or no to photos placed
WRONG_PAIRS = DATA / "overrides" / "wrong_pairs.jsonl"  # pairs the user marked wrong
MATCH_RADIUS_M = 60  # a placement from a caption can be 25 m off
MATCH_PANORAMAS = 16  # ranked by a quick first pass, the best views matched in full
# every panorama a photo's match may be found from stands this near its caption's place: those
# it is compared with, and those its focusing pass may take (its camera's reach, then its own)
SCENE_REACH_M = max(MATCH_RADIUS_M, match.MOST_MOVED_M + match.FOCUS_NEAR_M)
EARLIEST_YEAR = 1850  # an undated photo: only the buildings standing by then rule a camera out
# The gate before matching real photos: views of panoramas, taken for old photos, found
# within this of their true camera (median).
CHECK_M = 1.0
CHECK_DEG = 2.0
CHECK_SHARE = 0.7  # and at least this share of them situated
CHECK_LEAST = 10  # views, for an area's check to pass: fewer tell nothing
CHECK_KEPT = (0.25, 0.35)  # the share of the true view a band keeps
FAR_M = 200  # a decoy: a view of a street this far away
CHECK_FOV = 60  # its views' width (and height: they are square)
MARKED_TO = 0.015  # a landmark's column, marked by hand: off by this (a share of the width)
RIGHT_M, RIGHT_DEG = 3.0, 5.0  # a camera found farther off than this is a wrong one
WRONG_SHARE = 0.05  # of all the views, bands and decoys: at most this share wrong


def _above(z: float, height: float | None) -> float | None:
    """A height from `z` (none: none)."""
    return None if height is None else z + height


def _digest(data: object) -> str:
    """Data that is JSON, hashed (16 hex digits)."""
    return hashlib.sha1(json.dumps(data).encode()).hexdigest()[:16]


def _hash_of(*modules: ModuleType) -> str:
    """The source of `modules`, hashed (16 hex digits)."""
    digest = hashlib.sha1()
    for module in modules:
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()[:16]


@functools.cache
def _code() -> str:
    """The code a camera is found by (the check's own landmarks among it): a match or a check
    found by other code is stale."""
    return _hash_of(match, buildings, landmarks)


@functools.cache  # the panoramas do not move while a command runs
def _setup() -> str:
    """Every panorama whose picture is at hand, where it stands (lined up and levelled), and
    the code: what chaining is still keyed on (a photo's own match depends on its own panoramas
    only, see _scene_of)."""
    with closing(db.connect(DB_PATH)) as conn:
        levelled = _levelled(panoramas.load(conn))
    where = sorted(
        (p.id, round(p.lon, 7), round(p.lat, 7), round(p.azimuth or 0.0, 2), p.roll, p.pitch)
        for p in levelled
        if (PANORAMAS / f"{p.id}.jpg").exists()
    )
    return hashlib.sha1(f"{where!r}{_code()}".encode()).hexdigest()[:16]


def _scene_of(near: Sequence[match.Panorama], here: tuple[float, float], code: str) -> str:
    """What one photo's match is found from besides the photo itself: the panoramas within
    SCENE_REACH_M of it (`near`: where each stands, faces and leans), its place (`here`: where
    its camera may stand) and the `code`. A panorama added out of their reach, or moved there,
    leaves it be: photos and areas are added step by step."""
    seen = [
        (
            p.id,
            round(p.x, 2),
            round(p.y, 2),
            round(p.azimuth, 2),
            round(p.roll or 0.0, 3),
            round(p.pitch or 0.0, 3),
        )
        for p in near
    ]
    place = (round(here[0], 1), round(here[1], 1))
    return hashlib.sha1(f"{seen!r}{place!r}{code}".encode()).hexdigest()[:16]


def _scenes(walk: Iterable[export.WalkPhoto], near_by: list[match.Panorama]) -> dict[str, str]:
    """Each photo's scene (see _scene_of), among the panoramas `near_by` as matching has them."""
    code, reached = _code(), _reach(near_by, SCENE_REACH_M, None)
    return {me.id: _scene_of(reached(here := _place_of(me)), here, code) for me in walk}


@functools.cache  # for one run, as _image
def _scenes_now() -> dict[str, str]:
    """Each walk photo's scene as it is now, for what reads placements without matching: the
    panoramas where matching has them (their height aside: it does not choose them)."""
    with closing(db.connect(DB_PATH)) as conn:
        walk = export.photos(conn, DATA / "pilot", REVIEWS)
        pictures = _levelled(panoramas.load(conn))
    return _scenes(walk, _panoramas(pictures, lambda x, y: 0.0))


def _exported(photo_id: str) -> Path:
    """A photo's image as the walk shows it and the matcher reads it (cropped): the site's,
    else, for a photo only seeded, the one kept apart from it."""
    shown = WEB_DATA / export.image_name(photo_id)
    return shown if shown.exists() else SEEDED / export.image_name(photo_id)


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


def _photo_part(photo_id: str) -> str:
    """What a photo itself gives its match: its image as exported (cropped), and the landmarks
    marked in it."""
    part = _image(photo_id)
    if sights := _sights().get(photo_id):
        part += f":{hashlib.sha1(repr(sights).encode()).hexdigest()[:8]}"
    return part


def _inputs(scene: str, photo_id: str) -> str:
    """What a photo's match is found from: its `scene` (see _scene_of), then the photo itself
    (see _photo_part)."""
    return f"{scene}:{_photo_part(photo_id)}"


def _matches() -> dict[str, dict | None]:
    """Each photo matched: its placement (a Situated's fields and the `matcher` that found it;
    none: DISK, before LoMa-G), or, left unplaced, its `inputs` and the matchers `tried`."""
    return register.load(MATCHES)


def _situation(kept: dict) -> match.Situated:
    """A placement as kept in matches.json, its matcher aside."""
    return match.Situated(**{f.name: kept[f.name] for f in fields(match.Situated)})


def _placed_inputs(inputs: str, matcher: str | None) -> str:
    """What a placement is found from: the photo's `inputs` (see _inputs), and its matcher
    unless DISK (as before LoMa-G): a placement of each (a verdict may still follow a camera
    found again by the other: see _Judged.on)."""
    return inputs if matcher in (None, DISK) else f"{inputs}:{matcher}"


def _found_inputs(placed: str, matcher: str | None) -> str:
    """What a placement's points were found from: its inputs, its matcher's mark taken off (see
    _placed_inputs)."""
    return placed if matcher in (None, DISK) else placed.removesuffix(f":{matcher}")


def _tried(kept: dict | None, inputs: str) -> list[str]:
    """The matchers a photo was tried with from these `inputs` (none: from others, or never;
    before LoMa-G, only DISK was): a placement kept though they found nothing says so
    (`tried_from`, see _still_shown)."""
    if not kept:
        return []
    if "x" in kept:
        if _placed_now(kept, inputs):
            return kept.get("tried", [kept.get("matcher") or DISK])
        return kept.get("tried", []) if kept.get("tried_from") == inputs else []
    return kept.get("tried", [DISK]) if kept.get("inputs") == inputs else []


def _placed_now(kept: dict | None, inputs: str) -> bool:
    """Whether a kept match is a placement found from these `inputs`."""
    return (
        bool(kept) and "x" in kept and kept["inputs"] == _placed_inputs(inputs, kept.get("matcher"))
    )


def _left_to_try(kept: dict | None, inputs: str, refused: bool = False) -> list[str]:
    """The matchers a photo is still to be tried with, in order: those it was not tried with,
    unless it is placed as it is now and the user did not turn that placement down."""
    tried = _tried(kept, inputs)
    if _placed_now(kept, inputs) and not refused:
        return []
    return [m for m in MATCHERS if m not in tried]


def _is_proposal(inputs: str) -> bool:
    """Whether a placement's inputs are a camera proposed (see PROPOSED_CAMERAS)."""
    return re.search(rf":{PROPOSAL}-[0-9a-f]+$", inputs) is not None


def _situated(marks: bool = True) -> dict[str, match.Situated]:
    """The old photos whose camera was found against today's panoramas, from what they are
    now (a match found from other panoramas or another crop of the photo is left out, unless
    the user said yes to it and the photo is the same: see _Judged.holds), and not turned down
    by the user (as placed, or as found again: see _Judged.on); and those whose camera matching
    turned down the user said yes to (see PROPOSED_CAMERAS). Without `marks`: see _found_as_now."""
    scenes, judged, matches = _scenes_now(), _Judged(), _matches()
    found = {}
    for photo_id, m in matches.items():
        if m and "x" in m and photo_id in scenes and _exported(photo_id).exists():
            if _stands(photo_id, m, _inputs(scenes[photo_id], photo_id), judged, marks):
                found[photo_id] = _situation(m)
    for photo_id, p in _proposed_now(scenes, judged, marks).items():
        if judged.said_yes(photo_id, p) and not (
            photo_id in found and judged.said_yes(photo_id, matches[photo_id])
        ):  # the user's yes wins over a placement not judged
            found[photo_id] = _situation(p)
    return found


def _proposed_now(scenes: dict[str, str], judged: "_Judged", marks: bool = True) -> dict[str, dict]:
    """The cameras proposed (see PROPOSED_CAMERAS) for photos as they are now (`scenes`: each one's,
    see _scene_of) the user did not turn down, or said yes to while the photo is the same (see
    _stands, `marks`)."""
    return {
        k: p
        for k, p in register.load(PROPOSED_CAMERAS).items()
        if k in scenes
        and _exported(k).exists()
        and _stands(k, p, _inputs(scenes[k], k), judged, marks)
    }


def _confirmed() -> dict[str, match.Situated]:
    """The situated photos the user said yes to as they are placed now: the photo walk takes
    their place and the 3D walk shows them (unless their place was set by hand since: a
    reviewed place wins, see export.photos). A placement not judged (as placed, or as found
    again: see _Judged.on) waits for a yes (a camera can collapse onto a panorama's spot and
    still look right). A photo placed by chaining said yes to is among them, where match has
    none said yes to (see _chained)."""
    judged = _Judged()
    situated = _situated(marks=False)  # shown: marks aside
    found = {k: m for k, m in situated.items() if judged.said_yes(k, asdict(m))}
    for photo_id, c in _chained_said_yes(_scenes_now(), judged).items():
        found.setdefault(photo_id, _situation(c))  # where match has none said yes to
    return found


def _turned_down() -> set[str]:
    """The photos the user's last word on is not a yes: chaining leaves them out, as close
    placements of them come back from the same details through a point or two of another
    photo (they wait for their camera to be set right, or a new match to be judged)."""
    return {  # a camera proposed and turned down: not (the photo waits for another)
        k for k, v in _verdicts().items() if v["verdict"] != "yes" and not _is_proposal(v["inputs"])
    }


def _size_of(image: Path) -> tuple[int, int]:
    """An image's height and width (its header only)."""
    with Image.open(image) as opened:
        return opened.size[::-1]


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

    @property
    def standing(self) -> list[dict]:
        """The buildings standing then (BD TOPO's features)."""
        return self._standing

    @functools.cached_property
    def _built(self) -> list[dict]:
        return scene.buildings_of(self._standing, (0.0, 0.0, self._heights.origin_z))

    @functools.cached_property
    def blocks(self) -> Blocks:
        return Blocks.of([(b["footprint"], b["base"], b["height"]) for b in self._built])

    @functools.cached_property
    def footprints(self) -> list[list[tuple[float, float]]]:
        """Their footprints on the plan (Lambert-93)."""
        return [b["footprint"] for b in self._built]


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


def _per_area(path: Path, matcher: str) -> dict[str, dict]:
    """A matcher's file kept area by area (a known-answer check, its traces): those of the areas."""
    return {k: v for k, v in register.load(_of_matcher(path, matcher)).items() if k in pilot.AREAS}


def _area_of(here: tuple[float, float]) -> str:
    """The area a place belongs to (see pilot.AREAS): the one it stands in, else the nearest."""
    point = Point(*here)
    return min(pilot.AREAS, key=lambda name: pilot.AREAS[name].distance(point))


def _check_setup(matcher: str, area: str, near_by: list[match.Panorama]) -> str:
    """What a matcher's known-answer check in an `area` depends on: the panoramas within reach
    of it (SCENE_REACH_M: its views', those they are compared with and those their focusing
    pass may take, see _check_area; where each stands, faces and leans: see _scene_of), the
    code, and for LoMa-G its own code and LoMa's revision. Panoramas added or moved elsewhere
    leave it be."""
    reach = pilot.AREAS[area].buffer(SCENE_REACH_M)
    inside = sorted((p for p in near_by if reach.contains(Point(p.x, p.y))), key=lambda p: p.id)
    centre = pilot.AREAS[area].centroid
    setup = f"{area}:{_scene_of(inside, (centre.x, centre.y), _code())}"
    if matcher == DISK:
        return setup
    return f"{setup}:{_lomag_code()}"


def _lomag_code() -> str:
    """LoMa-G's own code and LoMa's revision, hashed."""
    from importlib import metadata

    code = Path(__file__).with_name("lomag.py").read_bytes()
    revision = metadata.distribution("lomatch").read_text("direct_url.json") or ""
    return hashlib.sha1(code + revision.encode()).hexdigest()[:8]


def _of_matcher(path: Path, matcher: str) -> Path:
    """A file of DISK's (as before LoMa-G), or its like for another matcher."""
    return path if matcher == DISK else path.with_stem(f"{path.stem}_{matcher}")


class Matching(NamedTuple):
    """A matcher, as matching uses it."""

    label: str
    kept: int  # panoramas' views kept in memory (about 13 MB each with DISK)
    views: Path  # their features, kept on disk
    photos: Path  # the old photos' features, observations, points (a chain.Store's folder)


LOMAG, DISK = "lomag", "disk"
MATCHERS = {  # tried in this order (see lomag); LoMa-G's descriptors are twice DISK's
    LOMAG: Matching("LoMa-G", 16, DATA / "raw" / "view_features_lomag", DATA / "raw" / LOMAG),
    DISK: Matching("DISK", 64, DATA / "raw" / "view_features", DATA / "raw"),
}


def _lomag_matcher() -> match.Matcher:
    from flaneur import lomag  # torch: never with pycolmap

    return lomag.LoMaGMatcher(DATA / "raw" / "models")


def _views(kept: int | None = None, matcher: str = DISK) -> match.Views:
    """The panoramas' views' features as `matcher` finds them (`kept` panoramas' in memory:
    by default, the matcher's)."""
    if matcher == LOMAG:
        found = _lomag_matcher()
    else:
        with httpx.Client(timeout=300) as client:
            found = match.Matcher(match.fetch_models(client, DATA / "raw" / "models"))
    matching = MATCHERS[matcher]
    return match.Views(found, matching.views, matching.kept if kept is None else kept)


def _origin(m: match.Situated, round_: dict, kept: dict | None) -> dict:
    """Who placed a photo, as the confirmation page says it, and why to check it closely: a
    camera proposed (`round_` carries why it was turned down; said yes to: its inputs say
    so), the matcher that found it against today, nothing for chaining."""
    if "proposed" in round_:
        return {"by": "proposée", "check": f"caméra refusée par le contrôle ({round_['proposed']})"}
    if _is_proposal(m.inputs):
        return {"by": "proposée, dite juste"}
    return _found_by(kept) if round_["round"] == 0 and kept and "x" in kept else {}


def _found_by(kept: dict) -> dict:
    """Which matcher placed a photo (`by`, as the user reads it), and why to `check` it
    closely: found by a matcher once others tried found nothing right."""
    matcher = kept.get("matcher")
    if matcher is None:
        return {"by": f"{MATCHERS[DISK].label} (avant {MATCHERS[LOMAG].label})"}
    found = {"by": MATCHERS[matcher].label}
    if before := [m for m in kept.get("tried", []) if m != matcher]:
        found["check"] = ", ".join(MATCHERS[m].label for m in before) + " n'a rien trouvé de juste"
    return found


def _to_match(walk: list[export.WalkPhoto]) -> list[export.WalkPhoto]:
    """The walk's photos `match` works on: those exported."""
    return [me for me in walk if _exported(me.id).exists()]


def _compared_with(
    near_by: list[match.Panorama], here: tuple[float, float]
) -> list[match.Panorama]:
    """The panoramas `match` compares a photo with: MATCH_PANORAMAS of those within
    MATCH_RADIUS_M of its place (`here`: the caption's, or the review's, never a previous
    match's: see _place_of), nearest first (ties: in `near_by`'s order)."""
    return _reach(near_by)(here)


def _reach(
    near_by: list[match.Panorama],
    radius: float = MATCH_RADIUS_M,
    most: int | None = MATCH_PANORAMAS,
) -> Callable[[tuple[float, float]], list[match.Panorama]]:
    """_compared_with for many photos among the same panoramas: where they stand read once
    (or, for another `radius`, the `most` nearest within it: all, for None)."""
    xs = np.array([p.x for p in near_by])
    ys = np.array([p.y for p in near_by])

    def compared_with(here: tuple[float, float]) -> list[match.Panorama]:
        distance = np.hypot(xs - here[0], ys - here[1])
        within = np.flatnonzero(distance <= radius)
        nearest = within[np.argsort(distance[within], kind="stable")][:most]
        return [near_by[i] for i in nearest]

    return compared_with


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
CONFIRM_WIDTH = 960  # pixels: the photo and today's view on the confirmation page
SHEET_PANORAMAS = 3  # today's view from the nearest of these whose picture is at hand
SHEET_MOST_M = 30  # and no further from the camera: a view from further shows another street
OUTLINE_COLOUR = (0, 200, 255)  # BGR amber (a golden yellow): the photo's outlines over today
LABEL_BAND = 26  # pixels: the dark strip the sheet's labels are written on, above the pictures
FRAME_COVER = 0.6  # a row or column this full of edges is a print's or mount's edge, not a street's
SHEET_JPEG = [  # colour not subsampled: the thin amber lines keep their colour
    cv2.IMWRITE_JPEG_QUALITY,
    85,
    cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
    cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
]
LINE_SHARE = 0.25  # of a sheet pixel an edge must cross to be drawn: texture stays lines


def _unframed(edges: np.ndarray) -> np.ndarray:
    """Edges without a print's own border or its mount's where they are found whole (the crop
    may keep them, anywhere in the picture): a row or column they cover for the most part,
    with its neighbours. Measured on the 297 photos exported: only mounts' and prints' edges
    go; a border found in pieces stays (a thin frame, easy to tell from the street)."""
    rows = np.flatnonzero(edges.mean(axis=1) > FRAME_COVER)
    columns = np.flatnonzero(edges.mean(axis=0) > FRAME_COVER)
    edges = edges.copy()
    for i in rows:
        edges[max(i - 1, 0) : i + 2] = False
    for i in columns:
        edges[:, max(i - 1, 0) : i + 2] = False
    return edges


def _outlines(old: np.ndarray, today: np.ndarray) -> np.ndarray:
    """Today's view (at the sheet's size) in grey, dimmed, with the old photo's outlines drawn
    on it in amber: the edges its camera was scored on (outlines.edges_of, at the size it is
    matched at), less the print's own border. Where they line up, it is placed right."""
    drawn = _dimmed(today)
    drawn[_edges_at(old, today.shape[:2])] = OUTLINE_COLOUR
    return drawn


def _edges_at(old: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The old photo's outlines (see _outlines), as a mask of a picture of `size`."""
    edges = _unframed(outlines.edge_mask(_grey(old)))
    h, w = size
    return cv2.resize(edges.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= LINE_SHARE


def _dimmed(today: np.ndarray) -> np.ndarray:
    """Today's view in grey, dimmed: what outlines are drawn on."""
    grey = cv2.cvtColor(today, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.convertScaleAbs(grey, alpha=0.8), cv2.COLOR_GRAY2BGR)


def _toward(camera: match.View, panorama: match.Panorama) -> match.View:
    """How the page's view of today is cut from `panorama` for a `camera` found: framed as it
    (heading, width, horizon; see match.real_view), from where the panorama stands."""
    return replace(camera, x=panorama.x, y=panorama.y, z=panorama.z)


SKY_M = 1e5  # what meets nothing (the sky, a spire over the roofs) stands this far: no parallax
DEPTH_STEP = 4  # pixels: today's view's depth is read every few, and in between interpolated


def _depth_of_view(
    view: match.View,
    size: tuple[int, int],
    walls: facades.Scene | None,
    bdtopo: Blocks,
    street: Callable[[np.ndarray, np.ndarray], np.ndarray],
    step: int = DEPTH_STEP,
    kinds: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Where what a view of today (`view`: cut from a panorama, of `size`) shows stands, every
    `step` pixels (rows × columns × east, north, up; see _where_shown); the walls raised as far
    as the panorama sees building over them (`kinds`: what rays from it see, segment.BUILT...;
    see _up_the_walls)."""
    h, w = size
    rows = np.arange(step / 2 - 0.5, h, step)
    columns = np.arange(step / 2 - 0.5, w, step)
    pixels = np.stack(np.meshgrid(columns, rows), axis=-1).reshape(-1, 2)
    stands, on_wall = _where_shown(view, size, pixels, walls, bdtopo, street)
    shape = (len(rows), len(columns))
    stands = stands.reshape(*shape, 3)
    if kinds is not None:
        directions = match.rays(pixels, size, view.fov, view.horizon, view.heading)
        seen = kinds(directions).reshape(shape)
        stands = _up_the_walls(stands, on_wall.reshape(shape), seen, view)
    return stands


def _up_the_walls(
    stands: np.ndarray, on_wall: np.ndarray, seen: np.ndarray, view: match.View
) -> np.ndarray:
    """Each column of a view's map (`stands`, rows top down; the view upright, each column a
    vertical plane) raised over its topmost wall, up the wall's face: as long as the panorama
    sees building there (`seen`: BD TOPO's walls stop at the gutter, a roof, its dormers and
    chimneys above it; fitted walls, topped already, only gain their dormers), then the sky
    right over it (measured 13/11: else the photo's own roofs, looked for far off, come out in
    today's sky). Trees, or a building seen past the sky (a spire, a tower), stay as they were:
    behind. A panorama not segmented (building everywhere, as segment.seen says it) leaves them
    all as they were, as facades.topped does."""
    if not np.any(seen == segment.SKY):
        return stands
    eye = np.array([view.x, view.y, view.z])
    rows = np.arange(len(on_wall))[:, None]
    top = on_wall.argmax(axis=0)  # each column's topmost wall (none: 0, nothing over it)
    roof = np.where((rows < top) & (seen != segment.BUILT), rows, -1).max(axis=0)  # roof ends
    sky = np.where((rows <= roof) & (seen != segment.SKY), rows, -1).max(axis=0)  # its sky ends
    raised = (rows > sky) & (rows < top)
    reach = np.hypot(*(stands[top, np.arange(len(top)), :2] - eye[:2]).T)  # the wall's face, level
    d = stands - eye
    with np.errstate(divide="ignore", invalid="ignore"):  # straight up: never raised, never read
        face = eye + d * (reach / np.hypot(d[..., 0], d[..., 1]))[..., None]
    return np.where(raised[..., None], face, stands)


def _where_shown(
    view: match.View,
    size: tuple[int, int],
    pixels: np.ndarray,
    walls: facades.Scene | None,
    bdtopo: Blocks,
    street: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Where what `pixels` of a view of today (`view`, from a panorama; of `size`) show stands
    (east, north, up): on the panorama's own walls (`walls`, fitted from its neighbours and seen
    from it, as the placing page takes them), else on BD TOPO's seen from it, else on the
    street (`street`: its height at points), else SKY_M off; and which stand on a wall."""
    directions = match.rays(pixels, size, view.fov, view.horizon, view.heading)
    eye = np.array([view.x, view.y, view.z])
    away = np.full(len(directions), np.inf)
    if walls is not None and walls.walls:
        metres, met = facades.cast(walls, directions)
        away = np.where(met >= 0, metres, np.inf)  # its walls only: the street, BD TOPO's first
    left = np.flatnonzero(~np.isfinite(away))
    met, edge = bdtopo.hit(eye, directions[left])
    away[left[edge >= 0]] = np.linalg.norm(met[edge >= 0] - eye, axis=1)
    to_street = match._to_street(eye, directions, street)
    on_wall = np.isfinite(away) & (away <= to_street)
    away = np.minimum(away, to_street)
    away = np.where(np.isfinite(away), away, SKY_M)
    return eye + away[:, None] * directions, on_wall


BREAK_PX = 4 * DEPTH_STEP  # samples of a map this far apart are a break (a wall's edge, the sky)


def _spread(at: np.ndarray, size: tuple[int, int], step: int) -> tuple[np.ndarray, np.ndarray]:
    """A map sampled every `step` pixels (rows × columns × 2: the pixel each sample is looked
    for at; NaN: nowhere) spread over a picture of `size`, each pixel between its four samples:
    interpolated, except across a break (one of them nowhere, or they lie more than BREAK_PX
    apart), where the nearest is taken."""
    h, w = size
    rows, columns = at.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    grid_x, grid_y = (grid_x - (step / 2 - 0.5)) / step, (grid_y - (step / 2 - 0.5)) / step
    valid = np.isfinite(at).all(axis=2)
    filled = np.where(valid[..., None], at, -10.0 * (h + w)).astype(np.float32)  # far off it
    padded = np.pad(filled, ((0, 1), (0, 1), (0, 0)), mode="edge")
    corners = [padded[a : a + rows, b : b + columns] for a in (0, 1) for b in (0, 1)]
    span = np.max(corners, axis=0) - np.min(corners, axis=0)
    whole = np.pad(valid, ((0, 1), (0, 1)), mode="edge")
    whole = np.all([whole[a : a + rows, b : b + columns] for a in (0, 1) for b in (0, 1)], axis=0)
    broken = ~whole | (span.max(axis=2) > BREAK_PX)
    cell_y = np.clip(np.floor(grid_y), 0, rows - 1).astype(int)
    cell_x = np.clip(np.floor(grid_x), 0, columns - 1).astype(int)
    nearest_too = broken[cell_y, cell_x]
    spread = []
    for k in (0, 1):
        smooth = cv2.remap(
            filled[..., k], grid_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        nearest = cv2.remap(
            filled[..., k], grid_x, grid_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE
        )
        spread.append(np.where(nearest_too, nearest, smooth))
    return spread[0], spread[1]


def _parallax_outlines(
    old: np.ndarray,
    today: np.ndarray,
    camera: match.View,
    seen_from: match.View,
    walls: facades.Scene | None,
    bdtopo: Blocks,
    street: Callable[[np.ndarray, np.ndarray], np.ndarray],
    kinds: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Today's view as _outlines draws it, the old photo's outlines drawn where today's view
    (`seen_from` its panorama) sees what they show: each of its pixels taken where it stands
    (see _depth_of_view; `kinds`: what its rays see), then looked for in the photo through the
    `camera` found (see _spread). What the panorama sees from elsewhere (parallax) no longer
    shifts them; what is far stays."""
    size = today.shape[:2]
    stands = _depth_of_view(seen_from, size, walls, bdtopo, street, kinds=kinds)
    rows, columns = stands.shape[:2]
    at = match.project(camera, stands.reshape(-1, 3), size).reshape(rows, columns, 2)
    to_x, to_y = _spread(at, size, DEPTH_STEP)
    edges = _edges_at(old, size).astype(np.uint8) * 255
    drawn = _dimmed(today)
    drawn[cv2.remap(edges, to_x, to_y, cv2.INTER_LINEAR, borderValue=0) > 64] = OUTLINE_COLOUR
    return drawn


def _inside(at: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Which pixels (rows of column, row; NaN: none) fall in a picture of `size`."""
    h, w = size
    return (
        np.isfinite(at).all(axis=1)
        & (at[:, 0] >= 0)
        & (at[:, 0] < w)
        & (at[:, 1] >= 0)
        & (at[:, 1] < h)
    )


PAIRS_SHOWN = 12  # pairs on the confirmation page: as many as the eye follows
# today's view comes from the panorama COLMAP placed next to the camera when it stands this much
# nearer than the source: it frames the photo as its camera does, and shows where the camera is
# off along its view (109472: from 2.2 m, 8 m too far forward; from its source 37.7 m behind,
# nothing seen); its pairs are projected, the source's matched ones a key away
NEARER_M = 10.0
PLAN_M = 40.0  # the confirmation page's plan: so far each way from its camera


def _most_spread(at: np.ndarray, most: int) -> list[int]:
    """Up to `most` of the points `at` (pixels), the most spread: the first, then each in turn
    the farthest from those chosen (none where one is already: never twice the same)."""
    chosen: list[int] = []
    if len(at):
        chosen.append(0)
        gap = np.hypot(*(at - at[0]).T)
        while len(chosen) < min(most, len(at)):
            far = int(np.argmax(gap))
            if gap[far] == 0:
                break
            chosen.append(far)
            gap = np.minimum(gap, np.hypot(*(at - at[far]).T))
    return chosen


OFFERED = 24  # pairs of a placement said near laid on the placing page, the most spread


def _said_near(photos: dict[str, dict]) -> dict[str, dict]:
    """The photos the user's last word on is near (the right place, the camera to set right),
    on the image as it is now (`photos`: the placing page's), with the placement said near:
    match's, once judged its word is the photo's (it came after any camera proposed); else a
    camera once proposed (`presque`). The placing page puts them first."""
    try:
        kept, judged = _matches(), _Judged()
    except Exception as error:  # a file written wrong: none first, the queue goes on
        typer.echo(f"photos jugées « près » illisibles ({error})", err=True)
        return {}
    try:
        proposed = register.load(PROPOSED_CAMERAS)
    except Exception as error:  # match's placements said near come first all the same
        typer.echo(f"caméras proposées illisibles ({error})", err=True)
        proposed = {}

    def said_on(k: str, m: dict | None) -> str | None:
        if m and "x" in m and _image_of(m["inputs"]) == photos[k]["inputs"]:
            return (judged.on(k, m) or {}).get("verdict")
        return None

    near = {}
    for k in sorted(photos):
        word = said_on(k, kept.get(k))
        if word == "near" or (word is None and said_on(k, proposed.get(k)) == "near"):
            near[k] = kept[k] if word else proposed[k]
    return near


def _offers(
    near: dict[str, dict],
    photos: dict[str, dict],
    by_id: dict[str, match.Panorama],
    fixed: Set[str],
) -> dict[str, dict]:
    """The placements said near (see _said_near) as the placing page opens them (see
    _offer_of): those with pairs of their own (a camera proposed kept no points), none of the
    points the user marked wrong (see confirming.marked)."""
    try:
        wrong = confirming.marked(WRONG_PAIRS)
    except (ValueError, KeyError, TypeError) as error:  # as the confirmation page: none marked
        typer.echo(f"paires marquées fausses illisibles ({error})", err=True)
        wrong = {}
    offers = {}
    for k, m in near.items():
        if _is_proposal(m["inputs"]):
            continue
        try:
            found = _offer_of(
                k, m, by_id, fixed, _size_of(photos[k]["image"]), wrong.get((k, m["inputs"]), set())
            )
        except Exception as error:  # this one opens as any other photo: the others go on
            typer.echo(f"{k} : pas de paires proposées ({error})", err=True)
            continue
        if found:
            offers[k] = found
    return offers


def _offer_of(
    photo_id: str,
    kept: dict,
    by_id: dict[str, match.Panorama],
    fixed: Set[str],
    size: tuple[int, int],
    wrong: Set[tuple[float, float]] = frozenset(),
) -> dict | None:
    """A placement of match, as the placing page opens it (see placing.Sources.offer): in the
    first of its panoramas (see _sources_first) where it has enough pairs of its own (see
    bridging.paired), up to OFFERED of them, the most spread, the photo's pixels at `size` (the
    page's image of it), none of the points at the pixels `wrong` (marked so by the user, see
    confirming.marked); its camera as the starting pin. None
    without its points (kept for other inputs) or pairs."""
    if (found := _found_from(photo_id, kept)) is None:
        return None
    m, points, observed = found
    if wrong:  # marked so on the confirmation page: never laid again
        keys = (confirming.point_key(xy) for xy in points.pixels)
        right = np.fromiter((k not in wrong for k in keys), bool, len(points.pixels))
        points = match.Points(points.pixels[right], points.world[right], points.seen[right])
    for sphere in _sources_first(points, by_id, m, fixed):
        if (paired := bridging.paired(points, observed, sphere, size)) is None:
            continue
        photo_px, view_px = np.round(paired.photo_px, 2), np.round(paired.view_px, 2)
        chosen = _most_spread(photo_px, OFFERED)  # as sent: never twice the same pixel
        return {
            "panorama": paired.panorama,
            "view": asdict(paired.view),
            "pairs": {"photo_px": photo_px[chosen].tolist(), "view_px": view_px[chosen].tolist()},
            **_places({photo_id: m})[photo_id],
        }
    return None


def _placing_ahead(near: Collection[str], carried: dict[str, dict]) -> dict[str, str]:
    """The photos the placing page shows first, with why: those said near (to set right, from
    match's pairs when it kept some), then those carried from a photo placed. Those said yes
    are no longer put first: they are registered on their own (see _registrations)."""
    carried_only = (i for i in carried if i not in near)
    return {
        **dict.fromkeys(sorted(near), "près"),
        **dict.fromkeys(carried_only, "portée"),
    }


def _pairs_shown(
    points: match.Points,
    photo: tuple[int, ...],
    size: tuple[int, int],
    seen_from: match.View,
    matched: np.ndarray | None = None,
) -> list[dict]:
    """Up to PAIRS_SHOWN of the points a camera was found from, the most spread over the photo:
    each one's `index`, its pixel on the `photo` (of that shape, matched at PHOTO_SIDE) shown at
    `size` (`old`), and on today's view (`seen_from` a panorama, `today`) where that panorama
    matched it (`matched`, see _matched_in: those alone if any), else where it sees its point in
    space — not where the camera puts it, which lines up by construction."""
    scale = size[1] / (photo[1] * match.PHOTO_SIDE / max(photo[:2]))
    old = points.pixels * scale
    today = match.project(seen_from, points.world, size)
    here = np.zeros(len(old), bool) if matched is None else ~np.isnan(matched).any(axis=1)
    there = today if matched is None else np.where(here[:, None], matched, today)
    seen = _inside(old, size) & here & _inside(there, size)
    if seen.any():  # matched in the view: those alone, where it matched them
        today, inside = there, seen
    else:
        inside = _inside(old, size) & _inside(today, size)
    kept = np.flatnonzero(inside)
    chosen = [int(kept[i]) for i in _most_spread(old[kept], PAIRS_SHOWN)]
    return [
        {
            "index": k,
            "point": list(confirming.point_key(points.pixels[k])),  # a mark's key
            "old": [round(float(v), 2) for v in old[k]],
            "today": [round(float(v), 2) for v in today[k]],
            "matched": bool(here[k]),
        }
        for k in chosen
    ]


def _plan(
    m: match.Situated,
    panorama: match.Panorama,
    sources: Sequence[tuple[float, float]],
    caption: tuple[float, float],
    footprints: Sequence[Sequence[tuple[float, float]]],
    points: Sequence[Sequence[float]] = (),
) -> dict:
    """The confirmation page's plan around a camera found (metres east and north of it; `reach`
    of them each way): its heading and width, the `panorama` today's view comes from and how
    far, the panoramas its points were seen from (`sources`), its caption's place, today's
    buildings within PLAN_M (`footprints`, Lambert-93) and where its points stand (`points`,
    in the order of the pairs' indexes)."""

    def local(x: float, y: float) -> list[float]:
        return [round(x - m.x, 1), round(y - m.y, 1)]

    def on_plan(f: Sequence[tuple[float, float]]) -> bool:  # its box meets the plan's square
        xs, ys = [x for x, _ in f], [y for _, y in f]
        east = max(xs) >= m.x - PLAN_M and min(xs) <= m.x + PLAN_M
        return east and max(ys) >= m.y - PLAN_M and min(ys) <= m.y + PLAN_M

    near = [f for f in footprints if on_plan(f)]
    return {
        "reach": PLAN_M,
        "camera": {"heading": round(m.heading, 1), "fov": round(m.fov, 1)},
        "panorama": {
            "at": local(panorama.x, panorama.y),
            "away_m": round(float(np.hypot(panorama.x - m.x, panorama.y - m.y)), 1),
        },
        "sources": [local(*at) for at in sources],
        "caption": local(*caption),
        "buildings": [[local(x, y) for x, y in f] for f in near],
        "points": [local(p[0], p[1]) for p in points],
    }


def _to_width(image: np.ndarray, width: int) -> np.ndarray:
    return cv2.resize(
        image, (width, round(width * image.shape[0] / image.shape[1])), interpolation=cv2.INTER_AREA
    )


class _Run(NamedTuple):
    """What one `match` run reads (the panoramas, where they stand, the buildings) and keeps
    (each photo's match, the matchers it was tried with, traces, those placed; the placements
    the user turned down)."""

    near_by: list[match.Panorama]
    then: Callable[[int], Then]
    heights: scene.CameraHeights
    blocks: Blocks
    fixed: set[str]  # the panoramas COLMAP placed (see match.around)
    depth: match.Depth  # where rays meet today's walls fitted, before BD TOPO's (see _wall_depth)
    fitted_from: dict[str, str]  # each photo's walls in reach, as they are (see _photo_walls)
    to_try: dict[str, list[str]]  # the matchers each photo is tried with this run, in order
    scenes: dict[str, str]  # what each photo's match is found from (see _scenes)
    found: dict
    tried: dict[str, list[str]]
    traces: dict
    run: dict
    placed: set[str]
    judged: "_Judged"


def _pictures_fixed() -> tuple[list[panoramas.Picture], set[str]]:
    """The panoramas known, and those COLMAP placed itself, as recorded now (see
    _fixed_by_colmap)."""
    with closing(db.connect(DB_PATH)) as conn:
        pictures = panoramas.load(conn)
    return pictures, _fixed_by_colmap(pictures, register.load(FIXES))


def _colmap_fixed() -> set[str]:
    """The panoramas COLMAP placed itself, as recorded now (see _fixed_by_colmap)."""
    return _pictures_fixed()[1]


def _match_with(matcher: str, photos: list[export.WalkPhoto], at: _Run) -> None:
    """Matches `photos` with one matcher (its own features, observations and points), writing
    each one's match as it goes. A redo that finds nothing leaves a placement made from the
    same inputs, unless turned down (a placement said yes to is never redone: see
    match_photos)."""
    views = _views(matcher=matcher)
    store = chain.Store(MATCHERS[matcher].photos, views.matcher)
    for me in photos:
        here = _place_of(me)
        near = _compared_with(at.near_by, here)
        inputs = _inputs(at.scenes[me.id], me.id)
        image = cv2.imread(str(_exported(me.id)))
        now = at.then(_then_year(me))
        where = _where(me, at.heights, now)
        trace = match.Trace()
        mine = store.features(me.id, _image(me.id), lambda image=image: match.shrunk(image))
        observed = match.observe(
            image, near, at.blocks, views, where, trace, mine, at.heights.origin_z, at.depth
        )
        first = match.place(observed, where, trace)
        observed, (camera, walls, points), focus = match.focus(
            image,
            observed,
            first,
            here,
            at.near_by,
            at.fixed,
            {p.id for p in near},
            at.blocks,
            views,
            where,
            trace,
            mine,
            at.heights.origin_z,
            at.depth,
        )
        store.save_observed(me.id, inputs, observed)
        at.traces[me.id] = at.run[me.id] = {  # the caption's place; what it was matched from
            **_traced(trace),
            "place": list(here),
            "inputs": inputs,
            "focus": focus,
        }
        at.tried[me.id] = [m for m in MATCHERS if m in at.tried[me.id] or m == matcher]
        if camera is None:  # kept, with what it was tried from: not tried again for nothing
            before = at.found.get(me.id)
            if not _stands(me.id, before, inputs, at.judged):
                at.found[me.id] = (
                    {**before, "tried": at.tried[me.id], "tried_from": inputs}
                    if _still_shown(me.id, before, inputs, at.judged)  # near, or marked since
                    else {"inputs": inputs, "tried": at.tried[me.id]}
                )
            typer.echo(f"{me.id} : rien de sûr ({MATCHERS[matcher].label})")
        else:
            moved = float(np.hypot(camera.x - here[0], camera.y - here[1]))
            outline = outlines.contrast(_grey(image), camera, now.blocks)
            situated = match.Situated.of(
                camera,
                street=at.heights.street(camera.x, camera.y),
                walls=walls,
                moved_m=moved,
                inputs=_placed_inputs(inputs, matcher),
                outline=outline,
            )
            at.found[me.id] = {
                **asdict(situated),
                "matcher": matcher,
                "tried": at.tried[me.id],
            }
            at.placed.add(me.id)
            store.save_points(me.id, inputs, points)
            typer.echo(
                f"{me.id} : situé par {MATCHERS[matcher].label} ({camera.agree} points, "
                f"{camera.panoramas} panoramas, déplacé de {moved:.1f} m, "
                f"contours {outline:.2f})"
            )
        if camera is not None or matcher == at.to_try[me.id][-1]:  # done with these walls
            at.found[me.id] = {**at.found[me.id], "fitted_from": at.fitted_from[me.id]}
        write_json(MATCHES, at.found)
        write_json(MATCH_TRACE, at.traces)  # each photo's last run: a stop loses the one in hand


def _checked_in(near_by: list[match.Panorama]) -> dict[str, list[str]]:
    """For each area, the matchers whose known-answer check passed there for its panoramas as
    they are: how sure a placement of each is there (see placed.Matched), not whether it is
    tried (the chaining's check is a measure only: the chained photos listed are said yes to).
    Says which checks are still to make (with the command), and which failed."""
    checked: dict[str, list[str]] = {name: [] for name in pilot.AREAS}
    to_make, failed = defaultdict(list), []
    for name in pilot.AREAS:
        for m in MATCHERS:
            check = _per_area(MATCH_CHECK, m).get(name, {})
            if not _check_current(check, m, name, near_by):
                to_make[m].append(name)
            elif check.get("passed"):
                checked[name].append(m)
            else:
                failed.append(f"{MATCHERS[m].label} ({name})")
    for m, areas in to_make.items():
        typer.echo(
            f"{MATCHERS[m].label} pas encore contrôlé pour ces panoramas ({', '.join(areas)}) : "
            f"`flaneur match-check --perimes --matcher {m}`"
        )
    if failed:
        typer.echo(f"contrôle non passé : {', '.join(failed)} (leurs placements le disent)")
    return checked


def _check_current(check: dict, matcher: str, area: str, near_by: list[match.Panorama]) -> bool:
    """Whether a matcher's known-answer check in an area (as kept) was made for its panoramas
    and the code as they are (see _check_setup)."""
    return check.get("setup") == _check_setup(matcher, area, near_by)


@app.command("match")
def match_photos(
    photo: Annotated[list[str] | None, typer.Option(help="seulement ces photos")] = None,
    again: Annotated[
        bool,
        typer.Option(
            help="refaire les photos déjà faites (une place trouvée reste si rien n'est trouvé)"
        ),
    ] = False,
) -> None:
    """Situe les photos en les comparant aux panoramas d'aujourd'hui autour d'elles : position,
    direction, champ et horizon, là où des façades d'alors sont encore debout. Avec LoMa-G
    d'abord, puis DISK pour celles qu'il n'a pas situées (la page de confirmation dit lequel),
    dans tous les quartiers : le contrôle à réponse connue dit seulement combien chacun y est
    sûr (data/pilot/checked_areas.json, lu par la liste des photos placées)."""
    with closing(db.connect(DB_PATH)) as conn:
        walk = export.photos(conn, DATA / "pilot", REVIEWS)
    heights, near_by, then = _matching()
    write_json(CHECKED_AREAS, _checked_in(near_by))
    scenes = _scenes(walk, near_by)
    blocks = then(date.today().year).blocks
    found, judged, proposed = _matches(), _Judged(), register.load(PROPOSED_CAMERAS)
    chosen = [me for me in _to_match(walk) if not photo or me.id in photo]
    fixed = _colmap_fixed()
    walls = _wall_scenes(near_by, fixed)
    walls_of, fitted_from = _photo_walls(walls, near_by), {}
    tried, to_try = {}, {}  # the matchers each photo was tried with, and is to be
    for me in chosen:
        kept, inputs = found.get(me.id), _inputs(scenes[me.id], me.id)
        if any(p and judged.holds(me.id, p) for p in (kept, proposed.get(me.id))):
            tried[me.id], to_try[me.id] = _tried(kept, inputs), []  # said yes to: never redone
            continue
        fitted_from[me.id] = walls_of(_place_of(me))
        if again or photo or (kept and kept.get("fitted_from") != fitted_from[me.id]):
            tried[me.id], to_try[me.id] = [], list(MATCHERS)  # new walls: redone, still shown
        else:
            turned_down = bool(kept) and "x" in kept and judged.turned_down(me.id, kept)
            tried[me.id] = _tried(kept, inputs)
            to_try[me.id] = _left_to_try(kept, inputs, turned_down)
    at = _Run(
        near_by=near_by,
        then=then,
        heights=heights,
        blocks=blocks,
        fixed=fixed,
        depth=_wall_depth(walls),
        fitted_from=fitted_from,
        to_try=to_try,
        scenes=scenes,
        found=found,
        tried=tried,
        traces=register.load(MATCH_TRACE),
        run={},
        placed=set(),
        judged=judged,
    )

    chosen = sorted((me for me in chosen if to_try[me.id]), key=_visit)
    for matcher in MATCHERS:  # one model at a time: each let go before the next
        if photos := [me for me in chosen if matcher in to_try[me.id] and me.id not in at.placed]:
            _match_with(matcher, photos, at)
            from flaneur import lomag

            lomag.released()  # the views' cache holds the matcher in a cycle: only gc frees it
    typer.echo(f"où ces photos s'arrêtent : {_funnel(at.run.values())}")
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


def _said_on(placed: list[dict]) -> dict[str, dict]:
    """For each photo `placed` (id, inputs, camera), the user's word that holds for it as it is
    placed (see _Judged.on), read anew."""
    judged = _Judged()
    return {
        p["id"]: said
        for p in placed
        if (said := judged.on(p["id"], {"inputs": p["inputs"], **p["camera"]}))
    }


def _verdicts() -> dict[str, dict]:
    """The user's last word on each photo placed (ANCHORS: yes, near or no)."""
    return review.latest(ANCHORS, key="photo_id")


# A verdict holds for a camera found again this near the one judged, and turned this little:
# placed from other panoramas, it is the same camera (to the eye, on the confirmation page).
CARRY_M = 1.0
CARRY_DEG = 2.0
CARRY_HORIZON = 0.02  # and its horizon this near (a share of the photo's height): framed alike


def _whole(camera: dict, other: dict) -> bool:
    """Whether two cameras are kept with their height, width and horizon: comparable."""
    return {"x", "above_street", "fov", "horizon"} <= camera.keys() & other.keys()


def _same_camera(camera: dict, other: dict) -> bool:
    """Whether two cameras (x, y, above_street, heading, fov, horizon) are one found again:
    within CARRY_M across and up, CARRY_DEG both turned and widened, CARRY_HORIZON tilted. A
    camera judged without its height, width and horizon (before they were kept) is none: the
    verdict holds for its inputs only."""
    if not _whole(camera, other):
        return False
    return bool(
        np.hypot(camera["x"] - other["x"], camera["y"] - other["y"]) <= CARRY_M
        and abs(camera["above_street"] - other["above_street"]) <= CARRY_M
        and abs(angle(camera["heading"], other["heading"])) <= CARRY_DEG
        and abs(camera["fov"] - other["fov"]) <= CARRY_DEG
        and abs(camera["horizon"] - other["horizon"]) <= CARRY_HORIZON
    )


def _stands(
    photo_id: str, kept: dict | None, inputs: str, judged: "_Judged", marks: bool = True
) -> bool:
    """Whether a placement as kept stands as it is now (see _found_as_now, `marks`) and the
    user did not turn it down, unless the yes holds (see _Judged)."""
    return _found_as_now(photo_id, kept, inputs, judged, marks) and (
        judged.holds(photo_id, kept) or not judged.turned_down(photo_id, kept)
    )


def _found_as_now(
    photo_id: str, kept: dict | None, inputs: str, judged: "_Judged", marks: bool = True
) -> bool:
    """Whether a placement as kept is one of the photo as it is now: the user said yes to it and
    the photo is the same, or it was made from `inputs` (the photo and its panoramas now).
    Without `marks`, its landmarks' marks aside, for what shows it: they only help match find
    the camera, so one marked since stays shown, on the same image, until match redoes it (an
    annotation never takes a placement away)."""
    if not kept or "x" not in kept:
        return False
    if marks:
        return judged.holds(photo_id, kept) or _placed_now(kept, inputs)
    if judged.said_yes(photo_id, kept) and _image_of(kept["inputs"]) == _image_of(inputs):
        return True
    placed_from = _placed_inputs(inputs, kept.get("matcher"))
    return _marks_aside(_unchained(kept["inputs"])) == _marks_aside(placed_from)


def _still_shown(photo_id: str, kept: dict | None, inputs: str, judged: "_Judged") -> bool:
    """Whether a placement a run found nothing better than is kept all the same: shown as the
    photo is now (see _found_as_now, its marks aside) and not refused (said near: the right
    place, the camera to set by hand). Kept, it says what it was tried from (see _tried):
    not tried again every night."""
    said = judged.on(photo_id, kept) if kept else None
    return _found_as_now(photo_id, kept, inputs, judged, marks=False) and not (
        said and said["verdict"] == "no"
    )


def _marks_aside(inputs: str) -> str:
    """A placement's inputs (see _inputs, _placed_inputs) without its landmarks' mark."""
    fields = inputs.split(":")
    if len(fields) > 2 and re.fullmatch(r"[0-9a-f]{8}", fields[2]):
        del fields[2]
    return ":".join(fields)


def _image_of(inputs: str) -> str:
    """The photo's image a placement's inputs were made from (see _inputs)."""
    return _photo_of(inputs).split(":")[0]


def _photo_of(inputs: str) -> str:
    """The part of a placement's inputs the photo itself gives (see _inputs): its image, and
    its landmarks' mark when it has any."""
    fields = inputs.split(":")
    if len(fields) > 2 and re.fullmatch(r"[0-9a-f]{8}", fields[2]):
        return f"{fields[1]}:{fields[2]}"
    return fields[1] if len(fields) > 1 else ""


class _Judged:
    """Every verdict the user gave (ANCHORS), by photo, in the order given: what each says
    of a placement."""

    def __init__(self, path: Path | None = None) -> None:
        self.by_photo: dict[str, list[dict]] = {}
        for row in jsonl_rows(path or ANCHORS):
            self.by_photo.setdefault(row["photo_id"], []).append(row)

    def on(self, photo_id: str, placed: dict) -> dict | None:
        """The user's last word on a placement (`placed`: its inputs, and its camera's x, y and
        heading when placed): given to it, or to a camera of the same photo found within CARRY_M
        and CARRY_DEG of it (placed again from other panoramas, or by the other matcher), unless
        the user since took back a camera that near the one it was given to (a yes followed by
        a no beside it holds for that very camera only). A yes holds for its inputs (never
        redone); a no or near for its inputs and camera, kept whole."""
        said_since: list[dict] = []
        for said in reversed(self.by_photo.get(photo_id, [])):
            camera = said.get("camera")  # verdicts older than their camera's record: none
            other = camera and _whole(camera, placed) and not _same_camera(camera, placed)
            if said["inputs"] == placed["inputs"] and (said["verdict"] == "yes" or not other):
                return said  # a no or near: not on another camera new walls found from them
            if (
                camera
                and "x" in placed
                and _photo_of(said["inputs"]) == _photo_of(placed["inputs"])
                and _same_camera(camera, placed)
            ):
                taken_back = any(
                    later["verdict"] != said["verdict"]
                    and "camera" in later
                    and _same_camera(later["camera"], camera)
                    for later in said_since
                )
                return None if taken_back else said
            said_since.append(said)
        return None

    def said_yes(self, photo_id: str, placed: dict) -> bool:
        return bool(said := self.on(photo_id, placed)) and said["verdict"] == "yes"

    def turned_down(self, photo_id: str, placed: dict) -> bool:
        """Whether the user did not take a placement as it is (no; or near: the right buildings,
        the camera off)."""
        return bool(said := self.on(photo_id, placed)) and said["verdict"] != "yes"

    def holds(self, photo_id: str, placed: dict) -> bool:
        """Whether a placement the user said yes to stands as it is: the photo is the same (see
        _photo_part), whatever became of the panoramas (its camera is in the world)."""
        return (
            "x" in placed
            and self.said_yes(photo_id, placed)
            and _photo_of(placed["inputs"]) == _photo_part(photo_id)
        )


MERGE_PX = 1.5  # a photo's LoMa-G keypoint this near one of its DISK keypoints: that detail


def _chain_setup() -> str:
    """What chaining and its check now depend on: what matching does, chaining's own code
    (chain.py, and what of it lives here, with its numbers), LoMa-G's (its anchors are paired
    through it) and the areas (the check's confidence is said for each)."""
    here = [_Chaining, _merged, _rounds, _chain_passes, _inconsistent, _placed_from, _too_far]
    here += [_moved, _rays_at, _kept_store, _features, _for_pages, _Judged, _is_chained]
    here += [_situation, _keep_chained, _chained_wins]
    check = [_degraded, _replay, _check_passed, chain_check, _check_fold, _off_row, _by_area]
    numbers = (MERGE_PX, CHAIN_PASSES, CHECK_M, CHECK_DEG, CHECK_BLOCK_M, CHECK_FOLDS)
    numbers += (CHECK_HALVINGS, CHECK_FEWEST_RAYS, DECOY_M, CHAIN_CHECK_LEAST, CHAIN_CHECK_WRONG)
    code = [inspect.getsource(f) for f in here + check] + [repr(numbers), _lomag_code()]
    areas = {name: square.wkt for name, square in pilot.AREAS.items()}
    digest = _digest([Path(chain.__file__).read_text(), *code, areas])[:8]
    return f"{_setup()}:{digest}"


def _rays_at(rays: match.Rays, rows: np.ndarray) -> match.Rays:
    return match.Rays(*(getattr(rays, f.name)[rows] for f in fields(match.Rays)))


def _merged(disk: np.ndarray, lomag: np.ndarray, offset: int) -> np.ndarray:
    """The rows of a photo's details its LoMa-G keypoints are: a DISK keypoint's within
    MERGE_PX (the same detail), else their own, after `offset`."""
    own = offset + np.arange(len(lomag))
    if not len(disk) or not len(lomag):
        return own
    pixels, nearest = cKDTree(disk).query(lomag)
    return np.where(pixels <= MERGE_PX, nearest, own)


class _LoadedWhenNeeded:
    """A matcher made only when first asked for features or pairs (its model is loaded then):
    those kept are read without it. `precision`: the made matcher's (lomag.LoMaGMatcher's
    for LOMAG), which names the folder its pairs are kept in."""

    def __init__(self, precision: str, make: Callable[[], match.Matcher]) -> None:
        self.precision, self._matcher = precision, functools.cache(make)

    def features(self, bgr: np.ndarray) -> match.Features:
        return self._matcher().features(bgr)

    def match(self, a: match.Features, b: match.Features) -> tuple[np.ndarray, np.ndarray]:
        return self._matcher().match(a, b)


CHAINED = ":from-"  # a chained placement's inputs: then its anchors' (see _placed_from)


def _is_chained(inputs: str) -> bool:
    return CHAINED in inputs


CHAINED_POINTS = DATA / "raw" / "chained"  # the points chained photos were placed from


def _keep_chained(photo_id: str, inputs: str, points: match.Points) -> None:
    """Keeps a chained photo's points apart from match's (one file a photo each: a placement
    set aside by eye keeps its own), for the pages (see _kept_store); one placed without
    anchors is a placement against today: nothing kept."""
    if _is_chained(inputs):
        chain.Store(CHAINED_POINTS, None).save_points(photo_id, inputs, points)


def _unchained(inputs: str) -> str:
    """What the photo of a placement was observed from: its inputs, a chained one's anchors off."""
    return inputs.split(CHAINED, 1)[0]


def _placed_from(inputs: str, parents: list[chain.Anchor]) -> str:
    """What a chained photo was placed from: its own inputs, and its anchors' (a verdict on
    it holds while they do); without any, what it was placed from against today."""
    if not parents:
        return inputs
    from_ = repr(sorted((a.id, a.inputs) for a in parents))
    return f"{inputs}{CHAINED}{hashlib.sha1(from_.encode()).hexdigest()[:8]}"


class _Chaining:
    """What chaining needs: the walk's photos and where each could stand, the store of what
    each was matched with, the copies of each negative, and the photos placed against today
    (round 0) that the user did not turn down."""

    def __init__(self) -> None:
        self.key = _chain_setup()
        with closing(db.connect(DB_PATH)) as conn:
            walk = export.photos(conn, DATA / "pilot", REVIEWS)
            self.negatives = dedup.negatives(conn)
        self.heights, self.near_by, then = _matching()
        self.scenes = _scenes(walk, self.near_by)  # what each photo's own match is found from
        # by year: the buildings of a year are built once (a few kept at a time)
        self.walk = sorted((me for me in walk if _exported(me.id).exists()), key=_then_year)
        self.where = {me.id: _where(me, self.heights, then(_then_year(me))) for me in self.walk}
        self.then = then
        self.store = chain.Store(DATA / "raw", _views().matcher)
        self.lomag_store = chain.Store(
            MATCHERS[LOMAG].photos, _LoadedWhenNeeded(LOMAG, _lomag_matcher)
        )
        self.by_id = {me.id: me for me in self.walk}
        self.refused = set(_set_aside())  # judged wrong by eye: not placed so again
        self.turned_down = _turned_down()  # by the user: not chained at all
        self.situated, matches = _situated(), _matches()
        # proposed and said yes to: no points kept, neither anchors nor chained
        self.elsewhere = {k for k, m in self.situated.items() if _is_proposal(m.inputs)}
        self.by_lomag = {  # those paired through LoMa-G
            k for k in self.situated if (matches.get(k) or {}).get("matcher") == LOMAG
        } - self.elsewhere
        self.cut: dict[str, match.Observed] = {}  # observations made for the check (_replay)
        self._seen: dict[str, match.Observed | None] = {}  # each photo's, read once
        self._lomag_keypoints: dict[str, np.ndarray] = {}
        self._disk_keypoints: dict[str, int] = {}  # how many: an anchor's LoMa-G rows' start
        self._rows_of: dict[str, np.ndarray] = {}
        self.round_zero: dict[str, chain.Anchor] = {}
        aside = _set_aside()  # by eye: neither an anchor nor a truth to check against
        for photo_id, m in self.situated.items():
            if (
                photo_id in self.by_id
                and photo_id not in self.elsewhere
                and (photo_id, m.inputs) not in aside
            ):
                if (kept := _kept_store(m, matches.get(photo_id))) is None:
                    continue  # proposed: no points (left out as `elsewhere` already)
                if (points := kept[0].points(photo_id, kept[1])) is not None:
                    view = m.view(self.heights.street(m.x, m.y))
                    self.round_zero[photo_id] = self.anchor(photo_id, view, points, m.inputs)
        # the photos chained before the user said yes to, as placed: anchors, never redone
        self.kept_chained: dict[str, dict] = {}
        self.judged, store = _Judged(), chain.Store(CHAINED_POINTS, None)
        for k, row in register.load(CHAINS).get("photos", {}).items():
            c = row["situated"]
            today = asdict(self.situated[k]) if k in self.round_zero else None
            if (
                k in self.by_id
                and self.judged.holds(k, c)
                and _chained_wins(self.judged, k, c, today)
            ):
                if (
                    _is_chained(c["inputs"])
                    and (points := store.points(k, c["inputs"])) is not None
                ):
                    m = _situation(c)
                    view = m.view(self.heights.street(m.x, m.y))
                    self.round_zero[k] = self.anchor(k, view, points, c["inputs"])
                    self.kept_chained[k] = row

    def features(self, photo_id: str) -> match.Features:
        return _features(self.store, photo_id)

    def lomag_features(self, photo_id: str) -> match.Features:
        return _features(self.lomag_store, photo_id)

    def lomag_keypoints(self, photo_id: str) -> np.ndarray:
        if photo_id not in self._lomag_keypoints:
            self._lomag_keypoints[photo_id] = self.lomag_features(photo_id).keypoints
        return self._lomag_keypoints[photo_id]

    def anchor(
        self, photo_id: str, view: match.View, points: match.Points, inputs: str
    ) -> chain.Anchor:
        """An old photo placed, as others are placed from: its keypoints (DISK's, then
        LoMa-G's for one LoMa-G placed, paired through it; descriptors stay in the stores),
        its camera and points."""
        features = self.features(photo_id)
        keypoints = features.keypoints
        self._disk_keypoints[photo_id] = len(keypoints)
        if photo_id in self.by_lomag:
            keypoints = np.concatenate([keypoints, self.lomag_keypoints(photo_id)])
        return chain.Anchor(photo_id, view, keypoints, features.size, points, inputs)

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
        """A photo's details and their rays in today's panoramas (DISK's), its LoMa-G keypoints
        after them (as a LoMa-G anchor's keypoints are); as cut for the check, if so."""
        if me.id in self.cut:
            return self.cut[me.id]
        if me.id not in self._seen:
            now = _inputs(self.scenes[me.id], me.id)
            observed = self.store.observed(me.id, now)
            if observed is not None:
                pixels = np.concatenate([observed.pixels, self.lomag_keypoints(me.id)])
                observed = replace(observed, pixels=pixels)
            else:  # LoMa-G's only (it placed the photo): its rays on the same rows
                placed = self.situated[me.id].inputs if me.id in self.situated else now
                observed = self._lomag_observed(me.id, _found_inputs(placed, LOMAG))
            self._seen[me.id] = observed
        return self._seen[me.id]

    def _lomag_observed(self, photo_id: str, inputs: str) -> match.Observed | None:
        """A photo's observation by LoMa-G with its details as chaining has them (its DISK
        keypoints, then its LoMa-G ones): the rays on its LoMa-G keypoints, on their rows
        (see _merged); those on the bands zoomed on its landmarks left out."""
        seen = self.lomag_store.observed(photo_id, inputs)
        if seen is None:
            return None
        disk, lomag = self.features(photo_id).keypoints, self.lomag_keypoints(photo_id)
        rays = _rays_at(seen.rays, np.flatnonzero(seen.rays.keys < len(lomag)))
        rays = replace(rays, keys=self.rows_of(photo_id, len(disk) + len(lomag))[rays.keys])
        return replace(seen, pixels=np.concatenate([disk, lomag]), rays=rays)

    def rows_of(self, photo_id: str, details: int) -> np.ndarray:
        """The rows of a photo's `details` its LoMa-G keypoints are: they come last (see
        observed), those beside its DISK keypoints are theirs (see _merged)."""
        if photo_id not in self._rows_of:
            mine = self.lomag_keypoints(photo_id)
            disk = self.features(photo_id).keypoints
            self._rows_of[photo_id] = _merged(disk, mine, details - len(mine))
        return self._rows_of[photo_id]

    def paired(
        self, me: export.WalkPhoto, observed: match.Observed
    ) -> Callable[[chain.Anchor], tuple[np.ndarray, np.ndarray]]:
        """How a photo's details pair with an anchor's (rows in the photo, rows in the
        anchor): through the matcher that placed the anchor (LoMa-G's, by_lomag), else DISK."""
        image = (me.id, _image(me.id))

        def pairs(anchor: chain.Anchor) -> tuple[np.ndarray, np.ndarray]:
            other = (anchor.id, _image(anchor.id))
            if anchor.id not in self.by_lomag:
                rows = self.store.pairs(image, other, self.features)
                return rows[:, 0], rows[:, 1]
            found = self.lomag_store.pairs(image, other, self.lomag_features)
            rows_of = self.rows_of(me.id, len(observed.pixels))
            after = self._disk_keypoints[anchor.id]  # its LoMa-G keypoints' rows (see anchor)
            rows = np.column_stack([rows_of[found[:, 0]], after + found[:, 1]])
            rows = rows[np.sort(np.unique(rows[:, 0], return_index=True)[1])]  # a detail once
            return rows[:, 0], rows[:, 1]

        return pairs

    def place(
        self,
        me: export.WalkPhoto,
        observed: match.Observed,
        anchors: list[chain.Anchor],
        trace: match.Trace,
    ) -> match.Found:
        """A photo's camera from what it was observed with and from its anchors, standing
        where it could."""
        rays, known = chain.from_anchors(anchors, self.paired(me, observed))
        return match.place(observed, self.where[me.id], trace, rays, known)


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
def chain_photos() -> None:
    """Situe, tour après tour, les photos que les panoramas seuls n'ont pas situées, à partir
    des vieilles photos déjà situées près d'elles (leurs ancres) : au tour k, celles des
    tours précédents. Seules les caméras sûres sont gardées ; chacune garde son tour et ses
    ancres. Puis chaque photo est re-située avec les points des autres (pas de celles situées
    depuis elle) : la photo enchaînée qui bouge le plus est écartée et les tours refaits sans
    elle ; celles du tour 0 qui bougent sont signalées, à juger. Écrit chains.json. Le contrôle
    (`chain-check`) n'arrête rien : il dit, par quartier, la confiance des photos enchaînées."""
    work = _Chaining()
    typer.echo(f"tour 0 : {len(work.round_zero)} photos situées contre les panoramas")
    results, placed, doubts, set_aside = _chain_passes(work)
    _for_pages(work, results, placed)
    for photo_id, why in doubts.items():
        if photo_id in work.round_zero:
            typer.echo(f"{photo_id} : en désaccord avec les autres ({why})")
    write_json(
        CHAINS,
        {
            "setup": work.key,
            "photos": work.kept_chained | results,
            "doubts": {k: str(why) for k, why in doubts.items() if k in work.round_zero},
            "set_aside": set_aside,
        },
    )
    write_json(CHECKED_AREAS, _checked_in(work.near_by))  # what the list says of each area: now
    typer.echo(
        f"{len(results)} photos situées par enchaînement ; {len(set_aside)} écartées ; "
        f"écrit dans {CHAINS}"
    )


def _for_pages(work: _Chaining, results: dict[str, dict], placed: dict[str, chain.Anchor]) -> None:
    """What the pages show of the photos chained, once chaining is done: the points each was
    placed from (kept apart: see _keep_chained) and how well the buildings then fit it."""
    for photo_id, row in results.items():
        anchor, me = placed[photo_id], work.by_id[photo_id]
        _keep_chained(photo_id, anchor.inputs, anchor.points)
        row["situated"]["outline"] = outlines.contrast(
            _grey(cv2.imread(str(_exported(photo_id)))),
            anchor.view,
            work.then(_then_year(me)).blocks,
        )


def _chain_passes(
    work: _Chaining,
) -> tuple[dict[str, dict], dict[str, chain.Anchor], dict[str, "_Doubt"], dict[str, str]]:
    """Chaining whole (see chain_photos): its rounds run again without the chained photo most
    in doubt, at most CHAIN_PASSES times. The photos chained (their outline not yet
    measured), all those placed, the doubts left and the photos set aside, with why."""
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
    return results, placed, doubts, set_aside


def _rounds(work: _Chaining, left_out: set[str]) -> tuple[dict[str, dict], dict[str, chain.Anchor]]:
    """The rounds of chaining from the photos placed against today (round 0), photos
    `left_out` never placed: each photo chained (a Situated's fields, its outline not yet
    measured: NaN; its round and anchors), and all those placed."""
    placed, results = dict(work.round_zero), {}
    tried: dict[str, set[str]] = {}  # the anchors each photo was last tried with
    for k in range(1, chain.MOST_ROUNDS + 1):
        new, why = {}, []
        for me in work.walk:
            if me.id in placed or me.id in left_out or me.id in work.turned_down:
                continue
            if me.id in work.elsewhere:
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
                inputs = _placed_from(_inputs(work.scenes[me.id], me.id), parents)
                if (me.id, inputs) in work.refused:
                    trace.verdict = "refusée"
                else:
                    new[me.id] = (found, inputs)
            why.append({"verdict": trace.verdict})
        typer.echo(f"tour {k} : {len(new)} photos situées ; les autres : {_funnel(why)}")
        for photo_id, (found, inputs) in new.items():
            camera = found.camera
            x, y = work.where[photo_id].around
            situated = match.Situated.of(
                camera,
                street=work.heights.street(camera.x, camera.y),
                walls=found.walls,
                moved_m=float(np.hypot(camera.x - x, camera.y - y)),
                inputs=inputs,
                outline=float("nan"),
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


def _too_far(metres: float, degrees: float) -> bool:
    """Whether a camera found again stands or turns too far from where it was to be the same."""
    return metres > CHECK_M or degrees > CHECK_DEG


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
        elif _too_far(*(off := _moved(camera, anchor.view))):
            doubts[photo_id] = _Doubt(off[0], f"bouge de {off[0]:.1f} m, {off[1]:.1f}°")
    return doubts


def _chained_from(results: dict[str, dict], photo_id: str) -> set[str]:
    """The photos chained, directly or not, from `photo_id` (results in round order)."""
    found = {photo_id}
    for k, row in results.items():
        if found & set(row["from"]):
            found.add(k)
    return found - {photo_id}


CHECK_BLOCK_M = 150  # the chaining's check: photos in a block this wide held out together
CHECK_FOLDS = 5  # the blocks, in this many sets: one held out at a time
CHECK_HALVINGS = 12  # a photo held out: its rays halved at most this often
CHECK_FEWEST_RAYS = 8  # one matching still places from this few: left out of the check
DECOY_M = (30, 150)  # a decoy: a photo confirmed this far from the one it is given for
CHAIN_CHECK_LEAST = 10  # photos held out and placed again, for the check to tell anything
CHAIN_CHECK_WRONG = 0.1  # of them, at most this share off by more than CHECK_M or CHECK_DEG


def _check_fold(x: float, y: float) -> int:
    """The set a camera's block falls in (see CHECK_BLOCK_M, CHECK_FOLDS)."""
    return hash((int(x // CHECK_BLOCK_M), int(y // CHECK_BLOCK_M))) % CHECK_FOLDS


def _degraded(
    observed: match.Observed, where: match.Where, rng: np.random.Generator
) -> match.Observed | None:
    """A photo placed against today made like those chaining has to place: its rays drawn at
    random and halved until matching alone no longer places it (at most CHECK_HALVINGS
    times); none if it still does from CHECK_FEWEST_RAYS rays or fewer."""
    rays = observed.rays
    for _ in range(CHECK_HALVINGS + 1):
        cut = replace(observed, rays=rays)
        if match.place(cut, where, match.Trace()).camera is None:
            return cut
        if len(rays.keys) <= CHECK_FEWEST_RAYS:
            return None
        rays = _rays_at(rays, np.sort(rng.permutation(len(rays.keys))[: len(rays.keys) // 2]))
    return None


def _replay(
    work: _Chaining,
    held: dict[str, match.Observed],
    decoys: dict[str, tuple[match.Observed, str]],
) -> _Chaining:
    """Chaining with confirmed photos held out of round 0: each `held` observed as cut (see
    _degraded), each decoy as cut and given for being where another held photo stands
    (`decoys`: its cut observation and that photo). What was read is shared with `work`."""
    replay = shallow(work)
    replay.cut = held | {d: seen for d, (seen, _) in decoys.items()}
    replay.round_zero = {k: a for k, a in work.round_zero.items() if k not in replay.cut}
    replay.where = work.where | {
        d: replace(work.where[as_], sights=()) for d, (_, as_) in decoys.items()
    }
    return replay


def _off_row(camera: match.View, truth: match.View) -> dict:
    """How far a camera found again stands and turns from the truth, and whether too far."""
    metres, degrees = _moved(camera, truth)
    return {"off": [round(metres, 2), round(degrees, 1)], "wrong": _too_far(metres, degrees)}


def _camera_row(view: match.View) -> list[float]:
    """A camera as the check keeps it: east, north (metres), heading and width (degrees)."""
    return [round(view.x, 2), round(view.y, 2), round(view.heading, 1), round(view.fov, 1)]


class _Gate(NamedTuple):
    passed: bool
    wanting: list[str]
    back: int  # photos held out and placed again
    wrong: int  # of them, too far
    fooled: int  # decoys taken


def _check_passed(rows: list[dict]) -> _Gate:
    """Whether the chaining's check passes, and what it lacks: enough photos held out placed
    again (fewer tell nothing), few of them off, no decoy taken."""
    back = [r for r in rows if r["kind"] == "vérité" and "off" in r]
    wrong = sum(1 for r in back if r["wrong"])
    fooled = sum(1 for r in rows if r["kind"] == "leurre" and r.get("wrong"))
    wanting = [
        need
        for met, need in [
            (len(back) >= CHAIN_CHECK_LEAST, f"au moins {CHAIN_CHECK_LEAST} photos revenues"),
            (wrong <= CHAIN_CHECK_WRONG * len(back), f"au plus {CHAIN_CHECK_WRONG:.0%} fausses"),
            (fooled == 0, "aucun leurre pris"),
        ]
        if not met
    ]
    return _Gate(not wanting, wanting, len(back), wrong, fooled)


def _by_area(rows: list[dict]) -> dict[str, list[dict]]:
    """The chaining's check's rows, by area (each its `area`)."""
    by_area: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["area"]):
        by_area[r["area"]].append(r)
    return dict(by_area)


@app.command("chain-check")
def chain_check(seed: int = 0) -> None:
    """Contrôle à réponse connue de l'enchaînement. Les photos du tour 0 dites justes par
    l'utilisateur sont ôtées du tour 0 par blocs (CHECK_BLOCK_M, CHECK_FOLDS ensembles), rendues
    semblables aux photos à enchaîner (leurs rayons réduits jusqu'à ce que les panoramas seuls
    ne les situent plus), puis l'enchaînement entier tourne sans elles : chacune revenue est
    comparée à sa place (juste à CHECK_M et CHECK_DEG). Un leurre par photo : une autre photo
    confirmée à DECOY_M de là, donnée pour être à sa place, ne doit pas y être située. La porte
    (assez de photos revenues, peu de fausses, aucun leurre pris), par quartier, est la
    mesure de l'enchaînement (chain_check.json) : elle n'arrête rien, et seules les photos
    enchaînées dites justes entrent dans la liste des photos placées."""
    work = _Chaining()
    rng = np.random.default_rng(seed)
    truths = sorted(  # placed against today: the chaining is not checked against its own
        k
        for k in work.round_zero
        if k not in work.kept_chained and work.judged.holds(k, asdict(work.situated[k]))
    )
    at = {k: np.array([work.round_zero[k].view.x, work.round_zero[k].view.y]) for k in truths}
    area = {k: _area_of(tuple(at[k])) for k in truths}
    cuts: dict[str, match.Observed | None] = {}  # each photo's, the same in every role

    def cut(photo_id: str) -> match.Observed | None:
        if photo_id not in cuts:
            seen = work.observed(work.by_id[photo_id])
            cuts[photo_id] = seen and _degraded(seen, work.where[photo_id], rng)
        return cuts[photo_id]

    rows: list[dict] = []
    for fold in range(CHECK_FOLDS):
        held: dict[str, match.Observed] = {}
        for k in (k for k in truths if _check_fold(*at[k]) == fold):
            if (seen := cut(k)) is None:
                rows.append(
                    {
                        "photo": k,
                        "kind": "vérité",
                        "fold": fold,
                        "area": area[k],
                        "why": "hors test",
                    }
                )
            else:
                held[k] = seen
        decoys: dict[str, tuple[match.Observed, str]] = {}
        for k in held:
            near = [
                o
                for o in truths
                if o not in held
                and o not in decoys
                and DECOY_M[0] <= np.hypot(*(at[o] - at[k])) <= DECOY_M[1]
            ]
            for d in rng.permutation(near):  # the first that can be cut
                if (seen := cut(str(d))) is not None:
                    decoys[str(d)] = (seen, k)
                    break
        typer.echo(f"pli {fold} : {len(held)} photos ôtées, {len(decoys)} leurres")
        if not held:
            continue
        replay = _replay(work, held, decoys)
        results, placed, _, set_aside = _chain_passes(replay)
        for k, seen in held.items():
            row = {"photo": k, "kind": "vérité", "fold": fold, "rays": len(seen.rays.keys)}
            row |= {"area": area[k], "truth": _camera_row(work.round_zero[k].view)}
            row["lomag"] = k in work.by_lomag  # placed by LoMa-G
            row["anchors"] = len(replay.anchors(work.by_id[k], replay.round_zero))
            if k in results:
                row |= _off_row(placed[k].view, work.round_zero[k].view)
                row |= {"from": results[k]["from"], "agree": results[k]["situated"]["agree"]}
                row["camera"] = _camera_row(placed[k].view)
            else:
                row["why"] = set_aside.get(k, "pas revenue")
            rows.append(row)
        for d, (_, k) in decoys.items():
            row = {"photo": d, "kind": "leurre", "as": k, "fold": fold}
            row["area"] = area[k]  # where it is given for being
            if d in results:
                row |= _off_row(placed[d].view, work.round_zero[d].view)
                row["camera"] = _camera_row(placed[d].view)
            rows.append(row)
    by_area = _by_area(rows)
    gates = {a: _check_passed(mine) for a, mine in by_area.items()}
    areas = {a: gate._asdict() for a, gate in gates.items()}
    write_json(CHAIN_CHECK, {"setup": work.key, "areas": areas, "rows": rows})
    write_json(CHECKED_AREAS, _checked_in(work.near_by))  # what the list says of each area: now
    for a, gate in gates.items():
        mine = by_area[a]
        tried = sum(1 for r in mine if r["kind"] == "vérité" and "rays" in r)
        typer.echo(
            f"{a} : {tried} photos confirmées testées : {gate.back} revenues (dont fausses : "
            f"{gate.wrong}) ; leurres pris : {gate.fooled} sur "
            f"{sum(1 for r in mine if r['kind'] == 'leurre')} ; "
            + ("passé" if gate.passed else f"pas encore ({', '.join(gate.wanting)})")
        )
        for r in mine:
            if r["kind"] == "vérité" and "off" in r:
                typer.echo(f"  {r['photo']} : {r['off'][0]} m, {r['off'][1]}° (depuis {r['from']})")


@app.command("match-check")
def match_check(
    tries: int = 20,
    seed: int = 0,
    matcher: Annotated[str, typer.Option(help="lomag ou disk")] = DISK,
    area: Annotated[str | None, typer.Option(help="un quartier (sinon : chacun)")] = None,
    stale: Annotated[
        bool, typer.Option("--perimes", help="seulement ceux dont le contrôle n'est plus à jour")
    ] = False,
) -> None:
    """Contrôle à réponse connue, sur `tries` panoramas. Une vue découpée dans l'un, prise
    pour une vieille photo, est située contre les autres et comparée à sa vraie caméra ; puis
    la même vue dont il ne reste qu'une bande (le reste : une autre rue), vieillie ; puis une
    vue vieillie d'une autre rue, qui ne doit pas être située. La porte (assez de vues
    situées, à moins de CHECK_M et CHECK_DEG en médiane, presque aucune fausse) est gardée pour
    chaque apparieur (`--matcher`) et chaque quartier (`--area`) : un placement de `match` dit
    si la porte de son apparieur est passée dans son quartier. `--perimes` : seulement les
    quartiers dont le contrôle a été fait pour d'autres panoramas ou un autre code."""
    if matcher not in MATCHERS:
        raise typer.BadParameter(f"apparieur : l'un de {', '.join(MATCHERS)}")
    if area is not None and area not in pilot.AREAS:
        raise typer.BadParameter(f"quartier : l'un de {', '.join(pilot.AREAS)}")
    heights, near_by, then = _matching()
    areas = [area] if area else list(pilot.AREAS)
    if stale:
        kept = _per_area(MATCH_CHECK, matcher)
        areas = [a for a in areas if not _check_current(kept.get(a, {}), matcher, a, near_by)]
        if not areas:
            typer.echo(f"{MATCHERS[matcher].label} : chaque contrôle est à jour")
            return
    year = date.today().year
    today = then(year)
    shown = functools.partial(  # only landmarks whose being hidden can be told
        landmarks.in_view,
        origin_z=heights.origin_z,
        blocks=today.blocks,
        year=year,
        landmarks=landmarks.among(today.blocks),
    )
    views = _views(matcher=matcher)
    # only panoramas COLMAP placed: another's place is borrowed from its run, or its phone's
    fixed = _colmap_fixed()
    walls = _wall_scenes(near_by, fixed)
    for name in areas:
        _check_area(name, tries, seed, matcher, near_by, fixed, heights, today, shown, views, walls)
    write_json(CHECKED_AREAS, _checked_in(near_by))  # what the list says of each area: now


def _wanting(
    tries: int, situated: int, metres: float, degrees: float, wrong: int, attempts: int
) -> list[str]:
    """What a known-answer check lacks to pass (none: it passes): of its `tries` views,
    `situated` found (`metres`, `degrees`: their median distance to the truth), `wrong` cameras
    among all its `attempts` (views, bands and decoys)."""
    return [
        need
        for met, need in [
            (tries >= CHECK_LEAST, f"au moins {CHECK_LEAST} vues"),
            (situated >= CHECK_SHARE * tries, f"au moins {CHECK_SHARE:.0%} situées"),
            (metres < CHECK_M, f"moins de {CHECK_M} m"),
            (degrees < CHECK_DEG, f"moins de {CHECK_DEG}°"),
            (wrong <= WRONG_SHARE * attempts, f"au plus {WRONG_SHARE:.0%} de fausses"),
        ]
        if not met
    ]


def _check_area(
    name: str,
    tries: int,
    seed: int,
    matcher: str,
    near_by: list[match.Panorama],
    fixed: set[str],
    heights: scene.CameraHeights,
    today: "Then",
    shown: Callable[[match.View], tuple[match.Sight, ...]],
    views,
    walls: _Scenes,
) -> None:
    """One area's known-answer check (see match_check): its views cut from the panoramas COLMAP
    placed in it, its decoys from its own streets far from them, its result kept for it alone;
    their points on today's `walls` fitted, never with the panorama a view is cut from."""
    rng = np.random.default_rng(seed)
    marking = np.random.default_rng([seed, 1])  # its own: the fakes stay the same
    square = pilot.AREAS[name]
    in_area = sorted((p for p in near_by if square.contains(Point(p.x, p.y))), key=lambda p: p.id)
    at = np.array([(p.x, p.y) for p in in_area]).reshape(-1, 2)

    def far_from(p: match.Panorama) -> list[match.Panorama]:  # a decoy: another street
        return [in_area[i] for i in np.flatnonzero(np.hypot(*(at - (p.x, p.y)).T) > FAR_M)]

    placed = [p for p in in_area if p.id in fixed and far_from(p)]
    tries = min(tries, len(placed))
    if not tries:  # the other areas are checked all the same
        typer.echo(f"{name} : aucun panorama placé par COLMAP avec un leurre possible")
        return
    found: dict[str, list[Off | None]] = {"vue": [], "bande": [], "leurre": []}
    traces: dict[str, list[dict]] = {kind: [] for kind in found}
    for k in rng.choice(len(placed), tries, replace=False):
        me = placed[k]
        heading = float(rng.uniform(0, 360))
        far = far_from(me)
        true = _cut_view(me, heading)
        there, facing = far[rng.integers(len(far))], float(rng.uniform(0, 360))
        elsewhere = _cut_view(there, facing)
        keep = rng.uniform(*CHECK_KEPT)
        span = (0.0, keep) if rng.random() < 0.5 else (1 - keep, 1.0)  # one side or the other
        band = fakes.aged(fakes.mixed(true, elsewhere, span), rng)
        # as match compares a photo (see _compared_with): within reach of the area only
        others = _nearest(near_by, me.x, me.y, MATCH_RADIUS_M, but=me.id)[:MATCH_PANORAMAS]
        compared = {me.id} | {p.id for p in others}  # not matched again (see match.focus)
        depth = _wall_depth(walls, but=frozenset({me.id}))  # never its own walls
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
            mine = views.matcher.features(match.shrunk(fake))
            observed = match.observe(
                fake, others, today.blocks, views, where, trace, mine, heights.origin_z, depth
            )
            first = match.place(observed, where, trace)
            _, (camera, _, _), _ = match.focus(
                fake,
                observed,
                first,
                (me.x, me.y),
                near_by,
                fixed,
                compared,
                today.blocks,
                views,
                where,
                trace,
                mine,
                heights.origin_z,
                depth,
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
    wanting = _wanting(tries, len(situated), metres, degrees, len(wrong), attempts)
    passed = not wanting
    write_json(_of_matcher(CHECK_TRACE, matcher), {**_per_area(CHECK_TRACE, matcher), name: traces})
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
    checks = _per_area(MATCH_CHECK, matcher)
    checks[name] = {
        "setup": _check_setup(matcher, name, near_by),
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
    }
    write_json(_of_matcher(MATCH_CHECK, matcher), checks)
    typer.echo(
        f"{name} — vues : {len(situated)} situées sur {tries}, écart médian {metres:.1f} m et "
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


class Framed(NamedTuple):
    """A photo situated (as exported; and `old`, at a width) and today's true view framed as
    its camera (`today`, of the same size), from a panorama `away_m` from the camera."""

    photo: np.ndarray
    old: np.ndarray
    today: np.ndarray
    away_m: float
    camera: match.View  # as found, standing on the street
    panorama: match.Panorama  # the one today's view is cut from


def _away(views: Sequence[match.Panorama], m: match.Situated) -> float | None:
    """How far from a photo's camera stands the panorama its view of today comes from: the
    first of `views` (see _viewed_from) whose picture is at hand, as _framed takes it."""
    shown = _first_read(views)
    return None if shown is None else round(float(np.hypot(shown.x - m.x, shown.y - m.y)), 1)


def _viewed_from(
    near_by: list[match.Panorama],
    m: match.Situated,
    sources: Sequence[match.Panorama],
    fixed: Set[str],
    walled: Callable[[match.Panorama], bool] | None = None,
    nearest_first: bool = False,
) -> list[match.Panorama]:
    """The panoramas today's view of a placement may come from, the first whose picture is read
    taken: those its camera was found from (`sources`, see _sources_first), wherever they stand
    (in the frame of its points, they matched what the photo shows); then the nearest
    SHEET_PANORAMAS within SHEET_MOST_M, those within VIEW_WALLS_M whose walls are fitted
    (`walled`) before them, COLMAP's (`fixed`) before those that only borrow their run's place
    (see _placed). `nearest_first`: COLMAP's nearest the camera before them all, when it stands
    NEARER_M nearer than the one that would be shown."""
    near = sorted(_nearest(near_by, m.x, m.y, SHEET_MOST_M), key=lambda p: p.id not in fixed)
    walls_near = [
        p for p in near if np.hypot(p.x - m.x, p.y - m.y) <= VIEW_WALLS_M and walled and walled(p)
    ]
    views = [*sources, *walls_near, *near[:SHEET_PANORAMAS]]
    if nearest_first and (shown := _first_read(views)) is not None:
        close = _first_read(p for p in near if p.id in fixed)  # near: COLMAP's, nearest first

        def away(p: match.Panorama) -> float:
            return math.hypot(p.x - m.x, p.y - m.y)

        if close is not None and away(close) + NEARER_M <= away(shown):
            return [close, *views]
    return views


def _other_view(
    views: Sequence[match.Panorama], shown: match.Panorama, m: match.Situated
) -> dict | None:
    """The other panorama today's view of a placement may be shown from (see _viewed_from: the
    one next to its camera, or its source), how far from the camera; none when it is the one
    shown."""
    other = _first_read(views)
    if other is None or other.id == shown.id:
        return None
    return {"away_m": _away([other], m)}


def _first_read(views: Iterable[match.Panorama]) -> match.Panorama | None:
    """The first of `views` whose picture is at hand (as _framed takes it)."""
    return next((p for p in views if p.path.exists()), None)


SOURCE_SHARE = 0.5  # see _sources_first (measured 20/11: no placement of the day changed)


def _sources_first(
    points: match.Points | None,
    by_id: dict[str, match.Panorama],
    m: match.Situated,
    fixed: Set[str],
) -> list[match.Panorama]:
    """The panoramas a camera's points were seen from: COLMAP's (`fixed`) first among those that
    saw SOURCE_SHARE of the top one's points at least, then those that saw the most, then the
    nearest it."""
    if points is None:
        return []
    counted = Counter(i for seen in points.seen for i in seen if i in by_id)
    most = max(counted.values(), default=0)
    return sorted(
        (by_id[i] for i in counted),
        key=lambda p: (
            counted[p.id] < SOURCE_SHARE * most,
            p.id not in fixed,
            -counted[p.id],
            float(np.hypot(p.x - m.x, p.y - m.y)),
        ),
    )


RAY_FAR_M = 100.0  # a point this far along a matched ray: where the ray meets today's view


def _matched_in(
    points: match.Points,
    observed: match.Observed,
    panorama: match.Panorama,
    seen_from: match.View,
    size: tuple[int, int],
) -> np.ndarray:
    """Where a `panorama` matched each of a camera's points (its own ray, kept by `match` in
    `observed`; a point's pixel is its detail's, as chain.Store keeps it: see bridging.own_rows), on
    today's view from it (`seen_from`, `size`); NaN where it saw none."""
    rays, rows = observed.rays, bridging.own_rows(points, observed, panorama.id)
    found = np.full((len(points.pixels), 2), np.nan)
    hit = rows >= 0
    if hit.any():
        far = rays.origins[rows[hit]] + RAY_FAR_M * rays.directions[rows[hit]]
        found[hit] = match.project(seen_from, far, size)
    return found


def _framed(
    photo_id: str,
    m: match.Situated,
    heights: scene.CameraHeights,
    views: Sequence[match.Panorama],
    read: Callable[[Path], np.ndarray | None],
    width: int,
) -> Framed | None:
    """A photo situated, `width` wide, and today's true view framed as the camera found, from
    the first of `views` (see _viewed_from) whose picture is read; none without the photo or a
    panorama."""
    photo = cv2.imread(str(_exported(photo_id)))
    if photo is None:
        return None
    old = _to_width(photo, width)
    for panorama in views:  # one may be unreadable
        if (picture := read(panorama.path)) is not None:
            break
    else:
        return None
    camera = m.view(heights.street(m.x, m.y))
    real = match.real_view(camera, old.shape[:2], panorama, picture)
    away = float(np.hypot(panorama.x - m.x, panorama.y - m.y))
    return Framed(photo, old, real, away, camera, panorama)


def _sheet_row(
    photo_id: str,
    m: match.Situated,
    heights: scene.CameraHeights,
    views: Sequence[match.Panorama],
    read: Callable[[Path], np.ndarray | None],
    width: int = SHEET_COLUMN,
) -> np.ndarray | None:
    """A photo situated, then today's true view from the first of `views` at hand (see
    _viewed_from), framed as the camera found, and the photo's outlines drawn on that view;
    each `width` wide."""
    if (framed := _framed(photo_id, m, heights, views, read, width)) is None:
        return None
    photo, old, real, away = framed.photo, framed.old, framed.today, framed.away_m
    pictures = np.hstack([old, real, _outlines(photo, real)])
    row = np.vstack([np.full((LABEL_BAND, *pictures.shape[1:]), 40, np.uint8), pictures])
    for column, text in enumerate(
        [
            f"{photo_id}  {m.agree} pts, {m.panoramas} sources, a {m.moved_m:.0f} m de sa legende",
            f"aujourd'hui, panorama a {away:.0f} m",
            "contours de la photo sur aujourd'hui",
        ]
    ):
        wide = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        scale = 0.45 * min(1.0, (width - 12) / wide)  # within its own column
        cv2.putText(
            row, text, (column * width + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1
        )
    return row


@app.command("match-sheet")
def match_sheet(per_page: int = 10) -> None:
    """Planches de contrôle, une ligne par photo située : la photo ; la vraie vue d'un panorama
    voisin, cadrée comme sa caméra retrouvée ; les contours de la photo sur cette vue."""
    heights, near_by, _ = _matching()
    if not near_by:
        raise typer.BadParameter("aucun panorama téléchargé")
    fixed = _colmap_fixed()
    by_id, kept = {p.id: p for p in near_by}, _matches()
    read = functools.lru_cache(maxsize=SHEET_PANORAMAS + 1)(lambda path: cv2.imread(str(path)))

    def views(photo_id: str, m: match.Situated) -> list[match.Panorama]:
        points = _points_of(photo_id, m, kept.get(photo_id))
        sources = _sources_first(points, by_id, m, fixed)
        return _viewed_from(near_by, m, sources, fixed, nearest_first=True)  # as the page

    rows = [
        row
        for photo_id, m in sorted(_situated().items())
        if (row := _sheet_row(photo_id, m, heights, views(photo_id, m), read)) is not None
    ]
    for stale in (DATA / "pilot").glob("match_sheet_*.jpg"):  # an earlier run's pages
        stale.unlink()
    for page in range(0, len(rows), per_page):
        out = DATA / "pilot" / f"match_sheet_{page // per_page + 1}.jpg"
        cv2.imwrite(str(out), np.vstack(rows[page : page + per_page]), SHEET_JPEG)
    typer.echo(f"{len(rows)} photos situées ; planches dans {DATA / 'pilot'}/match_sheet_*.jpg")


def _anchoring(placements: dict[str, dict]) -> set[str]:
    """The photos chaining anchors on among `placements` against today (id: fields, its
    inputs among them) not turned down: neither proposed (no points) nor set aside by eye."""
    aside = _set_aside()
    return {
        k
        for k, p in placements.items()
        if not _is_proposal(p["inputs"]) and (k, p["inputs"]) not in aside
    }


def _chained_wins(judged: "_Judged", photo_id: str, chained: dict, today: dict | None) -> bool:
    """Whether a photo's chained placement is shown rather than its placement against `today`
    (none: there is none): the user said yes to it, and not to that one."""
    return judged.said_yes(photo_id, chained) and not (today and judged.said_yes(photo_id, today))


def _chained_said_yes(scenes: dict, judged: "_Judged") -> dict[str, dict]:
    """The last chaining's placements the user said yes to, of the photo as it is now (see
    _found_as_now): the only chained placements listed and walked. The others are proposals, on
    the confirmation page; a near would not last there (the next chaining leaves photos turned
    down aside and keeps only those said yes), unlike match's near placements."""
    said = {}
    for k, row in register.load(CHAINS).get("photos", {}).items():
        c = row["situated"]
        if (
            judged.said_yes(k, c)
            and _is_chained(c["inputs"])
            and k in scenes
            and _exported(k).exists()
            and _found_as_now(k, c, _inputs(scenes[k], k), judged, marks=False)
        ):
            said[k] = c
    return said


def _chained(scenes: dict, judged: "_Judged", anchors: Set[str]) -> dict:
    """What the last chaining found (chains.json): the photos it placed from others (`photos`,
    each its `situated`, a Situated's fields; `round`; the photos it was placed `from`), those
    still of the photo as it is now (`scenes`; see _found_as_now: others wait for the next
    chaining, as match's wait for the next match) whose anchors are all still placed (`anchors`:
    the placements against today now; then the photos kept, in round order), unless said yes to
    (a yes holds); and the `doubts` (why each). One placed without anchors keeps no points of its
    own (see _keep_chained): left out."""
    kept = register.load(CHAINS)
    placed_now, photos = set(anchors), {}
    for k, row in kept.get("photos", {}).items():
        c = row["situated"]
        if (
            _is_chained(c["inputs"])
            and k in scenes
            and _exported(k).exists()
            and _found_as_now(k, c, _inputs(scenes[k], k), judged, marks=False)
            and (judged.said_yes(k, c) or placed_now.issuperset(row["from"]))
        ):
            photos[k] = row
            placed_now.add(k)
    return {**kept, "photos": photos}


def _with_chained(
    situated: dict[str, match.Situated], chained: dict, shown: set[str], judged: "_Judged"
) -> tuple[dict[str, match.Situated], dict[str, dict]]:
    """The photos `shown` placed against today (round 0) or else by the last chaining, and
    each one's round and the photos it was placed `from`: a placement against today wins over
    a chaining, unless the user said yes to the chained one only (see _chained_wins)."""
    placed = {k: m for k, m in situated.items() if k in shown}
    rounds = {k: {"round": 0, "from": []} for k in placed}
    for k, row in chained.get("photos", {}).items():
        today = asdict(placed[k]) if k in placed else None
        if k in shown and (today is None or _chained_wins(judged, k, row["situated"], today)):
            placed[k] = match.Situated(**row["situated"])
            rounds[k] = {"round": row["round"], "from": row["from"]}
    return placed, rounds


VIEW_WALLS_M = 10.0  # today's view from the nearest panorama with its walls fitted this near the
# camera found (else the nearest): the parallax drawn from that panorama's own walls


def _no_matcher() -> match.Views:
    raise RuntimeError("la page de confirmation ne réveille pas l'apparieur")


@functools.cache
def _walls_code() -> str:
    """The code walls are fitted by (beside cli's _Scenes, see WALLS_VERSION), hashed."""
    return _hash_of(facades, depth, segment)


@functools.cache  # for one command: the area's buildings read once
def _street_segments() -> Callable[[float, float], list[facades.Segment]]:
    """BD TOPO's street facades the walls are fitted from, as the placing page has them (see
    _placing_world): for match, the check and the confirmation page alike."""
    with closing(db.connect(DB_PATH)) as conn:
        area = placements.area(conn, buildings.MARGIN_M)
    return _street_facades(_buildings(area))


def _wall_scenes(
    near_by: list[match.Panorama],
    fixed: set[str],
    views: Callable[[], match.Views] | None = None,
) -> _Scenes:
    """Today's walls around panoramas from the pairs on disk alone (listed once: the matcher
    never woken), segmented as `flaneur segmenter` left them, kept in WALLS_FITTED: those match
    places its points on, and the confirmation page draws the parallax from. With `views` (see
    _views: called once), pairs are matched as needed (flaneur murs): those on disk looked at
    each time."""
    masks = segment.Segmenter(VIEW_STRUCTURE, threading.Lock(), _levelled_name, make=False)
    made = None if views else frozenset(p.name for p in PAIR_MATCHES.glob("*.npz"))
    return _Scenes(
        near_by,
        fixed,
        functools.cache(views) if views else _no_matcher,  # the matcher loaded once
        threading.Lock(),
        _street_segments(),
        WALLS,
        seen=_seeing(False, masks),
        kept=WALLS_FITTED,
        made=made,
        masks=masks,
    )


def _wall_homes(
    spheres: list[match.Panorama],
    places: list[tuple[float, float]],
    placed: list[tuple[float, float]],
    areas: Sequence[BaseGeometry],
) -> list[match.Panorama]:
    """The panoramas whose walls match may use (within SCENE_REACH_M of a photo's place), in
    the order the nights make them: within reach of a placement first, then in an area
    (`areas`: grown by that reach), then the rest; the nearest a photo first."""
    if not places:
        return []
    xy = np.array([(p.x, p.y) for p in spheres]).reshape(-1, 2)

    def nearest(points: list[tuple[float, float]]) -> np.ndarray:
        if not points:
            return np.full(len(xy), np.inf)
        return cKDTree(np.asarray(points, float)).query(xy)[0]

    to_photo, to_placed = nearest(places), nearest(placed)
    rank = [
        0 if d <= SCENE_REACH_M else 1 if any(a.contains(Point(p.x, p.y)) for a in areas) else 2
        for p, d in zip(spheres, to_placed, strict=True)
    ]
    order = sorted(range(len(spheres)), key=lambda i: (rank[i], to_photo[i]))
    return [spheres[i] for i in order if to_photo[i] <= SCENE_REACH_M]


def _segment_homes(
    homes: list[match.Panorama],
    spheres: list[match.Panorama],
    fixed: set[str],
    masks: segment.Segmenter,
) -> None:
    """Segments, in the order of `homes`, each one and its neighbours walls are fitted from
    (see depth.candidates) that has no mask yet (`masks` makes them)."""
    seen: set[str] = set()
    wanted = [
        p
        for home in homes
        for p in [home, *depth.candidates(home, spheres, fixed)]
        if not (p.id in seen or seen.add(p.id))
    ]
    _segment_each([p for p in wanted if not masks.kept(p)], masks)


def _segment_each(panoramas: list[match.Panorama], masks: segment.Segmenter) -> None:
    """Segments each panorama in turn; one that fails (memory, a picture) is said and left,
    the others go on."""
    started = time.monotonic()
    for n, panorama in enumerate(panoramas, 1):
        try:
            if masks.made(panorama) is None:
                typer.echo(f"{panorama.id} : image illisible")
        except Exception as error:  # this one left: the others go on
            typer.echo(f"{panorama.id} : laissé ({error})", err=True)
        if n % 50 == 0 or n == len(panoramas):
            typer.echo(f"{n}/{len(panoramas)} ({time.monotonic() - started:.0f} s)")


def _make_walls(homes: list[match.Panorama], scenes: _Scenes) -> None:
    """Each home matched with its neighbours and its walls fitted and kept (see _Scenes.make),
    in order. Stopped at any time, it loses one panorama at most: each pair and walls kept are
    written whole."""
    for n, home in enumerate(homes, 1):
        started = time.monotonic()
        try:
            found = scenes.make(home)
        except Exception as error:  # this one left (memory, a picture): the others go on
            typer.echo(f"{home.id} : laissé ({error})", err=True)
            continue
        walls = 0 if found is None else len(found.walls)
        typer.echo(f"{n} {home.id} : {walls} murs en {time.monotonic() - started:.1f} s")


@app.command("murs")
def make_walls(
    masques: Annotated[bool, typer.Option(help="seulement la segmentation qui manque")] = False,
    page: Annotated[int, typer.Option(help="le port de la page pour placer")] = 8772,
) -> None:
    """Fait les murs ajustés que `match` et la page de confirmation lisent : chaque panorama à
    moins de SCENE_REACH_M d'une photo, apparié à ses voisins (DISK, une fois par paire), ses
    murs ajustés et gardés ; près des placements d'abord, puis dans les quartiers, puis le
    reste, le plus près d'une photo d'abord. `--masques` : seulement la segmentation qui leur
    manque (le modèle et l'apparieur ne tiennent pas ensemble en mémoire : deux passages).
    Reprend où il s'était arrêté ; refusé pendant que la page pour placer tourne (mémoire)."""
    _page_closed(page)
    heights, near_by, then = _matching()  # as match has them: the walls kept are its own
    with closing(db.connect(DB_PATH)) as conn:
        walk = export.photos(conn, DATA / "pilot", REVIEWS)
    fixed = _colmap_fixed()
    placed = [(m["x"], m["y"]) for m in _matches().values() if m and "x" in m]
    areas = [square.buffer(SCENE_REACH_M) for square in pilot.AREAS.values()]
    homes = _wall_homes(near_by, [_place_of(me) for me in walk], placed, areas)
    typer.echo(f"{len(homes)} panoramas à moins de {SCENE_REACH_M:.0f} m d'une photo")
    if masques:
        masks = segment.Segmenter(VIEW_STRUCTURE, threading.Lock(), _levelled_name)
        _segment_homes(homes, near_by, fixed, masks)
        return
    _make_walls(homes, _wall_scenes(near_by, fixed, lambda: _views(kept=8)))


def _wall_depth(scenes: _Scenes, but: frozenset[str] = frozenset()) -> match.Depth:
    """Where rays from a panorama meet its walls fitted (see match.Depth), each panorama's
    scene made once a run (see _Scenes.scene: only the segmented fit walls). `but`: panoramas
    left out."""

    @functools.cache
    def scene_of(panorama: str) -> facades.Scene | None:
        return scenes.scene(panorama, None, made_only=True, but=but)

    def depth_of(panorama: match.Panorama, directions: np.ndarray) -> tuple:
        return facades.depth_of(scene_of(panorama.id), directions)

    return depth_of


def _photo_walls(
    scenes: _Scenes, near_by: list[match.Panorama]
) -> Callable[[tuple[float, float]], str]:
    """The walls a photo's match may meet, fitted and drawn, for the panoramas within
    SCENE_REACH_M of its place (heights from each panorama's, to 10 cm): other walls there
    redo it (see match_photos), not a change of code or of pairs that leaves them as they are;
    each panorama's made once a run."""
    reached = _reach(near_by, SCENE_REACH_M, None)

    def rounded(v: float | None, z: float = 0.0) -> float | None:
        return None if v is None else round(v - z, 1)

    @functools.cache
    def of(panorama: match.Panorama) -> list:
        found = scenes.scene(panorama.id, None, made_only=True)
        if found is None:
            return []
        walls = sorted(
            (
                [*(rounded(c) for c in (*w.a, *w.b)), rounded(w.top, panorama.z)]
                for w in found.walls
            ),
            key=str,  # a top may be none
        )
        return [rounded(found.ground, panorama.z), walls]

    return lambda here: _digest([of(p) for p in reached(here)])


class _ViewWalls:
    """The walls the confirmation page draws the parallax from: each panorama's own, fitted
    from the neighbours it is matched with already (see _wall_scenes: those match saw), and
    BD TOPO's (`blocks`) where they say nothing."""

    def __init__(self, near_by: list[match.Panorama], fixed: set[str], blocks: Blocks):
        self.blocks = blocks
        self.scenes = _wall_scenes(near_by, fixed)
        self._fitted: dict[str, bool] = {}

    def fitted(self, panorama: match.Panorama) -> bool:
        """Whether a panorama may have walls of its own: a neighbour matched already, or walls
        drawn by hand near it (whether any could be fitted is known once they are)."""
        if panorama.id not in self._fitted:
            drawn = facades.saved_near(WALLS, panorama.x, panorama.y, now=self.scenes.now)[0]
            self._fitted[panorama.id] = bool(drawn) or self.scenes.has_pairs(panorama)
        return self._fitted[panorama.id]

    def of(self, panorama: match.Panorama) -> tuple[facades.Scene | None, Blocks]:
        """A panorama's own walls (none: none fitted nor drawn) and BD TOPO's in its reach."""
        own = None
        if self.fitted(panorama):
            own = self.scenes.scene(panorama.id, None, made_only=True)
        bdtopo = self.blocks.near(panorama.x, panorama.y, match.NEAR_M)
        return (own if own is not None and own.walls else None), bdtopo


def _kept_store(m: match.Situated, kept: dict | None) -> tuple[chain.Store, str] | None:
    """Where a placement's points and rays are kept, and under which inputs: as `match` kept
    them, a chained one's as the chaining did (see _keep_chained); none for a camera
    proposed."""
    if _is_proposal(m.inputs):
        return None
    if _is_chained(m.inputs):
        return chain.Store(CHAINED_POINTS, None), m.inputs
    matcher = (kept or {}).get("matcher") or DISK
    return chain.Store(MATCHERS[matcher].photos, None), _found_inputs(m.inputs, matcher)


REGISTERED = DATA / "pilot" / "registered.json"  # each placement of match registered on its own
PLACED_PHOTOS = DATA / "pilot" / "placed.json"  # the photos placed, each with its status
CHECKED_AREAS = DATA / "pilot" / "checked_areas.json"  # each area's matchers passing the check


SURVEY_PRODUCER = "Immergis pour la Ville de Paris"  # the survey: panoramas sure of their pose


def _sure_panoramas() -> tuple[set[str], set[str]]:
    """The panoramas sure of their pose (the survey's, COLMAP's), and all those known."""
    pictures, fixed = _pictures_fixed()
    survey = {p.id for p in pictures if p.producer == SURVEY_PRODUCER}
    return survey | fixed, {p.id for p in pictures}


def _placed_photos(write: bool = False, answering: tuple | None = None) -> dict[str, placed.Placed]:
    """The photos placed now (see placed.build): the placing page's answers on the image each
    has now; match's placements of each photo as it is now (see _found_as_now), the yes
    given to cameras matching turned down (as _situated), and the last chaining's placements the
    user said yes to (the others are only proposed, on the confirmation page), each with the
    user's word,
    whether its method passes the check in the area its camera stands in, its landmarks' state
    (on that image). `write`: kept in PLACED_PHOTOS too, when changed; `answering`: the placing
    page's answers, already read (see _placing_answers)."""
    _, answers, far = answering or _placing_answers()
    scenes, judged, checked = _scenes_now(), _Judged(), register.load(CHECKED_AREAS)
    marks = marking.states(SIGHTS)
    sure = functools.cache(_sure_panoramas)  # read once, if a placement has its points

    def unsure(k: str, m: dict) -> bool:
        points = _points_of(k, _situation(m), m)
        return points is not None and placed.unsure(points, *sure())

    def weighed(k: str, m: dict) -> placed.Matched:
        said, image = judged.on(k, m), _image(k)
        mark, marked_on = marks.get(k, (None, None))
        if _photo_of(m["inputs"]) != _photo_part(k):  # placed before these marks: not on them
            mark = None
        chained = _is_chained(m["inputs"])
        source = placed.CHAINED if chained else placed.MATCHED
        method = source if chained else (m.get("matcher") or DISK)
        return placed.Matched(
            m,
            said and said["verdict"],
            image,
            method in checked.get(_area_of((m["x"], m["y"])), []),
            mark if marked_on == image else None,
            unsure(k, m),
            source,
        )

    matched = {
        k: weighed(k, m)
        for k, m in _matches().items()
        if k in scenes
        and _exported(k).exists()
        and _found_as_now(k, m, _inputs(scenes[k], k), judged, marks=False)
    }
    for k, p in _proposed_now(scenes, judged, marks=False).items():
        if judged.said_yes(k, p) and not (k in matched and matched[k].verdict == "yes"):
            matched[k] = weighed(k, p)  # the user's yes wins over a placement not judged
    for k, c in _chained_said_yes(scenes, judged).items():  # the others: proposed only
        if k not in matched or matched[k].verdict != "yes":  # a yes wins over a placement not
            matched[k] = weighed(k, c)  # said yes to (see _chained_wins)
    found = placed.build(_on_image_now(answers), far, matched, _registrations)
    rows = {k: asdict(p) for k, p in found.items()}
    if write and rows != register.load(PLACED_PHOTOS):
        write_json(PLACED_PHOTOS, rows)
    return found


def _shown_in_3d(answering: tuple | None = None) -> dict[str, dict]:
    """The photos placed with a registration (see _placed_photos; `answering`: the placing
    page's answers, already read), as the 3D walk, the export and the copies carried take them
    (see placed.Placed.for_scene)."""
    return {
        k: p.for_scene() for k, p in _placed_photos(answering=answering).items() if p.registration
    }


@functools.cache
def _registering_code() -> str:
    """The code a placement of match is registered by, hashed."""
    return _hash_of(bridging, registration, placed)


def _registered_as(kept: dict) -> str:
    """What a placement's registration is made from: its inputs, its camera and the code."""
    camera = [round(kept[k], 3) for k in ("x", "y", "heading", "fov", "horizon")]
    return _digest([kept["inputs"], camera, _registering_code()])


def _registrations(standing: dict[str, dict]) -> dict[str, placed.Registered]:
    """Each placement of match (`standing`: as kept in matches.json) registered on its own in a
    panorama it was found from (see placed.registered), as the placing page writes registrations;
    none where none fits. With its standing camera where that lays the photo better (see _stood).
    Kept (REGISTERED), made again only when the placement changes."""
    was = register.load(REGISTERED)
    made = {k: _registered_as(m) for k, m in standing.items()}
    kept = {k: was[k] for k in standing if was.get(k, {}).get("made") == made[k]}
    if todo := [k for k in standing if k not in kept]:
        by_id, fixed = _registering_world()
        for k in todo:
            try:
                found = _register_one(k, standing[k], by_id, fixed)
            except Exception as error:  # this one left, tried again next time: the others go on
                typer.echo(f"{k} : pas recalée ({error})", err=True)
                continue
            kept[k] = {"made": made[k], "registration": found and registration.to_json(found)}
    if any(r["registration"] for r in kept.values()):  # the walls read only then
        kept = _stood(standing, kept)
    if kept != was:  # made, or left (no longer placed)
        write_json(REGISTERED, kept)
    return {k: _registered_shown(kept.get(k) or {}) for k in standing}


def _stood(standing: dict[str, dict], kept: dict[str, dict]) -> dict[str, dict]:
    """Registered placements (`kept`, see _registrations) given their standing camera (see
    placing.stood_against_flat) in the panorama registered in: kept with the walls it stood
    against (see _walls_of), made again when they change (drawn by hand, fitted again)."""
    out = dict(kept)
    for k, row in kept.items():
        if row["registration"] is None:
            continue
        home = _standing_home(row["registration"]["panorama"])
        walls = home and _walls_of(home)
        if (row.get("standing") or {}).get("walls") == walls:
            continue
        try:
            stood = _stand_one(k, standing[k], home) if home else None
        except Exception as error:  # this one left, tried again next time: the others go on
            typer.echo(f"{k} : pas debout ({error})", err=True)
            continue
        out[k] = {**row, "standing": {"walls": walls, **_standing_as(stood, home)}}
    return out


def _walls_of(home: "_Home") -> str:
    """What a standing camera is fitted from, beside its pairs, hashed: the walls and the
    street around its panorama, from it (see scene.walls_shown), the panorama's levelling
    (its rays), and the code."""
    around, sphere = home.scene, home.sphere
    made = [w.made for w in around.walls]
    ground = round(around.ground - around.origin[2], 2)
    levelled = [round(sphere.azimuth, 2), round(sphere.roll, 3), round(sphere.pitch, 3)]
    return _digest([scene.walls_shown(around), made, ground, levelled, _standing_code()])


@functools.cache
def _standing_code() -> str:
    """The code a standing camera is fitted and compared by, hashed (not the placing page's)."""
    fitting = (placing.stood_against_flat, placing._depth, placing._rays)
    return _digest([inspect.getsource(f) for f in fitting] + [_hash_of(registration, facades)])


def _standing_as(stood: placing.Stood | None, home: "_Home | None") -> dict:
    """A standing camera as kept: placed from its panorama (east, north, up metres: the frame
    of heights moves as the area grows), its errors held out against the flat fit's (none: not
    measured) and how many pairs were held out (see _standing_drawn)."""
    if stood is None or stood.camera is None:
        return {"camera": None, "held": stood.held if stood else 0}
    camera = registration.standing_to_json(stood.camera)
    camera["position"] = [
        round(a - b, 3) for a, b in zip(camera["position"], home.scene.origin, strict=True)
    ]
    errors = {"flat": stood.flat, "parallax": stood.parallax}
    return {
        "camera": camera,
        **{k: round(e, 5) if np.isfinite(e) else None for k, e in errors.items()},
        "held": stood.held,
    }


def _standing_drawn(kept: dict, gain: float = 0.0, held: int = 0) -> dict | None:
    """A standing camera as kept (see _standing_as) where it lays the photo better than the flat
    fit, by `gain` at least (shares of the photo's diagonal), on `held` held-out pairs at least;
    else none."""
    flat, parallax = kept.get("flat"), kept.get("parallax")
    if not kept.get("camera") or None in (flat, parallax) or kept.get("held", held) < held:
        return None
    better = flat - parallax >= gain - 1e-9 if gain else parallax < flat  # stored to 1e-5
    return kept["camera"] if better else None


def _registered_shown(row: dict) -> placed.Registered:
    """A placement's registration, and its standing camera where that lays the photo better
    than the flat fit (held out, see placing.stood_against_flat)."""
    return placed.Registered(row.get("registration"), _standing_drawn(row.get("standing") or {}))


def _standing_walls(panorama: str) -> facades.Scene | None:
    """The walls a photo standing off `panorama` is laid on in the 3D walk."""
    return (home := _standing_home(panorama)) and home.scene


class _Home(NamedTuple):
    sphere: match.Panorama
    scene: facades.Scene


@functools.cache  # for one command: the walls read once
def _standing_homes() -> Callable[[str], _Home | None]:
    """Each panorama (by id) as matched against, with its walls (see _wall_scenes): where a
    photo's standing camera is fitted, and laid in the 3D walk."""
    _, near_by, _ = _matching()
    walls = _wall_scenes(near_by, _colmap_fixed())
    by_id = {p.id: p for p in near_by}

    @functools.cache
    def home(panorama: str) -> _Home | None:
        around = walls.scene(panorama, None, made_only=True) if panorama in by_id else None
        return _Home(by_id[panorama], around) if around is not None else None

    return home


def _standing_home(panorama: str) -> _Home | None:
    return _standing_homes()(panorama)


def _stand_one(
    photo_id: str,
    kept: dict,
    home: _Home,
    against: registration.Registration | None = None,
    again: Callable[[match.Panorama, tuple[int, int]], bridging.Paired | None] | None = None,
    least: int = 0,
) -> placing.Stood | None:
    """A placement's standing camera in a panorama (`home`, with its walls), from the pairs it
    matched there (see bridging.paired; else `again`: the photo matched again there), scored
    against the flat fit (see placing.stood_against_flat: `against` a copy's matrix, on
    `least` held-out pairs at least); none without points or pairs."""
    if (found := _found_from(photo_id, kept)) is None:
        return None
    m, points, observed = found
    paired = bridging.paired(points, observed, home.sphere, observed.size)
    if paired is None and again is not None:
        paired = again(home.sphere, observed.size)
    if paired is None:
        return None
    return placing.stood_against_flat(
        paired.photo_px,
        paired.view_px,
        paired.view,
        home.sphere,
        observed.size,
        home.scene,
        (m.x, m.y),
        against=against,
        least=least,
    )


def _registering_world() -> tuple[dict[str, match.Panorama], set[str]]:
    """The panoramas as the 3D walk shows them (levelled, their picture at hand), by id, and
    those COLMAP placed."""
    with closing(db.connect(DB_PATH)) as conn:
        pictures = panoramas.load(conn)
    spheres = _panoramas(_lined_up(pictures), lambda x, y: 0.0)
    return {p.id: p for p in spheres}, _fixed_by_colmap(pictures, register.load(FIXES))


def _observed_of(store: chain.Store, photo_id: str, inputs: str, kept: dict | None):
    """A placement's observation (see chain.Store.observed): as found from its `inputs` (a
    chained one's: the photo's own, see _unchained), else, for one kept though a run found
    nothing better (see _still_shown), as that run left it: the same photo and panoramas, its
    marks aside, the rays its points were found on. A chained one's: in DISK's store."""
    if _is_chained(inputs):
        store = chain.Store(MATCHERS[DISK].photos, None)
    found = store.observed(photo_id, _unchained(inputs))
    if found is None and (tried := (kept or {}).get("tried_from")):
        found = store.observed(photo_id, tried)
    return found


def _found_from(photo_id: str, kept: dict) -> tuple | None:
    """A placement of match (as kept), the points it was found from and its observation; none
    without them (a camera proposed: no points kept)."""
    m = _situation(kept)
    if (at := _kept_store(m, kept)) is None:
        return None
    store, inputs = at
    points, observed = store.points(photo_id, inputs), _observed_of(store, photo_id, inputs, kept)
    return None if points is None or observed is None else (m, points, observed)


def _register_one(
    photo_id: str, kept: dict, by_id: dict[str, match.Panorama], fixed: set[str]
) -> registration.Registration | None:
    """A placement of match registered on its own in a panorama it was found from (`by_id`:
    those the 3D walk can show), COLMAP's first among equals (see _sources_first)."""
    if (found := _found_from(photo_id, kept)) is None:
        return None
    m, points, observed = found
    return placed.registered(points, observed, _sources_first(points, by_id, m, fixed))


def _points_of(photo_id: str, m: match.Situated, kept: dict | None) -> match.Points | None:
    """The points a placement against today was found from, as `match` kept them (none: a
    camera proposed, or the points kept for another placement)."""
    if (at := _kept_store(m, kept)) is None:
        return None
    store, inputs = at
    return store.points(photo_id, inputs)


def _placement_points(
    photo_id: str, m: match.Situated, kept: dict | None, chained: bool
) -> match.Points | None:
    """The points of a placement shown (see _points_of); none for one the chaining placed
    without anchors (its inputs a placement's against today: match's points are not its own)."""
    if chained and not _is_chained(m.inputs):
        return None
    return _points_of(photo_id, m, kept)


@app.command("confirmer")
def confirm_placed(port: int = 8768, open_browser: bool = True) -> None:
    """Ouvre la page de confirmation (locale) : oui ou non à chaque photo située, contre les
    panoramas ou par enchaînement, à côté de la vraie vue d'aujourd'hui cadrée comme sa caméra
    et de ses contours posés sur cette vue. Un « non » la retire ; l'enchaînement suivant
    re-situe sans elle les photos qu'elle avait aidé à situer."""
    heights, near_by, then = _matching()
    with closing(db.connect(DB_PATH)) as conn:
        walk = export.photos(conn, DATA / "pilot", REVIEWS)
        pictures = panoramas.load(conn)
    fixed = _fixed_by_colmap(pictures, register.load(FIXES))
    credit = {p.id: (p.producer, (p.taken or "")[:4] or None) for p in pictures}  # who, when
    titles = {me.id: me.title for me in walk}
    places = {me.id: _place_of(me) for me in walk}
    today = then(date.today().year)
    walls = _ViewWalls(near_by, fixed, today.blocks)
    by_id = {p.id: p for p in near_by}
    shown_now, judged_now = _situated(marks=False), _Judged()
    anchors = _anchoring({k: asdict(m) for k, m in shown_now.items()})
    chained = _chained(_scenes_now(), judged_now, anchors)
    situated, rounds = _with_chained(shown_now, chained, set(titles), judged_now)  # as shown
    for k, why in chained.get("doubts", {}).items():
        if k in rounds:
            rounds[k]["doubt"] = why
    for (k, inputs), why in _set_aside().items():
        if k in rounds and situated[k].inputs == inputs:
            rounds[k]["doubt"] = f"jugée fausse à l'œil : {why}"
    kept = _matches()  # round 0: each placement as kept, its matcher with it
    scenes, judged = _scenes_now(), _Judged()
    for k, p in _proposed_now(scenes, judged).items():  # turned down by matching: yes or no?
        if k in titles and k not in situated:
            situated[k] = _situation(p)
            rounds[k] = {"round": 0, "from": [], "proposed": p["why"]}

    @functools.cache
    def points_of(photo_id: str) -> match.Points | None:
        chained = rounds[photo_id]["round"] > 0
        return _placement_points(photo_id, situated[photo_id], kept.get(photo_id), chained)

    def first_of(photo_id: str) -> list[match.Panorama]:
        return _sources_first(points_of(photo_id), by_id, situated[photo_id], fixed)

    def views_of(photo_id: str, source: bool = False) -> list[match.Panorama]:
        """Today's view of a placement: from COLMAP's panorama next to its camera, or (`source`)
        from those it was found from."""
        m = situated[photo_id]
        return _viewed_from(
            near_by, m, first_of(photo_id), fixed, walls.fitted, nearest_first=not source
        )

    placed = [
        {
            "id": k,
            "title": titles[k],
            "inputs": m.inputs,
            **_origin(m, rounds[k], kept.get(k)),
            "camera": {
                "x": round(m.x, 2),
                "y": round(m.y, 2),
                "heading": round(m.heading, 2),
                "fov": round(m.fov, 2),
                "horizon": round(m.horizon, 4),
                "above_street": round(m.above_street, 2),
            },
            "agree": m.agree,
            "moved_m": m.moved_m,
            "away_m": _away(views_of(k), m),
            **rounds[k],
        }
        for k, m in situated.items()
    ]
    read = functools.lru_cache(maxsize=16)(lambda path: cv2.imread(str(path)))

    @functools.lru_cache(
        maxsize=4
    )  # the heavy part, in arrays: the photo shown and the next, each from either panorama
    def framed(photo_id: str, source: bool = False) -> Framed | None:
        m = situated[photo_id]
        return _framed(photo_id, m, heights, views_of(photo_id, source), read, CONFIRM_WIDTH)

    @functools.lru_cache(maxsize=192)  # the pictures as sent, a few hundred kB a photo
    def frame(photo_id: str, part: str, source: bool = False) -> bytes | None:
        if (f := framed(photo_id, source)) is None:
            return None
        if part == "parallax":
            seen_from = _toward(f.camera, f.panorama)
            own, bdtopo = walls.of(f.panorama)
            kinds = functools.partial(walls.scenes.seen, f.panorama)
            args = (f.camera, seen_from, own, bdtopo, heights.streets, kinds)
            picture = _parallax_outlines(f.photo, f.today, *args)
        elif part == "outlines":
            picture = _outlines(f.photo, f.today)
        else:
            picture = getattr(f, part)
        return cv2.imencode(".jpg", picture, SHEET_JPEG)[1].tobytes()

    @functools.lru_cache(maxsize=192)
    def scene(photo_id: str, source: bool = False) -> dict | None:
        if (f := framed(photo_id, source)) is None:
            return None
        m = situated[photo_id]
        chained = rounds[photo_id]["round"] > 0
        points = points_of(photo_id)
        size = f.today.shape[:2]
        pairs, seen_from, counted = None, _toward(f.camera, f.panorama), {}
        if points is not None:
            at = _kept_store(m, kept.get(photo_id))
            observed = _observed_of(at[0], photo_id, at[1], kept.get(photo_id)) if at else None
            matched = observed and _matched_in(points, observed, f.panorama, seen_from, size)
            pairs = _pairs_shown(points, f.photo.shape, size, seen_from, matched)
            counted = {
                "points": sum(f.panorama.id in s for s in points.seen),
                "of": len(points.pixels),
            }
        why = "enchaînée" if chained else "proposée" if _is_proposal(m.inputs) else "perdus"
        producer, year = credit.get(f.panorama.id, (None, None))
        return {
            "size": [size[1], size[0]],
            "walls": "fitted" if walls.fitted(f.panorama) else "bdtopo",
            "view": {
                "panorama": f.panorama.id[:8],
                "producer": producer,
                "year": year,
                "colmap": f.panorama.id in fixed,
                "away_m": round(f.away_m, 1),
                "source": source,
                "other": _other_view(views_of(photo_id, not source), f.panorama, m),
                **counted,
            },
            "pairs": pairs,
            "no_pairs": None if pairs is not None else why,
            "plan": _plan(
                m,
                f.panorama,
                [(p.x, p.y) for p in first_of(photo_id)],
                places.get(photo_id, (m.x, m.y)),
                today.footprints,
                () if points is None else points.world,
            ),
        }

    _serve_page(
        lambda: confirming.serve(
            placed, ANCHORS, WRONG_PAIRS, functools.partial(_said_on, placed), frame, scene, port
        ),
        f"photos situées ({len(placed)})",
        port,
        open_browser,
    )


PROPOSALS = DATA / "pilot" / "proposals_3d.json"  # each photo proposed in its nearest panorama
PROPOSING_M = placing.NEAR_M  # a panorama farther from the photo's place is not tried


def _spheres(corrected: bool = True) -> list[match.Panorama]:
    """The panoramas where they were taken, levelled, for cutting views (no height needed)."""
    with closing(db.connect(DB_PATH)) as conn:
        pictures = _lined_up(panoramas.load(conn), corrected)
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
                "at": list(poses.at(home.x, home.y, home.azimuth)),  # stale once it moves
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


def _still_proposed(
    known: dict, photo_id: str, panorama: str, stands: tuple[float, float, float] | None
) -> bool:
    """Whether a proposal made by `flaneur proposer` holds: on this panorama and the photo's
    image now, the panorama standing where it was made (`stands`: see placings.where,
    poses.stands)."""
    if (known.get("panorama"), known.get("inputs")) != (panorama, _image(photo_id)):
        return False
    return poses.stands(known.get("at"), stands)


def _answered_as(row: dict) -> str:
    """What a copy or a proposal was made from: a placed photo's registration (the placing
    page's or match's: see placed.Placed.for_scene) on its image, the way round it was shown."""
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
    refaites. Puis, pour les photos posées sur les murs chez elles, les copies aussi
    (data/pilot/carried_standing.json)."""
    held = _shown_in_3d()  # placed by hand, by match or by chaining, on the image each has now
    with closing(db.connect(DB_PATH)) as conn:
        pictures = _lined_up(panoramas.load(conn))  # refused when tilts were measured otherwise
    spheres = {p.id: p for p in _panoramas(pictures, lambda x, y: 0.0)}
    now = _poised(spheres.values())
    carried = _carried(held, now)
    todo = [i for i in held if i not in carried]
    if todo:
        street = panoramas.links(pictures)  # the next ones along the street, every way
        views = _carrying_views()
        for photo_id in todo:
            r = placings.registration_of(held[photo_id])
            home = spheres.get(r.panorama)
            copies, tried = (
                ([], set()) if home is None else _to_neighbours(r, home, spheres, street, views)
            )
            carried[photo_id] = {
                "made": _answered_as(held[photo_id]),
                "copies": copies,
                "at": {p: now.get(p) for p in sorted({r.panorama, *tried})},
            }
            write_json(CARRIED, carried)  # kept photo after photo: a stop loses one at most
    write_json(CARRIED, carried)
    counts = [len(c["copies"]) for c in carried.values()]
    typer.echo(
        f"{len(carried)} photos placées ({len(todo)} refaites) portées sur {sum(counts)} "
        f"panoramas voisins ({np.median(counts) if counts else 0:.0f} par photo, médiane)"
    )
    _copies_standing(held, carried)


CARRIED_STANDING = DATA / "pilot" / "carried_standing.json"  # the copies laid on the walls
COPY_HELD = 10  # held-out pairs a copy's standing camera is judged on, at least
COPY_GAIN = 0.001  # and how much better than the copy's matrix it lays them (photo diagonals)


@functools.cache
def _carrying_views() -> match.Views:
    """The panoramas' views, a panorama and its neighbours in memory (the matcher loaded once)."""
    return _views(kept=8)


def _photo_store() -> chain.Store:
    """The photos' features as matching reads them, extracted where not kept (the matcher)."""
    return chain.Store(MATCHERS[DISK].photos, _carrying_views().matcher)


@functools.cache
def _copying_code() -> str:
    """The code a copy's own pairs and standing camera are found by, hashed (not the code
    that only chooses which copies or writes them: changing it refits nothing)."""
    made = (_stand_one, _matched_again)
    return _digest([inspect.getsource(f) for f in made] + [_hash_of(carrying, bridging)])


def _copy_standing(kept: dict) -> dict | None:
    """A copy's standing camera as kept (see _copies_standing) where it is to be drawn."""
    return _standing_drawn(kept, COPY_GAIN, COPY_HELD)


def _copy_made(carried: dict, copy: dict, registered_as: str, walls: str) -> str:
    """What a copy's standing camera is made from (see _copies_standing): the photo as carried
    (`carried`: its row, see _answered_as), the copy, the placement of match (as registered:
    see _registered_as), the walls of its panorama (see _walls_of), the code, and the least
    held-out pairs a camera is fitted on (the gain is judged on reading)."""
    made = [copy["registration"], registered_as, walls, _copying_code(), COPY_HELD]
    return _digest([carried["made"], *made])


def _copies_to_stand(
    held: dict[str, dict], carried: dict[str, dict], placements: dict[str, dict]
) -> Iterator[tuple[str, int, str, dict, Callable[[], str | None]]]:
    """The copies that may stand (see _copies_standing): those of a photo standing in its home
    panorama (`held`) as match placed it (`placements`, as registered: not a camera proposed
    and said yes to); each with its place among the photo's copies and what its camera is made
    from (see _copy_made: worked out when asked, the walls read once a panorama; none for a
    panorama without walls)."""
    registered = register.load(REGISTERED)
    walls = functools.cache(
        lambda panorama: _walls_of(home) if (home := _standing_home(panorama)) else None
    )
    for photo_id, row in held.items():
        placement = placements.get(photo_id) or {}
        if not row.get("standing") or photo_id not in carried or "x" not in placement:
            continue
        registered_as = _registered_as(placement)
        if registered.get(photo_id, {}).get("made") != registered_as:  # shown otherwise
            continue
        for at, copy in enumerate(carried[photo_id]["copies"]):
            panorama = copy["registration"]["panorama"]

            def made(photo_id=photo_id, copy=copy, panorama=panorama, registered_as=registered_as):
                around = walls(panorama)
                return around and _copy_made(carried[photo_id], copy, registered_as, around)

            yield photo_id, at, panorama, copy, made


def _copies_drawn(shown: dict[str, dict], carried: dict[str, dict]) -> dict[str, list[dict]]:
    """Each photo's copies (see porter) as the 3D walk draws them: with the camera where the
    photographer stood (see _copies_standing) where it lays the photo better (see
    _copy_standing) and was made from what the copy and its panorama's walls are now; else its
    matrix (a wall drawn since: until the next night)."""
    drawn = {i: list(c["copies"]) for i, c in carried.items()}
    if not (stood := register.load(CARRIED_STANDING)):
        return drawn
    for photo_id, at, panorama, copy, made in _copies_to_stand(shown, carried, _matches()):
        found = stood.get(photo_id, {}).get(panorama)
        if found and (camera := _copy_standing(found)) and found["made"] == made():
            drawn[photo_id][at] = {**copy, "standing": camera}
    return drawn


def _copies_standing(held: dict[str, dict], carried: dict[str, dict]) -> None:
    """Each copy (see porter) of a placement of match standing in its home panorama (`held`:
    as the 3D walk takes them; none for one placed by hand or resting on panoramas nothing
    checks) given the camera where the photographer stood, from the photo's own pairs in the
    copy's panorama (see _stand_one: else the photo matched again there) and that panorama's
    walls, scored against the copy's matrix (drawn where _copy_standing says so). Kept in
    CARRIED_STANDING with what each was made from (see _copy_made), made again when that
    changes; copies never stood are done too, panorama after panorama (one 360° picture at a
    time)."""
    was = register.load(CARRIED_STANDING)
    kept = _matches()
    out: dict[str, dict] = {}
    todo = []
    for photo_id, _, panorama, copy, made in _copies_to_stand(held, carried, kept):
        if (now := made()) is None:  # no walls there
            continue
        if (before := was.get(photo_id, {}).get(panorama)) and before["made"] == now:
            out.setdefault(photo_id, {})[panorama] = before
        else:
            todo.append((panorama, photo_id, copy, now))
    if todo:
        store = functools.cache(_photo_store)  # the matcher loaded only if one is matched again
        picture = functools.lru_cache(maxsize=1)(lambda path: cv2.imread(str(path)))
        for panorama, group in itertools.groupby(sorted(todo, key=lambda t: t[:2]), lambda t: t[0]):
            home = _standing_home(panorama)
            for _, photo_id, copy, made in group:
                against = registration.from_json(copy["registration"])

                def again(sphere, size, photo_id=photo_id, against=against):
                    return _matched_again(photo_id, against, sphere, store(), picture, size)

                try:
                    stood = _stand_one(photo_id, kept[photo_id], home, against, again, COPY_HELD)
                except Exception as error:  # this copy left, tried again next time
                    typer.echo(f"{photo_id} sur {panorama[:8]} : pas debout ({error})", err=True)
                    continue
                out.setdefault(photo_id, {})[panorama] = {
                    "made": made,
                    **_standing_as(stood, home),
                }
            write_json(CARRIED_STANDING, out)  # panorama after panorama: a stop loses one
    if out != was:
        write_json(CARRIED_STANDING, out)
    standing = sum(1 for p in out.values() for c in p.values() if _copy_standing(c))
    typer.echo(f"{standing} copies posées sur les murs, sur {sum(map(len, out.values()))}")


def _matched_again(
    photo_id: str,
    copy: registration.Registration,
    sphere: match.Panorama,
    store: chain.Store,
    picture: Callable[[Path], np.ndarray | None],
    size: tuple[int, int],
) -> bridging.Paired | None:
    """The photo (its features as matching reads them, from `store`) matched against a view of
    `sphere` (its 360° picture: `picture`) framed on its copy there (see carrying.framed),
    shown at placing.SHOWN: the pairs that agree on one flat warp (raw matches, unlike match's
    own pairs, which its geometry kept: measured 20/11, unfiltered 26 copies standing instead
    of 39), the photo's pixels at `size` (the placement's); none from fewer than
    MIN_MATCHED. Its features or picture missing: an error (not judged: done next time)."""
    mine = _features(store, photo_id) if store is not None else None
    seen = picture(sphere.path)
    if mine is None or seen is None:
        raise FileNotFoundError(f"{photo_id} ou {sphere.path} : pas à portée")
    view = carrying.framed(copy, sphere.axes)
    cut = match.perspective(seen, sphere.axes, view.heading, view.fov, placing.SHOWN, view.pitch)
    theirs = store.matcher.features(cut)
    pairs = store.matcher.pairs(mine, theirs)
    scale = max(size) / max(mine.size)
    photo_px = mine.keypoints[pairs[:, 0]].astype(float) * scale
    view_px = theirs.keypoints[pairs[:, 1]].astype(float)
    try:
        _, agree = registration.agreeing(photo_px, view_px)
    except (registration.NotFitted, cv2.error):  # too few agree, or a degenerate set
        return None
    return bridging.Paired(sphere.id, view, photo_px[agree], view_px[agree])


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
) -> tuple[list[dict], set[str]]:
    """A photo placed on `home`, carried outward along the street to the panoramas that see
    the same details (the home view's under the photo, matched in their views around its
    direction, the nearest first), and the panoramas it was carried to or tried on."""
    n = proposing.around(r.view.heading)[1]  # the kept view it was placed in, or the nearest
    home_view = registration.ViewOf(heading=float(match.VIEW_HEADINGS[n]))
    seen = views.of(home)
    if not seen:
        return [], set()
    kept = seen[n]
    try:
        inverse = np.linalg.inv(registration.in_view(r, home_view, home.axes))
    except np.linalg.LinAlgError:
        return [], set()
    if not len(kept.keypoints):  # a view with no details (sky): nothing to carry it by
        return [], set()
    under = cv2.perspectiveTransform(
        (kept.keypoints / proposing.SIDE)[None].astype(float), inverse
    )[0]
    inside = ((under >= 0) & (under <= 1)).all(axis=1)  # on the photo, warped into the view
    patch = match.Features(kept.keypoints[inside], kept.descriptors[inside], kept.size)
    left, nearest, right = proposing.around(home_view.heading)  # the nearest tried first

    tried: set[str] = set()  # those carried to, or tried and not

    def carry(panorama: str) -> registration.Registration | None:
        tried.add(panorama)
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
        linked = street.get(panorama, [])
        tried.update(q for q, _, _ in linked)  # those out of reach too: in reach later, redone
        return [(q, metres) for q, _, metres in linked if q in spheres]

    def from_home(panorama: str) -> float:
        p = spheres[panorama]
        return float(np.hypot(p.x - home.x, p.y - home.y))

    copies = [
        {"registration": registration.to_json(copy), "strength": carrying.strength(from_home(q))}
        for q, copy in carrying.outward(home.id, links, from_home, carry)
    ]
    return copies, tried


def _page_closed(page: int) -> None:
    """Refused while the placing page runs on `page`: the two do not fit in memory together."""
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", page)) == 0:
            typer.echo(f"la page tourne sur le port {page} : l'arrêter d'abord (mémoire)", err=True)
            raise typer.Exit(1)


def _paired_homes(spheres: list[match.Panorama], folder: Path) -> list[match.Panorama]:
    """The panoramas with three pairs or more kept in `folder` under their name as they stand
    (see _levelled_name: a panorama turned since has others), either way round, by id."""
    pairs = {frozenset(f.stem.split("__")) for f in folder.glob("*.npz")}
    made = Counter(name for pair in pairs for name in pair)
    return sorted((p for p in spheres if made[_levelled_name(p)] >= 3), key=lambda p: p.id)


def _within(answers: Path, reach: float, corrected: bool = True) -> list[match.Panorama]:
    """The panoramas within `reach` of a photo of the walk, the nearest to one first."""
    with closing(db.connect(DB_PATH)) as conn:
        places = _placing_answers(answers, corrected)[0]
        spots = np.array([_place_of(me) for me in _walk(conn, places)])
    if not len(spots):
        return []
    spheres = _spheres(corrected)
    nearest, _ = cKDTree(spots).query([(p.x, p.y) for p in spheres])
    return [spheres[i] for i in np.argsort(nearest) if nearest[i] <= reach]


def _in_street_or_not(client: httpx.Client | None) -> Callable[[float, float], bool]:
    """Whether a place is in the street (see poses.in_street); a LiDAR patch that cannot be
    read (network, an answer not understood): not, the panorama left where it stands."""

    def in_street(x: float, y: float) -> bool:
        try:
            return poses.in_street(x, y, LIDAR_MNS, client)
        except (httpx.HTTPError, ValueError) as error:
            typer.echo(f"LiDAR illisible en {x:.0f}, {y:.0f} : laissé en place ({error})", err=True)
            return False

    return in_street


@app.command("releve")
def set_on_survey() -> None:
    """Remet d'aplomb les panoramas des bénévoles sur ceux du relevé (Immergis pour la Ville de
    Paris), par les paires déjà faites (sans apparieur) : un bénévole vu par des panoramas du
    relevé qui s'accordent prend leur cap, et sa place où leurs directions se croisent ; remis
    sur deux références ou plus, il passe sa correction, jusqu'à quatre sauts. Un déplacement de
    plus de 5 m n'est gardé que si le LiDAR met la nouvelle place dans la rue. Jugés tels que
    COLMAP les a laissés (une fois appliquées, les paires d'un panorama tourné sont rangées sous
    son nom tourné : un bénévole arrivé depuis ne passe plus par lui) ; écrit
    data/pilot/panorama_survey.json, lu selon survey.APPLIED (après la mesure E du plan)."""
    spheres, segments = _placing_world(corrected=False)
    with closing(db.connect(DB_PATH)) as conn:
        pictures = panoramas.load(conn)
    producer = {p.id: p.producer for p in pictures}
    fixed = _fixed_by_colmap(pictures, register.load(FIXES))
    scenes = _Scenes(spheres, fixed, _no_matcher, threading.Lock(), segments, WALLS)
    by_id = {p.id: p for p in spheres}

    @functools.cache  # each pair fitted once a run, though asked each round
    def seen(home: str, other: str) -> survey.Seen | None:
        h, o = by_id[home], by_id[other]
        if not scenes._made(h, o):  # pairs on disk only
            return None
        rays_h, rays_o = scenes._rays(h, o)
        pair = (
            depth.set_right(placing._at(h), placing._at(o), rays_h, rays_o) if len(rays_h) else None
        )
        if pair is None:
            return None
        turn = np.asarray(pair.turn)
        at = np.asarray(pair.at)
        return survey.Seen(math.degrees(math.atan2(turn[1, 0], turn[0, 0])), (at[0], at[1]))

    @functools.cache
    def neighbours(i: str) -> list[str]:
        return [p.id for p in depth.candidates(by_id[i], spheres, fixed)]

    refs = {p.id: (p.x, p.y) for p in spheres if producer.get(p.id) == survey.PRODUCER}
    others = {p.id: (p.x, p.y) for p in spheres if p.id not in refs}
    with httpx.Client() as client:
        found = survey.set_right(refs, others, neighbours, seen, _in_street_or_not(client))
    survey.save(SURVEYED, found, {p.id: poses.at(p.x, p.y, p.azimuth) for p in spheres})
    turned = survey.fixes(found, "all")
    hops = Counter(f.hops for f in found.values())
    moved = sum(1 for f in turned.values() if f[:2] != (0.0, 0.0))
    typer.echo(f"{len(found)} panoramas de bénévoles remis sur le relevé")
    typer.echo("  " + ", ".join(f"{n} à {h} saut(s)" for h, n in sorted(hops.items())))
    typer.echo(f"  tournés de plus de {survey.LEAST_TURN_DEG:g}° : {len(turned)}, déplacés {moved}")
    typer.echo(f"  appliqués : {survey.APPLIED} (survey.APPLIED)")


@app.command("accorder")
def agree_panoramas() -> None:
    """Accorde les panoramas des bénévoles avec leurs voisins, par les paires déjà faites (sans
    apparieur) : chaque lien de la balade dont la paire se recale bien dit de combien tourner le
    voisin ; une petite rotation par panorama en est tirée, pour tous à la fois, ceux du relevé
    tenus fixes. Calculé sur les panoramas mis à niveau tels qu'avant tout accord (ce qu'il
    écrit ne revient pas dans ce qu'il calcule) ; écrit data/pilot/panorama_agreed.json, lu
    selon agreement.APPLIED (après la mesure E8a du plan)."""
    spheres, _ = _placing_world(agreed=False)
    with closing(db.connect(DB_PATH)) as conn:
        pictures = panoramas.load(conn)
    by_id = {p.id: p for p in spheres}
    scenes = _wall_scenes(spheres, _colmap_fixed())
    links, seen = [], set()
    for home, near in panoramas.links(pictures).items():
        for other, _, _ in near:
            both = frozenset((home, other))
            if home not in by_id or other not in by_id or both in seen:
                continue
            seen.add(both)
            h, o = by_id[home], by_id[other]
            if scenes._stored(h, o) is None:  # pairs on disk only: none made
                continue
            turn = agreement.measured(placing._at(h), placing._at(o), *scenes._rays(h, o))
            if turn is not None:
                links.append((home, other, turn))
    held = {p.id for p in pictures if p.producer == survey.PRODUCER}
    found = agreement.averaged(links, held)
    against = {i: (by_id[i].azimuth, by_id[i].pitch, by_id[i].roll) for i in found}
    agreement.save(AGREED, found, against)
    turned = [float(np.linalg.norm(c)) for c in found.values()]
    typer.echo(
        f"liens mesurés : {len(links)} ; panoramas accordés : {len(found)} ; correction "
        f"médiane {np.median(turned) if turned else 0.0:.2f}°"
    )
    typer.echo(f"  appliqués : {agreement.APPLIED} (agreement.APPLIED) ; écrit dans {AGREED}")


@app.command("segmenter")
def segment_panoramas(answers: Path = PLACINGS, page: int = 8772) -> None:
    """Segmente d'avance (bâti, végétation, ciel...) les panoramas que la page pour placer peut
    atteindre : ceux à moins de 40 m d'une photo, et leurs voisins à moins de 15 m. La page ne
    charge pas le modèle (plusieurs gigaoctets) : un panorama pas segmenté y a ses murs sans
    la segmentation. Reprend où il s'était arrêté ; refuse de tourner la page ouverte (sur le
    port `page`) : les deux ne tiennent pas en mémoire ensemble."""
    _page_closed(page)
    segmenter = segment.Segmenter(VIEW_STRUCTURE, threading.Lock(), _levelled_name, make=True)
    reach = queueing.NEAR_M + depth.NEAR_M[1]
    reached = _within(answers, reach)
    if poses.APPLIED != "none":  # recaler judges them as COLMAP left them: its masks too
        names = {_levelled_name(p) for p in reached}
        reached += [p for p in _within(answers, reach, False) if _levelled_name(p) not in names]
    if not reached:
        typer.echo("aucune photo, ou aucun panorama à portée : rien à faire")
        return
    left = [p for p in reached if not segmenter.kept(p)]
    typer.echo(f"{len(reached)} panoramas à moins de {reach:g} m d'une photo, {len(left)} à faire")
    _segment_each(left, segmenter)


LIDAR_MNS = DATA / "raw" / "lidar_mns"  # IGN's LiDAR HD surface model, patches around panoramas


def _used_by_matching(
    walk: list[export.WalkPhoto], near_by: list[match.Panorama]
) -> list[match.Panorama]:
    """The panoramas `match` compares the walk's photos with (see _to_match, _compared_with),
    those of more photos first."""
    used = Counter(p for me in _to_match(walk) for p in _compared_with(near_by, _place_of(me)))
    return sorted(used, key=lambda p: (-used[p], p.id))


@app.command("recaler")
def set_panoramas_right(
    only: Annotated[list[str] | None, typer.Argument(help="préfixes de panoramas")] = None,
    again: Annotated[bool, typer.Option(help="rejuger ceux déjà jugés tels qu'ils sont")] = False,
    photos: Annotated[
        bool, typer.Option(help="ceux que `match` compare aux photos, les plus employés d'abord")
    ] = False,
    page: Annotated[int, typer.Option(help="le port de la page pour placer")] = 8772,
) -> None:
    """Recale sur le plan les panoramas nommés (préfixes), ou (--photos) ceux que `match`
    compare aux photos de la balade (leurs paires faites au besoin), sinon ceux dont les paires
    sont déjà faites (sans en faire d'autres) : la forme entière de leurs détails de bâti
    cherchée sur les façades de la BD TOPO, à toute rotation et tout décalage, comme le fait
    l'œil ; une correction n'est gardée que si elle dépasse l'écart des deux relevés et que les
    contours du LiDAR HD de l'IGN la confirment (les détails à moins de 4 m d'eux, et pas plus
    loin qu'avant ; voir poses). Les panoramas sont jugés tels que COLMAP les a laissés (le
    reste les lit corrigés, selon poses.APPLIED : appliquer des corrections périme les places de
    `match`, fait repasser son contrôle, et les panoramas tournés refont leurs caches) ; chaque
    verdict est gardé au fur et à mesure (POSES), avec où était le panorama et la règle qui l'a
    donné : un panorama déjà jugé là, par la règle d'aujourd'hui, n'est pas rejugé (sauf
    --again). Un panorama dont un voisin n'est pas segmenté est laissé (`flaneur segmenter`
    d'abord). Refusé pendant que la page pour placer tourne : le modèle et elle ne tiennent pas
    ensemble en mémoire."""
    if only and photos:
        raise typer.BadParameter("des préfixes, ou --photos : pas les deux")
    if only or photos:  # pairs made as needed: the matcher wakes
        _page_closed(page)
    spheres, segments = _placing_world(corrected=False)  # as COLMAP left them
    with closing(db.connect(DB_PATH)) as conn:
        fixed = _fixed_by_colmap(panoramas.load(conn), register.load(FIXES))
        walk = export.photos(conn, DATA / "pilot", REVIEWS) if photos else []
    made_only = False  # their pairs made as needed, unless taken for having them on disk
    if only:
        homes = [p for p in spheres if any(p.id.startswith(o) for o in only)]
    elif photos:  # chosen where match will compare them (corrected), judged as COLMAP left them
        by_id = {p.id: p for p in spheres}
        homes = [by_id[p.id] for p in _used_by_matching(walk, _spheres()) if p.id in by_id]
        if not homes:
            typer.echo("aucune photo exportée à portée d'un panorama : rien à juger")
            return
        typer.echo(f"{len(homes)} panoramas que `match` compare aux photos, à juger")
    else:  # those whose pairs are on disk (three or more)
        homes, made_only = _paired_homes(spheres, PAIR_MATCHES), True
    views = functools.cache(lambda: _views(kept=8))  # the matcher, only if a pair must be made
    scenes = _Scenes(
        spheres, fixed, views, threading.Lock(), segments, WALLS, seen=_seeing(make=False)
    )
    masks = segment.Segmenter(VIEW_STRUCTURE, threading.Lock(), _levelled_name, make=False)
    verdicts = poses.load(POSES)
    with httpx.Client() as client:
        for home in homes:
            placed = poses.at(home.x, home.y, home.azimuth)
            if not again and (kept := verdicts.get(home.id)) and poses.still(kept, placed):
                continue  # judged as it stands
            neighbours = depth.candidates(home, spheres, fixed)
            if not all(masks.kept(p) for p in [home, *neighbours]):
                typer.echo(f"{home.id[:8]} : un voisin pas segmenté, laissé (`flaneur segmenter`)")
                continue
            details = scenes.details(home, neighbours, made_only=made_only)
            origin = np.array([home.x, home.y, home.z])
            try:
                found = poses.judge(
                    details,
                    origin,
                    home.z - PANORAMA_HEIGHT_M,
                    segments(home.x, home.y),
                    LIDAR_MNS,
                    client,
                )
            except (httpx.HTTPError, ValueError) as error:  # this one left: the others go on
                typer.echo(f"{home.id[:8]} : laissé ({error})", err=True)
                continue
            verdicts[home.id] = replace(found, placed=placed)
            poses.save(POSES, verdicts)
            said = f"corrigé {found.correction}" if found.correction else "laissé"
            how = "COLMAP" if home.id in fixed else "emprunté"
            typer.echo(f"{home.id[:8]} ({how}) : {said} — BD TOPO {found.bdtopo}")
    where = placings.where(spheres)
    standing = poses.to_apply(verdicts, where, "all")  # judged where COLMAP still puts them
    judged = len(
        poses.still_standing(verdicts, where)
    )  # by the rule of now, where COLMAP puts them
    typer.echo(
        f"{len(standing)} panorama(s) corrigé(s) sur {judged} jugés par la règle d'aujourd'hui"
    )
    applied = len(poses.to_apply(verdicts, where, poses.APPLIED))
    typer.echo(
        f"appliquées ({poses.APPLIED}) : {applied}"
        if poses.APPLIED != "none"
        else "mesurées, pas encore appliquées (poses.APPLIED)"
    )


@app.command("parallaxe")
def compare_parallax(answers: Path = PLACINGS) -> None:
    """Compare, sur les photos posées par points (leurs paires gardées), les deux recalages
    refaits : à plat (l'appareil au panorama) et en parallaxe (l'appareil où se tenait le
    photographe, cherché depuis là où son recalage était parti). Pour chacune, l'erreur médiane
    des paires avec profondeur, chacune refaite sans elle (part de la diagonale de la photo ;
    un recalage refusé compte comme le pire), et l'écart entre la place trouvée et le départ.
    Avec le modèle (les murs d'aujourd'hui, vus depuis les panoramas voisins)."""
    if not any(r.get("pairs") and r["registration"] for r in placings.holding(answers).values()):
        typer.echo("aucune réponse par points avec ses paires")
        return
    spheres, segments = _placing_world()
    held = placings.holding(answers, placings.where(spheres))
    rows = [r for r in held.values() if r.get("pairs") and r["registration"]]
    fixed = _colmap_fixed()
    views = _views(kept=8)  # a panorama and its neighbours
    masks = segment.Segmenter(VIEW_STRUCTURE, threading.Lock(), _levelled_name, make=True)
    made = _Scenes(
        spheres, fixed, lambda: views, threading.Lock(), segments, WALLS, seen=_seeing(True, masks)
    )
    depth_made = depthmaps.Depthmaps(threading.Lock())  # live (MoGe-2), as on the page
    fill = depthmaps.fills(
        lambda p: depth_made.maps(made.by_id[p]), lambda p: masks.parts(made.by_id[p])
    )
    found, skipped = [], 0
    for row in sorted(rows, key=lambda r: r["registration"]["panorama"]):  # its views kept
        panorama, image = row["registration"]["panorama"], _exported(row["photo_id"])
        try:
            if not image.exists() or row.get("inputs") != _image(row["photo_id"]):
                raise ValueError("son image a changé depuis (recadrée) : à refaire")
            chosen = (row.get("standing") or {}).get("chosen")  # as fitted
            walls = made.scene(panorama, chosen)
            if walls is None:
                raise ValueError("pas de panorama voisin")
            size = _size_of(image)
            one = placing.compared(
                row,
                made.by_id[panorama],
                size,
                walls,
                made.by_id,
                lambda other, at=panorama: made.pair_of(at, other),
                fill=fill(walls),  # as the page: where no wall is met
            )
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


def _framing(photo_id: str, exported: dict) -> dict:
    """An Albert-Kahn photo's plate and the museum's label (shares of its image): as the export
    found them, else found now on its image (one seeded, or exported since)."""
    if not photo_id.startswith(export.ALBERT_KAHN):
        return {}
    if "plate" in exported:
        return {k: exported[k] for k in ("plate", "label") if k in exported}
    found = plates.albert_kahn(Image.open(_exported(photo_id)))
    return {"plate": list(found.plate), **({"label": list(found.label)} if found.label else {})}


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
    site = WEB_DATA / "walk.json"
    framed = {p["id"]: p for p in json.loads(site.read_text())["photos"]} if site.exists() else {}
    photos = {
        me.id: {
            "title": me.title,
            "years": me.years,
            "lon": me.lon,
            "lat": me.lat,
            "heading": me.heading,
            "image": _exported(me.id),
            "inputs": _image(me.id),  # answers say which image they were made on
            **_framing(me.id, framed.get(me.id, {})),
        }
        for me in walk
        if _exported(me.id).exists()
    }
    if live:  # at their heights (parallax needs them), BD TOPO's facades read
        spheres, segments = _placing_world()
    else:
        spheres, segments = _spheres(), lambda x, y: []
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

    standing = placings.where(spheres)  # proposals, copies and answers made from them
    made: dict[tuple[str, str], dict[str, dict]] = {}  # by source and the answer it holds

    def from_source(source: str, row: dict, *rest) -> dict[str, dict]:  # kept while it holds
        key = (source, _answered_as(row))
        if key not in made:
            made[key] = carrying.from_source(source, row, *rest)
        return made[key]

    @functools.lru_cache(maxsize=1)
    def carried_at(answered: tuple[float, int]) -> dict[str, dict]:  # made again on new answers
        return carrying.proposals(
            placings.holding(answers, standing),
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
        if known and _still_proposed(known, photo_id, panorama, standing.get(panorama)):
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

    @functools.lru_cache(maxsize=1)
    def near_at(_: tuple) -> dict[str, dict]:  # made again on a new verdict or match
        return _said_near(photos)

    @functools.lru_cache(maxsize=1)
    def offers_at(stamp: tuple) -> dict[str, dict]:  # only for the photo opened: pairs are slow
        return _offers(near_at(stamp), photos, by_id, fixed)

    def stamp() -> tuple:
        return _stamp([MATCHES, ANCHORS, PROPOSED_CAMERAS, WRONG_PAIRS])

    def ahead() -> dict[str, str]:  # near verdicts, then photos carried from placed ones
        return _placing_ahead(near_at(stamp()), carried_now())

    def offer(photo_id: str) -> dict | None:
        return offers_at(stamp()).get(photo_id)

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

    matching = threading.Lock()  # the matcher: proposals, and walls when they must be made
    masks = segment.Segmenter(VIEW_STRUCTURE, threading.Lock(), _levelled_name, make=False)
    walls_made = _Scenes(
        spheres, fixed, models, matching, segments, WALLS, seen=_seeing(False, masks)
    )

    def walls_at(panorama: str, chosen: tuple[str, ...] | None) -> facades.Scene | None:
        """Today's walls around a panorama (see _Scenes); live only."""
        if not live:
            raise ValueError("pas de parallaxe sans le modèle : relancer sans --no-live")
        return walls_made.scene(panorama, chosen)

    def aimed_at(
        panorama: str, neighbours: tuple[str, ...], heading: float, fov: float
    ) -> list[str]:
        """The walls' neighbours, those giving most where a view looks first; live only."""
        return walls_made.aimed(panorama, neighbours, heading, fov) if live else list(neighbours)

    def pair_at(panorama: str, neighbour: str) -> depth.Pair | None:
        """A neighbour set right against a panorama (see _Scenes.pair_of); live only."""
        return walls_made.pair_of(panorama, neighbour) if live else None

    depth_made = depthmaps.Depthmaps(matching)  # live: the model loaded at the first need

    def maps_of(panorama: str) -> np.ndarray | None:
        """A panorama's views' depth by the model (MoGe-2), made the first time it is asked."""
        return depth_made.maps(by_id[panorama]) if panorama in by_id else None

    def parts_of(panorama: str) -> np.ndarray | None:
        """What each pixel of a panorama's views shows (segmented ahead: `flaneur segmenter`)."""
        return masks.parts(by_id[panorama]) if panorama in by_id else None

    sources = placing.Sources(
        photos,
        spheres,
        answers,
        proposal,
        ahead,
        graph,
        carried_now,
        offer=offer,
        full=full,
        scene=walls_at,
        pair=pair_at,
        aimed=aimed_at,
        segments=segments,
        fixed=fixed,
        walls=WALLS,
        matching=matching,
        fill=depthmaps.fills(maps_of, parts_of),
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
    """Écrit la scène de la balade en 3D : les panoramas d'aujourd'hui, et les photos placées
    dans l'un d'eux (à la main, ou par `match` et pas refusées : chacune dit son statut), avec
    leurs copies portées sur les voisins (`flaneur porter`) ; copie sur le site les panoramas
    qui portent une photo."""
    _write_scene(_placing_answers())


def _write_scene(answering: tuple) -> None:
    """The 3D scene written (see write_scene), from the placing page's answers as read."""
    with closing(db.connect(DB_PATH)) as conn:
        walk = _walk(conn, answering[0])
        pictures = _lined_up(panoramas.load(conn))  # refused when tilts were measured otherwise
    site = json.loads((WEB_DATA / "walk.json").read_text())
    marked = {p["id"] for p in site["photos"] if p["in_3d"]}  # by the export, on its images
    shown = {i: r for i, r in _shown_in_3d(answering).items() if i in marked}
    now = _poised(_panoramas(pictures, lambda x, y: 0.0))
    copies = _copies_drawn(shown, _carried(shown, now))
    built, holding = scene.build3d(
        [p for p in pictures if (PANORAMAS / f"{p.id}.jpg").exists()],
        shown,
        copies,
        walk,
        _standing_walls,
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


ANSWER_FILES = (ANCHORS, PLACINGS, SIGHTS, REVIEWS)  # the answers the light update reads
WATCH_S, QUIET_S = 5, 10  # answers looked at every WATCH_S; the update once QUIET_S calm
REFRESH_LOCK = DATA / "pilot" / ".mise-a-jour.lock"  # one light update at a time
NIGHT_STEPS = (  # each in its own process (one model in memory at a time), in this order
    ("releve",),  # first: views, pairs and walls follow where the panoramas stand
    ("segmenter",),
    ("murs", "--masques"),
    ("murs",),
    ("reperes-proposer", "--write"),
    *(("match-check", "--perimes", "--matcher", m) for m in MATCHERS),
    ("match",),
    ("porter",),  # before chaining, which the hour cuts first: chained photos carried a night late
    ("chain-check",),
    ("chain",),
)


@contextmanager
def _refreshing() -> Iterator[None]:
    """Held while a light update writes: a second one waits for it (REFRESH_LOCK)."""
    REFRESH_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with REFRESH_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # let go when the file closes
        yield


def _update_light() -> None:
    """The walk exported, then the list of the photos placed and the 3D scene, judged on the
    images it wrote (a photo cropped since leaves the 3D until match places it again), one
    update at a time."""
    with _refreshing():
        export_walk()
        _image.cache_clear()  # the export's images: read again
        _sights.cache_clear()
        answering = _placing_answers()
        _placed_photos(write=True, answering=answering)
        _write_scene(answering)


def _flaneur(*args: str) -> list[str]:
    """A flaneur command, run in a process of its own."""
    return [sys.executable, "-m", "flaneur", *args]


def _until(hour: str | None) -> float | None:
    """The next time it is `hour` ("HH:MM"), in epoch seconds; none: never."""
    if hour is None:
        return None
    at = datetime.combine(date.today(), datetime.strptime(hour, "%H:%M").time())
    return (at if at > datetime.now() else at + timedelta(days=1)).timestamp()


def _run_night(
    until: float | None, run: Callable = subprocess.run, clock: Callable[[], float] = time.time
) -> None:
    """The night's steps (NIGHT_STEPS) in order (all but chaining refresh what is stale only;
    chaining is redone whole): one that fails is said and the next goes on; none starts after
    `until` (epoch seconds), and the one running then is stopped (each writes its files whole:
    one panorama or photo lost at most). Then the light update, whatever the hour: it is what
    shows the night's work."""
    for step in NIGHT_STEPS:
        name, left = " ".join(step), None if until is None else until - clock()
        if left is not None and left <= 0:
            typer.echo(f"{name} : pas lancée, l'heure est passée")
            break
        typer.echo(f"{name} : lancée {datetime.now():%H:%M}")
        try:
            failed = run(_flaneur(*step), timeout=left).returncode
        except subprocess.TimeoutExpired:
            failed = "arrêtée à l'heure"
        if failed:
            typer.echo(f"{name} : échouée ({failed}), la suite continue", err=True)
    run(_flaneur("mettre-a-jour"), timeout=None)


def _stamp(paths: Iterable[Path]) -> tuple:
    """When each file was last written, and how long it is (none: not there)."""

    def of(path: Path) -> tuple[int, int] | None:
        try:
            found = path.stat()
        except FileNotFoundError:
            return None
        return found.st_mtime_ns, found.st_size

    return tuple(of(p) for p in paths)


def _follow(
    update: Callable[[], None],
    paths: Sequence[Path],
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    stamp: Callable[[Sequence[Path]], object] = _stamp,
) -> None:
    """Looks at `paths` every WATCH_S; once one changed and all were calm QUIET_S since, runs
    `update` (once for many changes). Never ends."""
    seen, changed = stamp(paths), None
    while True:
        sleep(WATCH_S)
        now = stamp(paths)
        if now != seen:
            seen, changed = now, clock()
        elif changed is not None and clock() - changed >= QUIET_S:
            changed = None
            update()


@app.command("mettre-a-jour")
def update(
    suivre: Annotated[
        bool, typer.Option(help="refaire le léger après chaque annotation enregistrée")
    ] = False,
    nuit: Annotated[bool, typer.Option(help="les étapes lourdes, puis le léger")] = False,
    jusqua: Annotated[
        str | None, typer.Option(help="la nuit : aucune étape après cette heure (HH:MM)")
    ] = None,
) -> None:
    """Remet à jour ce qui dépend des annotations. Par défaut le léger : la liste des photos
    placées, la balade, la scène en 3D (une mise à jour à la fois). `--suivre` : le léger
    refait dès que les pages ont enregistré une annotation (après 10 s de calme).
    `--nuit` : les étapes lourdes dans l'ordre (segmenter, murs, repères proposés, contrôles
    périmés, match, porter : chacune ne refait que ce qui est périmé ; puis l'enchaînement,
    refait en entier), puis le léger ; une étape qui échoue n'arrête pas les suivantes."""
    if suivre and nuit:
        raise typer.BadParameter("--suivre ou --nuit : pas les deux")
    if nuit:
        _run_night(_until(jusqua))
    elif suivre:
        typer.echo(f"regarde {', '.join(p.name for p in ANSWER_FILES)} (Ctrl-C pour arrêter)")
        _follow(lambda: subprocess.run(_flaneur("mettre-a-jour")), ANSWER_FILES)
    else:
        _update_light()


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
