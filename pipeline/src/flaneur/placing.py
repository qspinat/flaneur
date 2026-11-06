"""The placing page: a local page where the user places each photo, on the map (where the
photographer stood, which way they looked), then in its nearest panorama (the proposal checked,
or points clicked in pairs), or says that no panorama sees it. Answers go to
data/overrides/placings.jsonl (see placings). Local use only (see localpage)."""

import sys
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from http.server import HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import cv2
import numpy as np
from shapely.geometry import Point

from flaneur import depth, facades, localpage, match, placings, queueing, registration
from flaneur.gazetteer import to_degrees, to_metres
from flaneur.panoramas import PANORAMA_HEIGHT_M
from flaneur.registration import ViewOf
from flaneur.sources import number

PAGE = Path(__file__).with_name("placing_page.html")
NEAR_M = queueing.NEAR_M  # the panoramas offered around a photo's spot (where proposals are made)
SHOWN = 2 * match.VIEW_SIDE  # a view's side as the page shows it: 1280 px, what 75° of a
# panorama 6144 px around holds (see Sources.full)
FULL_WIDTH = round(SHOWN * 360 / match.VIEW_FOV)  # 6144 px around: SHOWN px in a 75° view
KEPT_PICTURES = 6  # 360° pictures kept decoded (57 MB each at FULL_WIDTH)
KEPT_VIEWS = 8  # views' depth kept (6.6 MB each at SHOWN, float32), refitted on the same view
RETRY_S = 60.0  # a larger picture not had (offline...) is asked for again after this
EPIPOLAR_M = (1.0, 200.0)  # a detail clicked in a view may stand this far: its ray, sampled
EPIPOLAR_SAMPLES = 1000
TINT_M = (2.0, 60.0)  # the view tinted by depth: warm this near, cool this far
WARM, COOL = np.array([40, 120, 255], np.float32), np.array([255, 160, 40], np.float32)  # BGR
Second = tuple[match.Panorama, np.ndarray]  # a pair's second click: its neighbour, its ray
NOT_MET = "les deux rayons ne se rencontrent pas : cliquer le même détail dans les deux vues"


@dataclass
class Sources:
    photos: dict[str, dict]  # id: title, years, lon, lat, heading, image (its file), inputs
    spheres: list[match.Panorama]  # levelled
    placings: Path
    # (photo, panorama): {registration, carried_from} or None: proposed, not carried
    proposal: Callable[[str, str], dict | None]
    # the photos to come first, with why, in order (asked after each answer)
    ahead: Callable[[], dict[str, str]] = dict
    graph: dict[str, set[str]] = field(default_factory=dict)  # old neighbours, by photo
    # the photos proposed from the placed ones: {registration, carried_from}, by photo
    carried: Callable[[], dict[str, dict]] = dict
    # a panorama's picture larger than its own (see SHOWN), the views shown cut from it; none to
    # be had, its own
    full: Callable[[str], np.ndarray | None] = lambda _: None
    # today's walls around a panorama (see facades), from the neighbours chosen (else picked by
    # their geometry: see depth.neighbours); none without neighbours or the matcher
    scene: Callable[[str, tuple[str, ...] | None], facades.Scene | None] = lambda *_: None
    # BD TOPO's facades near a spot (Lambert-93): shown under the walls, to draw along
    segments: Callable[[float, float], list[facades.Segment]] = lambda *_: []
    fixed: set[str] = field(default_factory=set)  # the panoramas COLMAP placed itself
    walls: Path | None = None  # where walls drawn and removed by hand are kept (see facades)
    # the matcher, one thing at a time: proposals, and depth when it must be made
    matching: threading.Lock = field(default_factory=threading.Lock)


class Around:
    """The panoramas near a spot, found at once among all of them."""

    def __init__(self, spheres: list[match.Panorama]):
        self.spheres = spheres
        self.at = np.array([(p.x, p.y) for p in spheres]).reshape(-1, 2)

    def near(
        self, x: float, y: float, radius: float = NEAR_M
    ) -> list[tuple[match.Panorama, float]]:
        """Those within `radius` metres of (x, y), nearest first, with how far."""
        metres = np.hypot(*(self.at - (x, y)).T)
        return [
            (self.spheres[i], float(metres[i])) for i in np.argsort(metres) if metres[i] <= radius
        ]


def settings() -> dict:
    """What the page shows the user by the server's rules."""
    return {
        "min_points": registration.MIN_POINTS,
        "min_standing": registration.MIN_STANDING,
        "view_step": 360 // len(match.VIEW_HEADINGS),
        "near_m": NEAR_M,
    }


def _listed(values: np.ndarray) -> list:
    """Numbers as JSON has them: NaN (none) as null, infinite (a fit refused) as "refusé"."""
    return [None if np.isnan(v) else "refusé" if np.isinf(v) else float(v) for v in values]


def flat(
    photo_px: np.ndarray, view_px: np.ndarray, view: ViewOf, sphere: match.Panorama, size: tuple
) -> tuple[dict, registration.Registration]:
    """The photo laid by its camera at the panorama, from pairs clicked in a view shown at
    SHOWN: as the page has it (with each pair's error, fitted and held out), and the fit."""
    fitted, _ = registration.fit(photo_px, view_px, size, SHOWN, sphere.id, view, sphere.axes)
    errors = registration.errors_flat(fitted, photo_px, view_px, size, SHOWN, view, sphere.axes)
    held_out = registration.held_out_flat(photo_px, view_px, size, SHOWN, view, sphere.axes)
    found = {
        "registration": registration.to_json(fitted),
        "errors": _listed(errors),
        "held_out": _listed(held_out),
    }
    return found, fitted


@dataclass
class _Standing:
    """A camera fitted where the photographer stood, and what it was fitted from."""

    camera: registration.Standing
    used: np.ndarray  # the pairs with a depth
    world: np.ndarray  # their points of today (NaN without depth)
    metres: np.ndarray
    source: list[str | None]  # what each depth came from (see depth_at)
    fitting: tuple  # what registration.fit_standing was given: refitted to hold pairs out


def _rays(view_px: np.ndarray, view: ViewOf, sphere: match.Panorama) -> np.ndarray:
    """The world directions, from the panorama, of points clicked in a view shown at SHOWN."""
    return registration.own_directions(view_px, view, sphere.axes, SHOWN) @ sphere.axes


def depth_at(
    scene: facades.Scene, sphere: match.Panorama, view_px: np.ndarray, view: ViewOf
) -> tuple[np.ndarray, list[str | None]]:
    """How far from the panorama the points clicked in a view stand, where their rays meet
    today's walls (see facades.cast), and what gave each its depth: a wall fitted ("wall") or
    drawn ("drawn"), the street ("ground"); NaN and None meeting nothing."""
    return _depth(scene, _rays(view_px, view, sphere))


def _depth(scene: facades.Scene, rays: np.ndarray) -> tuple[np.ndarray, list[str | None]]:
    metres, met = facades.cast(scene, rays)
    named = {facades.GROUND: "ground", facades.NOTHING: None} | {
        i: "drawn" if w.made == "drawn" else "wall" for i, w in enumerate(scene.walls)
    }
    return metres, [named[m] for m in met]


def _at(p: match.Panorama) -> np.ndarray:
    """Where a panorama's camera stands (Lambert-93, height)."""
    return np.array([p.x, p.y, p.z])


def _second_depth(sphere: match.Panorama, ray: np.ndarray, second: Second) -> float:
    """How far along `ray` (from the panorama) its second click's ray meets it; refused when
    they do not meet."""
    other, seen = second
    met = depth.meeting(_at(sphere), ray, _at(other), seen)
    if met is None:
        raise ValueError(NOT_MET)
    return met


def _pairs_depth(
    scene: facades.Scene,
    sphere: match.Panorama,
    rays: np.ndarray,
    seconds: list[Second | None],
    refuse: bool = True,
) -> tuple[np.ndarray, list[str | None]]:
    """Each pair's depth along its ray from the panorama (`rays`): where its second click's
    ray meets it ("click"), else where it meets today's walls (see _depth). A second click
    whose rays do not meet is refused, or (`refuse` False: one saved, the panoramas placed
    again since) left to the walls."""
    metres, source = _depth(scene, rays)
    for i, second in enumerate(seconds):
        if second is not None:
            try:
                metres[i], source[i] = _second_depth(sphere, rays[i], second), "click"
            except ValueError as refused:
                if refuse:
                    raise ValueError(f"paire {i + 1} : {refused}") from refused
    return metres, source


def _standing(
    photo_px: np.ndarray,
    view_px: np.ndarray,
    view: ViewOf,
    sphere: match.Panorama,
    size: tuple,
    scene: facades.Scene,
    pin: tuple[float, float],
    start: registration.Registration,
    seconds: list[Second | None] = (),
    refuse: bool = True,
) -> _Standing:
    """The camera where the photographer stood: from the pairs whose panorama point has a
    depth (see _pairs_depth), started from the flat fit `start` and the user's `pin`."""
    rays = _rays(view_px, view, sphere)
    metres, source = _pairs_depth(scene, sphere, rays, seconds, refuse)
    world = np.array(scene.origin) + metres[:, None] * rays
    fitting = (photo_px, world, size, start, sphere.axes, pin, scene.ground)
    camera, used = registration.fit_standing(*fitting)
    return _Standing(camera, used, world, metres, source, fitting)


def _flat_without(
    photo_px: np.ndarray, view_px: np.ndarray, view: ViewOf, sphere: match.Panorama, size: tuple
) -> Callable[[int], registration.Registration]:
    """The flat fit made without pair i: where a pair held out starts its standing camera."""

    def made(i: int) -> registration.Registration:
        rest = np.arange(len(photo_px)) != i
        fitted = registration.fit(
            photo_px[rest], view_px[rest], size, SHOWN, sphere.id, view, sphere.axes
        )
        return fitted[0]

    return made


def parallax(
    photo_px: np.ndarray,
    view_px: np.ndarray,
    view: ViewOf,
    sphere: match.Panorama,
    size: tuple,
    scene: facades.Scene,
    pin: tuple[float, float],
    held_out: bool = True,
    seconds: list[Second | None] = (),
) -> dict:
    """The photo's camera where the photographer stood (see _standing), as the page has it:
    where it stood, each pair's depth and what gave it, the neighbours the walls were fitted
    from, and its error, fitted and held out (both fits; `held_out` False, none: when only the
    camera is wanted)."""
    if held_out:
        shown, start = flat(photo_px, view_px, view, sphere, size)
    else:
        start, _ = registration.fit(photo_px, view_px, size, SHOWN, sphere.id, view, sphere.axes)
    try:
        s = _standing(photo_px, view_px, view, sphere, size, scene, pin, start, seconds)
    except registration.NotFitted as refused:
        if scene.walls:
            raise
        raise registration.NotFitted(f"{refused} ; {_no_walls(scene)}") from refused
    lon, lat = _degrees(*s.camera.position[:2])
    return {
        "registration": registration.to_json(start),  # the flat fit, which an answer keeps too
        "standing": registration.standing_to_json(s.camera),
        "lon": lon,
        "lat": lat,
        "height_above_ground": s.camera.position[2] - scene.ground,
        "heading": round(s.camera.heading, 1),
        "used": s.used.tolist(),
        "metres": _listed(s.metres),
        "source": s.source,
        "neighbours": list(scene.neighbours),
        "chosen": None if scene.chosen is None else list(scene.chosen),
        "errors": _listed(registration.errors_standing(s.camera, photo_px, s.world)),
        "held_out": held_out
        and {
            "flat": shown["held_out"],
            "parallax": _listed(
                registration.held_out_standing(
                    *s.fitting,
                    restart=_flat_without(photo_px, view_px, view, sphere, size),
                    shift=s.camera.shift,
                )
            ),
        },
    }


def kept_pairs(pairs: dict, sphere: match.Panorama, spheres: dict[str, match.Panorama]) -> dict:
    """Pairs clicked in a view shown at SHOWN, as an answer keeps them: the panorama's as
    directions in its own picture (unmoved when it is levelled again), and so their second
    clicks in the neighbours (`spheres`), if any."""
    view = ViewOf(**pairs["view"])

    def own(px: list, p: match.Panorama) -> list:
        return registration.own_directions(np.array(px, float), view, p.axes, SHOWN).tolist()

    found = {
        "photo_px": pairs["photo_px"],
        "own": own(pairs["view_px"], sphere),
        "view": {"heading": view.heading, "pitch": view.pitch, "fov": view.fov},
        "side": SHOWN,
    }

    def kept(s: dict) -> dict:
        return {"neighbour": s["neighbour"], "own": own([s["px"]], spheres[s["neighbour"]])[0]}

    if any(pairs.get("second_clicks") or []):
        found["second_clicks"] = [None if s is None else kept(s) for s in pairs["second_clicks"]]
    return found


def _kept_seconds(pairs: dict, spheres: dict[str, match.Panorama]) -> list[Second | None]:
    """An answer's second clicks (see kept_pairs), as rays of the neighbours as placed now."""

    def ray(s: dict) -> Second:
        other = spheres[s["neighbour"]]
        return other, np.array(s["own"]) @ other.axes

    return [None if s is None else ray(s) for s in pairs.get("second_clicks") or []]


def compared(
    row: dict,
    sphere: match.Panorama,
    size: tuple,
    scene: facades.Scene,
    spheres: dict[str, match.Panorama],
) -> dict:
    """An answer's two fits made again from its saved pairs: the flat one, and the camera where
    the photographer stood (started from the pin its parallax fit started from, else from the
    answer's place). For each, the median of each pair's error held out (shares of the
    photo's diagonal), on the same pairs, those with a depth; a refit refused counts as the
    worst, and a camera refused outright as infinitely bad. How far it stood from its start.
    Second clicks give depth as they did (their neighbours among `spheres`)."""
    pairs = row["pairs"]
    seconds = _kept_seconds(pairs, spheres)
    view = ViewOf(**pairs["view"])
    photo_px = np.array(pairs["photo_px"], float)
    view_px = registration.view_pixels(np.array(pairs["own"], float), view, sphere.axes, SHOWN)
    started = (row.get("standing") or {}).get("start") or row
    pin = _metres(started["lon"], started["lat"])
    found = {"photo_id": row["photo_id"], "fit": row.get("fit") or "flat", "pairs": len(photo_px)}
    rays = _rays(view_px, view, sphere)
    same = ~np.isnan(
        _pairs_depth(scene, sphere, rays, seconds, refuse=False)[0]
    )  # both fits: those with depth
    flat_off = registration.held_out_flat(
        photo_px[same], view_px[same], size, SHOWN, view, sphere.axes
    )
    found |= {"with_depth": int(same.sum()), "refused": {"flat": int(np.isinf(flat_off).sum())}}
    try:
        start = registration.fit(photo_px, view_px, size, SHOWN, sphere.id, view, sphere.axes)[0]
        s = _standing(photo_px, view_px, view, sphere, size, scene, pin, start, seconds, False)
    except registration.NotFitted as refused:
        flat_median = float(np.median(flat_off)) if len(flat_off) else np.inf
        return {
            **found,
            "flat": flat_median,
            "parallax": np.inf,
            "moved_m": None,
            "refused": {**found["refused"], "parallax": int(same.sum())},
            "why": str(refused),
        }
    restart = _flat_without(photo_px, view_px, view, sphere, size)
    standing_off = registration.held_out_standing(
        *s.fitting, restart=restart, shift=s.camera.shift
    )[same]
    standing_off[np.isnan(standing_off)] = np.inf  # a point it cannot see: the worst
    return {
        **found,
        "flat": float(np.median(flat_off)),
        "parallax": float(np.median(standing_off)),
        "refused": {**found["refused"], "parallax": int(np.isinf(standing_off).sum())},
        "moved_m": float(np.hypot(s.camera.position[0] - pin[0], s.camera.position[1] - pin[1])),
        "why": "",
    }


def _no_walls(scene: facades.Scene) -> str:
    if scene.chosen == ():
        return "aucun mur ici : aucun voisin choisi ; en dessiner sur le plan, ou en choisir"
    if not scene.neighbours:
        low, high = depth.NEAR_M
        return (
            f"aucun mur ici : pas de panorama voisin à {low:g}–{high:g} m pour les trouver ; "
            "en dessiner sur le plan, ou choisir des voisins"
        )
    return (
        f"aucun mur trouvé autour de ce panorama ({scene.details} détails vus avec ses "
        "voisins) : en dessiner sur le plan, ou choisir d'autres voisins"
    )


def _ends(sent: dict) -> facades.Segment:
    """A segment drawn on the map: its two ends (degrees), as Lambert-93."""
    if not isinstance(sent, dict):
        raise ValueError("un mur : ses deux bouts (longitude, latitude)")
    try:
        ends = tuple(_metres(*sent[k]) for k in ("a", "b"))
    except (KeyError, TypeError) as error:
        raise ValueError("un mur : ses deux bouts (longitude, latitude)") from error
    if ends[0] == ends[1]:
        raise ValueError("un mur : deux bouts distincts")
    return ends


def _since(sent: dict) -> int:
    """The save a wall or a removal sent back first was in (0: new)."""
    since = sent.get("since") if isinstance(sent, dict) else None
    if since is None:
        return 0
    if not (type(since) is int and since > 0):
        raise ValueError("un mur : l'enregistrement où il est apparu")
    return since


def _drawn(sent: dict, ground: float) -> facades.Wall:
    """A wall drawn on the map, with its top if the page gives one (else see with_drawn)."""
    top = sent.get("top") if isinstance(sent, dict) else None
    if top is not None and not (number(top) and top > ground):
        raise ValueError("un mur : son haut, au-dessus de la rue")
    top = None if top is None else float(top)
    return facades.Wall(*_ends(sent), top, "drawn", since=_since(sent))


def _tinted(view: np.ndarray, metres: np.ndarray) -> np.ndarray:
    """A view tinted by how far each pixel stands: warm near, cool far; the sky untouched."""
    low, high = TINT_M
    t = np.log(np.clip(metres, low, high).astype(np.float32) / low) / float(np.log(high / low))
    tint = (1 - t)[..., None] * WARM + t[..., None] * COOL
    mixed = cv2.addWeighted(view.astype(np.float32), 0.55, tint, 0.45, 0)
    return np.where(np.isfinite(metres)[..., None], mixed, view).astype(np.uint8)


def _degrees(x: float, y: float) -> list[float]:
    at = to_degrees(Point(x, y))
    return [at.x, at.y]


def _walls_shown(
    scene: facades.Scene, segments: list[facades.Segment], own: dict | None = None
) -> dict:
    """A panorama's walls as the page draws them on the map (degrees; a wall drawn, with the
    panorama where it was: the page sends back its own), with BD TOPO's facades near it, the
    neighbours they were fitted from and how many details; why there are none. A wall drawn at
    this panorama says where it stands among its own (`own`: by its ends; `mine`, else none)."""

    def mine(w: facades.Wall) -> int | None:
        here = w.made == "drawn" and w.at == scene.panorama
        return (own or {}).get((w.a, w.b)) if here else None

    return {
        "details": scene.details,
        "walls": [
            {
                "a": _degrees(*w.a),
                "b": _degrees(*w.b),
                "top": w.top,
                "made": w.made,
                "at": w.at,
                "mine": mine(w),
            }
            for w in scene.walls
        ],
        "neighbours": list(scene.neighbours),
        "segments": [[_degrees(*a), _degrees(*b)] for a, b in segments],
        "why": "" if scene.walls else _no_walls(scene),
    }


def _metres(lon: float, lat: float) -> tuple[float, float]:
    here = to_metres(Point(lon, lat))
    return here.x, here.y


def _shown(photo: dict) -> dict:
    """A photo as the page gets it (not its file nor its image's hash)."""
    return {k: v for k, v in photo.items() if k not in ("image", "inputs")}


def carried(sources: Sources) -> dict[str, dict]:
    """The photos proposed from those placed; none when making them failed (the page goes on)."""
    try:
        return sources.carried()
    except Exception:  # a pair or a feature kept wrong: the queue and the other proposals go on
        traceback.print_exc(file=sys.stderr)
        return {}


def next_photo(sources: Sources, nearest: dict[str, float], where: dict) -> dict:
    """The next photo to place, in the order of queueing.order (`nearest`: metres to the
    nearest panorama, `where`: x, y, by photo), how many are left, and the photos placed (to
    reopen from the map)."""
    held, skipped, answers = placings.state(sources.placings)
    # placed on an image the photo no longer has (cropped again): to see again, first
    stale = {
        i
        for i, r in held.items()
        if r["answer"] in placings.PLACED
        and r["inputs"] != sources.photos.get(i, {}).get("inputs", "")
    }
    held = {i: r for i, r in held.items() if i not in stale}
    found = queueing.order(
        list(sources.photos),
        where=where,
        held={
            i: {"answer": r["answer"], **dict(zip("xy", _metres(r["lon"], r["lat"]), strict=True))}
            for i, r in held.items()
        },
        skipped=skipped,
        ahead={**{i: "revoir" for i in sorted(stale)}, **sources.ahead()},
        graph=sources.graph,
        nearest_m=nearest,
        answered=sum(i in sources.photos for i in answers),
    )
    head = found and {"id": found[0][0], **_shown(sources.photos[found[0][0]]), "why": found[0][1]}
    proposed = head and carried(sources).get(head["id"])
    if proposed:  # opened where its source was placed, on the panorama it was carried onto
        head.update(
            registration=registration.to_json(proposed["registration"]),
            answer="fits",
            carried_from=proposed["carried_from"],
            **{k: proposed[k] for k in ("lon", "lat", "heading")},
        )
    placed = [
        {"id": i, "answer": r["answer"], "lon": r["lon"], "lat": r["lat"]}
        for i, r in held.items()
        if i in sources.photos
    ]
    return {"photo": head or None, "left": len(found), "placed": placed}


def reopened(sources: Sources, photo_id: str) -> dict:
    """A photo answered, as the page shows it again: its answer's place and registration."""
    row = placings.holding(sources.placings)[photo_id]
    same_image = row["inputs"] == sources.photos[photo_id].get("inputs", "")
    return {
        "id": photo_id,
        **_shown(sources.photos[photo_id]),
        **{k: row[k] for k in ("lon", "lat", "heading", "answer", "carried_from", "flipped")},
        "fit": row.get("fit"),  # kept in parallax: the page says so (its camera is not shown again)
        "registration": row["registration"] if same_image else None,  # made on another image
    }


def _jpeg(image: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def serve(sources: Sources, port: int) -> HTTPServer:
    spheres = {p.id: p for p in sources.spheres}
    around = Around(sources.spheres)
    where = {i: _metres(p["lon"], p["lat"]) for i, p in sources.photos.items()}
    nearest = {}
    for photo_id, (x, y) in where.items():
        found = around.near(x, y)
        nearest[photo_id] = found[0][1] if found else np.inf
    proposals: dict[tuple[str, str], dict | None] = {}  # made once per photo and panorama
    matching = sources.matching  # one proposal at a time (the matcher and its caches)

    def near(lon: float, lat: float) -> list[dict]:
        found = []
        for p, metres in around.near(*_metres(lon, lat)):
            east, north = _degrees(p.x, p.y)
            found.append(
                {
                    "id": p.id,
                    "lon": east,
                    "lat": north,
                    "metres": round(metres, 1),
                    "fixed": p.id in sources.fixed,  # placed by COLMAP itself: depth surer
                }
            )
        return found

    def proposal(photo_id: str, panorama: str) -> dict | None:
        """Carried from a photo placed (as the answers stand now), else made once; none when
        making it failed (the user places by points)."""
        found = carried(sources).get(photo_id)
        if found and found["registration"].panorama == panorama:
            return {
                "registration": registration.to_json(found["registration"]),
                "carried_from": found["carried_from"],
            }
        with matching:
            if (photo_id, panorama) not in proposals:
                try:
                    found = sources.proposal(photo_id, panorama)
                except Exception:  # the matcher, its models, memory: the page goes on
                    traceback.print_exc(file=sys.stderr)
                    found = None
                proposals[photo_id, panorama] = found and {
                    "registration": registration.to_json(found["registration"]),
                    "carried_from": found["carried_from"],
                }
            return proposals[photo_id, panorama]

    fetching = defaultdict(threading.Lock)  # each larger picture fetched once at a time
    missed: dict[str, float] = {}  # when one could not be had

    @lru_cache(maxsize=KEPT_PICTURES)
    def larger(panorama: str) -> np.ndarray:  # only those had: one not had is asked again
        found = sources.full(panorama)
        if found is None or found.shape[1] != 2 * found.shape[0]:  # none, or not all around
            missed[panorama] = time.monotonic()
            raise LookupError(panorama)
        return found

    @lru_cache(maxsize=KEPT_PICTURES)
    def own(panorama: str) -> np.ndarray:
        return cv2.imread(str(spheres[panorama].path))

    def picture(panorama: str) -> np.ndarray:
        """The panorama's larger picture, else its own (asked for the larger again RETRY_S
        after it could not be had); one fetched never holds up the views of another."""
        with fetching[panorama]:
            if time.monotonic() - missed.get(panorama, -np.inf) >= RETRY_S:
                try:
                    return larger(panorama)
                except LookupError:
                    pass
            return own(panorama)

    @lru_cache(maxsize=2)  # the photo being placed, as it is and turned over
    def photo_image(photo_id: str, flipped: bool) -> np.ndarray:
        image = cv2.imread(str(sources.photos[photo_id]["image"]))
        return image[:, ::-1].copy() if flipped else image

    def cut(panorama: str, view: ViewOf) -> np.ndarray:
        sphere = spheres[panorama]
        return match.perspective(
            picture(panorama), sphere.axes, view.heading, view.fov, SHOWN, view.pitch
        )

    def known(body: dict) -> tuple[str, match.Panorama | None]:
        photo_id = body.get("photo_id")
        if photo_id not in sources.photos:
            raise ValueError(f"photo inconnue : {photo_id}")
        panorama = body.get("panorama")
        if panorama is not None and panorama not in spheres:
            raise ValueError(f"panorama inconnu : {panorama}")
        return photo_id, spheres.get(panorama)

    def here(sphere: match.Panorama | None) -> match.Panorama:
        if sphere is None:
            raise ValueError("le panorama où poser la photo manque")
        return sphere

    def clicked(body: dict, photo_id: str) -> tuple[np.ndarray, np.ndarray, ViewOf, tuple]:
        """The pairs a request sends, the view they were clicked in, and the photo's size."""
        size = photo_image(photo_id, bool(body.get("flipped"))).shape[:2]
        pairs = (np.array(body["photo_px"], float), np.array(body["view_px"], float))
        return *pairs, ViewOf(**body["view"]), size

    def neighbour(body: dict) -> match.Panorama:
        other = spheres.get(body.get("neighbour"))
        if other is None:
            raise ValueError(f"panorama voisin inconnu : {body.get('neighbour')}")
        return other

    def ray_at(px: list, view: ViewOf, sphere: match.Panorama) -> np.ndarray:
        return _rays(np.array([px], float), view, sphere)[0]

    def second_of(sent: dict, view: ViewOf) -> Second:
        if not isinstance(sent, dict):
            raise ValueError("un second clic : le panorama voisin et le point cliqué")
        other = neighbour(sent)
        return other, ray_at(sent["px"], view, other)

    def seconds_of(body: dict, view: ViewOf) -> list[Second | None]:
        """The pairs' second clicks a request sends (each none, or a neighbour and where in its
        view facing as `view`), as rays."""
        sent = body.get("second_clicks") or []
        if sent and not (isinstance(sent, list) and len(sent) == len(body["photo_px"])):
            raise ValueError("un second clic par paire, ou aucun")
        return [None if s is None else second_of(s, view) for s in sent]

    def walls_shown(sphere: match.Panorama, body: dict) -> dict:
        """A panorama's walls as the page draws them (see _walls_shown), with its own walls
        drawn and removed as saved (`mine`, however far: what the page edits and sends back to
        /api/walls)."""
        drawn, removed, ages = (
            facades.saved_at(sources.walls, sphere.id) if sources.walls else ([], [], [])
        )
        scene = scene_of(sphere, chosen(body))
        mine = {  # with the save each first was in, sent back unchanged (see facades.record)
            "drawn": [
                {"a": _degrees(*w.a), "b": _degrees(*w.b), "top": w.top, "since": w.since}
                for w in drawn
            ],
            "removed": [
                {"a": _degrees(*a), "b": _degrees(*b), "since": s}
                for (a, b), s in zip(removed, ages, strict=True)
            ],
        }
        own = {(w.a, w.b): k for k, w in enumerate(drawn)}
        shown = _walls_shown(scene, sources.segments(sphere.x, sphere.y), own)
        return shown | {"mine": mine, "best": best(sphere, scene, body.get("heading"))}

    def best(sphere: match.Panorama, scene: facades.Scene, heading) -> str | None:
        """The neighbour of the walls that sees best what a view facing `heading` shows (see
        depth.neighbours): the one shown beside, for second clicks."""
        used = [spheres[n] for n in scene.neighbours if n in spheres]
        ranked = (
            depth.neighbours(sphere, used, sources.fixed, float(heading)) if number(heading) else []
        )
        return ranked[0].id if ranked else next(iter(scene.neighbours), None)

    def chosen(body: dict) -> tuple[str, ...] | None:
        """The neighbours the user chose for the walls (none: picked by their geometry)."""
        named = body.get("neighbours")
        if named is not None and not (placings.names(named) and all(n in spheres for n in named)):
            raise ValueError("des panoramas voisins connus, ou aucun")
        return None if named is None else tuple(sorted(set(named)))  # (): none, walls drawn only

    def scene_of(sphere: match.Panorama, neighbours: tuple[str, ...] | None) -> facades.Scene:
        """Today's walls around a panorama; refused without neighbours."""
        try:  # made with the matcher: whatever fails there, the page is told why
            found = sources.scene(sphere.id, neighbours)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"profondeur impossible à faire ici : {error}") from error
        if found is None:
            low, high = depth.NEAR_M
            raise ValueError(f"pas de panorama voisin à {low:g}–{high:g} m : pas de parallaxe ici")
        return found

    @lru_cache(maxsize=KEPT_VIEWS)
    def metres_in(scene: facades.Scene, view: ViewOf) -> np.ndarray:
        # the same view and walls, refitted: made once (walls drawn since: another scene)
        return facades.in_view(scene, view, SHOWN).astype(np.float32)

    def laid_standing(body: dict, image: np.ndarray, sphere: match.Panorama) -> np.ndarray:
        """The view with the photo laid by its standing camera, through the walls there."""
        view = ViewOf(**body["view"])
        camera = registration.standing_from_json(body["standing"])
        origin = _at(sphere)
        metres = metres_in(scene_of(sphere, chosen(body)), view)
        return registration.composite_standing(
            cut(sphere.id, view), image, camera, origin, view, metres
        )

    def answered_in_parallax(body: dict, photo_id: str, sphere: match.Panorama) -> dict:
        """The camera an answer kept in parallax keeps: made again from its pairs and the pin
        the fit started from (`parallax`: lon, lat; the answer's own place may be the position
        accepted since), never what the page sent back; with each pair's depth and what gave
        it, and the neighbours the walls were fitted from."""
        photo_px, view_px, view, size = clicked({**body, **body["pairs"]}, photo_id)
        start = _metres(body["parallax"]["lon"], body["parallax"]["lat"])
        scene = scene_of(sphere, chosen(body))
        seconds = seconds_of(body["pairs"], view)
        found = parallax(photo_px, view_px, view, sphere, size, scene, start, False, seconds)
        return {
            "camera": found["standing"],
            "metres": found["metres"],
            "source": found["source"],
            "neighbours": found["neighbours"],
            "chosen": found["chosen"],
            "start": body["parallax"],  # where the fit started (a fair comparison starts there)
        }

    class Handler(localpage.Handler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
            url = urlsplit(self.path)
            route, ask = unquote(url.path), parse_qs(url.query)
            parts = route.strip("/").split("/")
            try:
                if route == "/":
                    self._send(200, localpage.page(PAGE), "text/html; charset=utf-8")
                elif route == "/api/queue":
                    self._json({"settings": settings(), **next_photo(sources, nearest, where)})
                elif parts[:2] == ["api", "reopen"] and len(parts) == 3:
                    self._json(reopened(sources, parts[2]))
                elif route == "/api/near":
                    self._json(near(float(ask["lon"][0]), float(ask["lat"][0])))
                elif parts[:2] == ["api", "proposal"] and len(parts) == 4:
                    self._json(proposal(parts[2], parts[3]))
                elif parts[0] == "photo" and len(parts) == 2 and parts[1] in sources.photos:
                    flipped = ask.get("flipped", ["0"])[0] == "1"
                    self._send(200, _jpeg(photo_image(parts[1], flipped)), "image/jpeg")
                elif parts[0] == "view" and len(parts) == 2 and parts[1] in spheres:
                    view = ViewOf(
                        float(ask["heading"][0]),
                        float(ask.get("pitch", [match.VIEW_PITCH])[0]),
                        float(ask.get("fov", [match.VIEW_FOV])[0]),
                    )
                    self._send(200, _jpeg(cut(parts[1], view)), "image/jpeg")
                else:
                    self._send(404, b"not found", "text/plain")
            except (KeyError, ValueError) as error:
                self._send(400, str(error).encode(), "text/plain; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._body()
                photo_id, sphere = known(body)
                if self.path == "/api/answer":
                    found = body.get("registration")
                    found = found and registration.from_json(found)
                    if found and found.panorama not in spheres:
                        raise ValueError(f"panorama inconnu : {found.panorama}")
                    pairs = body.get("pairs")
                    if pairs is not None and not found:
                        raise ValueError("des paires vont avec leur panorama")
                    kept = spheres[found.panorama] if found else None
                    standing = None
                    if body.get("parallax"):
                        if pairs is None:
                            raise ValueError("une réponse en parallaxe garde ses paires")
                        standing = answered_in_parallax(body, photo_id, kept)
                    saved = None if pairs is None else kept_pairs(pairs, kept, spheres)
                    held = placings.holding(sources.placings).get(photo_id)
                    if (  # saved again on the same fit (reopened, its cone turned): kept
                        saved is None
                        and body.get("answer") == "points"
                        and held
                        and held.get("pairs")
                        and placings.registration_of(held) == found
                    ):
                        saved, standing = held["pairs"], held.get("standing")
                    self._json(
                        placings.record(
                            sources.placings,
                            photo_id,
                            body.get("answer"),
                            body.get("seconds"),
                            lon=body.get("lon"),
                            lat=body.get("lat"),
                            heading=body.get("heading"),
                            registration=found,
                            flipped=body.get("flipped", False),
                            carried_from=body.get("carried_from"),
                            inputs=sources.photos[photo_id].get("inputs", ""),
                            pairs=saved,
                            standing=standing,
                        )
                    )
                elif self.path == "/api/fit":
                    sphere = here(sphere)
                    photo_px, view_px, view, size = clicked(body, photo_id)
                    self._json(flat(photo_px, view_px, view, sphere, size)[0])
                elif self.path == "/api/depth":
                    self._json(walls_shown(here(sphere), body))
                elif self.path == "/api/walls":  # the panorama's own walls drawn and removed
                    sphere = here(sphere)
                    if sources.walls is None:
                        raise ValueError("pas de fichier où garder les murs dessinés")
                    ground = sphere.z - PANORAMA_HEIGHT_M
                    sent = [body.get(k) for k in ("drawn", "removed")]
                    if not all(isinstance(s, list) for s in sent):
                        raise ValueError("les murs dessinés et ceux retirés : deux listes")
                    drawn = [_drawn(w, ground) for w in sent[0]]
                    removed = [_ends(r) for r in sent[1]]
                    ages = [_since(r) for r in sent[1]]
                    facades.record(sources.walls, sphere.id, drawn, removed, ages)
                    self._json(walls_shown(sphere, body))
                elif self.path == "/api/second":
                    sphere, view = here(sphere), ViewOf(**body["view"])
                    second = second_of({**body, "px": body["neighbour_px"]}, view)
                    met = _second_depth(sphere, ray_at(body["view_px"], view, sphere), second)
                    self._json({"metres": met})
                elif self.path == "/api/epipolar":
                    sphere, other, view = here(sphere), neighbour(body), ViewOf(**body["view"])
                    along = np.geomspace(*EPIPOLAR_M, EPIPOLAR_SAMPLES)[:, None]
                    seen = _at(sphere) + along * ray_at(body["view_px"], view, sphere) - _at(other)
                    seen /= np.linalg.norm(seen, axis=1, keepdims=True)
                    px = registration.view_pixels(seen @ other.axes.T, view, other.axes, SHOWN)
                    self._json({"line": px[np.isfinite(px).all(axis=1)].tolist()})
                elif self.path == "/api/depthview":
                    sphere, view = here(sphere), ViewOf(**body["view"])
                    metres = metres_in(scene_of(sphere, chosen(body)), view)
                    self._send(200, _jpeg(_tinted(cut(sphere.id, view), metres)), "image/jpeg")
                elif self.path == "/api/parallax":
                    sphere = here(sphere)
                    photo_px, view_px, view, size = clicked(body, photo_id)
                    scene = scene_of(sphere, chosen(body))
                    pin = _metres(body["lon"], body["lat"])
                    seconds = seconds_of(body, view)
                    self._json(
                        parallax(photo_px, view_px, view, sphere, size, scene, pin, True, seconds)
                    )
                elif self.path == "/api/composite" and body.get("standing") is not None:
                    image = photo_image(photo_id, bool(body.get("flipped")))
                    self._send(200, _jpeg(laid_standing(body, image, here(sphere))), "image/jpeg")
                elif self.path == "/api/composite":
                    laid = registration.from_json(body["registration"])
                    image = photo_image(photo_id, bool(body.get("flipped")))
                    axes = spheres[laid.panorama].axes
                    view = cut(laid.panorama, laid.view)
                    self._send(
                        200,
                        _jpeg(registration.composite(view, image, laid, laid.view, axes)),
                        "image/jpeg",
                    )
                else:
                    self._send(404, b"not found", "text/plain")
            except (ValueError, KeyError, TypeError) as error:
                self._send(400, str(error).encode(), "text/plain; charset=utf-8")

    return localpage.server(Handler, port, threaded=True)
