"""The user's answers on the placing page (data/overrides/placings.jsonl, versioned with the
code, appended to): where the photographer stood and looked, and the photo's registration in
its panorama (see registration). Answers: "fits" (the proposal accepted), "points" (placed by
clicking pairs of points), "far" (no panorama sees it: no 3D view), "skip" (comes back later).
The last answer on a photo holds, a skip leaving the one before it; the rows before stay."""

from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from shapely.geometry import Point

from flaneur.gazetteer import to_degrees, to_metres
from flaneur.registration import (
    LEAST_STANDING,
    Registration,
    Standing,
    from_json,
    standing_from_json,
    to_json,
)
from flaneur.sources import append_row, number, rows, seconds_taken

ANSWERS = ("fits", "points", "far", "skip")
PLACED = ("fits", "points")  # those with a registration
# what gave a pair its depth in parallax: a wall fitted or drawn, the street, a second click
DEPTH_SOURCES = ("wall", "drawn", "ground", "click")


def record(
    path: Path,
    photo_id: str,
    answer: str,
    seconds: float,
    *,
    lon: float | None = None,
    lat: float | None = None,
    heading: float | None = None,
    registration: Registration | None = None,
    flipped: bool = False,
    carried_from: str | None = None,
    inputs: str = "",
    pairs: dict | None = None,
    standing: dict | None = None,
    panorama_at: tuple[float, float, float] | None = None,
) -> dict:
    """Append the user's answer on a photo, with the seconds it took; but for a skip, the spot
    and direction set on the map; placed in its panorama ("fits": its proposal accepted,
    "points": clicked by hand), its registration and, for a proposal carried from another old
    photo, that photo. Placed by points, the `pairs` clicked (the photo's pixels; the panorama's
    as directions in its own picture, `own`; the `view` and `side` they were clicked in); kept
    in parallax, its `standing` camera (`camera`, and for each pair the depth it was given,
    `metres`, and the neighbouring panoramas it came `from`, None without). A row by points says
    which fit the user kept (`fit`: "flat" or "parallax"); its `registration` is always flat.
    Where its panorama stood then (`panorama_at`: east, north, heading), for the answer to
    follow it if it is set right since (see followed)."""
    if answer not in ANSWERS:
        raise ValueError(f"une réponse est l'une de {ANSWERS}")
    if answer == "skip":
        if (lon, lat, heading) != (None, None, None):
            raise ValueError("un « skip » n'a pas de place sur le plan")
    elif not (number(lon) and number(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError("une réponse a sa place sur le plan : longitude et latitude")
    if heading is not None and not (number(heading) and 0 <= heading < 360):
        raise ValueError("une direction est un angle de 0 à 360°")
    if answer in PLACED and heading is None:
        raise ValueError("une photo posée dans son panorama a sa direction sur le plan")
    if (answer in PLACED) != (registration is not None):
        raise ValueError("un enregistrement dans le panorama va avec « fits » ou « points »")
    if not isinstance(flipped, bool):
        raise ValueError("« retournée » est vrai ou faux")
    if carried_from is not None and (answer != "fits" or carried_from == photo_id):
        raise ValueError("seule une proposition acceptée est portée depuis une autre photo")
    if pairs is not None:
        _check_pairs(pairs, answer)
    if standing is not None:
        _check_standing(standing, pairs)
    return append_row(
        path,
        {
            "photo_id": photo_id,
            "answer": answer,
            "seconds": seconds_taken(seconds),
            "lon": lon,
            "lat": lat,
            "heading": heading,
            "registration": registration and to_json(registration),
            "flipped": flipped,
            "carried_from": carried_from,
            "inputs": inputs,
            "pairs": pairs,
            "fit": None if pairs is None else "parallax" if standing else "flat",
            "standing": standing,
            "panorama_at": None if panorama_at is None else [float(v) for v in panorama_at],
        },
    )


class State(NamedTuple):
    held: dict[str, dict]  # the answer that holds on each photo answered: its last but skips
    skipped: set[str]  # the photos only skipped so far (they come back after the others)
    answers: list[str]  # the photos answered, one per answer given (not skips), in order


def state(path: Path, now: dict[str, tuple[float, float, float]] | None = None) -> State:
    """The answers given so far, in one reading; with where the panoramas stand `now` (east,
    north, heading, by id), each following its panorama (see followed)."""
    written = rows(path)
    held = {
        row["photo_id"]: followed(row, now) if now else row
        for row in written
        if row["answer"] != "skip"
    }
    answers = [row["photo_id"] for row in written if row["answer"] != "skip"]
    return State(held, {row["photo_id"] for row in written} - held.keys(), answers)


def holding(
    path: Path, now: dict[str, tuple[float, float, float]] | None = None
) -> dict[str, dict]:
    """The answer that holds on each photo answered (see state)."""
    return state(path, now).held


def followed(row: dict, now: dict[str, tuple[float, float, float]]) -> dict:
    """An answer as it stands now that its panorama stands `now` (east, north, heading, by id):
    its place and heading on the plan moved and turned as its panorama was since the answer,
    about it (its registration, in the panorama's own picture, holds). Left as it is without
    where its panorama stood then, or its panorama unknown."""
    reg = registration_of(row) if row.get("registration") else None
    if not row.get("panorama_at") or reg is None or reg.panorama not in now:
        return row
    (x0, y0, a0), (x1, y1, a1) = row["panorama_at"], now[reg.panorama]
    if (x0, y0, a0) == (x1, y1, a1):
        return row
    turn = np.radians(a1 - a0)  # compass: clockwise
    spot = to_metres(Point(row["lon"], row["lat"]))
    east, north = spot.x - x0, spot.y - y0
    there = to_degrees(
        Point(
            x1 + east * np.cos(turn) + north * np.sin(turn),
            y1 - east * np.sin(turn) + north * np.cos(turn),
        )
    )
    heading = None if row.get("heading") is None else (row["heading"] + a1 - a0) % 360
    return {**row, "lon": there.x, "lat": there.y, "heading": heading, "panorama_at": [x1, y1, a1]}


def registration_of(row: dict) -> Registration | None:
    return row["registration"] and from_json(row["registration"])


def standing_of(row: dict) -> Standing | None:
    """The camera where the photographer stood, of an answer kept in parallax (none before)."""
    found = row.get("standing")
    return standing_from_json(found["camera"]) if found else None


def _vectors(found, n: int) -> bool:
    """Whether `found` is a list of lists of `n` numbers."""
    return isinstance(found, list) and all(
        isinstance(v, list) and len(v) == n and all(map(number, v)) for v in found
    )


def _check_pairs(pairs: dict, answer: str) -> None:
    if answer != "points":
        raise ValueError("seule une réponse par points garde ses paires")
    if not isinstance(pairs, dict):
        raise ValueError("des paires : les points de la photo et ceux du panorama")
    photo, own = pairs.get("photo_px"), pairs.get("own")
    shaped = _vectors(photo, 2) and _vectors(own, 3) and len(photo) == len(own) >= 2
    if not shaped or any(not any(d) for d in own):  # a direction is not nothing
        raise ValueError(
            "des paires : autant de points dans la photo que dans le panorama, 2 au moins"
        )
    view = pairs.get("view")
    if not (
        isinstance(view, dict) and all(number(view.get(k)) for k in ("heading", "pitch", "fov"))
    ):
        raise ValueError("des paires : la vue où elles ont été cliquées")
    if not (0 < view["fov"] < 180 and -90 <= view["pitch"] <= 90):
        raise ValueError("des paires : le champ et la hauteur de la vue où elles ont été cliquées")
    side = pairs.get("side")
    if not (type(side) is int and side > 0):
        raise ValueError("des paires : la taille de la vue où elles ont été cliquées")
    seconds = pairs.get("second_clicks")
    if seconds is not None and not (
        isinstance(seconds, list) and len(seconds) == len(photo) and all(map(_second, seconds))
    ):
        raise ValueError(
            "un second clic par paire, ou aucun : le panorama voisin et la direction cliquée"
        )


def _second(s: Any) -> bool:
    """Whether a pair's second click is none, or a neighbour and a direction in its picture."""
    return s is None or (
        isinstance(s, dict)
        and isinstance(s.get("neighbour"), str)
        and _vectors([s.get("own")], 3)
        and any(s["own"])
    )


def _check_standing(standing: dict, pairs: dict | None) -> None:
    if pairs is None:
        raise ValueError("une caméra en parallaxe va avec les paires d'une réponse par points")
    try:
        camera = standing_from_json(standing["camera"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            "une caméra en parallaxe : sa position, son orientation, sa focale"
        ) from error
    older = "source" not in standing  # kept before walls: each depth's neighbours instead
    metres, sources = standing.get("metres"), standing.get("from" if older else "source")
    n = len(pairs["photo_px"])
    if not (
        isinstance(metres, list) and isinstance(sources, list) and len(metres) == len(sources) == n
    ):
        raise ValueError("une profondeur et sa source par paire, ou aucune")
    if not all(m is None or (number(m) and m > 0) for m in metres):
        raise ValueError("une profondeur est un nombre de mètres")
    if not all(
        (m is None) == (s is None)
        and (s is None or (names(s) and bool(s) if older else s in DEPTH_SOURCES))
        for m, s in zip(metres, sources, strict=True)
    ):
        raise ValueError("une profondeur dit d'où elle vient : " + ", ".join(sorted(DEPTH_SOURCES)))
    chosen = standing.get("chosen")
    if not (older or (names(standing.get("neighbours")) and (chosen is None or names(chosen)))):
        raise ValueError("une caméra en parallaxe : les panoramas voisins de ses murs")
    start = standing.get("start")
    if start is not None and not (
        isinstance(start, dict) and number(start.get("lon")) and number(start.get("lat"))
    ):
        raise ValueError("une caméra en parallaxe : le point d'où son recalage est parti")
    if sum(m is not None for m in metres) < LEAST_STANDING:  # (an answer kept under older rules)
        raise ValueError(
            f"une caméra en parallaxe vient de {LEAST_STANDING} paires avec profondeur"
        )
    h, w = camera.size
    if any(not (0 <= x <= w and 0 <= y <= h) for x, y in pairs["photo_px"]):
        raise ValueError("une caméra en parallaxe : les points cliqués tombent hors de sa photo")


def names(value: Any) -> bool:
    """Whether a value is a list of panoramas' ids."""
    return isinstance(value, list) and all(isinstance(i, str) for i in value)


def carried(rows: dict[str, dict]) -> dict[str, list[str]]:
    """For each old photo, the photos carried from it that the user accepted (answers that
    hold, `rows`): they stay when it is answered again, judged by eye on their own."""
    found = defaultdict(list)
    for photo_id, row in sorted(rows.items()):
        if row["answer"] in PLACED and row["carried_from"]:
            found[row["carried_from"]].append(photo_id)
    return dict(found)
