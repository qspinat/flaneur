"""The user's answers on the placing page (data/overrides/placings.jsonl, versioned with the
code, appended to): where the photographer stood and looked, and the photo's registration in
its panorama (see registration). Answers: "fits" (the proposal accepted), "points" (placed by
clicking pairs of points), "far" (no panorama sees it: no 3D view), "skip" (comes back later).
The last answer on a photo holds, a skip leaving the one before it; the rows before stay."""

from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from flaneur.registration import (
    MIN_STANDING,
    Registration,
    Standing,
    from_json,
    standing_from_json,
    to_json,
)
from flaneur.sources import append_row, number, rows, seconds_taken

ANSWERS = ("fits", "points", "far", "skip")
PLACED = ("fits", "points")  # those with a registration


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
) -> dict:
    """Append the user's answer on a photo, with the seconds it took; but for a skip, the spot
    and direction set on the map; placed in its panorama ("fits": its proposal accepted,
    "points": clicked by hand), its registration and, for a proposal carried from another old
    photo, that photo. Placed by points, the `pairs` clicked (the photo's pixels; the panorama's
    as directions in its own picture, `own`; the `view` and `side` they were clicked in); kept
    in parallax, its `standing` camera (`camera`, and for each pair the depth it was given,
    `metres`, and the neighbouring panoramas it came `from`, None without). A row by points says
    which fit the user kept (`fit`: "flat" or "parallax"); its `registration` is always flat."""
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
        },
    )


class State(NamedTuple):
    held: dict[str, dict]  # the answer that holds on each photo answered: its last but skips
    skipped: set[str]  # the photos only skipped so far (they come back after the others)
    answers: list[str]  # the photos answered, one per answer given (not skips), in order


def state(path: Path) -> State:
    """The answers given so far, in one reading."""
    written = rows(path)
    held = {row["photo_id"]: row for row in written if row["answer"] != "skip"}
    answers = [row["photo_id"] for row in written if row["answer"] != "skip"]
    return State(held, {row["photo_id"] for row in written} - held.keys(), answers)


def holding(path: Path) -> dict[str, dict]:
    """The answer that holds on each photo answered (see state)."""
    return state(path).held


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


def _check_standing(standing: dict, pairs: dict | None) -> None:
    if pairs is None:
        raise ValueError("une caméra en parallaxe va avec les paires d'une réponse par points")
    try:
        camera = standing_from_json(standing["camera"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            "une caméra en parallaxe : sa position, son orientation, sa focale"
        ) from error
    metres, sources = standing.get("metres"), standing.get("from")
    n = len(pairs["photo_px"])
    if not (
        isinstance(metres, list) and isinstance(sources, list) and len(metres) == len(sources) == n
    ):
        raise ValueError("une profondeur et sa source par paire, ou aucune")
    if not all(m is None or (number(m) and m > 0) for m in metres):
        raise ValueError("une profondeur est un nombre de mètres")
    named = all(
        (m is None) == (s is None)
        and (s is None or (isinstance(s, list) and s and all(isinstance(i, str) for i in s)))
        for m, s in zip(metres, sources, strict=True)
    )
    if not named:
        raise ValueError("une profondeur va avec les panoramas voisins d'où elle vient")
    start = standing.get("start")
    if start is not None and not (
        isinstance(start, dict) and number(start.get("lon")) and number(start.get("lat"))
    ):
        raise ValueError("une caméra en parallaxe : le point d'où son recalage est parti")
    if sum(m is not None for m in metres) < MIN_STANDING:
        raise ValueError(f"une caméra en parallaxe vient de {MIN_STANDING} paires avec profondeur")
    h, w = camera.size
    if any(not (0 <= x <= w and 0 <= y <= h) for x, y in pairs["photo_px"]):
        raise ValueError("une caméra en parallaxe : les points cliqués tombent hors de sa photo")


def carried(rows: dict[str, dict]) -> dict[str, list[str]]:
    """For each old photo, the photos carried from it that the user accepted (answers that
    hold, `rows`): they stay when it is answered again, judged by eye on their own."""
    found = defaultdict(list)
    for photo_id, row in sorted(rows.items()):
        if row["answer"] in PLACED and row["carried_from"]:
            found[row["carried_from"]].append(photo_id)
    return dict(found)
