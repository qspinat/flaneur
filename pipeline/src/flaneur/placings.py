"""The user's answers on the placing page (data/overrides/placings.jsonl, versioned with the
code, appended to): where the photographer stood and looked, and the photo's registration in
its panorama (see registration). Answers: "fits" (the proposal accepted), "points" (placed by
clicking pairs of points), "far" (no panorama sees it: no 3D view), "skip" (comes back later).
The last answer on a photo holds, a skip leaving the one before it; the rows before stay."""

from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from flaneur.registration import Registration, from_json, to_json
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
) -> dict:
    """Append the user's answer on a photo, with the seconds it took; but for a skip, the spot
    and direction set on the map; placed in its panorama ("fits": its proposal accepted,
    "points": clicked by hand), its registration and, for a proposal carried from another old
    photo, that photo."""
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


def carried(rows: dict[str, dict]) -> dict[str, list[str]]:
    """For each old photo, the photos carried from it that the user accepted (answers that
    hold, `rows`): they stay when it is answered again, judged by eye on their own."""
    found = defaultdict(list)
    for photo_id, row in sorted(rows.items()):
        if row["answer"] in PLACED and row["carried_from"]:
            found[row["carried_from"]].append(photo_id)
    return dict(found)
