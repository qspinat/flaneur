"""Volunteers' panoramas set right on the survey's (Immergis for the City of Paris: ~0.8 m and 0.5°
off their GPS; with each other, 0.01° in median, measured 18/11) through the pairs on disk: each
pair says how far a neighbour turns to agree with a panorama, and where it stands seen from it.
A volunteer seen by survey panoramas that agree takes their heading, and its place where their
bearings cross; one so set by two or more passes it on, up to MOST_HOPS from the survey.
Measured 18/11 (plan panoramas-sur-le-releve): the heading of one survey neighbour left out
within 2° in 98 % of 3,276 tries, the bearing within 1 m in 97 %; through 1 to 5 hops, 95, 90,
85, 81, 76 % within 2° of the survey's own answer. The map (BD TOPO, LiDAR) only checks: fitted
on its own it agreed with the survey 58 % of the time."""

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from flaneur.poses import Applied
from flaneur.sources import write_json

PRODUCER = "Immergis pour la Ville de Paris"  # the survey
APPLIED: Applied = (
    "none"  # what reads panoramas' poses takes of these (cli._placed): "all", "turns"
)
# (the heading only) or "none" — once a sandbox night shows match gains (plan, measure E)
AGREE_DEG = 2.0  # the turns a panorama's references give agree this well, or it is left
MOST_HOPS = 4  # from the survey (5: 76 % within 2°)
SPREAD_DEG = 20.0  # bearings this far apart at least cross well enough to place it
LEAST_TURN_DEG = 2.0  # a panorama turned less is left as it is (its caches stay)
MOST_SHIFT_M = 5.0  # a move beyond this only where the new place is in the street (LiDAR)


class Seen(NamedTuple):
    """A neighbour as a panorama sees it (depth.set_right): how far it turns to agree with the
    panorama as both stand (degrees, counter-clockwise on the plan), and where it stands seen
    from it (east, north; the distance recorded kept)."""

    turn: float
    at: tuple[float, float]


@dataclass(frozen=True)
class SetRight:
    """A panorama set right: its turn (degrees, counter-clockwise), its move (east, north;
    none: left where it stands), how many hops from the survey, on how many references."""

    turn: float
    shift: tuple[float, float] | None
    hops: int
    references: int


def set_right(
    survey: dict[str, tuple[float, float]],
    volunteers: dict[str, tuple[float, float]],
    neighbours: Callable[[str], Iterable[str]],
    seen: Callable[[str, str], Seen | None],
    in_street: Callable[[float, float], bool],
) -> dict[str, SetRight]:
    """The `volunteers` (where each stands, by id) set right, round after round: on the
    `survey` panoramas (right as they stand) first, then on volunteers set right on two
    references or more, up to MOST_HOPS; `seen(home, other)`: the pair of two panoramas
    (none: not on disk); `in_street`: whether a place is in the street."""
    origin = {i: np.asarray(at, float) for i, at in {**survey, **volunteers}.items()}
    turn_of = dict.fromkeys(survey, 0.0)  # those set right: their turn, and where they stand
    place_of = {i: origin[i] for i in survey}
    passing = set(survey)  # those that pass their correction on
    found: dict[str, SetRight] = {}
    for hops in range(1, MOST_HOPS + 1):
        new = {}
        for v in volunteers.keys() - found.keys():
            words = [(w, s) for w in neighbours(v) if w in passing and (s := seen(w, v))]
            if got := _one(words, turn_of, place_of, origin, in_street, origin[v]):
                new[v] = SetRight(*got[:2], hops, got[2])
        if not new:
            break
        for v, f in new.items():
            found[v], turn_of[v] = f, f.turn
            place_of[v] = origin[v] + (f.shift or (0.0, 0.0))
            if f.references >= 2:
                passing.add(v)
    return found


def _one(words, turn_of, place_of, origin, in_street, at):
    """A panorama standing `at` set right on what its references say (`words`), if they
    agree: its turn, its move (where their bearings cross; a big one only into the street),
    how many references."""
    if not words:
        return None
    turns = [turn_of[w] + s.turn for w, s in words]
    if max(turns) - min(turns) > AGREE_DEG:
        return None
    place = crossing(
        [place_of[w] for w, _ in words],
        [_turned(np.asarray(s.at, float) - origin[w], turn_of[w]) for w, s in words],
    )
    shift = None
    if place is not None:
        move = place - at
        if np.hypot(*move) <= MOST_SHIFT_M or in_street(*place):
            shift = (float(move[0]), float(move[1]))
    return float(np.median(turns)), shift, len(words)


def _turned(v: np.ndarray, deg: float) -> np.ndarray:
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]) @ v


def crossing(origins: list[np.ndarray], toward: list[np.ndarray]) -> np.ndarray | None:
    """Where lines from `origins` along `toward` cross (least squares), if two of them are
    SPREAD_DEG apart at least; none else."""
    if len(origins) < 2:
        return None
    d = [v / np.linalg.norm(v) for v in toward]
    angles = [math.degrees(math.atan2(v[1], v[0])) % 180 for v in d]
    spread = max(
        min(abs(a - b), 180 - abs(a - b)) for i, a in enumerate(angles) for b in angles[i + 1 :]
    )
    if spread < SPREAD_DEG:
        return None
    across = [np.eye(2) - np.outer(v, v) for v in d]
    return np.linalg.solve(sum(across), sum(m @ o for m, o in zip(across, origins, strict=True)))


def fixes(found: dict[str, SetRight], applied: Applied) -> dict[str, tuple[float, float, float]]:
    """The panoramas to move and turn, as register.applied takes them (east, north, change of
    compass heading: clockwise, the turn's opposite): those turned beyond LEAST_TURN_DEG;
    `applied` (see APPLIED) "turns": not moved, "none": none."""
    if applied == "none":
        return {}
    return {
        i: (*(f.shift if f.shift and applied == "all" else (0.0, 0.0)), -f.turn)
        for i, f in found.items()
        if abs(f.turn) > LEAST_TURN_DEG
    }


def save(path: Path, found: dict[str, SetRight], where: dict[str, tuple]) -> None:
    """The panoramas set right, each with where it stood (see poses.at)."""
    write_json(path, {i: {**asdict(f), "placed": where[i]} for i, f in sorted(found.items())})


def load(path: Path) -> dict[str, tuple[SetRight, tuple]]:
    """What `save` kept: each panorama set right, and where it stood."""
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {
        i: (
            SetRight(r["turn"], r["shift"] and tuple(r["shift"]), r["hops"], r["references"]),
            tuple(r["placed"]),
        )
        for i, r in rows.items()
    }
