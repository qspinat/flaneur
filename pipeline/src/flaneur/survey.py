"""Volunteers' panoramas set right on the survey's (Immergis for the City of Paris: ~0.8 m and 0.5°
off their GPS; with each other, 0.01° in median, measured 18/11) through the pairs on disk: each
pair says how far a neighbour turns to agree with a panorama, and where it stands seen from it.
A volunteer seen by survey panoramas that agree takes their heading, and its place where their
bearings cross; one so set by two or more passes it on, up to MOST_HOPS from the survey.
Measured 18/11 (plan panoramas-sur-le-releve): the heading of one survey neighbour left out
within 2° in 98 % of 3,276 tries, the bearing within 1 m in 97 %; through 1 to 5 hops, 95, 90,
85, 81, 76 % within 2° of the survey's own answer. The map (BD TOPO, LiDAR) only checks: fitted
on its own it agreed with the survey 58 % of the time."""

import math
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

PRODUCER = "Immergis pour la Ville de Paris"  # the survey
APPLIED = False  # whether what reads panoramas' poses takes these corrections (cli._placed):
# once a sandbox night shows match gains from them (plan panoramas-sur-le-releve, measure E)
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
    known: dict[str, tuple[float, np.ndarray, int, bool]] = {
        i: (0.0, np.asarray(at, float), 0, True) for i, at in survey.items()
    }  # turn, where it stands set right, hops, whether it passes its correction on
    found: dict[str, SetRight] = {}
    for hops in range(1, MOST_HOPS + 1):
        new = {}
        for v, at in volunteers.items():
            if v in found:
                continue
            words = []
            for w in neighbours(v):
                if w in known and known[w][3] and (s := seen(w, v)):  # passed on so far
                    words.append((w, s))
            if got := _one(
                words, known, volunteers, survey, np.asarray(at, float), in_street, hops
            ):
                new[v] = got
        if not new:
            break
        for v, f in new.items():
            found[v] = f
            moved = np.asarray(volunteers[v], float) + (f.shift if f.shift else (0.0, 0.0))
            known[v] = (f.turn, moved, hops, f.references >= 2)
    return found


def _one(words, known, volunteers, survey, at, in_street, hops: int) -> SetRight | None:
    """A panorama standing `at` set right on what its references say (`words`), if they
    agree: its turn, and where their bearings cross (a big move only into the street)."""
    if not words:
        return None
    turns = [known[w][0] + s.turn for w, s in words]
    if max(turns) - min(turns) > AGREE_DEG:
        return None
    origins, toward = [], []
    for w, s in words:
        turn, there, _, _ = known[w]
        stood = np.asarray(volunteers.get(w, survey.get(w)), float)
        origins.append(there)
        toward.append(_turned(np.asarray(s.at, float) - stood, turn))
    place = crossing(origins, toward)
    shift = None
    if place is not None:
        move = place - at
        if np.hypot(*move) <= MOST_SHIFT_M or in_street(*place):
            shift = (float(move[0]), float(move[1]))
    return SetRight(float(statistics.median(turns)), shift, hops, len(words))


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


def fixes(found: dict[str, SetRight]) -> dict[str, tuple[float, float, float]]:
    """The panoramas to move and turn, as register.applied takes them (east, north, change of
    compass heading: clockwise, the turn's opposite): those turned beyond LEAST_TURN_DEG."""
    return {
        i: (*(f.shift or (0.0, 0.0)), -f.turn)
        for i, f in found.items()
        if abs(f.turn) > LEAST_TURN_DEG
    }
