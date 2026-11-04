"""Landmarks proposed in photos not yet marked: predicted, then looked for.

A photo's caption puts it near a place: the landmarks standing in its years within SEEN_M
of it may show over the roofs. Each is looked for against pictures of it taken today from
many sides and distances (the references, chosen and boxed by the user: see References).
The photo's upper part is matched at full size, in tiles (a far spire is a few pixels wide),
against each box the user drew around the landmark, cut from its picture (the matcher sees
the landmark only, nothing around it): the pairs LightGlue is sure of that agree on one
two-view geometry found it. The best box with at least MIN_FOUND proposes the landmark, in the
column of the photo where those pairs stand, for the user to check on the marking page ("À
vérifier"). A wrong mark could only turn the photo's right camera down, never place it;
proposals are measured against the user's own marks first, judged as the solver takes a mark
(see agree).

Measured on the 182 photos the user marked or said show none (49 landmarks marked), 44 s a
photo on the graphics chip (alone), 2.7 GB at most: 28 proposals are marks, 1 is none (Notre-Dame,
0.04 right of its spire), and 13 marks are not found: 8 of Notre-Dame far off in the haze (a
few grey pixels), 3 of the Sainte-Chapelle's spire, 2 of the Tribunal de commerce's dome.
Notre-Dame's parts looked for apart: 30 marks, 3 none (its south transept, between them,
taken for its towers), 17 not found. SURE and MIN_FOUND were set on these same photos:
proposals on others are checked by the user all the same."""

import json
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from flaneur import marking, match
from flaneur.landmarks import BUILDINGS, LANDMARKS, SPIRES

SEEN_M = 2500.0  # landmarks this near a photo's place are looked for in it
MOST = 6  # and at most this many, the nearest
TILE = 800  # the photo, as exported (1600), is matched in tiles this big, overlapping by half,
# over its upper part (a landmark stands over the roofs: all 49 marked on the 182, by eye, none
# of them taken from high up), all of it when no taller than a tile
UPPER = match.ZOOM_UPPER
REFERENCE_SIDE = 1024  # a box cut from a reference is matched at most this big (longest side),
REFERENCE_KEYPOINTS = 1024  # its strongest keypoints
SURE = 0.7  # LightGlue's confidence in a pair, for it to count
MIN_FOUND = 18  # pairs agreeing, for a landmark to be proposed (on the 182, 28 marks found
# and 1 proposal none from 12 to 18, 24 and 1 at 30; Notre-Dame's parts apart, 33 and 20 at
# 12, 30 and 3 at 18: the margin kept for photos not marked)
# a box's top: a landmark's column is told by the pairs on this share of its height (on the
# 182, Notre-Dame's parts apart: 82% of finds within 0.03 of the mark, against 79% from all
# the pairs, 58-61% from the box's middle carried by a fit of the pairs, as the user's boxes do
# not centre their spires)
TOP_SHARE = 1 / 3
NEAR = 0.03  # spires proposed this near (a share of the photo's width) are one, either
# a building's parts are looked for as one (landmarks.BUILDINGS): each picture's boxes of them
# joined, a find proposing either part standing then
PART_OF = {part: building for building, parts in BUILDINGS.items() for part in parts}

Box = tuple[float, float, float, float]  # left, top, right, bottom: shares of the picture
# the way landmarks are looked for: photos looked at another way are looked at again
METHOD = "boxes cut from the references (a building's joined), column from their top"
SOURCE_KEYS = ("title", "page", "licence")  # a reference's, on Wikimedia Commons


@dataclass(frozen=True)
class Proposal:
    landmarks: tuple[str, ...]  # keys in landmarks.LANDMARKS: either (a building's parts)
    column: float  # a share of the photo's width
    found: int  # pairs agreeing on it


def candidates(
    place: tuple[float, float], years: Sequence[int | None], known: Collection[str]
) -> list[str]:
    """The landmarks (or buildings: see BUILDINGS) a photo taken at `place` in `years` may
    show, nearest first (as the marking page offers them), among those `known` (with
    references to be looked for)."""
    near = (k for k in marking.offered(place, years) if math.dist(LANDMARKS[k].at, place) <= SEEN_M)
    return [k for k in dict.fromkeys(PART_OF.get(k, k) for k in near) if k in known][:MOST]


def parts_of(key: str, years: Sequence[int | None]) -> tuple[str, ...]:
    """What a find of a landmark (or building) stands for in a photo of `years`: the landmark,
    or the building's parts standing then."""
    if key not in BUILDINGS:
        return (key,)
    return tuple(part for part in BUILDINGS[key] if LANDMARKS[part].stands(years))


def looked_for(
    place: tuple[float, float], years: Sequence[int | None], known: Collection[str]
) -> list[tuple[str, tuple[str, ...]]]:
    """The landmarks (or buildings) looked for in a photo (see candidates), each with what a
    find of it stands for (see parts_of)."""
    return [(key, parts_of(key, years)) for key in candidates(place, years, known)]


def joined(listed: dict[str, dict[str, list[Box]]]) -> dict[str, dict[str, list[Box]]]:
    """References by landmark (see boxes_of), a building's parts by building: each picture's
    boxes of them as one, around them all."""
    out = {key: of_one for key, of_one in listed.items() if key not in PART_OF}
    for building, parts in BUILDINGS.items():
        boxes: dict[str, list[Box]] = {}
        for part in parts:
            for file, of_part in listed.get(part, {}).items():
                boxes.setdefault(file, []).extend(of_part)
        if boxes:
            out[building] = {file: [around(of_file)] for file, of_file in boxes.items()}
    return out


def around(boxes: Sequence[Box]) -> Box:
    """The box around boxes."""
    lefts, tops, rights, bottoms = zip(*boxes, strict=True)
    return min(lefts), min(tops), max(rights), max(bottoms)


def boxes_of(path: Path) -> dict[str, dict[str, list[Box]]]:
    """The references (data/overrides/landmark_refs.jsonl, one row a picture: its `file`,
    under data/raw/landmark_refs, its `title`, `page` and `licence` on Wikimedia Commons, and
    the `boxes` the user drew around each landmark it shows), by landmark: each picture's
    boxes of it. A picture shows a landmark only in its boxes; one without is not used."""
    found: dict[str, dict[str, list[Box]]] = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not all(isinstance(row.get(k), str) and row[k] for k in SOURCE_KEYS):
                raise ValueError("une référence porte son titre, sa page et sa licence")
            for b in row["boxes"]:
                if b["landmark"] not in LANDMARKS:
                    raise ValueError(f"repère inconnu : {b['landmark']}")
                left, top, right, bottom = b["box"]
                if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
                    raise ValueError(f"cadre hors de l'image ou à l'envers : {b['box']}")
                boxes = found.setdefault(b["landmark"], {}).setdefault(row["file"], [])
                boxes.append((left, top, right, bottom))
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"{path}:{n} : référence mal écrite ({e})") from e
    return found


def cut(picture: np.ndarray, box: Box) -> np.ndarray:
    """The part of a picture in a box, no longer than REFERENCE_SIDE (shrunk if it is)."""
    h, w = picture.shape[:2]
    left, top, right, bottom = box
    x, y = min(int(left * w), w - 1), min(int(top * h), h - 1)  # a pixel at least, however thin
    part = picture[y : max(y + 1, round(bottom * h)), x : max(x + 1, round(right * w))]
    scale = REFERENCE_SIDE / max(part.shape[:2])
    if scale < 1:
        part = cv2.resize(part, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return part


def tiles(photo: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """The photo's upper part in tiles of TILE (fewer and smaller when it is smaller), every
    half tile, the last reaching its edge (one nearer the last than a quarter tile is left
    out: the last does its work): (its left, its top, the tile)."""
    h, w = photo.shape[:2]

    def starts(length: int) -> list[int]:
        last = max(0, length - TILE)
        return [s for s in range(0, last, TILE // 2) if s == 0 or last - s >= TILE // 4] + [last]

    return [
        (x, y, photo[y : y + TILE, x : x + TILE]) for y in starts(int(UPPER * h)) for x in starts(w)
    ]


class References:
    """The references of each landmark (see boxes_of), a building's parts joined (see
    joined): the features of each of its boxes, cut from its picture, worked out the first
    time they are needed. Pictures missing from `folder`, or that are not images (a download
    cut short), are left out (`missing`), and landmarks left without any."""

    def __init__(self, path: Path, folder: Path, matcher: match.Matcher):
        self.folder, self.matcher = folder, matcher
        drawn = boxes_of(path)
        files = {f for of_one in drawn.values() for f in of_one}
        # the landmarks each picture shows: a building's box in it is looked at for those only
        self.shows = {f: {key for key, of_one in drawn.items() if f in of_one} for f in files}
        listed = joined(drawn)
        self.missing = {f for f in files if cv2.imread(str(folder / f)) is None}
        self.boxes = {
            key: kept
            for key, of_one in listed.items()
            if (kept := {f: b for f, b in of_one.items() if f not in self.missing})
        }
        self._features: dict[tuple[str, Box], match.Features] = {}

    def of(self, landmark: str, parts: Collection[str]) -> list[match.Features]:
        """The features of a landmark's (or building's) boxes, in the pictures showing one of
        its `parts` (a building's standing then: not a spire before it was raised)."""
        chosen = {
            file: boxes
            for file, boxes in self.boxes.get(landmark, {}).items()
            if self.shows[file] & set(parts)
        }
        for file, boxes in chosen.items():
            if any((file, box) not in self._features for box in boxes):
                picture = cv2.imread(str(self.folder / file))
                for box in boxes:
                    found = self.matcher.features(cut(picture, box))
                    self._features[file, box] = found.top(REFERENCE_KEYPOINTS)
        return [self._features[f, b] for f, boxes in chosen.items() for b in boxes]


def seen(
    pairs: np.ndarray,
    sure: np.ndarray,
    in_photo: match.Features,
    in_box: match.Features,
    least: int = MIN_FOUND,
) -> tuple[np.ndarray, np.ndarray]:
    """The pairs (rows: index in the photo's features, index in the box's; `sure`: LightGlue's
    confidence in each) that found the landmark: those it is sure of, agreeing on one two-view
    geometry; none if fewer than `least`. Their pixels in the photo, and where they are in the
    box (shares of its width and height)."""
    none = np.empty((0, 2), np.float32)
    keep = sure >= SURE
    if keep.sum() < least:  # the geometry only drops some: too few already
        return none, none
    a, b = in_photo.keypoints[pairs[keep, 0]], in_box.keypoints[pairs[keep, 1]]
    agree = match.agreeing(a, b)
    if agree.sum() < least:
        return none, none
    h, w = in_box.size
    return a[agree], b[agree] / np.array([w, h], np.float32)


def column_of(in_photo: np.ndarray, in_box: np.ndarray) -> float:
    """The column of the photo (pixels) where a landmark found by these pairs stands (see
    seen): the middle of those on the top TOP_SHARE of its box, as its top is what is marked
    (the middle of all on a wide front, as Notre-Dame's, may be off it); of all when none are."""
    on_top = in_box[:, 1] <= TOP_SHARE
    return float(np.median(in_photo[on_top if on_top.any() else slice(None), 0]))


def look(
    photo: np.ndarray,
    place: tuple[float, float],
    years: Sequence[int | None],
    matcher: match.Matcher,
    references: References,
) -> list[Proposal]:
    """The landmarks found in a photo (BGR, as exported) taken at `place` in `years`."""
    looked = [
        (standing, references.of(key, standing))
        for key, standing in looked_for(place, years, references.boxes)
    ]
    if not looked:
        return []
    width = photo.shape[1]
    parts = [(x, matcher.features(tile)) for x, _, tile in tiles(photo)]
    found = []
    for standing, boxes in looked:
        best: Proposal | None = None
        for box in boxes:
            for left, mine in parts:
                least = MIN_FOUND if best is None else best.found + 1
                at, there = seen(*matcher.scored_pairs(mine, box), mine, box, least)
                if len(at):
                    best = Proposal(standing, (column_of(at, there) + left) / width, len(at))
        if best is not None:
            found.append(best)
    return found


def agree(proposal: Proposal, mark: dict, place: tuple[float, float]) -> bool:
    """Whether a proposal is a mark (a row of the marks file) in a photo taken at `place`:
    one of the same landmarks, as near its column as a camera found for the photo must see it
    (match.Sight), however wide it saw and however far from its caption it stood (the widest
    the solver tries, the same columns the most degrees apart; the furthest: the least slack)."""
    shared = set(proposal.landmarks) & set(mark.get("landmarks", ()))
    fov = match.FOVS[-1]
    off = abs(match.turn_of(proposal.column, fov) - match.turn_of(mark["column"], fov))
    seen = (LANDMARKS[k] for k in shared)
    return any(
        off <= match.slack(math.dist(lm.at, place) + match.MOST_MOVED_M, lm.spread) for lm in seen
    )


@dataclass(frozen=True)
class Score:
    right: int  # proposals that are marks
    wrong: int  # proposals that are none
    missed: int  # marks of a landmark looked for, not proposed

    def __add__(self, other: "Score") -> "Score":
        return Score(self.right + other.right, self.wrong + other.wrong, self.missed + other.missed)


def score(
    found: list[Proposal], marks: list[dict], looked: Sequence[str], place: tuple[float, float]
) -> Score:
    """Proposals for a photo taken at `place` against the user's marks of landmarks in it
    (rows of the marks file), the landmarks `looked` for being the only ones that could be
    found: each mark taken by one proposal at most (a second on it is wrong); a mark of a
    landmark a proposal found elsewhere (a building's other part) is not missed."""
    taken: dict[int, Proposal] = {}
    for f in found:
        for k, m in enumerate(marks):
            if k not in taken and agree(f, m, place):
                taken[k] = f
                break
    wanted = set(looked) - {key for f in taken.values() for key in f.landmarks}
    missed = sum(
        1 for k, m in enumerate(marks) if k not in taken and wanted & set(m.get("landmarks", ()))
    )
    return Score(len(taken), len(found) - len(taken), missed)


def marks_of(found: list[Proposal]) -> list[dict]:
    """Proposals as marks (rows of the marks file): spires found in one column, one mark of
    either (as the user marks those a photo cannot tell apart; a building's other parts left
    out: two spires in one column, it is a spire)."""
    marks: list[dict] = []
    for f in sorted(found, key=lambda f: -f.found):
        either = next(
            (
                m
                for m in marks
                if set(f.landmarks) & set(SPIRES)
                and set(m["landmarks"]) & set(SPIRES)
                and abs(m["column"] - f.column) <= NEAR
            ),
            None,
        )
        if either is None:
            marks.append({"landmarks": list(f.landmarks), "column": round(f.column, 4)})
        else:
            spires = [k for k in [*either["landmarks"], *f.landmarks] if k in SPIRES]
            either["landmarks"] = list(dict.fromkeys(spires))
    return marks
