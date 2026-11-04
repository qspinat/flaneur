"""Duplicates: the same negative printed several times, kept under different numbers (the
Musée Carnavalet holds two or three prints of many Atget photos). The prints differ in tone,
trimming and mount, so they are compared on the print alone (see flaneur.crop).

Two steps: pairs are shortlisted, by a coarse fingerprint (which way the brightness changes
across a small grid) or by their captions placing them near each other, then matching points
between the two prints decides. Two prints of the same negative overlap point for point:
hundreds of matches agree on one similarity (one is the other scaled, shifted, turned). Two
shots of the same street taken from nearby agree only on a facade, seen in perspective; two
cards with the same handwriting give a few.
"""

import functools
import json
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

from flaneur import crop, db
from flaneur.gazetteer import from_geojson

FINGERPRINT_SIDE = 12  # the fingerprint compares neighbours on a 12 × 12 grid: 144 bits
# Pairs whose fingerprints differ by at most this many bits are compared point by point
# (prints of one negative: as close as 15 bits, but up to 81 when their tone or trimming
# differ, so they are also compared when their captions place them near: see NEAR_M).
CANDIDATE_BITS = 40
# Captions' places (a point, else a street) this near: all 28 copies the fingerprint missed
# on the pilot were within 54 m.
NEAR_M = 60
FEATURE_SIDE = 800  # the print is scaled to this size to find points
MAX_POINTS = 2000
# Two prints are of one negative when at least this many matched points agree on one
# similarity (a scale, a shift and a turn: all that sets two prints of a negative apart).
# Measured on the pilot, 36 pairs judged by eye: 300 to 1,379 for prints of one negative;
# up to 127 for two shots of one corner taken moments apart from nearby, whose points
# agree on one perspective transform (one facade) up to 210 times.
MIN_MATCHES = 200
# Points this close to the print's edges are left out: logos, stamps and writing sit there
# and are the same on every print of a collection.
EDGE = 0.1


def print_of(image: Image.Image) -> Image.Image:
    """The print alone, in grey levels (the mount around it is cut off when found)."""
    grey = image.convert("L")
    frame = crop.detect(grey)
    return crop.apply(grey, frame) if frame else grey


def fingerprint(print_: Image.Image) -> np.ndarray:
    """Which way the brightness changes between neighbours, on a coarse grid (blurred and
    equalised first, so that tone and grain do not count)."""
    small = ImageOps.equalize(print_.filter(ImageFilter.GaussianBlur(3)))
    grid = np.asarray(small.resize((FINGERPRINT_SIDE + 1, FINGERPRINT_SIDE)), dtype=float)
    return (grid[:, 1:] > grid[:, :-1]).ravel()


def _features(print_: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    """The print's ORB points (their pixels) and descriptors, away from its edges."""
    grey = np.asarray(print_)
    scale = FEATURE_SIDE / max(grey.shape)
    grey = cv2.equalizeHist(cv2.resize(grey, None, fx=scale, fy=scale))
    height, width = grey.shape
    inside = np.zeros_like(grey)
    inside[
        round(EDGE * height) : round((1 - EDGE) * height),
        round(EDGE * width) : round((1 - EDGE) * width),
    ] = 255
    points, descriptors = cv2.ORB_create(MAX_POINTS).detectAndCompute(grey, inside)
    return np.float32([p.pt for p in points]).reshape(-1, 2), descriptors


def matches(a: tuple, b: tuple) -> int:
    """How many points of two prints match and agree on one similarity."""
    (points_a, descriptors_a), (points_b, descriptors_b) = a, b
    if descriptors_a is None or descriptors_b is None:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors_a, descriptors_b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < MIN_MATCHES:  # no more can agree
        return 0
    source = points_a[[m.queryIdx for m in good]]
    target = points_b[[m.trainIdx for m in good]]
    _, agree = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=5000
    )
    return int(agree.sum()) if agree is not None else 0


def _close(marks: np.ndarray) -> Iterable[tuple[int, int]]:
    """The pairs of prints (their indices, the first lower) whose fingerprints differ by
    CANDIDATE_BITS at most."""
    for a in range(len(marks)):
        for b in np.flatnonzero((marks[a + 1 :] != marks[a]).sum(axis=1) <= CANDIDATE_BITS):
            yield a, int(b) + a + 1


def _near(places: Mapping[str, BaseGeometry], ids: list[str]) -> Iterable[tuple[int, int]]:
    """The pairs of `ids` (their indices, the first lower) whose places are within NEAR_M."""
    placed = [k for k, i in enumerate(ids) if i in places]
    if not placed:
        return ()
    where = [places[ids[k]] for k in placed]
    a, b = STRtree(where).query(where, predicate="dwithin", distance=NEAR_M)
    return ((placed[x], placed[y]) for x, y in zip(a, b, strict=True) if x < y)


def groups(
    prints: Mapping[str, Image.Image], places: Mapping[str, BaseGeometry] | None = None
) -> list[set[str]]:
    """The photos that are prints of one negative, by id: groups of two or more (`places`:
    where their captions put them, Lambert-93 metres, for those it is known)."""
    ids = sorted(prints)
    marks = np.array([fingerprint(prints[i]) for i in ids])
    pairs = set(_close(marks)) | set(_near(places or {}, ids))
    features: dict[str, tuple] = {}
    parent = {i: i for i in ids}

    def root(i: str) -> str:
        while parent[i] != i:
            i = parent[i]
        return i

    for a, b in sorted(pairs):
        first, second = ids[a], ids[b]
        if root(first) == root(second):
            continue
        for i in (first, second):
            if i not in features:
                features[i] = _features(prints[i])
        if matches(features[first], features[second]) >= MIN_MATCHES:
            parent[root(second)] = root(first)
    found: dict[str, set[str]] = {}
    for i in ids:
        found.setdefault(root(i), set()).add(i)
    return [g for g in found.values() if len(g) > 1]


def find(photos_dir: Path, places: Mapping[str, BaseGeometry]) -> list[set[str]]:
    """The groups of prints of one negative among the photos prepared in `photos_dir` (one
    folder per photo, see flaneur.pilot), by photo id (`places`: see groups)."""
    prints = {
        json.loads((folder / "info.json").read_text())["photo_id"]: print_of(
            Image.open(folder / "photo.jpg")
        )
        for folder in photos_dir.iterdir()
        if (folder / "info.json").exists() and (folder / "photo.jpg").exists()
    }
    return groups(prints, places)


def places(conn: sqlite3.Connection) -> dict[str, BaseGeometry]:
    """Where each located photo's caption puts it (Lambert-93): its point, else its street
    (each street read once, however many photos it holds)."""
    read = functools.cache(from_geojson)
    return {
        r["photo_id"]: read(r["place"])
        for r in conn.execute(
            "SELECT l.photo_id, COALESCE(l.point, s.geometry) AS place FROM locations l "
            "LEFT JOIN streets s ON s.id = l.street_id WHERE place IS NOT NULL"
        )
    }


def save(conn: sqlite3.Connection, found: list[set[str]]) -> int:
    """Replace the stored groups with these. Each negative is named after the print that
    stands for it, on which it is placed and shown: a CC0 print first, then one already
    placed as a street view, then one matched to a picture of today, then one whose caption
    says where the camera stood."""
    ids = json.dumps(sorted(i for g in found for i in g))
    rank = {
        r["id"]: (
            r["license"] != "CC0",
            not r["is_street_view"],
            r["match"] is None,
            r["viewpoint"] is None,
            r["id"],
        )
        for r in conn.execute(
            "SELECT p.id, p.license, pl.is_street_view, pl.match, r.viewpoint FROM photos p "
            "LEFT JOIN placements pl ON pl.photo_id = p.id "
            "LEFT JOIN readings r ON r.photo_id = p.id "
            "WHERE p.id IN (SELECT value FROM json_each(?))",
            [ids],
        )
    }
    return db.replace(
        conn,
        "copies",
        (
            {"photo_id": i, "negative": min(g, key=lambda j: rank.get(j, (True,) * 4 + (j,)))}
            for g in found
            for i in g
        ),
    )


def negatives(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """For each print of a negative printed several times, by id: all the negative's prints,
    the one that stands for it first."""
    found: dict[str, list[str]] = {}
    for r in conn.execute(
        "SELECT negative, photo_id FROM copies ORDER BY photo_id != negative, photo_id"
    ):
        found.setdefault(r["negative"], []).append(r["photo_id"])
    return {i: prints for prints in found.values() for i in prints}


def years(ranges: list[tuple[int | None, int | None]]) -> tuple[int | None, int | None]:
    """When a negative was taken, from its prints' dates: the years all of them allow, or,
    when they contradict each other, the span covering them all (one of them is wrong)."""
    known = [(start or end, end or start) for start, end in ranges if start or end]
    if not known:
        return None, None
    start, end = max(s for s, _ in known), min(e for _, e in known)
    if start <= end:
        return start, end
    return min(s for s, _ in known), max(e for _, e in known)


def one_per_negative(rows: list, negatives: dict[str, list[str]]) -> list:
    """Rows (with a "photo_id"), one per negative, in their order: the print that stands
    for the negative when it is among them, else the first of its prints that is."""
    kept: dict[str, int] = {}
    for i, r in enumerate(rows):
        first = negatives.get(r["photo_id"], [r["photo_id"]])[0]
        if first not in kept or r["photo_id"] == first:
            kept[first] = i
    return [rows[i] for i in sorted(kept.values())]
