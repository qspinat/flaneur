"""Old photos placed by old photos already placed: chaining, round after round.

A photo B that today's panoramas alone cannot place is matched with the old photos placed
near it (its anchors), which share far more with it than today does. An anchor A gives B
two kinds of points, never a point from A's rays alone (A's camera would explain them all,
whatever their distance, and B would be put where A stood):

- the points that placed A (seen from today's panoramas, so placed independently of A's
  camera), on the details B shares with them: they keep the pictures that saw them;
- the rays of A's details, which make a point only where they meet another picture's ray
  (another anchor's, or a panorama's: see match.placed).

The camera is then found and checked as any other (match.place), so a photo is kept only
when it is sure; the photos it was placed from are its provenance."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from flaneur import match, register
from flaneur.pilot import safe_name
from flaneur.sources import write_atomic, write_json

TRANSFER_PX = 0.5  # a detail matched in an anchor is one of its points within this
CHAIN_M = 80  # a photo is chained to old photos placed this near its caption's place
MOST_ROUNDS = 5
KEPT = 128  # photos' features kept in memory (about 1 MB each)


@dataclass(frozen=True)
class Anchor:
    """An old photo placed: its id, its camera, its own details (the keypoints of the photo
    as matched, of `size`), the points that placed it, and what it was placed from
    (`inputs`)."""

    id: str
    view: match.View
    keypoints: np.ndarray
    size: tuple[int, int]
    points: match.Points
    inputs: str = ""

    @property
    def source(self) -> str:
        return f"photo:{self.id}"


def rays_from(anchor: Anchor, in_photo: np.ndarray, in_anchor: np.ndarray) -> match.Rays:
    """The rays of an anchor's details matched in a photo (`in_photo`, `in_anchor`: their
    indices): from its camera, never cast (the walls it saw may since have gone)."""
    v, n = anchor.view, len(in_photo)
    ahead = match.rays(
        anchor.keypoints[in_anchor].astype(float),
        anchor.size,
        v.fov,
        v.horizon,
        v.heading,
    )
    return match.Rays(
        in_photo,
        np.full(n, anchor.source, dtype=object),
        np.tile([v.x, v.y, v.z], (n, 1)),
        ahead,
        np.full((n, 3), np.nan),
        np.full((n, 6), np.nan),
    )


def transferred(anchor: Anchor, in_photo: np.ndarray, in_anchor: np.ndarray) -> match.Known:
    """The points that placed an anchor, on the details a photo shares with it: passed on
    through the anchor (match.VIA), seen by the pictures that saw them."""
    if not len(anchor.points.pixels) or not len(in_anchor):
        return match.Known.none()
    matched = anchor.keypoints[in_anchor]
    gap = np.linalg.norm(matched[:, None] - anchor.points.pixels[None], axis=2)
    nearest = gap.argmin(axis=1)
    on_point = gap[np.arange(len(matched)), nearest] <= TRANSFER_PX
    seen = [anchor.points.seen[k] | {match.VIA + anchor.id} for k in nearest[on_point]]
    return match.Known(
        in_photo[on_point], anchor.points.world[nearest[on_point]], np.array(seen, dtype=object)
    )


def from_anchors(
    anchors: Iterable[Anchor], pairs: Callable[[Anchor], tuple[np.ndarray, np.ndarray]]
) -> tuple[match.Rays, match.Known]:
    """What a photo's anchors give it (`pairs(anchor)`: their matched details, indices in the
    photo and in the anchor): rays, and points already placed."""
    rays, known = [], []
    for anchor in anchors:
        in_photo, in_anchor = pairs(anchor)
        if len(in_photo):
            rays.append(rays_from(anchor, in_photo, in_anchor))
            known.append(transferred(anchor, in_photo, in_anchor))
    return match.Rays.of(rays), match.Known.of(known)


def provenance(points: match.Points) -> list[str]:
    """The old photos a camera was found from: those that saw its points, or passed them on."""
    seen = frozenset().union(*points.seen) if len(points.seen) else frozenset()
    return sorted({s.split(":", 1)[1] for s in seen if s.startswith(("photo:", match.VIA))})


# Kept on disk, per photo: what it was matched with (its features, its rays in today's
# panoramas, the points that placed it), so that a round only matches it with new anchors.


class Store:
    """The features, observations and points of old photos, and the matches between them,
    in `folder` (data/raw: derived, not versioned)."""

    def __init__(self, folder: Path, matcher: match.Matcher):
        self.folder, self.matcher = folder, matcher
        self._features: dict[tuple[str, str], match.Features] = {}  # oldest first

    def _path(self, kind: str, photo_id: str, suffix: str) -> Path:
        return self.folder / kind / f"{safe_name(photo_id)}{suffix}"

    def features(
        self, photo_id: str, image: str, small: Callable[[], np.ndarray]
    ) -> match.Features:
        """A photo's features as matched (`image`: its image's hash; `small()`: the photo,
        shrunk, when they must be extracted): extracted once, kept in half precision as the
        views' are, so that every use sees the same."""
        key = (photo_id, image)
        if key in self._features:
            return self._features[key]
        path = self._path("photo_features", photo_id, f"_{image}.npz")
        if path.exists():
            with np.load(path) as kept:
                size = tuple(int(v) for v in kept["size"])
                found = match.Features(kept["k"], kept["d"].astype(np.float32), size)
        else:
            extracted = self.matcher.features(small())
            half = extracted.descriptors.astype(np.float16)
            _savez(path, k=extracted.keypoints, d=half, size=extracted.size)
            found = match.Features(extracted.keypoints, half.astype(np.float32), extracted.size)
        if len(self._features) >= KEPT:
            self._features.pop(next(iter(self._features)))
        self._features[key] = found
        return found

    def save_observed(self, photo_id: str, inputs: str, observed: match.Observed) -> None:
        rays = {f.name: getattr(observed.rays, f.name) for f in fields(match.Rays)}
        rays["sources"] = rays["sources"].astype(str)
        _savez(
            self._path("observed", photo_id, ".npz"),
            inputs=inputs,
            pixels=observed.pixels,
            size=observed.size,
            views=observed.views,
            **rays,
        )

    def observed(self, photo_id: str, inputs: str) -> match.Observed | None:
        """A photo's observation, if made from these `inputs` (see cli._inputs)."""
        path = self._path("observed", photo_id, ".npz")
        if not path.exists():
            return None
        with np.load(path) as kept:
            if str(kept["inputs"]) != inputs:
                return None
            rays = {f.name: kept[f.name] for f in fields(match.Rays)}
            rays["sources"] = rays["sources"].astype(object)
            size = tuple(int(v) for v in kept["size"])
            views = int(kept["views"]) if "views" in kept else 0
            return match.Observed(kept["pixels"], size, match.Rays(**rays), views)

    def save_points(self, photo_id: str, inputs: str, points: match.Points) -> None:
        write_json(
            self._path("points", photo_id, ".json"),
            {
                "inputs": inputs,
                "pixels": np.round(points.pixels, 3).tolist(),
                "world": np.round(points.world, 3).tolist(),
                "seen": [sorted(s) for s in points.seen],
            },
        )

    def points(self, photo_id: str, inputs: str) -> match.Points | None:
        kept = register.load(self._path("points", photo_id, ".json"))
        if kept.get("inputs") != inputs:
            return None
        return match.Points(
            np.array(kept["pixels"], float).reshape(-1, 2),
            np.array(kept["world"], float).reshape(-1, 3),
            np.array([frozenset(s) for s in kept["seen"]], dtype=object),
        )

    def pairs(
        self,
        a: tuple[str, str],
        b: tuple[str, str],
        features: Callable[[str], match.Features],
    ) -> np.ndarray:
        """The details two old photos share (`a`, `b`: id and image hash; `features(id)`:
        a photo's, needed only when they were never matched): rows of indices in a and in
        b, matched once in the matcher's precision and kept."""
        first, second = sorted([a, b])
        name = f"{safe_name(first[0])}_{first[1]}__{safe_name(second[0])}_{second[1]}.npy"
        path = self.folder / f"old_pairs_{self.matcher.precision}" / name
        if path.exists():
            found = np.load(path)
        else:
            pair = self.matcher.match(features(first[0]), features(second[0]))
            found = np.column_stack(pair).astype(np.int32)
            write_atomic(path, lambda out: np.save(out, found))
        return found if first == a else found[:, ::-1]


def _savez(path: Path, **arrays) -> None:
    write_atomic(path, lambda out: np.savez(out, **arrays))
