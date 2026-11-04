"""Chaining: old photos placed from old photos already placed, on built scenes."""

import numpy as np
import pytest

from flaneur import chain, match


def on_street(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.zeros_like(xs)


SIZE = (600, 800)


def facade(n: int = 80) -> np.ndarray:
    """Points of a street's facades, 18 to 26 m north, not one plane."""
    rng = np.random.default_rng(2)
    return np.column_stack([rng.uniform(-12, 12, n), rng.uniform(18, 26, n), rng.uniform(1, 14, n)])


def photo_from(view: match.View, world: np.ndarray) -> np.ndarray:
    return match.project(view, world, SIZE)


def anchor_at(view: match.View, world: np.ndarray, placed_by: np.ndarray) -> chain.Anchor:
    """An old photo placed at `view`, its details on `world`; those `placed_by` today's
    panoramas are its points."""
    pixels = photo_from(view, world).astype(np.float32)
    seen = np.array([frozenset({"pano:1", "pano:2"})] * placed_by.sum(), dtype=object)
    points = match.Points(pixels[placed_by], world[placed_by], seen)
    return chain.Anchor("a", view, pixels, SIZE, points)


B = match.View(0.0, 0.0, 1.6, 0.0, 0.5, 60.0)
A = match.View(7.0, -5.0, 1.6, -12.0, 0.5, 55.0)  # 8.6 m away, turned


def test_a_photo_chained_to_an_anchor_stands_where_it_was_taken_not_on_the_anchor():
    world = facade()
    anchor = anchor_at(A, world, np.arange(len(world)) % 2 == 0)  # half its details placed
    observed = match.Observed(photo_from(B, world), SIZE, match.Rays.of([]))
    shared = np.arange(len(world))

    rays, known = chain.from_anchors([anchor], lambda _: (shared, shared))
    found = match.place(observed, match.Where(on_street), match.Trace(), rays, known)

    assert found.camera is not None
    assert np.hypot(found.camera.x - B.x, found.camera.y - B.y) < 0.3
    assert chain.provenance(found.points) == ["a"]


def test_an_anchors_point_goes_to_the_detail_matched_on_it_with_who_saw_it():
    world = facade(10)
    anchor = anchor_at(A, world, np.isin(np.arange(10), [2, 5]))

    keys, where, seen = chain.transferred(anchor, np.array([10, 11, 12]), np.array([2, 3, 5]))

    assert keys.tolist() == [10, 12]
    assert where == pytest.approx(world[[2, 5]])
    assert seen[0] == {"pano:1", "pano:2", "via:a"}


def test_points_an_anchor_passes_on_from_one_panorama_are_not_seen_twice():
    world = facade()
    one = np.array([frozenset({"pano:1", "via:a"})] * len(world), dtype=object)
    observed = match.Observed(photo_from(B, world), SIZE, match.Rays.of([]))
    trace = match.Trace()

    found = match.place(
        observed,
        match.Where(on_street),
        trace,
        known=match.Known(np.arange(len(world)), world, one),
    )

    assert found.camera is None  # the anchor passed them on: it did not see them itself
    assert "vue d'une seule source" in {c.why for c in trace.candidates}


def test_an_anchors_rays_are_never_cast_on_todays_walls():
    world = facade(10)
    rays = chain.rays_from(anchor_at(A, world, np.zeros(10, bool)), np.arange(10), np.arange(10))

    assert np.isnan(rays.casts).all()
    assert set(rays.sources) == {"photo:a"}
    ahead = (world - [A.x, A.y, A.z]) / np.linalg.norm(world - [A.x, A.y, A.z], axis=1)[:, None]
    assert rays.directions == pytest.approx(ahead, abs=1e-3)


class Counting:
    """A matcher that counts its matches."""

    def __init__(self, precision="full"):
        self.calls, self.precision = 0, precision

    def match(self, a, b):
        self.calls += 1
        return np.array([0, 1]), np.array([3, 4])


def test_two_photos_are_matched_once_whichever_asks(tmp_path):
    matcher = Counting()
    store = chain.Store(tmp_path, matcher)

    ab = store.pairs(("a", "1"), ("b", "2"), lambda _: None)
    ba = store.pairs(("b", "2"), ("a", "1"), lambda _: None)

    assert matcher.calls == 1
    assert ab.tolist() == [[0, 3], [1, 4]]
    assert ba.tolist() == [[3, 0], [4, 1]]


def test_two_photos_are_matched_again_in_another_precision(tmp_path):
    full, half = Counting(), Counting("half")

    chain.Store(tmp_path, full).pairs(("a", "1"), ("b", "2"), lambda _: None)
    chain.Store(tmp_path, half).pairs(("a", "1"), ("b", "2"), lambda _: None)

    assert (full.calls, half.calls) == (1, 1)


def test_an_observation_is_kept_for_the_inputs_it_was_made_from(tmp_path):
    store = chain.Store(tmp_path, None)
    rays = match.Rays(
        np.array([0, 1]),
        np.array(["pano:1", "pano:2"], dtype=object),
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.full((2, 3), np.nan),
        np.full((2, 6), np.nan),
    )
    store.save_observed("pm:1", "x", match.Observed(np.ones((2, 2)), SIZE, rays))

    kept = store.observed("pm:1", "x")

    assert kept.size == SIZE and kept.rays.sources.tolist() == ["pano:1", "pano:2"]
    assert store.observed("pm:1", "y") is None
