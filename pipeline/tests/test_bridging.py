import numpy as np
import pytest
from test_match import facade_seen_from_the_origin

from flaneur import bridging, match, placing, registration

A = match.Panorama("a", 0.0, 0.0, 1.6, 30.0, None, roll=1.0, pitch=-2.0)  # turned, leaning
B = match.Panorama("b", 6.0, 0.0, 1.6, 0.0, None)


def _units(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _seen(world: np.ndarray, pixels: np.ndarray, by: dict[str, list[int]]):
    """The photo's points and what each panorama in `by` matched of them (their rows)."""
    rows = [(name, i) for name, kept in by.items() for i in kept]
    at = {"a": A, "b": B}
    origins = np.array([[at[n].x, at[n].y, at[n].z] for n, _ in rows])
    keys = np.array([i for _, i in rows])
    rays = match.Rays(
        keys,
        np.array([n for n, _ in rows], dtype=object),
        origins,
        _units(world[keys] - origins),
        np.full((len(rows), 3), np.nan),
        np.full((len(rows), 6), np.nan),
    )
    observed = match.Observed(pixels.copy(), (600, 800), rays)
    seen = np.array(
        [frozenset(n for n, kept in by.items() if i in kept) for i in range(len(world))]
    )
    return match.Points(np.round(pixels, 3), world, seen), observed


def test_a_pair_lies_on_the_panorama_s_own_ray():
    world, pixels = facade_seen_from_the_origin()
    points, observed = _seen(world, pixels, {"a": list(range(60))})

    p = bridging.paired(points, observed, A, (600, 800))

    which = [int(np.argmin(np.hypot(*(points.pixels - px).T))) for px in p.photo_px]
    own = registration.own_directions(p.view_px, p.view, A.axes, placing.SHOWN)
    assert own @ A.axes == pytest.approx(_units(world[which] - [A.x, A.y, A.z]), abs=1e-6)


def test_only_the_points_the_panorama_matched_itself_are_paired():
    world, pixels = facade_seen_from_the_origin()
    points, observed = _seen(world, pixels, {"a": list(range(40)), "b": list(range(60))})

    p = bridging.paired(points, observed, A, (600, 800))

    which = {int(np.argmin(np.hypot(*(points.pixels - px).T))) for px in p.photo_px}
    assert which and which <= set(range(40))


def test_the_photo_s_pixels_are_at_the_size_asked():
    world, pixels = facade_seen_from_the_origin()
    points, observed = _seen(world, pixels, {"a": list(range(60))})

    p = bridging.paired(points, observed, A, (1200, 1600))

    shown = p.photo_px / (1600 / match.PHOTO_SIDE)
    assert all(np.hypot(*(points.pixels - px).T).min() < 1e-9 for px in shown)


def test_fewer_than_fifteen_pairs_pair_nothing():
    world, pixels = facade_seen_from_the_origin()
    points, observed = _seen(world, pixels, {"a": list(range(14)), "b": list(range(60))})

    assert bridging.paired(points, observed, A, (600, 800)) is None
