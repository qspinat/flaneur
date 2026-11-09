"""The depth behind what passes (a car, a person) rebuilt from the street around it."""

import numpy as np

from flaneur import behind, match

SIDE = 160
EYE = 2.5  # the camera above the street
FACADE_M = 8.0  # a facade running north, this far east of the camera
TOP_M = 18.0  # its height


def _street(slope: float = 0.0):
    """A view looking east at a facade (x = FACADE_M, up to TOP_M) over a street (z = slope·x),
    from a camera EYE above the street: its rays, true depth, and which pixels are building or
    ground."""
    rays = match.view_directions(match.pixels_of((SIDE, SIDE)), 90.0, side=SIDE)
    origin = np.array([0.0, 0.0, EYE])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_wall = np.where(rays[:, 0] > 0, FACADE_M / rays[:, 0], np.inf)
        z_wall = origin[2] + t_wall * rays[:, 2]
        t_wall = np.where((z_wall >= slope * FACADE_M) & (z_wall <= TOP_M), t_wall, np.inf)
        den = rays[:, 2] - slope * rays[:, 0]
        t_ground = np.where(den < 0, -EYE / den, np.inf)
    depth = np.minimum(t_wall, t_ground)
    built = np.isfinite(t_wall) & (t_wall <= t_ground)
    ground = np.isfinite(t_ground) & (t_ground < t_wall)
    depth = np.where(np.isfinite(depth), depth, np.nan)
    shape = (SIDE, SIDE)
    return rays, origin, depth.reshape(shape), built.reshape(shape), ground.reshape(shape)


def _hole(rows: slice, cols: slice) -> np.ndarray:
    hole = np.zeros((SIDE, SIDE), bool)
    hole[rows, cols] = True
    return hole


def _fill(hole, depth, built, ground, rays, origin):
    known = np.where(hole, np.nan, depth)
    return behind.filled(hole, known, built & ~hole, ground & ~hole, rays, origin, 0.0)


def test_a_car_at_the_foot_of_a_facade_is_seen_through():
    rays, origin, depth, built, ground = _street()
    foot = np.nanargmax(np.where(built, np.arange(SIDE)[:, None], -1), axis=0)[SIDE // 2]
    hole = _hole(slice(foot - 12, foot + 12), slice(60, 100))

    found = _fill(hole, depth, built, ground, rays, origin)

    assert np.nanmax(np.abs(found[hole] - depth[hole]) / depth[hole]) < 0.02


def test_a_hole_up_to_the_top_takes_the_facade_from_its_sides():
    rays, origin, depth, built, ground = _street()
    hole = _hole(slice(0, SIDE // 2 + 10), slice(70, 90))

    found = _fill(hole, depth, built, ground, rays, origin)

    on = hole & np.isfinite(depth)
    assert np.nanmedian(np.abs(found[on] - depth[on]) / depth[on]) < 0.02


def test_a_sloping_street_is_followed_not_taken_flat():
    rays, origin, depth, built, ground = _street(slope=0.06)
    hole = _hole(slice(SIDE - 30, SIDE - 5), slice(40, 120))
    flat = behind.filled(
        hole, np.where(hole, np.nan, depth), built & ~hole, np.zeros_like(ground), rays, origin, 0.0
    )

    found = _fill(hole, depth, built, ground, rays, origin)

    error = np.nanmedian(np.abs(found[hole] - depth[hole]) / depth[hole])
    error_flat = np.nanmedian(np.abs(flat[hole] - depth[hole]) / depth[hole])
    assert error < 0.02 < error_flat


def test_nothing_known_around_leaves_the_sky_empty_and_the_street_flat():
    rays, origin, *_ = _street()
    hole = _hole(slice(0, SIDE), slice(0, SIDE))
    nothing = np.zeros((SIDE, SIDE), bool)

    found = behind.filled(hole, np.full((SIDE, SIDE), np.nan), nothing, nothing, rays, origin, 0.0)

    up = (rays[:, 2] > 0).reshape(SIDE, SIDE)
    assert np.isnan(found[up]).all()
    assert np.isfinite(found[~up & hole]).any()


def test_a_street_seen_facing_west_is_rebuilt_as_facing_east(monkeypatch):
    monkeypatch.setattr(behind, "RING", 0.25)  # wide enough to see the facade spread on the plan
    rays, origin, depth, built, ground = _street()
    hole = _hole(slice(0, SIDE // 2 + 10), slice(50, 70))
    # the facade seen on one side only, its points on either side of the view's axis
    built = built & (np.arange(SIDE)[None, :] >= 70)
    east = _fill(hole, depth, built, ground, rays, origin)
    assert np.isfinite(east).any()
    west_rays = rays * np.array([-1.0, -1.0, 1.0])  # the same street turned half round

    west = _fill(hole, depth, built, ground, west_rays, origin)

    assert np.array_equal(np.isfinite(west), np.isfinite(east))
    assert np.allclose(west[np.isfinite(west)], east[np.isfinite(east)], rtol=1e-6)
