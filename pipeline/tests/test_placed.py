import numpy as np
from test_bridging import AT, A, B, _seen
from test_match import facade_seen_from_the_origin

from flaneur import placed, registration

HERE = A.__class__("here", 0.0, 0.0, 1.6, 30.0, None, roll=1.0, pitch=-2.0)  # at the camera
AT["here"] = HERE


def _in_the_photo():
    """The facade's points the photo (800 × 600) shows."""
    world, pixels = facade_seen_from_the_origin(120)
    inside = ((pixels >= 0) & (pixels <= [800, 600])).all(axis=1)
    return world[inside], pixels[inside]


def _off_in_view(r: registration.Registration, sphere, world, pixels) -> float:
    """How far (view pixels, worst) the registration puts the photo's points from where the
    panorama sees them."""
    h = registration.in_view(r, r.view, sphere.axes)
    unit = np.column_stack([pixels / [800, 600], np.ones(len(pixels))]) @ h.T
    at = unit[:, :2] / unit[:, 2:] * 1  # in [0, 1] of the view
    seen = world - [sphere.x, sphere.y, sphere.z]
    own = seen / np.linalg.norm(seen, axis=1, keepdims=True) @ sphere.axes.T
    truth = registration.view_pixels(own, r.view, sphere.axes, 1)
    return float(np.abs(at - truth).max())


def test_a_placement_is_registered_in_the_first_panorama_that_fits():
    world, pixels = _in_the_photo()
    points, observed = _seen(world, pixels, {"here": list(range(len(world)))})

    r = placed.registered(points, observed, [HERE])

    assert r.panorama == "here" and _off_in_view(r, HERE, world, pixels) < 1e-3


def test_a_panorama_without_enough_pairs_gives_way_to_the_next():
    world, pixels = _in_the_photo()
    points, observed = _seen(world, pixels, {"b": list(range(10)), "here": list(range(len(world)))})

    assert placed.registered(points, observed, [B, HERE]).panorama == "here"


def test_no_panorama_that_fits_registers_nothing():
    world, pixels = _in_the_photo()
    points, observed = _seen(world, pixels, {"a": list(range(10)), "b": list(range(10))})

    assert placed.registered(points, observed, [A, B]) is None
