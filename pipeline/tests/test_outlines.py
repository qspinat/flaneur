"""The outlines of the buildings then standing, against the edges of an old photo, on a built
scene."""

from dataclasses import replace

import numpy as np

from flaneur import match, outlines
from flaneur.blocks import Blocks


def drawn(blocks: Blocks, view: match.View, size: tuple[int, int]) -> np.ndarray:
    """A grey picture of the buildings seen by `view`: each wall shaded by the way it faces,
    the sky dark."""
    ahead = match.rays(match.pixels_of(size), size, view.fov, view.horizon, view.heading)
    _, edge = blocks.hit(np.array([view.x, view.y, view.z]), ahead)
    e = blocks.edges[np.maximum(edge, 0)]
    facing = np.abs(np.cos(np.arctan2(e[:, 3] - e[:, 1], e[:, 2] - e[:, 0])))
    return np.where(edge >= 0, 90 + 140 * facing, 20).astype(np.uint8).reshape(size)


BLOCKS = Blocks.of(
    [
        ([(-20, 25), (-4, 25), (-4, 40), (-20, 40)], 0.0, 14.0),
        ([(2, 30), (12, 22), (20, 32), (10, 40)], 0.0, 20.0),
        ([(-30, 60), (30, 60), (30, 70), (-30, 70)], 0.0, 9.0),
    ]
)
VIEW = match.View(0.0, 0.0, 1.6, heading=5.0, horizon=0.6, fov=60.0)
SIZE = (600, 800)


def test_the_outlines_fall_on_the_edges_of_what_the_camera_saw():
    photo = drawn(BLOCKS, VIEW, SIZE)

    right = outlines.fit(outlines.edges_of(photo), VIEW, BLOCKS)
    turned = outlines.fit(outlines.edges_of(photo), replace(VIEW, heading=VIEW.heading + 6), BLOCKS)

    assert right > 0.8
    assert turned < 0.5 * right


def test_a_camera_fits_better_than_the_same_camera_turned():
    photo = drawn(BLOCKS, VIEW, SIZE)
    elsewhere = drawn(BLOCKS, replace(VIEW, x=-9.0, heading=30.0), SIZE)

    assert outlines.contrast(photo, VIEW, BLOCKS) > 0.4
    assert outlines.contrast(elsewhere, VIEW, BLOCKS) < 0.15


def test_nothing_to_fit_scores_nothing():
    empty = Blocks.of([])

    assert outlines.fit(outlines.edges_of(np.zeros(SIZE, np.uint8)), VIEW, empty) == 0.0
    assert outlines.contrast(np.zeros(SIZE, np.uint8), VIEW, empty) == 0.0
