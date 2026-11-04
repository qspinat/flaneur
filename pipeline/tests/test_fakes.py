"""The fake old photos the known-answer check situates: a view of today, mostly replaced and
aged."""

import cv2
import numpy as np

from flaneur import fakes


def test_a_mixed_view_keeps_one_band_of_the_true_view():
    true = np.full((100, 200, 3), 255, np.uint8)
    other = np.zeros((100, 200, 3), np.uint8)

    mixed = fakes.mixed(true, other, (0.7, 1.0))

    assert (mixed == 255).all(axis=(0, 2)).nonzero()[0].tolist() == list(range(140, 200))


def test_an_aged_picture_is_toned_and_softer():
    rows, columns = np.mgrid[0:200, 0:200]
    board = np.where((rows // 4 + columns // 4) % 2, 230, 20).astype(np.uint8)
    picture = cv2.merge([board, board, board])

    aged = fakes.aged(picture, np.random.default_rng(0))

    blue, green, red = (aged[..., k].astype(float).mean() for k in range(3))
    assert red > green > blue  # sepia
    sharpness = cv2.Laplacian(cv2.cvtColor(aged, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert sharpness < 0.5 * cv2.Laplacian(board, cv2.CV_64F).var()
