"""Fake old photos for the known-answer check: a view cut from today's panorama, of which only
a band is kept (the rest of the street rebuilt since: a view of another street), then aged
like a print of 1900 (sepia, soft, grainy, faded)."""

import cv2
import numpy as np

SEPIA = np.array([0.55, 0.78, 1.0])  # B, G, R of a toned albumen print, against its grey
BLUR_PX = (1.0, 2.5)  # a soft old lens, and the copy of a copy
GRAIN = 8.0  # grey levels
FADED = (0.6, 0.85)  # the contrast left


def mixed(true: np.ndarray, other: np.ndarray, span: tuple[float, float]) -> np.ndarray:
    """`other` (the same size as `true`) with a band of `true`, its full height, between the
    fractions `span` of its width: the buildings of one side of the street still stand."""
    w = true.shape[1]
    first, last = round(span[0] * w), round(span[1] * w)
    out = other.copy()
    out[:, first:last] = true[:, first:last]
    return out


def aged(picture: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """`picture` (BGR) as an old print: grey, toned sepia, blurred, faded, grainy."""
    grey = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY).astype(float)
    grey = cv2.GaussianBlur(grey, (0, 0), rng.uniform(*BLUR_PX))
    grey = 128 + (grey - 128) * rng.uniform(*FADED)
    grey += rng.normal(0, GRAIN, grey.shape)
    return np.clip(grey[..., None] * SEPIA, 0, 255).astype(np.uint8)
