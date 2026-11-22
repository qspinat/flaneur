"""Albert-Kahn's autochrome plates as scanned: the plate in a black frame, the collection's
number handwritten beside it ("SEINE · A 13583"), and the museum's black label
("ALBERT-KAHN musée départemental") laid over the plate's bottom-left corner. What is shown of
a scan (the plate) and what is never drawn on today's street (the label), as shares of the
image; the image itself is left as it is (match reads it: its hash names the placements).

The plate's edge is a straight line across the scan, between the frame or the strip and the
picture: on each side, the strongest line within its outer BAND, none when no line stands out
(the plate reaches that side). The label is the same picture on every scan: found by its
template (LABEL), at a few scales. Measured 22/11: the edges right on 16 scans looked at; the
label found on all 104 scans on disk (normalised correlation ≥ 0.76, median 0.93)."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LABEL = Path(__file__).with_name("albert_kahn_label.png")  # cut from IF40077's scan
BAND = 0.12  # a side's plate edge is looked for this far in from the scan's edge, at most
STRONG = 3.0  # a line this many times the scan's typical gradient, at least: an edge
LABEL_CORNER = (0.35, 0.25)  # the label is looked for in this share of the width and height
LABEL_SCALES = (0.8, 0.9, 1.0, 1.1, 1.25)
LABEL_SCORE = 0.6  # found when the template matches this well at least
Box = tuple[float, float, float, float]  # x0, y0, x1, y1: shares of the width and height


@dataclass(frozen=True)
class Plate:
    shown: Box
    hidden: Box | None  # the museum's label, never drawn (none found: nothing hidden)


def albert_kahn(image: Image.Image) -> Plate:
    """What of an Albert-Kahn scan is the plate, and where the museum's label lies on it."""
    grey = np.asarray(image.convert("L"), float)
    return Plate(_plate(grey), _label(grey.astype(np.uint8)))


def _plate(grey: np.ndarray) -> Box:
    h, w = grey.shape
    smooth = cv2.GaussianBlur(grey, (3, 3), 0)
    across = np.abs(cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3))
    down = np.abs(cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3))
    # a line across the scan, not a word: the median along its middle
    rows = np.median(across[:, int(0.1 * w) : int(0.9 * w)], axis=1)
    columns = np.median(down[int(0.1 * h) : int(0.9 * h), :], axis=0)
    y0, y1 = _edge(rows, False), _edge(rows, True)
    x0, x1 = _edge(columns, False), _edge(columns, True)
    return x0 / w, y0 / h, x1 / w, y1 / h


def _edge(profile: np.ndarray, from_end: bool) -> int:
    """Where the plate starts, counted from one end of the profile (the scan's edge if no line
    stands out within BAND of it)."""
    n = len(profile)
    k = int(BAND * n)
    part = profile[n - k :][::-1] if from_end else profile[:k]
    i = int(np.argmax(part))
    if part[i] < STRONG * max(float(np.median(profile)), 1.0):
        return n if from_end else 0
    return n - i - 1 if from_end else i + 1


def _label(grey: np.ndarray) -> Box | None:
    h, w = grey.shape
    cw, ch = int(LABEL_CORNER[0] * w), int(LABEL_CORNER[1] * h)
    corner = grey[h - ch :, :cw]
    template = cv2.imread(str(LABEL), cv2.IMREAD_GRAYSCALE)
    best = (LABEL_SCORE, None)
    for scale in LABEL_SCALES:
        t = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if t.shape[0] >= corner.shape[0] or t.shape[1] >= corner.shape[1]:
            continue
        _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(corner, t, cv2.TM_CCOEFF_NORMED))
        if score >= best[0]:
            top = h - ch + y
            best = (score, (x / w, top / h, (x + t.shape[1]) / w, (top + t.shape[0]) / h))
    return best[1]
