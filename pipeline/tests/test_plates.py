"""Albert-Kahn plates: the plate shown, the museum's label hidden."""

import numpy as np
import pytest
from PIL import Image

from flaneur import plates

LABEL = np.asarray(Image.open(plates.LABEL).convert("L"))


def _scan(h=800, w=593, frame=(15, 12, 10), strip=28, label_at=(20, 12)):
    """A scan: a textured plate, a black frame on three sides (top, left, right), a pale strip
    with handwriting below, the museum's label at the plate's bottom-left."""
    rng = np.random.default_rng(1)
    top, left, right = frame
    scan = np.zeros((h, w, 3), np.uint8)
    y1, x1 = h - strip, w - right
    plate = rng.integers(60, 200, (y1 - top, x1 - left, 3), dtype=np.uint8)
    scan[top:y1, left:x1] = plate
    scan[y1:] = 190
    scan[y1 + 8 : y1 + 12, 100:300] = 20  # a stroke of handwriting
    lh, lw = LABEL.shape
    ly, lx = y1 - label_at[1] - lh, left + label_at[0]
    scan[ly : ly + lh, lx : lx + lw] = LABEL[..., None]
    return (
        Image.fromarray(scan),
        (left / w, top / h, x1 / w, y1 / h),
        (lx / w, ly / h, (lx + lw) / w, (ly + lh) / h),
    )


def test_the_plate_is_shown_without_its_frame_and_strip_and_its_label_hidden():
    scan, plate, label = _scan()

    found = plates.albert_kahn(scan)

    assert found.plate == pytest.approx(plate, abs=0.01)
    assert found.label == pytest.approx(label, abs=0.01)


def test_a_plate_reaching_the_scan_s_edges_is_shown_whole():
    scan, _, label = _scan(frame=(0, 0, 0), strip=0)

    found = plates.albert_kahn(scan)

    assert found.plate == pytest.approx((0.0, 0.0, 1.0, 1.0), abs=0.01)
    assert found.label == pytest.approx(label, abs=0.01)


def test_a_scan_without_the_museum_s_label_hides_nothing():
    rng = np.random.default_rng(2)
    plain = Image.fromarray(rng.integers(60, 200, (600, 800, 3), dtype=np.uint8))

    assert plates.albert_kahn(plain).label is None
