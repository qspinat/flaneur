"""The pilot's photos: their crop found again from their scan."""

import json

import numpy as np
import pytest
from PIL import Image

from flaneur import pilot


def test_a_photos_crop_and_its_doubts_are_found_again_from_its_scan(tmp_path):
    folder = tmp_path / "pm_1"
    folder.mkdir()
    (folder / "info.json").write_text(json.dumps({"photo_id": "pm:1", "suggested_crop": None}))
    card = np.full((500, 400, 3), (205, 200, 190), dtype=np.uint8)
    card[100:350, 90:310] = np.random.default_rng(0).integers(30, 160, (250, 220, 3))
    Image.fromarray(card).save(folder / "photo.jpg", quality=95)

    changed, doubts = pilot.crop_again(folder)

    info = json.loads((folder / "info.json").read_text())
    assert changed
    assert info["suggested_crop"][0] == pytest.approx(90 / 400, abs=0.02)
    assert info["crop_doubts"] == doubts == []
    assert info["photo_id"] == "pm:1"  # the rest of it kept
    assert not pilot.crop_again(folder)[0]  # found again, the same
