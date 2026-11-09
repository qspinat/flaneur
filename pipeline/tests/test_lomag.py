import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from loma.descriptor.dedode import DeDoDeDescriptor
from loma.detector.dad import DaD

from flaneur import lomag

URL = "https://example.org/weights/tiny.pth"


def _saved(tmp_path, kept) -> str:
    path = tmp_path / "tiny.pth"
    torch.save(kept, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_weights_load_when_their_hash_matches(tmp_path, monkeypatch):
    monkeypatch.setitem(lomag.WEIGHTS, "tiny.pth", _saved(tmp_path, {"w": torch.ones(3)}))
    loaded = lomag.pinned(URL, tmp_path)
    assert torch.equal(loaded["w"], torch.ones(3))


def test_weights_whose_hash_differs_are_refused(tmp_path, monkeypatch):
    _saved(tmp_path, {"w": torch.ones(3)})
    monkeypatch.setitem(lomag.WEIGHTS, "tiny.pth", "0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256"):
        lomag.pinned(URL, tmp_path)


def test_weights_not_pinned_are_refused(tmp_path):
    with pytest.raises(RuntimeError, match="épinglés"):
        lomag.pinned(URL, tmp_path)


def test_pinned_weights_holding_more_than_tensors_are_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(lomag.WEIGHTS, "tiny.pth", _saved(tmp_path, {"w": SimpleNamespace()}))
    with pytest.raises(Exception, match="weights_only|Weights only"):
        lomag.pinned(URL, tmp_path)


def test_loading_through_pinned_leaves_torch_as_it_was(tmp_path):
    before = torch.hub.load_state_dict_from_url
    with lomag.loading_pinned(tmp_path):
        assert torch.hub.load_state_dict_from_url is not before
    assert torch.hub.load_state_dict_from_url is before


@pytest.mark.parametrize("size", [(480, 640), (640, 427)])
def test_images_are_read_as_loma_reads_them_from_a_file(tmp_path, size):
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, (*size, 3), np.uint8)
    path = tmp_path / "im.png"
    cv2.imwrite(str(path), bgr)
    detector = SimpleNamespace(resize=1024, keep_aspect_ratio=True)
    by_file = DaD.load_image(detector, path, device="cpu")
    assert torch.equal(lomag.detector_input(lomag.picture(bgr), detector, "cpu"), by_file)
    by_file = DeDoDeDescriptor.read_image(None, path).cpu()
    assert torch.equal(lomag.descriptor_input(lomag.picture(bgr), "cpu"), by_file)


def test_lomag_is_never_loaded_beside_pycolmap_torch_left_unloaded():
    code = (
        "import sys; sys.modules['pycolmap'] = object()\n"
        "try:\n    import flaneur.lomag\nexcept ImportError:\n    print('torch' in sys.modules)"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert run.stdout.strip() == "False"


def test_weights_given_their_place_as_torch_hub_does_positionally_land_there(tmp_path, monkeypatch):
    monkeypatch.setitem(lomag.WEIGHTS, "tiny.pth", _saved(tmp_path, {"w": torch.ones(3)}))
    with lomag.loading_pinned(tmp_path):
        loaded = torch.hub.load_state_dict_from_url(URL, None, "meta")

    assert loaded["w"].device.type == "meta"


def test_a_download_that_does_not_match_its_hash_is_not_kept(tmp_path, monkeypatch):
    monkeypatch.setitem(lomag.WEIGHTS, "tiny.pth", "0" * 64)
    monkeypatch.setattr(
        torch.hub, "download_url_to_file", lambda url, dst, **_: Path(dst).write_bytes(b"garbled")
    )

    with pytest.raises(RuntimeError, match="SHA-256"):
        lomag.pinned(URL, tmp_path)

    assert not any(tmp_path.iterdir())
