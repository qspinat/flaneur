"""What the views cut from a panorama show where (building, vegetation...), read on rays."""

import threading

import cv2
import numpy as np
import pytest

from flaneur import match, segment

SIDE = 64


def _toward(heading: float, up: float = 0.0) -> np.ndarray:
    """A world direction (east, north, up) toward a heading, raised `up` degrees."""
    h, u = np.radians(heading), np.radians(up)
    return np.array([[np.sin(h) * np.cos(u), np.cos(h) * np.cos(u), np.sin(u)]])


def _kinds() -> np.ndarray:
    """The 12 views' kinds: only the left half of the view facing north is building."""
    kinds = np.full((len(match.VIEW_HEADINGS), SIDE, SIDE), segment.OTHER, np.int8)
    kinds[0, :, : SIDE // 2] = segment.BUILT
    return kinds


def test_a_ray_on_a_building_in_its_view_sees_it():
    assert segment.seen(_kinds(), _toward(-10.0, 5.0)).tolist() == [segment.BUILT]


def test_a_ray_beside_it_in_the_same_view_does_not():
    assert segment.seen(_kinds(), _toward(10.0, 5.0)).tolist() == [segment.OTHER]


def test_a_ray_is_read_in_the_view_facing_it_most():
    rays = np.vstack([_toward(-10.0, 5.0), _toward(-100.0, 5.0)])  # west: another view's

    assert segment.seen(_kinds(), rays).tolist() == [segment.BUILT, segment.OTHER]


def test_without_kinds_every_ray_is_taken_for_a_building():
    assert segment.seen(None, _toward(10.0)).tolist() == [segment.BUILT]


def _kept(folder, labels: list[str]) -> segment.Segmenter:
    """A segmenter whose one panorama ("p") was segmented with `labels`: class 0 everywhere."""
    classes = np.zeros((len(match.VIEW_HEADINGS), SIDE, SIDE), np.uint8)
    np.savez_compressed(
        folder / "p.npz", classes=classes, labels=np.array(labels), model=segment.MODEL
    )
    return segment.Segmenter(folder, threading.Lock(), lambda p: p.id)


def _panorama(path) -> match.Panorama:
    return match.Panorama("p", 0.0, 0.0, 0.0, 0.0, path)


def test_kept_classes_give_what_pixels_show(tmp_path):
    kinds = _kept(tmp_path, ["Building", "Sky"]).kinds(_panorama(tmp_path / "p.jpg"))

    assert (kinds == segment.BUILT).all()


def test_a_model_without_the_structure_classes_is_refused(tmp_path):
    segmenter = _kept(tmp_path, ["Sky", "Road"])

    with pytest.raises(ValueError):
        segmenter.kinds(_panorama(tmp_path / "p.jpg"))


def test_an_unread_picture_is_asked_again(tmp_path):
    segmenter = segment.Segmenter(tmp_path, threading.Lock(), lambda p: p.id)
    panorama = _panorama(tmp_path / "missing.jpg")
    assert segmenter.kinds(panorama) is None

    _kept(tmp_path, ["Building"])

    assert (segmenter.kinds(panorama) == segment.BUILT).all()


def test_a_segmenter_that_does_not_segment_says_so(tmp_path):
    picture = tmp_path / "p.jpg"
    cv2.imwrite(str(picture), np.zeros((32, 64, 3), np.uint8))  # readable: the model would run
    segmenter = segment.Segmenter(tmp_path, threading.Lock(), lambda p: p.id, make=False)

    with pytest.raises(segment.NotSegmented):
        segmenter.kinds(_panorama(picture))


def test_classes_made_by_another_model_are_not_kept(tmp_path):
    classes = np.zeros((len(match.VIEW_HEADINGS), SIDE, SIDE), np.uint8)
    np.savez_compressed(tmp_path / "p.npz", classes=classes, labels=["Building"], model="old")
    segmenter = segment.Segmenter(tmp_path, threading.Lock(), lambda p: p.id, make=False)
    panorama = _panorama(tmp_path / "p.jpg")

    assert not segmenter.kept(panorama)
    with pytest.raises(segment.NotSegmented):
        segmenter.kinds(panorama)


def test_rays_see_building_sky_or_something_else(tmp_path):
    classes = np.full((len(match.VIEW_HEADINGS), SIDE, SIDE), 2, np.uint8)  # vegetation
    classes[0, : SIDE // 2, : SIDE // 2] = 1  # north view: sky top left
    classes[0, SIDE // 2 :, : SIDE // 2] = 0  # building bottom left
    np.savez_compressed(
        tmp_path / "p.npz",
        classes=classes,
        labels=["Building", "Sky", "Vegetation"],
        model=segment.MODEL,
    )
    segmenter = segment.Segmenter(tmp_path, threading.Lock(), lambda p: p.id, make=False)
    rays = np.vstack([_toward(-10, 20), _toward(-10, -10), _toward(10, 0), [[0.0, 0.0, 1.0]]])

    seen = segment.seen(segmenter.kinds(_panorama(tmp_path / "p.jpg")), rays)

    assert seen.tolist() == [segment.SKY, segment.BUILT, segment.OTHER, segment.UNSEEN]


class _Inputs(dict):
    def to(self, *_):
        return self


def test_the_model_is_given_a_plain_rgb_picture():
    torch = pytest.importorskip("torch")
    given = {}
    view = np.zeros((8, 8, 3), np.uint8)
    view[..., 0] = 255  # blue, as OpenCV reads pictures (BGR)

    class Processor:
        def __call__(self, images, **_):  # transformers' processors refuse reversed strides
            given["contiguous"], given["blue"] = images.flags.c_contiguous, images[0, 0, 2]
            return _Inputs()

        def post_process_semantic_segmentation(self, found, target_sizes):
            return [torch.zeros(target_sizes[0], dtype=torch.long)]

    segment.Segmenter._segmented(view, lambda **_: None, Processor(), "cpu", torch.float32)

    assert given == {"contiguous": True, "blue": 255}  # RGB: the blue channel last
