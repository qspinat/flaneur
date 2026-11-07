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


def _masks(left_of_north_only: bool = True) -> np.ndarray:
    """The 12 views' structure masks: only the left half of the view facing north is building."""
    masks = np.zeros((len(match.VIEW_HEADINGS), SIDE, SIDE), bool)
    masks[0, :, : SIDE // 2] = left_of_north_only
    return masks


def test_a_ray_on_a_building_in_its_view_is_on_structure():
    assert segment.on_structure(_masks(), _toward(-10.0, 5.0)).tolist() == [True]


def test_a_ray_beside_it_in_the_same_view_is_not():
    assert segment.on_structure(_masks(), _toward(10.0, 5.0)).tolist() == [False]


def test_a_ray_is_read_in_the_view_facing_it_most():
    rays = np.vstack([_toward(-10.0, 5.0), _toward(-100.0, 5.0)])  # west: another view's

    assert segment.on_structure(_masks(), rays).tolist() == [True, False]


def test_no_masks_keep_every_ray():
    assert segment.on_structure(None, _toward(10.0)).tolist() == [True]


def _kept(folder, labels: list[str]) -> segment.Segmenter:
    """A segmenter whose one panorama ("p") was segmented with `labels`: class 0 everywhere."""
    classes = np.zeros((len(match.VIEW_HEADINGS), SIDE, SIDE), np.uint8)
    np.savez_compressed(
        folder / "p.npz", classes=classes, labels=np.array(labels), model=segment.MODEL
    )
    return segment.Segmenter(folder, threading.Lock(), lambda p: p.id)


def _panorama(path) -> match.Panorama:
    return match.Panorama("p", 0.0, 0.0, 0.0, 0.0, path)


def test_kept_classes_give_the_structure_masks(tmp_path):
    masks = _kept(tmp_path, ["Building", "Sky"]).masks(_panorama(tmp_path / "p.jpg"))

    assert masks.all()


def test_a_model_without_the_structure_classes_is_refused(tmp_path):
    segmenter = _kept(tmp_path, ["Sky", "Road"])

    with pytest.raises(ValueError):
        segmenter.masks(_panorama(tmp_path / "p.jpg"))


def test_an_unread_picture_is_asked_again(tmp_path):
    segmenter = segment.Segmenter(tmp_path, threading.Lock(), lambda p: p.id)
    panorama = _panorama(tmp_path / "missing.jpg")
    assert segmenter.masks(panorama) is None

    _kept(tmp_path, ["Building"])

    assert segmenter.masks(panorama).all()


def test_a_segmenter_that_does_not_make_masks_says_so(tmp_path):
    picture = tmp_path / "p.jpg"
    cv2.imwrite(str(picture), np.zeros((32, 64, 3), np.uint8))  # readable: the model would run
    segmenter = segment.Segmenter(tmp_path, threading.Lock(), lambda p: p.id, make=False)

    with pytest.raises(segment.NotSegmented):
        segmenter.masks(_panorama(picture))


def test_classes_made_by_another_model_are_not_kept(tmp_path):
    classes = np.zeros((len(match.VIEW_HEADINGS), SIDE, SIDE), np.uint8)
    np.savez_compressed(tmp_path / "p.npz", classes=classes, labels=["Building"], model="old")
    segmenter = segment.Segmenter(tmp_path, threading.Lock(), lambda p: p.id, make=False)
    panorama = _panorama(tmp_path / "p.jpg")

    assert not segmenter.kept(panorama)
    with pytest.raises(segment.NotSegmented):
        segmenter.masks(panorama)
