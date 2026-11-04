"""The automatic crop, on built images: a textured print on a smooth card."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flaneur import crop
from flaneur.crop import Crop, detect, examine


def print_(width: int, height: int, seed: int = 0) -> np.ndarray:
    """A textured, darker 'photograph'."""
    rng = np.random.default_rng(seed)
    return rng.integers(30, 160, (height, width, 3)).astype(np.uint8)


def mounted(card=(205, 200, 190), size=(400, 500), box=(80, 60, 320, 380)) -> Image.Image:
    image = np.full((size[1], size[0], 3), card, dtype=np.uint8)
    left, top, right, bottom = box
    image[top:bottom, left:right] = print_(right - left, bottom - top)
    return Image.fromarray(image)


def test_the_print_is_found_on_its_card():
    crop = detect(mounted())

    assert overlap(crop, (80 / 400, 60 / 500, 320 / 400, 380 / 500)) > 0.95


def test_a_photo_without_a_mount_is_not_cropped():
    assert detect(Image.fromarray(print_(400, 300))) is None


def test_a_stereoscopic_pair_keeps_the_left_view():
    image = np.full((300, 600, 3), (40, 40, 40), dtype=np.uint8)  # a dark stereo card
    image[40:260, 30:280] = print_(250, 220, seed=1) + 80
    image[40:260, 320:570] = print_(250, 220, seed=2) + 80

    crop = detect(Image.fromarray(image))

    assert overlap(crop, (30 / 600, 40 / 300, 280 / 600, 260 / 300)) > 0.9


def test_layers_are_peeled_down_to_the_print():
    # white scanner background, then a blue-grey card, then the print
    image = np.full((500, 400, 3), (250, 250, 250), dtype=np.uint8)
    image[20:480, 30:370] = (150, 160, 175)
    image[100:350, 90:310] = print_(220, 250)

    crop = detect(Image.fromarray(image))

    assert overlap(crop, (90 / 400, 100 / 500, 310 / 400, 350 / 500)) > 0.9


def overlap(a: Crop, b: Crop) -> float:
    """Intersection over union of two crops (1: the same frame)."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area else 0.0


def test_a_handwritten_caption_under_the_print_is_left_out():
    image = np.asarray(mounted(size=(400, 500), box=(80, 60, 320, 380))).copy()
    rng = np.random.default_rng(3)
    for x in range(70, 330, 9):  # a line of handwriting across the card under the print
        if rng.random() < 0.7:
            image[395:412, x : x + 4] = (60, 50, 40)

    crop = detect(Image.fromarray(image))

    assert crop[3] < 390 / 500
    assert overlap(crop, (80 / 400, 60 / 500, 320 / 400, 380 / 500)) > 0.9


def test_a_thin_scanner_rim_does_not_stop_the_peeling():
    # a rim of a few pixels of scanner background, then the card, then the print
    image = np.full((500, 400, 3), (120, 130, 150), dtype=np.uint8)
    image[4:496, 4:396] = (205, 200, 190)
    image[100:350, 90:310] = print_(220, 250)

    crop = detect(Image.fromarray(image))

    assert overlap(crop, (90 / 400, 100 / 500, 310 / 400, 350 / 500)) > 0.9


def test_a_card_cut_by_the_scan_on_two_sides_is_still_peeled():
    # scanner background left and right only: the card reaches the top and the bottom
    image = np.full((500, 400, 3), (205, 200, 190), dtype=np.uint8)
    image[:, :30] = image[:, 370:] = (250, 250, 250)
    image[100:350, 90:310] = print_(220, 250)

    crop = detect(Image.fromarray(image))

    assert overlap(crop, (90 / 400, 100 / 500, 310 / 400, 350 / 500)) > 0.9


def test_a_rim_on_two_sides_only_does_not_stop_the_peeling():
    image = np.full((500, 400, 3), (205, 200, 190), dtype=np.uint8)
    image[:, 392:] = image[492:, :] = (40, 40, 45)  # the scanner's dark edge
    image[60:200, 40:160] = print_(120, 140)  # a small print on an album board

    crop = detect(Image.fromarray(image))

    assert overlap(crop, (40 / 400, 60 / 500, 160 / 400, 200 / 500)) > 0.9


def test_a_stereoscopic_plate_without_a_mount_keeps_the_left_view():
    # two views of the same street, framed in black on a white plate, reaching its edges
    view = print_(280, 300, seed=4)
    image = np.full((300, 620, 3), 250, dtype=np.uint8)
    image[:, 0:280] = view
    image[:, 280:290] = 10  # the plate's black edge
    image[:, 320:600] = np.roll(view, -6, axis=1)  # seen from 6 cm to the right
    image[:, 600:610] = 10

    crop = detect(Image.fromarray(image))

    assert overlap(crop, (0, 0, 280 / 620, 1)) > 0.9


def test_a_wide_street_is_not_taken_for_a_stereoscopic_pair():
    assert detect(Image.fromarray(print_(620, 300, seed=5))) is None


SCANS = Path(__file__).parent / "fixtures" / "scans"  # Paris Musées scans (CC0), at 600 px


def scan(number: int) -> Image.Image:
    with Image.open(SCANS / f"{number}.jpg") as image:
        return image.copy()


@pytest.mark.parametrize(
    ("number", "print_frame"),
    [
        (765372, (0.30, 0.23, 0.71, 0.69)),  # its sky as pale as the card: no edge there
        (108561, (0.16, 0.18, 0.85, 0.84)),  # a scanner rim and handwriting around the card
        (770571, (0.16, 0.19, 0.84, 0.77)),  # two cards, a printed label above and below
    ],
)
def test_the_print_is_found_on_real_cards(number, print_frame):
    assert overlap(detect(scan(number)), print_frame) > 0.8


def test_a_real_stereoscopic_pair_with_rounded_views_keeps_the_left_view():
    assert overlap(detect(scan(91949)), (0.0, 0.0, 0.5, 1.0)) > 0.9


@pytest.mark.parametrize(
    "number",
    [
        112658,  # a strong upright edge down its middle: one photo, not a pair
        921318,  # a pale sky and a pale street, on no mount
        108438,  # a thin dark rim and a pale sky
    ],
)
def test_a_real_photo_without_a_mount_keeps_nearly_all_of_it(number):
    frame = detect(scan(number)) or (0.0, 0.0, 1.0, 1.0)
    assert (frame[2] - frame[0]) * (frame[3] - frame[1]) > 0.85


def test_a_pale_sky_unlike_the_card_stays_with_the_print_found_from_within(monkeypatch):
    monkeypatch.setattr(crop, "_peeled", lambda image: None)  # as on a border it cannot read
    image = np.full((500, 400, 3), (205, 200, 190), dtype=np.uint8)  # a cream card
    image[100:180, 90:310] = (175, 190, 215)  # a pale blue sky, smooth
    image[180:350, 90:310] = print_(220, 170)

    found = detect(Image.fromarray(image))

    assert overlap(found, (90 / 400, 100 / 500, 310 / 400, 350 / 500)) > 0.9


def test_two_alike_prints_one_above_the_other_are_not_a_stereoscopic_pair(monkeypatch):
    monkeypatch.setattr(crop, "_peeled", lambda image: None)  # the pair is looked for alone
    view = print_(200, 160, seed=6)
    image = np.full((560, 600, 3), (205, 200, 190), dtype=np.uint8)
    for top, left in ((30, 40), (370, 330)):  # the same picture, lower and to the right
        image[top - 3 : top + 163, left - 3 : left + 203] = 20  # framed by a dark edge
        image[top : top + 160, left : left + 200] = view

    found = detect(Image.fromarray(image))

    assert found is None or found[3] > 0.6  # not the upper left print alone


def test_a_good_crop_raises_no_doubt():
    assert examine(mounted())[1] == []


def test_an_album_page_of_several_prints_is_doubted():
    image = np.full((500, 600, 3), (205, 200, 190), dtype=np.uint8)
    for n, (left, top) in enumerate(((60, 60), (380, 60), (60, 310), (380, 310))):
        image[top : top + 110, left : left + 150] = print_(150, 110, seed=10 + n)
    assert examine(Image.fromarray(image))[1] != []


def test_a_stereoscopic_pair_cut_to_its_left_view_raises_no_doubt():
    assert examine(scan(91949))[1] == []


def test_a_stereoscopic_pair_peeled_to_its_left_view_raises_no_doubt():
    assert examine(scan(91790))[1] == []  # its views are not framed by straight lines
