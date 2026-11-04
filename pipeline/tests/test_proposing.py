"""Where an old photo lies in its nearest panorama, proposed from matched details."""

import cv2
import numpy as np

from flaneur import match, proposing, registration
from flaneur.registration import ViewOf

AXES = match.camera_axes(198.0, -8.4, 6.9)
H = np.array([[0.6, 0.05, 0.2], [-0.02, 0.55, 0.25], [0.0001, 0.0002, 1.0]])  # photo to view
SIDE = proposing.SIDE


class Matcher:
    """Pairs the photo's keypoints one for one with those of the view they were laid in."""

    def __init__(self, true: match.Features, agreeing: int):
        self.true, self.agreeing = true, agreeing

    def pairs(self, a: match.Features, b: match.Features) -> np.ndarray:
        if b is self.true:
            return np.column_stack([np.arange(self.agreeing)] * 2)
        rng = np.random.default_rng(len(b.keypoints))  # elsewhere: pairs that agree on nothing
        n = min(20, len(a.keypoints))
        return np.column_stack(
            [rng.permutation(len(a.keypoints))[:n], rng.permutation(len(b.keypoints))[:n]]
        )


def _features(points: np.ndarray, size: tuple[int, int]) -> match.Features:
    return match.Features(points.astype(np.float32), np.zeros((len(points), 128), np.float32), size)


def _scene(h: np.ndarray, agreeing: int = 60):
    rng = np.random.default_rng(3)
    photo_points = rng.uniform(0.03, 0.97, (agreeing, 2))
    photo = _features(photo_points * [800, 600], (600, 800))
    seen = cv2.perspectiveTransform(photo_points[None], h)[0] * SIDE
    true = _features(seen, (SIDE, SIDE))
    other = _features(rng.uniform(0, SIDE, (80, 2)), (SIDE, SIDE))
    views = [
        (ViewOf(heading=110.0), other),
        (ViewOf(heading=140.0), true),
        (ViewOf(heading=170.0), other),
    ]
    return photo, views, Matcher(true, agreeing)


def test_the_photo_is_proposed_in_the_view_its_details_were_found_in():
    photo, views, matcher = _scene(H)

    r = proposing.best(photo, views, "p1", AXES, matcher)

    assert r.view.heading == 140.0
    assert np.allclose(registration.in_view(r, ViewOf(heading=140.0), AXES), H, atol=1e-3)


def test_too_few_agreeing_details_make_no_proposal():
    photo, views, matcher = _scene(H, agreeing=10)

    assert proposing.best(photo, views, "p1", AXES, matcher) is None


def test_a_proposal_that_fails_the_shape_check_is_not_made():
    photo, views, matcher = _scene(np.diag([0.2, 0.2, 1.0]))  # 4% of the view

    assert proposing.best(photo, views, "p1", AXES, matcher) is None


def test_the_cached_views_nearest_the_photos_direction_are_tried_or_all_when_unknown():
    assert proposing.around(350.0) == [11, 0, 1]  # the views at 330°, 0° and 30°
    assert proposing.around(100.0) == [2, 3, 4]
    assert proposing.around(None) == list(range(len(match.VIEW_HEADINGS)))


def test_a_photo_is_proposed_among_its_panoramas_cached_views():
    photo, views, matcher = _scene(H)
    cached = [views[0][1]] * len(match.VIEW_HEADINGS)
    cached[4] = views[1][1]  # the true view is the one at 120°

    r = proposing.propose(photo, "p1", AXES, cached, 130.0, matcher)

    assert r.view == ViewOf(heading=120.0)


def test_with_no_direction_every_cached_view_is_tried():
    photo, views, matcher = _scene(H)
    cached = [views[0][1]] * len(match.VIEW_HEADINGS)
    cached[9] = views[1][1]  # the true view is the one at 270°

    assert proposing.propose(photo, "p1", AXES, cached, None, matcher).view.heading == 270.0


def test_a_panorama_whose_views_could_not_be_cut_gives_no_proposal():
    photo, _, matcher = _scene(H)

    assert proposing.propose(photo, "p1", AXES, [], 130.0, matcher) is None
