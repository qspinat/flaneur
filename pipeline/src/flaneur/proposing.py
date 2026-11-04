"""Where an old photo lies in its nearest panorama, proposed for the user to check: the photo
matched (DISK and LightGlue, as match.py does) against the views of the levelled panorama that
match.Views keeps, those around the direction the photo looks, and registered (see
registration) in the view where most details agree, when they pass the checks. Old photos and
today's street match poorly (20 to 60 pairs is the norm): no proposal is better than a wrong
one, the user then clicks points."""

import cv2
import numpy as np

from flaneur import match, registration
from flaneur.registration import NotFitted, Registration, ViewOf

SIDE = match.VIEW_SIDE


def around(heading: float | None) -> list[int]:
    """Which of a panorama's views (match.VIEW_HEADINGS, every 30°) to try: the one nearest the
    photo's direction and either side of it (together at least ±52° around it), or all of them
    when the direction is unknown."""
    n = len(match.VIEW_HEADINGS)
    if heading is None:
        return list(range(n))
    step = 360 / n
    nearest = round(heading / step) % n
    return [(nearest + turn) % n for turn in (-1, 0, 1)]


def best(
    photo: match.Features,
    cut: list[tuple[ViewOf, match.Features]],
    panorama: str,
    axes: np.ndarray,
    matcher,
) -> Registration | None:
    """The registration in the view where most matched details agree, if any passes."""
    found, most = None, 0
    for view, seen in cut:
        pairs = matcher.pairs(photo, seen)
        try:
            r, kept = registration.fit(
                photo.keypoints[pairs[:, 0]],
                seen.keypoints[pairs[:, 1]],
                photo.size,
                SIDE,
                panorama,
                view,
                axes,
                robust=True,
            )
        except (NotFitted, cv2.error):  # OpenCV's estimators assert on some degenerate sets
            continue
        if kept.sum() > most:
            found, most = r, int(kept.sum())
    return found


def propose(
    photo: match.Features,
    panorama: str,
    axes: np.ndarray,
    cached: list[match.Features],
    heading: float | None,
    matcher,
) -> Registration | None:
    """The proposal for a photo (its features, as match.py reads old photos) in a panorama
    levelled with `axes`, among its views cached by match.Views (one per VIEW_HEADINGS; none
    when its picture could not be read)."""
    if len(cached) != len(match.VIEW_HEADINGS):
        return None
    tried = [(ViewOf(heading=float(match.VIEW_HEADINGS[n])), cached[n]) for n in around(heading)]
    return best(photo, tried, panorama, axes, matcher)
