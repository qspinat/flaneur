"""A photo the user placed in its panorama, carried further: to the neighbouring panoramas that
still see the same facades (today against today: details agree by the hundred), fading as
they lie farther from its home; and to the old photos sharing many details with it, as
proposals for the user to check (a carried photo is never more precise than its source)."""

from collections.abc import Callable

import cv2
import numpy as np
from shapely import STRtree
from shapely.geometry import Point

from flaneur import match, placings, queueing, registration
from flaneur.registration import Registration, ViewOf

MIN_AGREE_TODAY = 40  # details of today's views agreeing on one homography
FADE_M = (5.0, 20.0)  # full strength up to the first, none from the second
LEAST_STRENGTH = 0.1  # a copy fainter would hardly show: not looked for (beyond 18.5 m)
MIN_SHARED = queueing.MIN_SHARED  # an old photo is carried from another sharing as many


def strength(metres: float) -> float:
    """How strongly a carried copy shows, by its panorama's distance from the photo's home."""
    near, far = FADE_M
    return float(np.clip((far - metres) / (far - near), 0.0, 1.0))


def _agreeing(
    a_px: np.ndarray, a_size: tuple[int, int], b_px: np.ndarray, b_size: tuple[int, int]
) -> tuple[np.ndarray, int] | None:
    """The homography from points `a_px` to `b_px` (pixels of pictures of these sizes: height,
    width) most of them agree on, from [0, 1]² to [0, 1]², and how many do; none when OpenCV
    finds none."""
    if len(a_px) < 4:
        return None
    try:
        h, mask = cv2.findHomography(
            np.asarray(a_px, float),
            np.asarray(b_px, float),
            cv2.USAC_MAGSAC,
            registration.MATCHED_PX,
        )
    except cv2.error:  # its estimators assert on some degenerate sets
        return None
    if h is None or mask is None:
        return None
    unit = np.diag([1 / b_size[1], 1 / b_size[0], 1.0]) @ h @ np.diag([a_size[1], a_size[0], 1.0])
    return unit, int(mask.sum())


def carried_today(
    r: Registration,
    view: ViewOf,
    axes: np.ndarray,
    here_px: np.ndarray,
    there_px: np.ndarray,
    side: int,
    onto: str,
    onto_view: ViewOf,
    onto_axes: np.ndarray,
) -> Registration | None:
    """`r`, seen in `view` of its panorama (levelled with `axes`), carried onto panorama `onto`
    by the details under the photo matched in both views (`here_px`, `there_px`, pairs of
    pixels, views `side` square); none when too few agree or the photo leaves the view."""
    found = _agreeing(here_px, (side, side), there_px, (side, side))
    if found is None or found[1] < MIN_AGREE_TODAY:
        return None
    carried = registration.carried(r, view, axes, onto, onto_view, onto_axes, found[0])
    return (
        carried
        if registration.frame_ok(registration.in_view(carried, onto_view, onto_axes))
        else None
    )


def carried_old(
    placed: Registration,
    axes: np.ndarray,
    placed_px: np.ndarray,
    placed_size: tuple[int, int],
    other_px: np.ndarray,
    other_size: tuple[int, int],
) -> Registration | None:
    """Another old photo laid on the placed one's panorama (levelled with `axes`), through the
    details they share (`other_px` in it, `placed_px` in the placed one, pairs of pixels; sizes
    height, width): a proposal, none when they share too few or it leaves the view."""
    found = _agreeing(other_px, other_size, placed_px, placed_size)
    if found is None or found[1] < MIN_SHARED:
        return None
    proposed = registration.of(placed.panorama, placed.matrix @ found[0], placed.view, axes)
    if not registration.frame_ok(registration.in_view(proposed, placed.view, axes)):
        return None
    return proposed


def from_source(
    source: str,
    row: dict,
    others: set[str],
    pairs: Callable[[str, str], np.ndarray | None],
    features: Callable[[str], match.Features | None],
    axes_of: Callable[[str], np.ndarray | None],
    images: dict[str, str],
) -> dict[str, dict]:
    """The old photos (`others`) proposed from one placed photo (`row`: its answer), each with
    the details they share, and where the source stood, near which it is shown."""
    if row["answer"] not in placings.PLACED or row["inputs"] != images.get(source):
        return {}  # not placed, or placed on an image it no longer has
    placed = registration.from_json(row["registration"])
    axes, mine = axes_of(placed.panorama), features(source)
    if axes is None or mine is None:
        return {}
    keypoints = mine.keypoints.copy()
    if row["flipped"]:  # placed shown the right way round: its details as shown
        keypoints[:, 0] = mine.size[1] - keypoints[:, 0]
    found = {}
    for other in sorted(others):
        shared = pairs(source, other)
        theirs = shared is not None and len(shared) >= MIN_SHARED and features(other)
        if not theirs:
            continue
        proposed = carried_old(
            placed,
            axes,
            keypoints[shared[:, 0]],
            mine.size,
            theirs.keypoints[shared[:, 1]],
            theirs.size,
        )
        if proposed:
            found[other] = {
                "registration": proposed,
                "carried_from": source,
                "shared": len(shared),
                **{k: row[k] for k in ("lon", "lat", "heading")},
            }
    return found


def proposals(
    held: dict[str, dict],
    graph: dict[str, set[str]],
    pairs: Callable[[str, str], np.ndarray | None],
    features: Callable[[str], match.Features | None],
    axes_of: Callable[[str], np.ndarray | None],
    images: dict[str, str],
    each: Callable[..., dict[str, dict]] = from_source,
) -> dict[str, dict]:
    """The old photos not answered yet, proposed from the placed ones they share most details
    with (`held`: the answers that hold; `graph`: each photo's old neighbours; `pairs(a, b)`:
    the details they share, indices in a and in b; `images`: each photo's image's hash; `each`:
    from_source, or a kept copy of it). Made again from what holds now: a photo answered again
    no longer proposes what it did."""
    found: dict[str, dict] = {}
    for source, row in held.items():
        made = each(source, row, graph.get(source, set()), pairs, features, axes_of, images)
        for other, proposal in made.items():
            if other not in held and proposal["shared"] > found.get(other, {}).get("shared", 0):
                found[other] = proposal
    return found


def outward(
    home: str,
    links: Callable[[str], list[tuple[str, float]]],
    from_home: Callable[[str], float],
    carry: Callable[[str], Registration | None],
) -> list[tuple[str, Registration]]:
    """The panoramas a photo placed on `home` is carried to: tried outward from it, panorama
    after panorama next along the street (`links`), not past one that does not see it, nor
    where it would hardly show (`from_home`: metres); with their copies (`carry`)."""
    found, frontier, seen = [], [home], {home}
    while frontier:
        for there, _ in links(frontier.pop(0)):
            if there in seen:
                continue
            seen.add(there)
            if strength(from_home(there)) < LEAST_STRENGTH:
                continue
            copy = carry(there)
            if copy is not None:
                found.append((there, copy))
                frontier.append(there)
    return found


def to_compare(
    where: dict[str, tuple[float, float]],
    radius: float,
    copies: dict[str, list[str]],
    kept: Callable[[str, str], bool],
) -> list[tuple[str, str]]:
    """The pairs of old photos (`where`: their x, y in metres) standing within `radius` of each
    other, not prints of one negative (`copies`), not compared yet (`kept(a, b)`), each once."""
    ids = sorted(where)
    if not ids:  # the tree takes no empty query
        return []
    points = [Point(where[i]) for i in ids]
    a, b = STRtree(points).query(points, predicate="dwithin", distance=radius)
    return [
        (ids[i], ids[j])
        for i, j in sorted(zip(a.tolist(), b.tolist(), strict=True))
        if i < j and ids[j] not in copies.get(ids[i], ()) and not kept(ids[i], ids[j])
    ]
