"""The photos placed: in a panorama by hand, or by `match` and not refused; each with what the
user said of it (its status). The 3D walk shows them all, each saying how sure it is."""

from collections.abc import Sequence
from dataclasses import dataclass

from flaneur import bridging, match, placing, registration

CHECKED, NEAR, AUTOMATIC = "checked", "near", "automatic"  # said yes or placed by hand; near
STATUS_OF = {"yes": CHECKED, "near": NEAR, None: AUTOMATIC}  # a "no" leaves the list


@dataclass(frozen=True)
class Placed:
    """A photo placed: how sure (`status`), by whom (`source`: "page", "match"), its registration
    in a panorama as the placing page writes them (none: on the map only), the image it was
    placed on (`inputs`, its hash), shown the other way round (`flipped`); for one of match,
    whether its matcher passes the check in its area (`area_checked`, none: not known yet) and
    whether its landmarks are checked or only proposed (`marks`)."""

    photo_id: str
    status: str
    source: str
    registration: dict | None
    inputs: str
    flipped: bool
    area_checked: bool | None = None
    marks: str | None = None

    def for_scene(self) -> dict:
        """As the 3D walk takes a photo placed (see scene.build3d)."""
        return {
            "registration": self.registration,
            "inputs": self.inputs,
            "flipped": self.flipped,
            "status": self.status,
            "unchecked_area": self.area_checked is False,
            "proposed_marks": self.marks == "proposed",
        }


def build(
    answers: dict[str, dict],
    far: set[str],
    standing: dict[str, dict],
    verdicts: dict[str, str | None],
    registrations: dict[str, dict | None],
    images: dict[str, str],
    checked_areas: dict[str, list[str]],
    area_of: dict[str, str],
    marks: dict[str, str],
) -> dict[str, Placed]:
    """The photos placed: by the placing page's `answers` on the image each has now (`images`),
    checked; else by match (`standing`: its placements made from what the photo is now, or said
    yes to), with the user's word (`verdicts`), unless answered `far` (no panorama sees it) or
    refused. `checked_areas`: each area's matchers that pass the check (empty: not known)."""
    found = {
        k: Placed(
            k, CHECKED, "page", a["registration"], a["inputs"], a["flipped"], None, marks.get(k)
        )
        for k, a in answers.items()
        if a["inputs"] == images.get(k)
    }
    for k, m in standing.items():
        if k in found or k in far or (status := STATUS_OF.get(verdicts.get(k))) is None:
            continue
        passes = (m.get("matcher") or "disk") in checked_areas.get(area_of[k], [])
        found[k] = Placed(
            k,
            status,
            "match",
            registrations.get(k),
            images[k],
            False,
            passes if checked_areas else None,
            marks.get(k),
        )
    return found


def registered(
    points: match.Points, observed: match.Observed, spheres: Sequence[match.Panorama]
) -> registration.Registration | None:
    """A placement of `match` registered on its own, as the placing page registers matched
    pairs: in the panorama of `spheres` where most of its pairs (see bridging.paired) agree on
    one flat warp, the first of equals; none if no panorama fits."""

    def tries():
        for sphere in spheres:
            if (pairs := bridging.paired(points, observed, sphere, observed.size)) is not None:
                yield (
                    pairs.photo_px,
                    pairs.view_px,
                    observed.size,
                    placing.SHOWN,
                    sphere.id,
                    pairs.view,
                    sphere.axes,
                )

    return registration.most_agreeing(tries())
