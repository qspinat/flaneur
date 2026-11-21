"""The photos placed: in a panorama by hand, or by `match` and not refused; each with what the
user said of it (its status). The 3D walk shows them all, each saying how sure it is."""

from collections.abc import Callable, Sequence, Set
from dataclasses import asdict, dataclass
from typing import NamedTuple

from flaneur import bridging, match, placing, registration

CHECKED, NEAR, AUTOMATIC = "checked", "near", "automatic"  # said yes or placed by hand; near
MATCHED, CHAINED = "match", "chain"  # placed against today's panoramas; from old photos placed
TRUSTED_SHARE = 0.5  # of a placement's sightings by sure panoramas: under it, it says so
NOTES = ("unchecked_area", "proposed_marks", "unsure_sources")  # what one not seen yet may say


def status_of(verdict: str | None) -> str | None:
    """A placement of match's status from the user's word on it: yes, checked; near, near;
    none yet, automatic; no, none (left on the map)."""
    if verdict == "no":
        return None
    return {"yes": CHECKED, "near": NEAR, None: AUTOMATIC}[verdict]


@dataclass(frozen=True)
class Placed:
    """A photo placed: how sure (`status`), by whom (`source`: "page", MATCHED, CHAINED), its
    registration
    in a panorama as the placing page writes them (none: on the map only), the image it was
    placed on (`inputs`, its hash), shown the other way round (`flipped`); for one of match not
    seen yet, whether its matcher does not pass the check in its area yet (`unchecked_area`),
    whether it rests on landmarks only proposed (`proposed_marks`) and on panoramas nothing
    checks (`unsure_sources`: see TRUSTED_SHARE); the camera where the photographer stood, where
    the 3D walk lays it on the walls (`standing`, see registration.standing_to_json)."""

    photo_id: str
    status: str
    source: str
    registration: dict | None
    inputs: str
    flipped: bool = False
    unchecked_area: bool = False
    proposed_marks: bool = False
    unsure_sources: bool = False
    standing: dict | None = None

    def for_scene(self) -> dict:
        """As the 3D walk takes a photo placed (see scene.build3d)."""
        return {k: v for k, v in asdict(self).items() if k not in ("photo_id", "source")}


class Registered(NamedTuple):
    """A placement of match registered (see cli._registrations): the flat fit (none: no
    panorama fits), and the camera where the photographer stood where that lays it better."""

    registration: dict | None
    standing: dict | None = None


@dataclass(frozen=True)
class Matched:
    """A placement of match, or of the chaining (`source`), as the photo is now, as the list
    weighs it: as kept (`placement`), the user's word on it (`verdict`, none yet), the photo's
    image now (its hash), whether its method passes the check in its area (its matcher's, or the
    chaining's), its landmarks' marks ("checked", "proposed", none), whether it rests mostly on
    panoramas nothing checks (see unsure)."""

    placement: dict
    verdict: str | None
    image: str
    area_checked: bool
    marks: str | None = None
    unsure_sources: bool = False
    source: str = MATCHED


def build(
    answers: dict[str, dict],
    far: set[str],
    matched: dict[str, Matched],
    register: Callable[[dict[str, dict]], dict[str, Registered]],
) -> dict[str, Placed]:
    """The photos placed: by the placing page's `answers` (on the image each has now), checked;
    else by match, with the user's word, unless answered `far` (no panorama sees it) or
    refused; those of match registered (`register`: see cli._registrations) once chosen, and
    standing but on panoramas nothing checks (their poses unsure: the matrix holds better)."""
    found = {
        k: Placed(k, CHECKED, "page", a["registration"], a["inputs"], flipped=a["flipped"])
        for k, a in answers.items()
    }
    kept = {
        k: (m, status)
        for k, m in matched.items()
        if k not in found and k not in far and (status := status_of(m.verdict))
    }
    registrations = register({k: m.placement for k, (m, _) in kept.items()})
    for k, (m, status) in kept.items():
        seen = status != AUTOMATIC  # the notes are for those not seen yet
        unsure_sources = not seen and m.unsure_sources
        found[k] = Placed(
            k,
            status,
            m.source,
            registrations[k].registration,
            m.image,
            unchecked_area=not (seen or m.area_checked),
            proposed_marks=not seen and m.marks == "proposed",
            unsure_sources=unsure_sources,
            standing=None if unsure_sources else registrations[k].standing,
        )
    return found


def unsure(points: match.Points, sure: Set[str], panoramas: Set[str]) -> bool:
    """Whether under TRUSTED_SHARE of a placement's sightings (a point seen by one of
    `panoramas`: one; old photos left out) are by `sure` panoramas; not when none saw any."""
    seen = [i for s in points.seen for i in s if i in panoramas]
    return bool(seen) and sum(i in sure for i in seen) < TRUSTED_SHARE * len(seen)


def registered(
    points: match.Points, observed: match.Observed, spheres: Sequence[match.Panorama]
) -> registration.Registration | None:
    """A placement of `match` registered on its own: in the panorama of `spheres` where most of
    its pairs (see bridging.paired) agree, the camera turned through them (see
    registration.turned_through), the first of equals; none if no panorama fits."""

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

    return registration.most_agreeing(tries(), registration.turned_through)
