"""The photos placed: in a panorama by hand, or by `match` and not refused; each with what the
user said of it (its status). The 3D walk shows them all, each saying how sure it is."""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from flaneur import bridging, match, placing, registration

CHECKED, NEAR, AUTOMATIC = "checked", "near", "automatic"  # said yes or placed by hand; near


def status_of(verdict: str | None) -> str | None:
    """A placement of match's status from the user's word on it: yes, checked; near, near;
    none yet, automatic; no, none (left on the map)."""
    if verdict == "no":
        return None
    return {"yes": CHECKED, "near": NEAR, None: AUTOMATIC}[verdict]


@dataclass(frozen=True)
class Placed:
    """A photo placed: how sure (`status`), by whom (`source`: "page", "match"), its registration
    in a panorama as the placing page writes them (none: on the map only), the image it was
    placed on (`inputs`, its hash), shown the other way round (`flipped`); for one of match not
    seen yet, whether its matcher does not pass the check in its area yet (`unchecked_area`) and
    whether it rests on landmarks only proposed (`proposed_marks`)."""

    photo_id: str
    status: str
    source: str
    registration: dict | None
    inputs: str
    flipped: bool = False
    unchecked_area: bool = False
    proposed_marks: bool = False

    def for_scene(self) -> dict:
        """As the 3D walk takes a photo placed (see scene.build3d)."""
        return {k: v for k, v in asdict(self).items() if k not in ("photo_id", "source")}


@dataclass(frozen=True)
class Matched:
    """A placement of match as the photo is now, as the list weighs it: as kept (`placement`),
    the user's word on it (`verdict`, none yet), the photo's image now (its hash), whether its
    matcher passes the check in its area, its landmarks' marks ("checked", "proposed", none)."""

    placement: dict
    verdict: str | None
    image: str
    area_checked: bool
    marks: str | None = None


def build(
    answers: dict[str, dict],
    far: set[str],
    matched: dict[str, Matched],
    register: Callable[[dict[str, dict]], dict[str, dict | None]],
) -> dict[str, Placed]:
    """The photos placed: by the placing page's `answers` (on the image each has now), checked;
    else by match, with the user's word, unless answered `far` (no panorama sees it) or
    refused; those of match registered (`register`: see cli._registrations) once chosen."""
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
        found[k] = Placed(
            k,
            status,
            "match",
            registrations[k],
            m.image,
            unchecked_area=not (seen or m.area_checked),
            proposed_marks=not seen and m.marks == "proposed",
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
