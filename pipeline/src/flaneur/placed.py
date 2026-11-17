"""The photos placed: in a panorama by hand, or by `match` and not refused; each with what the
user said of it (its status). The 3D walk shows them all, each saying how sure it is."""

from collections.abc import Sequence

from flaneur import bridging, match, placing, registration


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
