"""The order photos come to the placing page in. First the photos asked ahead (an earlier
verdict to see again, a proposal carried from an old photo); then those that carry the most
old photos not placed yet, the old photos they share many details with (Blanc, Produit,
Ingensand 2018: place the central photos by hand, the rest follow), the nearest to a panorama
first among equals; every `every`-th answer instead the photo farthest from those placed, so
that the 3D walk also reaches the empty parts of the map; then the photos with no panorama
near; the skipped ones last."""

import numpy as np

from flaneur.placings import PLACED

MIN_SHARED = 100  # details two old photos share to be neighbours (copies share 300 and more)
ISOLATED_WITHIN_M = 15  # a photo far from the others is offered when a panorama is this near
NEAR_M = 40  # beyond, no panorama near (the placing page offers those within it)


def order(
    ids: list[str],
    *,
    where: dict[str, tuple[float, float]],
    held: dict[str, dict],
    skipped: set[str],
    ahead: dict[str, str],
    graph: dict[str, set[str]],
    nearest_m: dict[str, float],
    every: int = 4,
    answered: int | None = None,
) -> list[tuple[str, str]]:
    """The photos (`where`: their x, y in metres) not answered yet (`held`: the answers that
    hold, with `answer`, `x`, `y`), in order, each with why it comes there (`ahead`: photos
    to come first, with why, in order; `graph`: each photo's old neighbours; `answered`: the
    answers given so far, reopened photos' too, for the rhythm: those held when not given)."""
    todo = [i for i in ids if i not in held]
    first = [(i, why) for i, why in ahead.items() if i in todo and i not in skipped]
    taken = {i for i, _ in first}
    rest = [i for i in todo if i not in taken and i not in skipped]
    near = [i for i in rest if nearest_m[i] <= NEAR_M]

    def carries(i: str) -> int:
        return len(graph.get(i, set()) - held.keys())

    by_carrying = sorted(near, key=lambda i: (-carries(i), nearest_m[i]))
    placed = np.array([(r["x"], r["y"]) for r in held.values() if r["answer"] in PLACED])

    def apart(i: str) -> float:
        if not len(placed):
            return np.inf
        x, y = where[i]
        return float(np.min(np.hypot(placed[:, 0] - x, placed[:, 1] - y)))

    isolated = sorted(
        (i for i in near if nearest_m[i] <= ISOLATED_WITHIN_M),
        key=lambda i: (-apart(i), nearest_m[i]),
    )
    found = list(first)
    answered = len(held.keys() & set(ids)) if answered is None else answered
    main, far = iter(by_carrying), iter(isolated)
    for k in range(answered + 1, answered + 1 + len(near)):
        p = every and k % every == 0 and next((i for i in far if i not in taken), None)
        if p:
            why = "isolée"
        else:
            p = next(i for i in main if i not in taken)
            why = "voisins" if carries(p) else "proche"  # else only its panorama near
        taken.add(p)
        found.append((p, why))
    found += [(i, "sans panorama") for i in rest if nearest_m[i] > NEAR_M]
    found += [(i, "passée") for i in todo if i in skipped]
    return found
