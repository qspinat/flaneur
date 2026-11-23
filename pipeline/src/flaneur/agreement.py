"""Volunteers' panoramas turned to agree with their neighbours. Placed by COLMAP and the survey,
and levelled, they still disagree with the panoramas beside them by a degree or two, mostly in
tilt. Each walk link (panoramas.links) whose pair on disk sets right well says how far the
neighbour turns to agree with the panorama (measured); one small turn per panorama is averaged
from them all, the survey's held where they stand (averaged), and applied to the panoramas (applied,
by cli._levelled) while every panorama they were solved with stands as it did (holds). The pairs
are set right as the walls' are (depth.set_right): a neighbour standing under 2 m away is left
out, which E8a's refit kept (measured: 4 of its 5,361 links; their turns are the same
otherwise)."""

import functools
import hashlib
import inspect
from collections.abc import Callable, Set
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from flaneur import depth, match, panoramas, placing, poses, register
from flaneur.panoramax import Picture
from flaneur.sources import write_json

APPLIED: poses.Applied = "all"  # what cli._levelled takes of these: "all" or "none" (a turn is
# all there is: no "turns"). "all" chosen 23/11 (E8a, held out by 10-fold cross-validation over
# the links: each volunteer panorama's disagreement with its neighbours 1.91° → 0.76° in
# median, 733 better, 315 worse; the neighbour's picture brought through the walls matched
# better, +0.0086 NCC, 52 better, 19 worse)
if APPLIED not in ("all", "none"):  # a slip refused before anything reads the panoramas
    raise ValueError(f"agreement.APPLIED: all or none, not {APPLIED!r}")
MOST_TURNED_DEG = 10.0  # a pair set right by more (its rays or its baseline) says nothing
PULL = 0.01  # each free panorama's turn pulled to none this much (where no survey panorama
# reaches, they stay as they are on the whole)
SCALE_DEG = 0.5  # a link disagreeing by more than this with the others counts less (soft L1)


class Agreement(NamedTuple):
    """The corrections solved together: how they were computed (see setup), each free
    panorama's turn (a rotation vector, degrees, as `measured`), and how every panorama of the
    graph stood then, the survey's among them (see stance)."""

    setup: str
    turns: dict[str, tuple[float, float, float]]
    stood: dict[str, tuple[float, ...]]


def stance(picture: Picture) -> tuple[float, ...]:
    """How a panorama stands and leans (see poses.poised): its pairs, and so the corrections
    solved with it, hold while this does."""
    x, y = picture.point.x, picture.point.y
    return poses.poised(x, y, picture.azimuth or 0.0, picture.roll, picture.pitch)


def links(
    pictures: list[Picture],
    spheres: list[match.Panorama],
    made: Callable[[match.Panorama, match.Panorama], bool],
    rays: Callable[[match.Panorama, match.Panorama], tuple[np.ndarray, np.ndarray]],
) -> list[tuple[str, str, np.ndarray]]:
    """Each walk link between the panoramas (panoramas.links, as recorded) once, whose pair is
    `made` already (on disk: none made here) and says how far the second turns against the
    first (`rays`: the pair's, read; see measured): (i, j, r_ij), for `averaged`."""
    by_id = {p.id: p for p in spheres}
    found, seen = [], set()
    for home, near in panoramas.links(pictures).items():
        for other, _, _ in near:
            both = frozenset((home, other))
            if home not in by_id or other not in by_id or both in seen:
                continue
            seen.add(both)
            h, o = by_id[home], by_id[other]
            if not made(h, o):
                continue
            turn = measured(placing._at(h), placing._at(o), *rays(h, o))
            if turn is not None:
                found.append((home, other, turn))
    return found


def measured(
    a: np.ndarray, b: np.ndarray, rays_a: np.ndarray, rays_b: np.ndarray
) -> np.ndarray | None:
    """How far the neighbour at `b` turns to agree with the panorama at `a`, from the rays of
    the details both see (rows, made unit): the turn depth.set_right sets it right by (a
    rotation vector, degrees; in the world, its rays · R.T); None when the pair is not set
    right (too near among others), or set right by more than MOST_TURNED_DEG."""
    rays_a, rays_b = (r / np.linalg.norm(r, axis=1, keepdims=True) for r in (rays_a, rays_b))
    pair = depth.set_right(a, b, rays_a, rays_b)
    if pair is None:
        return None
    turn = Rotation.from_matrix(pair.turn)
    cos = (b - a) @ (pair.at - a) / np.sum((b - a) ** 2)  # the baseline's length kept
    turned = np.degrees(max(turn.magnitude(), np.arccos(np.clip(cos, -1.0, 1.0))))
    return None if turned > MOST_TURNED_DEG else np.degrees(turn.as_rotvec())


def averaged(edges: list[tuple[str, str, np.ndarray]], held: Set[str]) -> dict[str, np.ndarray]:
    """One turn per panorama (a rotation vector, degrees, as `measured`) from the links
    measured (`edges`: i, j, r_ij, j's turn with i held; see links), c_j − c_i as near r_ij as
    they all allow: those `held` (the survey's) at none, every other pulled to none by PULL, a
    link far from the others counting less (soft L1 beyond SCALE_DEG); the turns of all but
    those held."""
    free = sorted({p for i, j, _ in edges for p in (i, j)} - set(held))
    if not free:
        return {}
    column = {p: k for k, p in enumerate(free)}
    rows, columns, values = [], [], []
    for n, (i, j, _) in enumerate(edges):
        for p, v in ((j, 1.0), (i, -1.0)):
            if p in column:
                rows.append(n)
                columns.append(column[p])
                values.append(v)
    incidence = sparse.csr_matrix((values, (rows, columns)), shape=(len(edges), len(free)))
    pulled = PULL * sparse.identity(3 * len(free), format="csr")
    A = sparse.vstack([sparse.kron(incidence, sparse.identity(3), format="csr"), pulled]).tocsr()
    b = np.r_[np.ravel([r for _, _, r in edges]), np.zeros(3 * len(free))]
    x = least_squares(
        lambda x: A @ x - b,
        np.zeros(A.shape[1]),
        jac=lambda x: A,
        loss="soft_l1",
        f_scale=SCALE_DEG,
        tr_solver="lsmr",
    ).x
    return dict(zip(free, x.reshape(-1, 3), strict=True))


def corrected(picture: Picture, rotvec_deg: np.ndarray) -> Picture:
    """A panorama turned by its correction (a rotation vector, degrees: its rays · R.T), as it
    then faces and leans."""
    axes = match.camera_axes(picture.azimuth or 0.0, picture.pitch, picture.roll)
    turn = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    azimuth, pitch, roll = camera_angles(axes @ turn.T)
    return replace(picture, azimuth=azimuth, pitch=pitch, roll=roll)


def camera_angles(axes: np.ndarray) -> tuple[float, float, float]:
    """The azimuth (in [0, 360)), pitch and roll a camera's axes are made from (see
    match.camera_axes, which this undoes): ahead gives the first two, its right the roll."""
    ahead = axes[1]
    azimuth = float(np.degrees(np.arctan2(ahead[0], ahead[1])) % 360)
    pitch = float(np.degrees(np.arcsin(np.clip(ahead[2], -1.0, 1.0))))
    right, down, _ = match._view_axes(azimuth, pitch)
    roll = float(np.degrees(np.arctan2(axes[0] @ down, axes[0] @ right)))
    return azimuth, pitch, roll


def applied(pictures: list[Picture], kept: Agreement | None, taken: poses.Applied) -> list[Picture]:
    """The panoramas, each turned by its correction `kept` (see holds: whether they still
    stand), `taken` "all"; "none": as they are."""
    if taken not in ("all", "none"):
        raise ValueError(f"applied: all or none, not {taken!r}")
    if taken == "none" or kept is None:
        return list(pictures)
    return [corrected(p, np.array(kept.turns[p.id])) if p.id in kept.turns else p for p in pictures]


def holds(kept: Agreement, pictures: list[Picture]) -> bool:
    """Whether corrections `kept` still hold: computed as they are now (see setup), and every
    panorama of the graph they were solved with standing as it stood (see stance; a survey fix,
    recaler or another tilt makes it other: its pairs, and its neighbours' turns, are other).
    A panorama no longer known, or one come since, is not judged, nor are the pairs on disk: a
    panorama come since stands unturned, the others keep their turns, until `flaneur accorder`
    runs again (after `flaneur murs` has made its pairs)."""
    now = {p.id: stance(p) for p in pictures if p.id in kept.stood}
    return kept.setup == setup() and all(
        poses.stands(kept.stood[i], stood) for i, stood in now.items()
    )


@functools.cache  # the code does not change while a command runs
def setup() -> str:
    """How corrections are now computed: the code they come from (how a pair is set right, a
    link measured, the walk's links gathered and averaged), its terms, and those the walk's
    links are drawn by (panoramas.links); corrections computed otherwise no longer hold (see
    holds). Which pairs are on disk is not: `flaneur accorder` reads them again when run."""
    code = [inspect.getsource(f) for f in (depth.set_right, measured, links, averaged)]
    terms = (PULL, SCALE_DEG, MOST_TURNED_DEG, depth.LEAST_MATCHES, depth.WIDEST_DEG)
    terms += (depth.ROBUST_DEG, depth.NEAR_M, panoramas.LINK_M, panoramas.SECTOR_DEG)
    return hashlib.sha1(repr((code, terms)).encode()).hexdigest()[:8]


def save(path: Path, kept: Agreement) -> None:
    """The corrections solved together (see Agreement)."""
    turns, stood = (dict(sorted(d.items())) for d in (kept.turns, kept.stood))
    write_json(path, kept._replace(turns=turns, stood=stood)._asdict())


def load(path: Path) -> Agreement | None:
    """What `save` kept; none if nothing is. ValueError for what is not corrections kept so
    (unreadable, or kept otherwise before)."""
    kept = register.load(path)
    if not kept:
        return None
    try:
        turns, stood = ({i: tuple(v) for i, v in kept[f].items()} for f in ("turns", "stood"))
        return Agreement(str(kept["setup"]), turns, stood)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"{path}: not corrections kept as now ({error!r})") from error
