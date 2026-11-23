"""Volunteers' panoramas turned to agree with their neighbours. Placed by COLMAP and the survey,
and levelled, they still disagree with the panoramas beside them by a degree or two, mostly in
tilt. Each walk link (panoramas.links) whose pair on disk sets right well says how far the
neighbour turns to agree with the panorama (measured); one small turn per panorama is averaged
from them all, the survey's held where they stand (averaged), and applied to the panorama as
levelled when it was computed (applied, by cli._levelled). The pairs are set right as the walls'
are (depth.set_right): a neighbour standing under 2 m away is left out, which E8a's refit kept
(its turn is the same otherwise)."""

import hashlib
import json
from collections.abc import Set
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from flaneur import depth, match, poses
from flaneur.gazetteer import angle
from flaneur.panoramax import Picture
from flaneur.sources import write_json

APPLIED: poses.Applied = "all"  # what cli._levelled takes of these: "all" or "none" (a turn is
# all there is: "turns" takes all). "all" chosen 23/11 (E8a, held out by 10-fold
# cross-validation over the links: each volunteer panorama's disagreement with its neighbours
# 1.91° → 0.76° in median, 733 better, 315 worse; the neighbour's picture brought through the
# walls matched better, +0.0086 NCC, 52 better, 19 worse)
MOST_TURNED_DEG = 10.0  # a pair set right by more (its rays or its baseline) says nothing
PULL = 0.01  # each free panorama's turn pulled to none this much (where no survey panorama
# reaches, they stay as they are on the whole)
SCALE_DEG = 0.5  # a link disagreeing by more than this with the others counts less (soft L1)
SAME_DEG = 0.01  # a correction holds while its panorama faces and leans this near as it did


class Agreed(NamedTuple):
    """A panorama's correction (rotation vector, degrees, as `measured`), and the azimuth,
    pitch and roll it was computed against."""

    rotation: tuple[float, float, float]
    against: tuple[float, float, float]


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


def averaged(links: list[tuple[str, str, np.ndarray]], held: Set[str]) -> dict[str, np.ndarray]:
    """One turn per panorama (a rotation vector, degrees, as `measured`) from the `links`
    (i, j, r_ij: j's turn with i held), c_j − c_i as near r_ij as they all allow: those `held`
    (the survey's) at none, every other pulled to none by PULL, a link far from the others
    counting less (soft L1 beyond SCALE_DEG); the turns of all but those held."""
    free = sorted({p for i, j, _ in links for p in (i, j)} - set(held))
    if not free:
        return {}
    column = {p: k for k, p in enumerate(free)}
    rows, columns, values = [], [], []
    for n, (i, j, _) in enumerate(links):
        for p, v in ((j, 1.0), (i, -1.0)):
            if p in column:
                rows.append(n)
                columns.append(column[p])
                values.append(v)
    incidence = sparse.csr_matrix((values, (rows, columns)), shape=(len(links), len(free)))
    pulled = PULL * sparse.identity(3 * len(free), format="csr")
    A = sparse.vstack([sparse.kron(incidence, sparse.identity(3), format="csr"), pulled]).tocsr()
    b = np.r_[np.ravel([r for _, _, r in links]), np.zeros(3 * len(free))]
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


def applied(pictures: list[Picture], kept: dict[str, Agreed]) -> list[Picture]:
    """The panoramas, each turned by its correction `kept` while it faces and leans as when
    the correction was computed (within SAME_DEG); one levelled otherwise since is left."""
    return [
        corrected(p, np.array(k.rotation)) if (k := kept.get(p.id)) and _same(p, k.against) else p
        for p in pictures
    ]


def _same(picture: Picture, against: tuple[float, float, float]) -> bool:
    azimuth, pitch, roll = against
    return (
        abs(angle(azimuth, picture.azimuth or 0.0)) <= SAME_DEG
        and abs(picture.pitch - pitch) <= SAME_DEG
        and abs(picture.roll - roll) <= SAME_DEG
    )


def setup() -> str:
    """How corrections are now computed: those computed otherwise are not read (see load)."""
    terms = (
        PULL,
        SCALE_DEG,
        MOST_TURNED_DEG,
        depth.LEAST_MATCHES,
        depth.WIDEST_DEG,
        depth.ROBUST_DEG,
    )
    return hashlib.sha1(repr(terms).encode()).hexdigest()[:8]


def save(
    path: Path, found: dict[str, np.ndarray], against: dict[str, tuple[float, float, float]]
) -> None:
    """The corrections (by panorama), each with the azimuth, pitch and roll it was computed
    against, and how they were computed (see setup)."""
    rows = {
        i: {"rotation": [float(v) for v in c], "from": [float(v) for v in against[i]]}
        for i, c in sorted(found.items())
    }
    write_json(path, {"setup": setup(), "panoramas": rows})


def load(path: Path) -> dict[str, Agreed]:
    """What `save` kept, if computed as corrections are now (see setup); none else."""
    if not path.exists():
        return {}
    kept = json.loads(path.read_text())
    if kept.get("setup") != setup():
        return {}
    return {i: Agreed(tuple(r["rotation"]), tuple(r["from"])) for i, r in kept["panoramas"].items()}
