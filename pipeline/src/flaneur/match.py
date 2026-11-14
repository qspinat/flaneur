"""Find an old photo's camera from today's panoramas around it.

The photo is matched against views cut from the panoramas near its placement (DISK keypoints,
an ONNX export run by onnxruntime: LightGlue-ONNX, Apache-2.0; LightGlue matching, by kornia
on PyTorch). A point matched in a panorama's view is cast onto today's buildings from where
that panorama was taken (lined up by flaneur.register): a point in space. From those points
and where they are in the old photo, its camera is solved: position, heading, width of view
and horizon. Several panoramas seeing the same facades from different places make it hold;
where the buildings are gone, little matches and the photo keeps its placement."""

import functools
import heapq
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from operator import itemgetter
from pathlib import Path
from typing import NamedTuple

import cv2
import httpx
import numpy as np
import onnxruntime as ort

from flaneur.blocks import MOST_PAIRS
from flaneur.gazetteer import angle
from flaneur.sources import write_atomic

MODELS = {
    "disk.onnx": "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/disk.onnx",
}
KEYPOINTS = 2048  # the strongest kept, in each picture
PHOTO_SIDE = 800  # the old photo is matched at this size (longest side)
EPIPOLAR_PX = 1.5  # matches farther than this from agreeing with the others are dropped
# Views cut from a panorama: rectilinear, a little upwards (facades rather than the street).
VIEW_SIDE = 640
VIEW_FOV = 75
VIEW_PITCH = 5
VIEW_HEADINGS = range(0, 360, 30)
MIN_MATCHES = 15  # a view with fewer matches than this is not used
# A first pass with each picture's strongest points only ranks the views; the best are then
# matched in full (LightGlue's time grows with the square of the points).
QUICK_KEYPOINTS = 512
QUICK_MIN = 8
BEST_VIEWS = 9
# Solving the camera: a level camera (view cameras were levelled; the horizon moves with the
# lens) over a grid of widths of view and horizons.
FOVS = range(30, 95, 5)
HORIZONS = np.linspace(0.25, 0.75, 11)
REPROJECTION_PX = 6.0  # of the photo at PHOTO_SIDE
MIN_AGREE = 20  # points agreeing with the camera found
MIN_SOURCES = 2  # seen from at least this many pictures: panoramas, or old photos placed
VIA = "via:"  # a point an old photo passed on, seen through it: not a sight of its own
MOST_MOVED_M = 50  # from where its caption puts it (rarely 25 m off): further is a wrong camera
# its position known to this (one standard deviation): on the check, every camera found
# right was known to 0.31 m or better, every wrong one to 0.54 m or worse
MOST_SPREAD_M = 0.4
STREET_HEIGHT_M = (0.5, 4.0)  # the camera stands on the street, on a tripod or a cart
HORIZON_IN = (0.1, 0.9)  # its horizon, inside the photo
PAIRS = 1000  # pairs of points cameras are guessed from, for each width of view and horizon
APART_DEG = 1.0  # two points nearer in bearing than this place the camera badly
OFF_HORIZON = 0.01  # nor a point this near the horizon (up per metre ahead): its distance
TOP = 5  # the best guesses refined on all the points agreeing with them
# guesses nearer than this to a better one, seeing as wide, are not refined
SAME_M, SAME_DEG, SAME_FOV = 2.0, 5.0, 5.0
# a rival camera: this far from the best, and agreed with by points the best leaves out (this
# share of the best's), or nearly as many in all (on the check, a share from 0.5 to 0.8 and a
# tie from 0.95 to 0.97 left out every wrong camera found ambiguous, and no right one)
RIVAL_M, RIVAL_DEG = 3.0, 5.0
RIVAL_OWN, RIVAL_TIE = 0.6, 0.96
MIN_REFINED = 10  # a guess with fewer points agreeing is not refined (6 unknowns)


def fetch_models(client: httpx.Client, folder: Path) -> Path:
    """The models, downloaded once into `folder`."""
    for name, url in MODELS.items():
        path = folder / name
        if not path.exists():
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()
            folder.mkdir(parents=True, exist_ok=True)
            path.with_suffix(".tmp").write_bytes(response.content)
            path.with_suffix(".tmp").replace(path)
    return folder


@dataclass(frozen=True)
class Features:
    keypoints: np.ndarray  # (n, 2) pixels, float32
    descriptors: np.ndarray  # (n, 128) float32
    size: tuple[int, int]  # height, width

    def top(self, n: int) -> "Features":
        """The `n` strongest (they are kept strongest first)."""
        return Features(self.keypoints[:n], self.descriptors[:n], self.size)


class Matcher:
    """DISK finds a picture's keypoints (ONNX, on the CPU), LightGlue pairs those of two
    pictures (PyTorch, on the graphics chip when there is one: measured 3.7 times faster than
    on the CPU in full precision, the same pairs; it runs there in half precision, 1.2 times
    faster again, keeping 99% of the pairs and the landmarks found on the checked photos).
    Torch is imported here only: its OpenMP and pycolmap's abort a process that loads both
    (see reconstruct.reconstruct_tile)."""

    def __init__(self, folder: Path):
        if "pycolmap" in sys.modules:  # it would abort the process, without a word
            raise RuntimeError("pycolmap est chargé : torch ne peut pas l'être avec lui")
        import torch
        from kornia.feature import LightGlue

        torch.hub.set_dir(str(folder / "torch"))  # LightGlue's weights, with the other models
        cpu = ["CPUExecutionProvider"]
        self.disk = ort.InferenceSession(str(folder / "disk.onnx"), providers=cpu)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # all layers, all points: pruning them is faster but keeps only 41% of the pairs
        self.precision = "half" if self.device.type == "mps" else "full"  # CPU: not measured
        lightglue = LightGlue(
            "disk", depth_confidence=-1, width_confidence=-1, mp=self.precision == "half"
        )
        self.lightglue = lightglue.to(self.device).eval()

    def features(self, bgr: np.ndarray) -> Features:
        h, w = bgr.shape[:2]
        padded = np.zeros((-(-h // 16) * 16, -(-w // 16) * 16, 3), np.uint8)  # multiples of 16
        padded[:h, :w] = bgr
        image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None] / 255
        keypoints, scores, descriptors = self.disk.run(None, {"image": image.astype(np.float32)})
        inside = (keypoints[0][:, 0] < w) & (keypoints[0][:, 1] < h)
        best = np.argsort(-np.where(inside, scores[0], -np.inf))[: min(KEYPOINTS, inside.sum())]
        return Features(
            keypoints[0][best].astype(np.float32), descriptors[0][best].astype(np.float32), (h, w)
        )

    def pairs(self, a: Features, b: Features) -> np.ndarray:
        """LightGlue's matches between `a` and `b`: one row (index in a, index in b) each."""
        return self.scored_pairs(a, b)[0]

    def scored_pairs(self, a: Features, b: Features) -> tuple[np.ndarray, np.ndarray]:
        """LightGlue's matches between `a` and `b` (as `pairs`), and how sure it is of each
        (0 to 1)."""
        import torch

        if not len(a.keypoints) or not len(b.keypoints):
            return np.empty((0, 2), np.int64), np.empty(0, np.float32)

        def side(f: Features) -> dict:
            return {
                "keypoints": torch.from_numpy(f.keypoints)[None].to(self.device),
                "descriptors": torch.from_numpy(f.descriptors)[None].to(self.device),
                "image_size": torch.tensor([f.size[::-1]], dtype=torch.float32, device=self.device),
            }

        with torch.inference_mode():
            found = self.lightglue({"image0": side(a), "image1": side(b)})
        to = found["matches0"][0].cpu().numpy()
        paired = np.nonzero(to > -1)[0]
        scores = found["matching_scores0"][0].cpu().numpy()
        return np.column_stack([paired, to[paired]]), scores[paired]

    def match(self, a: Features, b: Features) -> tuple[np.ndarray, np.ndarray]:
        """The matched keypoints of `a` and of `b` (their indices) that agree on one two-view
        geometry."""
        pairs = self.pairs(a, b)
        keep = agreeing(a.keypoints[pairs[:, 0]], b.keypoints[pairs[:, 1]])
        return pairs[keep, 0], pairs[keep, 1]


def agreeing(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Which pairs of points (pixels of one picture, `pa`, and of another, `pb`) agree on one
    two-view geometry (none, from fewer than 8)."""
    none = np.zeros(len(pa), bool)
    if len(pa) < 8:
        return none
    try:
        _, mask = cv2.findFundamentalMat(pa, pb, cv2.USAC_MAGSAC, EPIPOLAR_PX, 0.999, 10000)
    except cv2.error:  # its estimator asserts on some degenerate sets: no geometry
        return none
    return none if mask is None else mask.ravel().astype(bool)


def focal(fov: float, side: float) -> float:
    return side / 2 / np.tan(np.radians(fov) / 2)


def _view_axes(heading: float, pitch: float) -> np.ndarray:
    """A view camera's right, down and forward axes in the world (east, north, up)."""
    r, p = np.radians(heading), np.radians(pitch)
    forward = np.array([np.sin(r) * np.cos(p), np.cos(r) * np.cos(p), np.sin(p)])
    right = np.array([np.cos(r), -np.sin(r), 0.0])
    return np.stack([right, np.cross(forward, right), forward])


def camera_axes(azimuth: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """A camera's right, ahead and up in the world (rows): facing `azimuth`, its nose raised
    `pitch`, then leaning `roll` to its right (degrees)."""
    right, down, ahead = _view_axes(azimuth, pitch)
    r = np.radians(roll)
    return np.stack(
        [np.cos(r) * right + np.sin(r) * down, ahead, np.sin(r) * right - np.cos(r) * down]
    )


def rays(
    pixels: np.ndarray,
    size: tuple[int, int],
    fov: float,
    horizon: float,
    heading: float,
    pitch: float = 0.0,
) -> np.ndarray:
    """The world directions (east, north, up; unit) of pixels (column, row) of a picture of
    `size` (height, width), taken by a camera seeing `fov` degrees across, its horizon at that
    fraction of the height from the top."""
    h, w = size
    local = np.column_stack(
        [
            pixels[:, 0] - w / 2 + 0.5,
            pixels[:, 1] - horizon * h + 0.5,
            np.full(len(pixels), focal(fov, w)),
        ]
    )
    world = local @ _view_axes(heading, pitch)
    return world / np.linalg.norm(world, axis=1, keepdims=True)


def view_directions(
    pixels: np.ndarray,
    heading: float,
    fov: float = VIEW_FOV,
    side: int = VIEW_SIDE,
    pitch: float = VIEW_PITCH,
) -> np.ndarray:
    """The world directions of pixels of a view cut from a panorama (see perspective)."""
    return rays(pixels, (side, side), fov, 0.5, heading, pitch)


def _in_panorama(
    directions: np.ndarray, axes: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Where world directions fall in a 360° picture of `size` taken by a camera of these
    `axes` (see camera_axes): the columns and rows to sample (for cv2.remap)."""
    h, w = size
    own = directions @ axes.T
    bearing = np.degrees(np.arctan2(own[:, 0], own[:, 1]))  # from its picture's centre
    elevation = np.degrees(np.arcsin(np.clip(own[:, 2], -1, 1)))
    return ((0.5 + bearing / 360) % 1) * w - 0.5, (0.5 - elevation / 180) * h - 0.5


def pixels_of(size: tuple[int, int]) -> np.ndarray:
    """Every pixel of a picture of `size` (height, width), as (column, row), row by row."""
    rows, columns = np.mgrid[0 : size[0], 0 : size[1]]
    return np.column_stack([columns.ravel(), rows.ravel()]).astype(float)


def _cut(
    picture: np.ndarray, axes: np.ndarray, directions: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """A picture of `size` whose pixels show these world directions (one per pixel, row by
    row) as seen in a 360° picture taken by a camera of these `axes`."""
    u, v = _in_panorama(directions, axes, picture.shape[:2])
    return _sample(picture, u, v).reshape(*size, picture.shape[2])


def perspective(
    equirectangular: np.ndarray,
    axes: np.ndarray,
    heading: float,
    fov: float = VIEW_FOV,
    side: int = VIEW_SIDE,
    pitch: float = VIEW_PITCH,
) -> np.ndarray:
    """The rectilinear view facing `heading` (raised `pitch` from the world's level: its rays
    are view_directions), cut from a 360° picture taken by a camera of these `axes`."""
    d = view_directions(pixels_of((side, side)), heading, fov, side, pitch)
    return _cut(equirectangular, axes, d, (side, side))


@dataclass(frozen=True)
class View:
    """Where a camera stood and how it saw (metres; degrees)."""

    x: float
    y: float
    z: float
    heading: float
    horizon: float  # row of the horizon, a fraction of the height from the top
    fov: float  # horizontal, degrees


@dataclass(frozen=True)
class Camera(View):
    agree: int = 0  # points agreeing with it
    panoramas: int = 0  # how many pictures those points were seen from (see MIN_SOURCES)
    inliers: np.ndarray = field(default_factory=lambda: np.empty(0, int), repr=False)
    # how far off its position and heading could be (metres, degrees; one standard deviation)
    spread_m: float = 0.0
    spread_deg: float = 0.0


@dataclass(frozen=True)
class Situated:
    """An old photo's camera found against today's panoramas, as kept (matches.json): where it
    stood (metres Lambert-93; height above the street), how it saw, how sure, the facades its
    points lie on, and what it was found from (`inputs`: stale when they change)."""

    x: float
    y: float
    above_street: float
    heading: float
    horizon: float
    fov: float
    agree: int
    panoramas: int
    moved_m: float  # from where its caption (or the review) put it
    walls: list[dict]  # {"line": [[x0, y0], [x1, y1]], "height"}
    inputs: str
    outline: float  # how much better the buildings then standing fit it than nearby cameras

    @classmethod
    def of(
        cls,
        camera: Camera,
        street: float,
        walls: np.ndarray,
        moved_m: float,
        inputs: str,
        outline: float,
    ) -> "Situated":
        return cls(
            round(camera.x, 2),
            round(camera.y, 2),
            round(camera.z - street, 2),
            round(camera.heading, 2),
            round(camera.horizon, 3),
            camera.fov,
            camera.agree,
            camera.panoramas,
            round(moved_m, 1),
            [
                {
                    "line": np.round(w[:4], 2).reshape(2, 2).tolist(),
                    "height": round(float(w[5] - w[4]), 2),
                }
                for w in walls
            ],
            inputs,
            round(outline, 3),
        )

    def view(self, street: float) -> View:
        return View(
            self.x, self.y, street + self.above_street, self.heading, self.horizon, self.fov
        )


def meet(origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, float]:
    """The point nearest to rays (from `origins` along unit `directions`), and how far the
    rays pass from it (mean, metres)."""
    across = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    point = np.linalg.lstsq(
        across.sum(axis=0), (across @ origins[:, :, None]).sum(axis=0), rcond=None
    )[0].ravel()
    off = point - origins
    miss = np.linalg.norm(off - (off * directions).sum(axis=1, keepdims=True) * directions, axis=1)
    return point, float(miss.mean())


def project(view: View, world: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Where a level camera sees world points (east, north, up), in pixels of a picture of
    `size`; NaN behind it."""
    h, w = size
    # pixel centres as in `rays`: column c covers c to c + 1, so its centre is c + 0.5
    centre = (w / 2 - 0.5, view.horizon * h - 0.5)
    camera = np.array([[view.x, view.y, view.z, view.heading]])
    return _projected(camera, focal(view.fov, w), centre, world)[0]


def _projected(
    cameras: np.ndarray, f: float, centre: tuple[float, float], world: np.ndarray
) -> np.ndarray:
    """Where level cameras (rows of x, y, z, heading) of one `f` (pixels) and horizon `centre`
    see world points: cameras × points × (column, row), NaN behind them."""
    r = np.radians(cameras[:, 3])[:, None]
    d = world[None] - cameras[:, None, :3]
    across = d[..., 0] * np.cos(r) - d[..., 1] * np.sin(r)
    ahead = d[..., 0] * np.sin(r) + d[..., 1] * np.cos(r)
    ahead = np.where(ahead > 0.1, ahead, np.nan)
    return np.stack([centre[0] + f * across / ahead, centre[1] - f * d[..., 2] / ahead], axis=-1)


@dataclass(frozen=True)
class _Level:
    """A level camera fitted to world points seen at pixels of a photo of `size`: its
    parameters are x, y, z, heading, focal length and horizon row (pixels)."""

    world: np.ndarray
    pixels: np.ndarray
    size: tuple[int, int]

    SMALL = np.array([0.01, 0.01, 0.01, 0.01, 0.5, 0.5])  # steps for the derivatives

    def pack(self, view: View) -> np.ndarray:
        h, w = self.size
        return np.array(
            [view.x, view.y, view.z, view.heading, focal(view.fov, w), view.horizon * h]
        )

    def unpack(self, p: np.ndarray) -> View:
        h, w = self.size
        fov = float(np.degrees(2 * np.arctan(w / 2 / max(p[4], 1.0))))
        return View(*p[:4], horizon=float(p[5] / h), fov=fov)

    def residuals(self, p: np.ndarray) -> np.ndarray:
        r = (project(self.unpack(p), self.world, self.size) - self.pixels).ravel()
        return np.nan_to_num(r, nan=1e3)

    def jacobian(self, p: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                (self.residuals(p + np.eye(6)[k] * self.SMALL[k]) - r) / self.SMALL[k]
                for k in range(6)
            ]
        )


def refine_level(
    view: View, world: np.ndarray, pixels: np.ndarray, size: tuple[int, int], steps: int = 30
) -> View:
    """The level camera (position, heading, width of view, horizon) that sees `world` at
    `pixels` best, from a first guess (Gauss-Newton, residuals beyond 3 px weighing less)."""
    fit = _Level(world, pixels, size)
    p = fit.pack(view)
    settled = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-2, 1e-2])  # moves this small: done
    damping = 1e-3
    r = fit.residuals(p)
    weight = np.sqrt(np.minimum(1, 3 / np.maximum(np.abs(r), 1e-9)))  # Huber at 3 px
    for _ in range(steps):
        a, b = fit.jacobian(p, r) * weight[:, None], r * weight
        normal = a.T @ a
        # Levenberg-Marquardt: a step that does not help is tried again, shorter
        while damping < 1e8:
            move = np.linalg.lstsq(
                normal + damping * np.diag(np.diag(normal) + 1e-9), -a.T @ b, rcond=None
            )[0]
            trial = fit.residuals(p + move)
            if (
                np.isfinite(move).all()
                and (np.abs(trial) * weight).sum() <= (np.abs(r) * weight).sum()
            ):
                p, r, damping = p + move, trial, damping / 3
                weight = np.sqrt(np.minimum(1, 3 / np.maximum(np.abs(r), 1e-9)))
                break
            damping *= 10
        else:
            break
        if (np.abs(move) < settled).all():
            break
    return fit.unpack(p)


def spread(
    view: View, world: np.ndarray, pixels: np.ndarray, size: tuple[int, int]
) -> tuple[float, float]:
    """How far off the position (metres, the way it is least sure of) and the heading
    (degrees) of a level camera fitted to these points could be, one standard deviation:
    from how the points' pixels move with it, and how far they are from where it puts them.
    Points on a single far facade leave it unsure how far back it stood (a wider view nearer
    sees them the same); points on one side of the photo, which way it looked."""
    fit = _Level(world, pixels, size)
    p = fit.pack(view)
    r = fit.residuals(p)
    jac = fit.jacobian(p, r)
    variance = (r @ r) / max(len(r) - len(p), 1)
    covariance = variance * np.linalg.pinv(jac.T @ jac)
    metres = np.sqrt(np.linalg.eigvalsh(covariance[:2, :2]).max())
    return float(metres), float(np.sqrt(covariance[3, 3]))


def upright_pairs(
    world: np.ndarray, pixels: np.ndarray, f: float, centre: tuple[float, float]
) -> np.ndarray:
    """The level cameras that see two world points (east, north, up) at their pixels, for
    each pair (`world`: n × 2 × 3, `pixels`: n × 2 × 2), with a known width of view (`f`,
    pixels) and horizon (the pixel `centre`): n × 2 rows of x, y, z, heading; NaN for none.

    The upright two-point problem (Kukelova, Bujňák and Pajdla, ACCV 2010): each pixel gives
    the bearing of its point from the camera, against the heading, and its slope, which makes
    its distance linear in the camera's height. The two distances and the angle between the
    bearings close a triangle with the two points: a quadratic in the camera's height."""
    p = (pixels[..., 0] - centre[0]) / f  # across, per metre ahead
    q = (centre[1] - pixels[..., 1]) / f  # up, per metre ahead
    bearing = np.arctan(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.sqrt(1 + p**2) / q  # metres away, per metre the point stands above the camera
        a1, a2 = a[:, 0], a[:, 1]
        z1, z2 = world[:, 0, 2], world[:, 1, 2]
        span = world[:, 1, :2] - world[:, 0, :2]
        length = np.linalg.norm(span, axis=1)
        angle = bearing[:, 1] - bearing[:, 0]
        c = np.cos(angle)
        # |span|² = d1² + d2² - 2 d1 d2 cos(angle), with d = a (z - height)
        qa = a1**2 + a2**2 - 2 * a1 * a2 * c
        qb = 2 * a1 * a2 * c * (z1 + z2) - 2 * (a1**2 * z1 + a2**2 * z2)
        qc = (a1 * z1) ** 2 + (a2 * z2) ** 2 - 2 * a1 * a2 * c * z1 * z2 - length**2
        root = np.sqrt(qb**2 - 4 * qa * qc)[:, None]
        height = (-qb[:, None] + np.array([-1.0, 1.0]) * root) / (2 * qa[:, None])
        d1, d2 = a1[:, None] * (z1[:, None] - height), a2[:, None] * (z2[:, None] - height)
        along = (d1**2 - d2**2 + length[:, None] ** 2) / (2 * length[:, None])
        unit = span / length[:, None]
    off = np.sqrt(np.maximum(d1**2 - along**2, 0))
    left = np.column_stack([-unit[:, 1], unit[:, 0]])
    side = -np.sign(angle)[:, None]  # the second point clockwise of the first: the right side
    xy = (
        world[:, None, 0, :2]
        + along[..., None] * unit[:, None]
        + (side * off)[..., None] * left[:, None]
    )
    to_first = world[:, None, 0, :2] - xy
    heading = np.degrees(np.arctan2(to_first[..., 0], to_first[..., 1]) - bearing[:, :1]) % 360
    well_placed = (np.abs(angle) > np.radians(APART_DEG)) & (np.abs(q) > OFF_HORIZON).all(axis=1)
    ok = (d1 > 0) & (d2 > 0) & well_placed[:, None]
    found = np.concatenate([xy, height[..., None], heading[..., None]], axis=-1)
    return np.where(ok[..., None], found, np.nan)


def _pairs(n: int) -> np.ndarray:
    """Pairs of n points to guess cameras from: every one, or PAIRS drawn at random."""
    if n * (n - 1) // 2 <= PAIRS:
        return np.column_stack(np.triu_indices(n, 1))
    rng = np.random.default_rng(0)
    first = rng.integers(n, size=PAIRS)
    return np.column_stack([first, (first + rng.integers(1, n, size=PAIRS)) % n])


def _agreeing(
    cameras: np.ndarray,
    f: float,
    centre: tuple[float, float],
    world: np.ndarray,
    pixels: np.ndarray,
) -> np.ndarray:
    """How many points each camera sees where they are in the photo (cameras × points, a
    batch at a time)."""
    if not len(cameras):
        return np.zeros(0, int)
    batch = max(1, MOST_PAIRS // len(world))
    counts = []
    for k in range(0, len(cameras), batch):
        off = _projected(cameras[k : k + batch], f, centre, world) - pixels
        counts.append(((off**2).sum(axis=-1) < REPROJECTION_PX**2).sum(axis=1))
    return np.concatenate(counts)


@dataclass(frozen=True)
class Candidate:
    """A camera checked for a photo, and why it was turned down ("" when it was not)."""

    x: float
    y: float
    heading: float
    fov: float
    horizon: float
    agree: int
    spread_m: float | None  # None when not measured
    why: str

    @classmethod
    def of(cls, view: View, agree: int, metres: float | None, why: str) -> "Candidate":
        return cls(view.x, view.y, view.heading % 360, view.fov, view.horizon, agree, metres, why)


@dataclass
class Trace:
    """How far a photo got, to measure where photos are lost: the views of today ranked
    well enough to be matched in full, those matched, the details matched and placed in
    space, the cameras checked, and the verdict."""

    views: int = 0
    matched: int = 0
    matches: int = 0
    modelled: int = 0  # matched rays the model gave a depth to (see _cast): none, no check
    points: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    verdict: str = ""


# A landmark, marked by hand to a few degrees, tells which way the camera looked, never where
# it stood to the decimetres its points tell (a camera 0.8 m further back sees a landmark 40 m
# away 0.4° off): it turns cameras down, it never makes up for a check.
SIGHT_DEG = 2.0  # a landmark is marked in a photo to this, either side
LANDMARK_M = 5.0  # and its point stands this near the part of it marked (a tower's middle)


def turn_of(column: float, fov: float) -> float:
    """How far right of a level camera's axis (degrees) it sees what stands at `column` (a
    fraction of the width) of its view `fov` wide."""
    return float(np.degrees(np.arctan((column - 0.5) / focal(fov, 1.0))))


def slack(away: float | np.ndarray, spread: float) -> float | np.ndarray:
    """How far (degrees) from its mark a camera may see a landmark `away` from it: as near as
    it was marked, and as far off its point as what was marked can be seen (the nearer, the
    larger; `spread`: metres from its point to each of its tops)."""
    near = LANDMARK_M + spread
    return SIGHT_DEG + np.degrees(np.arcsin(near / np.maximum(away, near)))


@dataclass(frozen=True)
class Sight:
    """A landmark marked in a photo: where it stands (`points`: its alternatives, east and
    north, when the photo cannot tell which it is), its column (a fraction of the width), the
    top of each (`tops`, metres NGF; none: not zoomed on) and the spread of each (`spreads`,
    see slack; none given: no spread)."""

    points: tuple[tuple[float, float], ...]
    column: float
    tops: tuple[float, ...] = ()
    spreads: tuple[float, ...] = ()

    def __post_init__(self):
        if not self.spreads:
            object.__setattr__(self, "spreads", (0.0,) * len(self.points))

    def seen_from(self, cameras: np.ndarray, fov: float) -> np.ndarray:
        """Which level cameras (rows of x, y, z, heading) of a width of view `fov` see one of
        the points at that column, as near as it was marked and as the point stands (an angle
        the larger the nearer)."""
        target = turn_of(self.column, fov)
        seen = np.zeros(len(cameras), bool)
        for (x, y), spread in zip(self.points, self.spreads, strict=True):
            east, north = x - cameras[:, 0], y - cameras[:, 1]
            turn = angle(cameras[:, 3], np.degrees(np.arctan2(east, north)))  # right of axis
            seen |= np.abs(angle(target, turn)) <= slack(np.hypot(east, north), spread)
        return seen

    def seen_by(self, view: View) -> bool:
        return bool(self.seen_from(np.array([[view.x, view.y, view.z, view.heading]]), view.fov)[0])


@dataclass(frozen=True)
class Where:
    """Where an old photo's camera could stand: on the street (`streets`: its height at many
    points), at a tripod's height, outside the buildings then standing (`free`: whether
    points are), within MOST_MOVED_M of where its caption puts it (`around`), and seeing the
    landmarks marked in the photo (`sights`)."""

    streets: Callable[[np.ndarray, np.ndarray], np.ndarray]
    free: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    around: tuple[float, float] | None = None
    sights: tuple[Sight, ...] = ()

    def could(self, cameras: np.ndarray) -> np.ndarray:
        """Which cameras (rows of x, y, z, …) could stand there."""
        x, y = cameras[:, 0], cameras[:, 1]
        above = cameras[:, 2] - self.streets(x, y)
        ok = (above >= STREET_HEIGHT_M[0]) & (above <= STREET_HEIGHT_M[1])
        if self.around is not None:
            ok &= np.hypot(x - self.around[0], y - self.around[1]) <= MOST_MOVED_M
        if self.free is not None and ok.any():
            ok[ok] = self.free(x[ok], y[ok])
        return ok

    def sees(self, cameras: np.ndarray, fov: float) -> np.ndarray:
        """Which cameras (rows of x, y, z, heading) of a width of view `fov` see the landmarks
        marked in the photo where they are (all of them, when none is)."""
        ok = np.ones(len(cameras), bool)
        for sight in self.sights:
            ok &= sight.seen_from(cameras, fov)
        return ok


def solve(
    world: np.ndarray,
    pixels: np.ndarray,
    sources: np.ndarray,
    size: tuple[int, int],
    where: Where,
    trace: Trace | None = None,
) -> Camera | None:
    """The level camera that sees the `world` points (east, north, up) at `pixels` of a
    photo of `size` (height, width), or None when no camera that could stand `where` agrees
    with enough of them, from enough panoramas (`sources`: the panoramas that saw each, as
    frozensets), or when two cameras far apart agree with them nearly as well.

    For each width of view and horizon, every pair of points (or PAIRS of them) gives the
    cameras that see both; each is scored by all the points; the best few are refined on all
    the points that agree with them (as LO-RANSAC does) and checked."""
    h, w = size
    trace = Trace() if trace is None else trace
    trace.points = len(world)
    if len(world) < MIN_AGREE:
        trace.verdict = "trop peu de points"
        return None
    pairs = _pairs(len(world))
    guesses = []  # (points agreeing, (camera row, horizon, fov))
    for fov in FOVS:
        f = focal(fov, w)
        for horizon in HORIZONS:
            centre = (w / 2 - 0.5, horizon * h - 0.5)
            cameras = upright_pairs(world[pairs], pixels[pairs], f, centre).reshape(-1, 4)
            cameras = cameras[np.isfinite(cameras).all(axis=1)]
            cameras = cameras[where.could(cameras)]
            cameras = cameras[where.sees(cameras, fov)]
            agreeing = _agreeing(cameras, f, centre, world, pixels)
            for k in np.argsort(agreeing)[-TOP:]:
                guesses.append((int(agreeing[k]), (cameras[k], horizon, fov)))
    guesses.sort(key=itemgetter(0), reverse=True)
    if not guesses:
        trace.verdict = "aucune hypothèse"  # no two points give a camera that could stand there
        return None
    found = [
        camera
        for row, horizon, fov in _apart([guess for _, guess in guesses])
        if (
            camera := _checked(View(*row, horizon, fov), world, pixels, sources, size, where, trace)
        )
    ]
    if not found:
        trace.verdict = "aucune caméra"
        return None
    best = max(found, key=lambda c: c.agree)
    if any(_rivals(other, best) for other in found):
        trace.verdict = "rivale"
        return None
    trace.verdict = "située"
    return best


def _rivals(other: Camera, best: Camera) -> bool:
    """Whether `other`, elsewhere or looking elsewhere, makes the photo ambiguous: points the
    best camera leaves out agree with it, nearly as many as agree with the best (two
    explanations of the photo), or it ties with the best (the points do not tell them
    apart). A camera nearly as good on the same points, a little further back and seeing a
    little wider, is the best one less well placed: no rival. No place beats a wrong one."""
    if (
        np.hypot(other.x - best.x, other.y - best.y) <= RIVAL_M
        and abs(angle(other.heading, best.heading)) <= RIVAL_DEG
    ):
        return False
    own = len(np.setdiff1d(other.inliers, best.inliers))
    return own >= RIVAL_OWN * best.agree or other.agree >= RIVAL_TIE * best.agree


def _apart(guesses: list) -> list:
    """The first TOP guesses (each (camera row, horizon, fov)), leaving out any too close to
    one already kept."""
    kept = []
    for guess in guesses:
        (x, y, _, heading), _, fov = guess
        if all(
            np.hypot(x - other[0][0], y - other[0][1]) > SAME_M
            or abs(angle(heading, other[0][3])) > SAME_DEG
            or abs(fov - other[2]) > SAME_FOV
            for other in kept
        ):
            kept.append(guess)
            if len(kept) == TOP:
                break
    return kept


def _checked(
    rough: View,
    world: np.ndarray,
    pixels: np.ndarray,
    sources: np.ndarray,
    size: tuple[int, int],
    where: Where,
    trace: Trace,
) -> Camera | None:
    """A guessed camera refined on the points that agree with it (twice: more agree once it
    is nearer), or None if it fails a check (`trace` keeps which)."""

    def note(why: str, view: View, agree: int, metres: float | None = None) -> None:
        trace.candidates.append(Candidate.of(view, agree, metres, why))

    view = rough
    for _ in range(2):
        off = np.linalg.norm(project(view, world, size) - pixels, axis=1)
        inliers = np.flatnonzero(off < REPROJECTION_PX)
        if len(inliers) < MIN_REFINED:
            note("trop peu de points pour affiner", view, len(inliers))
            return None
        view = refine_level(view, world[inliers], pixels[inliers], size)
    off = np.linalg.norm(project(view, world, size) - pixels, axis=1)
    inliers = np.flatnonzero(off < REPROJECTION_PX)
    agree = len(inliers)
    seen_from = sum(1 for s in frozenset().union(*sources[inliers]) if not s.startswith(VIA))
    checks = [  # the first failed says why it is turned down
        (agree < MIN_AGREE, "trop peu de points d'accord"),
        (not all(s.seen_by(view) for s in where.sights), "ne voit pas le repère"),  # refined
        (seen_from < MIN_SOURCES, "vue d'une seule source"),
        (not where.could(np.array([[view.x, view.y, view.z]]))[0], "ne peut se tenir là"),
        (not FOVS[0] <= view.fov <= FOVS[-1], "champ hors limites"),
        (not HORIZON_IN[0] <= view.horizon <= HORIZON_IN[1], "horizon hors limites"),
    ]
    why = next((why for failed, why in checks if failed), "")
    if why:
        note(why, view, agree)
        return None
    metres, degrees = spread(view, world[inliers], pixels[inliers], size)
    if metres > MOST_SPREAD_M:
        note("position incertaine", view, agree, metres)
        return None
    note("", view, agree, metres)
    return Camera(
        view.x,
        view.y,
        view.z,
        view.heading % 360,
        view.horizon,
        round(view.fov, 1),
        agree,
        seen_from,
        inliers,
        metres,
        degrees,
    )


@dataclass(frozen=True)
class Panorama:
    id: str
    x: float
    y: float
    z: float
    azimuth: float  # where its picture's centre faces
    path: Path  # its 360° picture
    roll: float = 0.0  # degrees its camera leaned to its right (see level)
    pitch: float = 0.0  # degrees it raised its nose

    @property
    def axes(self) -> np.ndarray:
        return camera_axes(self.azimuth, self.pitch, self.roll)


class Views:
    """The features of the views cut from panoramas (one per VIEW_HEADINGS): extracted once
    for a panorama as lined up and levelled (its azimuth, and roll and pitch if it leans, are
    in the name), kept on disk in `folder`, and in memory while photos near it are matched."""

    def __init__(self, matcher: Matcher, folder: Path, kept: int = 64):
        self.matcher = matcher
        self.folder = folder
        self.of = functools.lru_cache(maxsize=kept)(self._of)

    def _of(self, panorama: Panorama) -> list[Features]:
        name = f"{panorama.id}_{panorama.azimuth:.1f}"
        if panorama.roll or panorama.pitch:
            name += f"_{panorama.roll:.2f}_{panorama.pitch:.2f}"
        path = self.folder / f"{name}.npz"
        if path.exists():
            with np.load(path) as kept:
                size = tuple(kept["size"])
                return [
                    Features(kept[f"k{n}"], kept[f"d{n}"].astype(np.float32), size)
                    for n in range(len(VIEW_HEADINGS))
                ]
        equirectangular = cv2.imread(str(panorama.path))
        if equirectangular is None:
            return []
        found = [
            self.matcher.features(perspective(equirectangular, panorama.axes, heading))
            for heading in VIEW_HEADINGS
        ]
        write_atomic(
            path,
            lambda out: np.savez(
                out,
                size=np.array(found[0].size),
                **{f"k{n}": f.keypoints for n, f in enumerate(found)},
                **{f"d{n}": f.descriptors.astype(np.float16) for n, f in enumerate(found)},
            ),
        )
        return found


MEET_M = 0.3  # rays of one detail from several pictures: they pass this close, or no point
MEET_DEG = 2.0  # and at an angle: nearly parallel rays tell little of the distance
AGREE_M = 1.0  # otherwise its casts onto today's walls: they lie this close, or no point


def placed(
    keys: np.ndarray,
    sources: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    casts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One point in space per detail of the photo (`keys`: its keypoint, once per cast): where
    the rays of the pictures that saw it meet, in front of them all (the facade as it is,
    BD TOPO's walls aside); else, where its casts onto today's walls lie, if they agree (a
    ray not to be cast alone, an old photo's, has none). Returns the points, for each its
    detail (keypoint), the cast whose wall it is on (-1: none), and the pictures that saw it
    (frozensets of their ids)."""
    order = np.argsort(keys, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(keys[order]) != 0])
    points, details, walls, seen = [], [], [], []
    for mine in np.split(order, starts[1:]):
        on_wall = mine[~np.isnan(casts[mine]).any(axis=1)]
        point = None
        if len(set(sources[mine].tolist())) >= 2:
            d = directions[mine]
            spread = np.degrees(np.arccos(np.clip((d[:, None] * d[None]).sum(axis=2), -1, 1))).max()
            if spread >= MEET_DEG:
                at, miss = meet(origins[mine], d)
                ahead = ((at - origins[mine]) * d).sum(axis=1) > 0
                if miss <= MEET_M and ahead.all():
                    point = at
        if point is None and len(on_wall):
            spots = casts[on_wall]
            if np.linalg.norm(spots[:, None] - spots[None], axis=2).max() <= AGREE_M:
                point = spots.mean(axis=0)
        if point is None:
            continue
        points.append(point)
        details.append(keys[mine[0]])
        walls.append(on_wall[0] if len(on_wall) else -1)
        seen.append(frozenset(sources[mine].tolist()))
    return (
        np.array(points).reshape(-1, 3),
        np.array(details, int),
        np.array(walls, int),
        np.array(seen, dtype=object),
    )


NEAR_M = 150  # buildings a panorama's matched points are cast onto
# a point cast on a wall the model (see depthmaps) puts this many times nearer or farther is left
# out: the wall is not what was matched (measured on 3,231 rays whose distance two panoramas
# tell: 60 % of the casts off by half or more left out, 2 % of the right ones)
MODEL_OFF = 2.0
ZOOM_FROM = 4  # a marked landmark is zoomed on from this many panoramas, the nearest
ZOOM_SIDE = 640
ZOOM_FOV = (10.0, 60.0)  # a zoomed view sees the landmark whole, within these widths
ZOOM_BAND = 0.13  # the photo zoomed on: its band this wide either side of the mark
ZOOM_UPPER = 0.6  # and its upper part (a landmark stands over the roofs)
OWN_M = 15.0  # a wall this near a landmark, before it, is its own: it does not hide it


def aimed(panorama: "Panorama", at: tuple[float, float], top: float) -> tuple[float, float, float]:
    """The heading, pitch and width of a view of `panorama` zoomed on a landmark standing at
    `at`, its top `top` high (the panoramas' frame): aimed at its middle, seeing it whole."""
    east, north = at[0] - panorama.x, at[1] - panorama.y
    away, rise = float(np.hypot(east, north)), top - panorama.z
    fov = float(np.clip(2 * np.degrees(np.arctan2(0.7 * rise, away)), *ZOOM_FOV))
    return (
        float(np.degrees(np.arctan2(east, north))),
        float(np.degrees(np.arctan2(rise / 2, away))),
        fov,
    )


def onto(origin: np.ndarray, directions: np.ndarray, at: tuple[float, float]) -> np.ndarray:
    """Where rays from `origin` meet the upright plane through `at` facing it: points on a
    landmark, as far as it stands (a tower a few metres across, hundreds away); NaN for rays
    turned away."""
    towards = np.asarray(at) - origin[:2]
    away = float(np.hypot(*towards))
    along = directions[:, :2] @ (towards / away)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(along > 0, away / along, np.nan)
    return origin + directions * t[:, None]


def _band(photo: np.ndarray, column: float) -> tuple[np.ndarray, float, float]:
    """The part of a photo zoomed on for a landmark marked at `column`, and how its pixels
    map to the photo as matched (shrunk): left edge (pixels of the photo), scale."""
    h, w = photo.shape[:2]
    left = max(0, int((column - ZOOM_BAND) * w))
    right = min(w, int((column + ZOOM_BAND) * w))
    return photo[: int(ZOOM_UPPER * h), left:right], left, PHOTO_SIDE / max(h, w)


def shrunk(photo: np.ndarray) -> np.ndarray:
    """An old photo at the size it is matched at (PHOTO_SIDE, its longest side)."""
    scale = PHOTO_SIDE / max(photo.shape[:2])
    return cv2.resize(photo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class Rays:
    """Rays of an old photo's details, matched in other pictures, one row each: the detail
    (`keys`, a row of the photo's pixels), the picture (`sources`: a panorama's id, or
    "photo:<id>" for an old photo placed; a point it passed on counts it as VIA<id>, not as a
    sight: see MIN_SOURCES), where it was taken from, the ray's direction, and
    where it meets a wall of today (`casts`, NaN: none, or not to be cast) with that wall
    (`walls`, a row of Blocks edges, NaN: none)."""

    keys: np.ndarray
    sources: np.ndarray
    origins: np.ndarray
    directions: np.ndarray
    casts: np.ndarray
    walls: np.ndarray

    @classmethod
    def of(cls, parts: list["Rays"]) -> "Rays":
        if not parts:
            return cls(
                np.empty(0, int),
                np.empty(0, object),
                np.empty((0, 3)),
                np.empty((0, 3)),
                np.empty((0, 3)),
                np.empty((0, 6)),
            )
        return cls(*(np.concatenate([getattr(r, f.name) for r in parts]) for f in fields(cls)))


@dataclass(frozen=True)
class Observed:
    """An old photo's details (`pixels`, at PHOTO_SIDE: its keypoints, then those of the bands
    zoomed on its landmarks) and their rays in today's panoramas (see Rays), from `views`
    ranked well enough to be matched; the photo matched is `size` (height, width)."""

    pixels: np.ndarray
    size: tuple[int, int]
    rays: Rays
    views: int = 0


@dataclass(frozen=True)
class Points:
    """The points a camera was found from: the details (`pixels`), where they stand in space
    (`world`) and the pictures that saw each (`seen`, frozensets of Rays sources)."""

    pixels: np.ndarray
    world: np.ndarray
    seen: np.ndarray


class Known(NamedTuple):
    """Points already placed, on details of a photo: the details (rows of its pixels), where
    they stand, the pictures that saw each (frozensets of Rays sources)."""

    keys: np.ndarray
    world: np.ndarray
    seen: np.ndarray

    @classmethod
    def none(cls) -> "Known":
        return cls(np.empty(0, int), np.empty((0, 3)), np.empty(0, object))

    @classmethod
    def of(cls, parts: list["Known"]) -> "Known":
        """Several together, a detail on two of them kept once (the first)."""
        joined = cls(*(np.concatenate(column) for column in zip(cls.none(), *parts, strict=False)))
        first = np.unique(joined.keys, return_index=True)[1]
        return cls(*(column[np.sort(first)] for column in joined))


class Found(NamedTuple):
    camera: Camera | None
    walls: np.ndarray  # rows of Blocks edges its points lie on: the facades to project it onto
    points: Points  # those that agree with it


def locate(
    photo: np.ndarray,
    near: list[Panorama],
    blocks,
    views: Views,
    where: Where,
    trace: Trace | None = None,
    origin_z: float = 0.0,
    model: Callable[[Panorama, np.ndarray], np.ndarray] | None = None,
) -> Found:
    """The camera of an old `photo` (BGR), from the panoramas `near` it, standing `where` it
    could (`trace` keeps how far it got; `origin_z`, `model`: see observe)."""
    trace = Trace() if trace is None else trace
    observed = observe(photo, near, blocks, views, where, trace, origin_z=origin_z, model=model)
    return place(observed, where, trace)


def observe(
    photo: np.ndarray,
    near: list[Panorama],
    blocks,
    views: Views,
    where: Where,
    trace: Trace,
    mine: Features | None = None,
    origin_z: float = 0.0,
    model: Callable[[Panorama, np.ndarray], np.ndarray] | None = None,
) -> Observed:
    """An old `photo` (BGR) matched against the views of the panoramas `near` it ranked best,
    and against views zoomed on the landmarks marked in it (`where.sights`); `mine`: its
    features, if already extracted; `origin_z`: how high the cameras' frame starts (metres
    NGF, as landmarks' tops are); `model`: how far rays from a panorama go by a model (see
    _cast), if any."""
    small = shrunk(photo)
    mine = views.matcher.features(small) if mine is None else mine
    quick = mine.top(QUICK_KEYPOINTS)
    candidates = (
        (len(views.matcher.pairs(quick, theirs.top(QUICK_KEYPOINTS))), n, heading, theirs)
        for n, panorama in enumerate(near)
        for heading, theirs in zip(VIEW_HEADINGS, views.of(panorama), strict=False)
    )
    best = heapq.nlargest(
        BEST_VIEWS, (c for c in candidates if c[0] >= QUICK_MIN), key=itemgetter(0)
    )
    trace.views = len(best)
    found = []
    for _, n, heading, theirs in best:
        in_photo, in_view = views.matcher.match(mine, theirs)
        if len(in_photo) < MIN_MATCHES:
            continue
        trace.matched += 1
        trace.matches += len(in_photo)
        ahead = view_directions(theirs.keypoints[in_view], heading)
        depth = None if model is None else model(near[n], ahead)
        trace.modelled += 0 if depth is None else int(np.isfinite(depth).sum())
        found.append(_cast(near[n], in_photo, ahead, blocks, model=depth))
    pixels = [mine.keypoints]
    pictures = functools.lru_cache(maxsize=ZOOM_FROM)(lambda path: cv2.imread(str(path)))
    for sight in (s for s in where.sights if s.tops):  # with no tops: not zoomed on
        band, rays = _zoom(
            photo, sight, near[:ZOOM_FROM], blocks, views.matcher, pictures, origin_z, trace
        )
        found += [replace(r, keys=r.keys + sum(map(len, pixels))) for r in rays]
        pixels.append(band)
    return Observed(np.concatenate(pixels), small.shape[:2], Rays.of(found), len(best))


def _cast(
    panorama: Panorama,
    keys: np.ndarray,
    ahead: np.ndarray,
    blocks,
    landmark: tuple[float, float] | None = None,
    model: np.ndarray | None = None,
) -> Rays:
    """Rays of details `keys` seen from `panorama` along `ahead`, cast onto today's walls near
    it; for a view zoomed on a `landmark`, a ray meeting no wall up to it (its own walls
    included) meets it. `model`: how far each ray goes by the model (NaN: unknown); a cast
    it puts MODEL_OFF times nearer or farther is left out (not for a zoomed view: a ray
    missing its wall meets the landmark anyway)."""
    around = blocks.near(panorama.x, panorama.y, NEAR_M)
    origin = np.array([panorama.x, panorama.y, panorama.z])
    cast, edge = around.hit(origin, ahead)
    on_wall = edge >= 0
    if model is not None:
        metres = np.linalg.norm(cast - origin, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = model / metres
            on_wall &= ~((ratio > MODEL_OFF) | (ratio < 1 / MODEL_OFF))
        cast = np.where(on_wall[:, None], cast, np.nan)
    if landmark is not None:
        away = float(np.hypot(landmark[0] - panorama.x, landmark[1] - panorama.y))
        on_wall &= np.hypot(*(cast[:, :2] - origin[:2]).T) <= away + OWN_M  # not beyond it
        cast = np.where(on_wall[:, None], cast, onto(origin, ahead, landmark))
    return Rays(
        keys,
        np.full(len(keys), panorama.id, dtype=object),
        np.tile(origin, (len(keys), 1)),
        ahead,
        cast,
        np.where(on_wall[:, None], around.edges[np.maximum(edge, 0)], np.nan),
    )


def place(
    observed: Observed,
    where: Where,
    trace: Trace,
    more: Rays | None = None,
    known: Known | None = None,
) -> Found:
    """The camera that sees the points an old photo's rays place (`observed`'s, and `more`:
    of old photos placed), and points already placed (`known`: details, where they stand,
    the pictures that saw each), standing `where` it could."""
    rays = Rays.of([observed.rays] + ([more] if more is not None else []))
    world, keys, wall_of, seen = placed(
        rays.keys, rays.sources, rays.origins, rays.directions, rays.casts
    )
    walls = np.where((wall_of >= 0)[:, None], rays.walls[np.maximum(wall_of, 0)], np.nan)
    if known is not None and len(known.keys):
        fresh = ~np.isin(known.keys, keys)  # a detail placed already keeps its point
        keys = np.concatenate([keys, known.keys[fresh]])
        world = np.concatenate([world, known.world[fresh]])
        seen = np.concatenate([seen, known.seen[fresh]])
        walls = np.concatenate([walls, np.full((fresh.sum(), 6), np.nan)])
    nothing = Found(None, np.empty((0, 6)), Points(np.empty((0, 2)), np.empty((0, 3)), seen[:0]))
    if not len(rays.keys) and not len(keys):
        trace.verdict = "aucune vue appariée" if observed.views else "aucune vue"
        return nothing
    pixels = observed.pixels[keys].astype(float)
    camera = solve(world, pixels, seen, observed.size, where, trace)
    if camera is None:
        return nothing
    on_walls = walls[camera.inliers]
    return Found(
        camera,
        np.unique(on_walls[~np.isnan(on_walls).any(axis=1)], axis=0),
        Points(pixels[camera.inliers], world[camera.inliers], seen[camera.inliers]),
    )


def _zoom(
    photo: np.ndarray,
    sight: Sight,
    near: list[Panorama],
    blocks,
    matcher: Matcher,
    pictures: Callable[[Path], np.ndarray | None],
    origin_z: float,
    trace: Trace,
) -> tuple[np.ndarray, list[Rays]]:
    """The band of `photo` around a landmark marked in it, matched against views of the
    panoramas `near` it zoomed on the landmark (`pictures`: their 360° pictures, by path):
    the band's details (pixels of the photo as matched) and their rays (keys: rows of
    those)."""
    band, left, scale = _band(photo, sight.column)
    mine = matcher.features(band)
    found = []
    for at, top in zip(sight.points, sight.tops, strict=True):
        for panorama in near:
            if (picture := pictures(panorama.path)) is None:
                continue
            heading, pitch, fov = aimed(panorama, at, top - origin_z)
            zoomed = perspective(picture, panorama.axes, heading, fov, ZOOM_SIDE, pitch)
            theirs = matcher.features(zoomed)
            in_band, in_view = matcher.match(mine, theirs)
            if len(in_band) < MIN_MATCHES:
                continue
            trace.matched += 1
            trace.matches += len(in_band)
            ahead = view_directions(theirs.keypoints[in_view], heading, fov, ZOOM_SIDE, pitch)
            found.append(_cast(panorama, in_band, ahead, blocks, at))
    return (mine.keypoints + np.array([left, 0], np.float32)) * scale, found


def _sample(picture: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The picture's colour at columns `u`, rows `v` (one row of colour per point); OpenCV's
    remap takes maps under 32 768 wide, so they are laid out in rows."""
    n, row = len(u), 1024
    if not n:
        return np.empty((0, picture.shape[2]), picture.dtype)
    padded = -(-n // row) * row
    maps = [np.pad(m.astype(np.float32), (0, padded - n)).reshape(-1, row) for m in (u, v)]
    found = cv2.remap(picture, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return found.reshape(-1, picture.shape[2])[:n]


HIDDEN_M = 0.5  # a spot is seen from a panorama if nothing stands this much before it
SEES_M = 200  # the walls a camera can meet, for the views of today
FIRST_STEP_M = 0.5  # where a ray meets the street: looked for at 0.5, 1, 2, 4… m
GROUND_ROUNDS = 10  # then narrowed down by halves this many times
UNSEEN, SKY = -1, -2  # the source of a pixel no panorama sees, and of the sky


def real_view(
    view: View, size: tuple[int, int], panorama: Panorama, picture: np.ndarray
) -> np.ndarray:
    """The panorama's own picture, framed as the old camera framed its photo (heading, width
    of view, horizon): a true view of today, from where the panorama stands."""
    directions = rays(pixels_of(size), size, view.fov, view.horizon, view.heading)
    return _cut(picture, panorama.axes, directions, size)


def _to_street(
    eye: np.ndarray, ahead: np.ndarray, street_z: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> np.ndarray:
    """How far each ray from `eye` goes before it meets the street (inf: up, or never within
    sight). The street may slope, even more steeply than a ray: the first step at which the
    ray is below it brackets the meeting, then halving finds it."""
    down = np.flatnonzero(ahead[:, 2] < 0)

    def below(t: np.ndarray) -> np.ndarray:
        at = eye + t[:, None] * ahead[down]
        return at[:, 2] <= street_z(at[:, 0], at[:, 1])

    near, far = np.zeros(len(down)), np.full(len(down), np.inf)
    t = FIRST_STEP_M
    while t < 2 * SEES_M:
        open_ = np.isinf(far)
        met = open_ & below(np.full(len(down), t))
        far[met] = t
        near[open_ & ~met] = t
        t *= 2
    found = np.isfinite(far)
    for _ in range(GROUND_ROUNDS):
        middle = np.where(found, (near + far) / 2, 0)
        under = below(middle)
        far = np.where(found & under, middle, far)
        near = np.where(found & ~under, middle, near)
    to_street = np.full(len(ahead), np.inf)
    to_street[down] = far
    return to_street


def today_from(
    view: View,
    size: tuple[int, int],
    panoramas: list[tuple[Panorama, np.ndarray]],
    blocks,
    street_z: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Today's street as the old camera saw it (a picture of `size`, height and width), made
    from the panoramas around (each with its picture, BGR), and for each pixel which of them
    gave it. Each pixel's ray ends on today's facades, on the
    street (its height at any point: `street_z`) or in the sky; of the panoramas that see
    that spot with nothing before it, the one looking at it most nearly as the camera does
    gives its colour (`SKY`: the sky, from the first panorama; `UNSEEN`: none sees it)."""
    h, w = size
    ahead = rays(pixels_of(size), size, view.fov, view.horizon, view.heading)
    eye = np.array([view.x, view.y, view.z])
    points, edge = blocks.near(view.x, view.y, SEES_M).hit(eye, ahead)
    to_wall = np.where(edge >= 0, np.linalg.norm(points - eye, axis=1), np.inf)
    to_street = _to_street(eye, ahead, street_z)
    reach = np.minimum(to_wall, to_street)
    sky = ~np.isfinite(reach)
    spot = eye + np.where(sky, 0, reach)[:, None] * ahead
    best = np.full(len(ahead), np.inf)  # the angle of the panorama chosen so far
    source = np.where(sky, SKY, UNSEEN)
    out = np.zeros((len(ahead), 3), np.uint8)
    # the sky: the same direction from every panorama, taken from the first
    first_panorama, first_picture = panoramas[0]
    u, v = _in_panorama(ahead[sky], first_panorama.axes, first_picture.shape[:2])
    out[sky] = _sample(first_picture, u, v)
    for k, (panorama, picture) in enumerate(panoramas):
        origin = np.array([panorama.x, panorama.y, panorama.z])
        towards = spot - origin
        distance = np.linalg.norm(towards, axis=1)
        with np.errstate(invalid="ignore"):  # the sky's spots are the eye itself
            towards /= distance[:, None]
            angle = np.arccos(np.clip((towards * ahead).sum(axis=1), -1, 1))
        could = np.flatnonzero(~sky & (angle < best))  # only where it would do better
        blocked, met = blocks.near(panorama.x, panorama.y, SEES_M).hit(origin, towards[could])
        first = np.where(met >= 0, np.linalg.norm(blocked - origin, axis=1), np.inf)
        seen = could[first >= distance[could] - HIDDEN_M]
        if not len(seen):
            continue
        u, v = _in_panorama(towards[seen], panorama.axes, picture.shape[:2])
        out[seen] = _sample(picture, u, v)
        best[seen] = angle[seen]
        source[seen] = k
    return out.reshape(h, w, 3), source.reshape(h, w)
