"""Today's buildings as upright prisms (BD TOPO footprints, ground and height), for casting
rays from a camera: the skyline seen from a point, the facade a ray meets first."""

from dataclasses import dataclass

import numpy as np

Block = tuple[list[tuple[float, float]], float, float]  # footprint, base, height
# Rays × edges worked on at once: each such array is 4 MB, a batch builds about ten (a whole
# picture against a district's walls at once took all the memory).
MOST_PAIRS = 500_000
SHARED_M = 0.5  # two buildings' corners closer than this are one, inside a facade


@dataclass(frozen=True)
class Blocks:
    """The footprints' edges: x0, y0, x1, y1, base and top of the wall, in the frame of the
    cameras looking at them (the CLI: metres Lambert-93, heights above the median ground)."""

    edges: np.ndarray
    # whether each edge's first end is a corner of its own: not shared with a neighbour along
    # a flush facade, where the picture shows nothing
    free: np.ndarray

    @classmethod
    def of(cls, blocks: list[Block]) -> "Blocks":
        edges = np.array(
            [
                (*f[i], *f[(i + 1) % len(f)], base, base + height)
                for f, base, height in blocks
                for i in range(len(f))
            ],
            dtype=float,
        ).reshape(-1, 6)
        spot = np.round(edges[:, :2] / SHARED_M).astype(np.int64)
        _, where, count = np.unique(spot, axis=0, return_inverse=True, return_counts=True)
        return cls(edges, count[where.ravel()] == 1)

    def near(self, x: float, y: float, metres: float) -> "Blocks":
        """The edges passing within `metres` of (x, y)."""
        e = self.edges
        along = np.stack([e[:, 2] - e[:, 0], e[:, 3] - e[:, 1]], axis=1)
        length = np.maximum((along**2).sum(axis=1), 1e-9)
        t = np.clip(((x - e[:, 0]) * along[:, 0] + (y - e[:, 1]) * along[:, 1]) / length, 0, 1)
        gap = np.hypot(e[:, 0] + t * along[:, 0] - x, e[:, 1] + t * along[:, 1] - y)
        close = gap < metres
        return Blocks(e[close], self.free[close])

    def _crossings(self, x: float, y: float, dx: np.ndarray, dy: np.ndarray):
        """For horizontal rays from (x, y) along (dx, dy) (one per row), the distance to each
        edge's line (t, in units of the ray's horizontal length) and where along the edge it
        is crossed (s, 0 to 1)."""
        e = self.edges
        ex, ey = (e[:, 2] - e[:, 0])[None], (e[:, 3] - e[:, 1])[None]
        rx, ry = (e[:, 0] - x)[None], (e[:, 1] - y)[None]
        dx, dy = dx[:, None], dy[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            den = dx * ey - dy * ex
            return (rx * ey - ry * ex) / den, (rx * dy - ry * dx) / den

    def _batched(self, work, *per_ray: np.ndarray):
        """`work` applied to the rays (`per_ray`: arrays with one row per ray) a batch at a
        time, so that rays × edges never exceeds MOST_PAIRS; the results put back together."""
        batch = max(1, MOST_PAIRS // max(len(self.edges), 1))
        if len(per_ray[0]) <= batch:
            return work(*per_ray)
        parts = [
            work(*(a[k : k + batch] for a in per_ray)) for k in range(0, len(per_ray[0]), batch)
        ]
        if isinstance(parts[0], tuple):
            return tuple(np.concatenate(found) for found in zip(*parts, strict=True))
        return np.concatenate(parts)

    def skyline(self, x: float, y: float, z: float, bearings: np.ndarray) -> np.ndarray:
        """Per bearing (radians clockwise from north), the highest elevation of a roof seen
        from (x, y, z); -inf where there is no building."""
        return self._batched(lambda b: self._skyline(x, y, z, b), bearings)

    def _skyline(self, x: float, y: float, z: float, bearings: np.ndarray) -> np.ndarray:
        t, s = self._crossings(x, y, np.sin(bearings), np.cos(bearings))
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (self.edges[:, 5][None] - z) / t  # the elevation grows with it, for t > 0
        hit = (t > 0.5) & (s >= 0) & (s <= 1)
        highest = np.where(hit, slope, -np.inf).max(axis=1, initial=-np.inf)
        return np.where(np.isfinite(highest), np.arctan(highest), -np.inf)

    def hit(self, origin: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where each ray from `origin` along `directions` (east, north, up; one per row) first
        meets a wall, and which edge: NaN and -1 for a ray that meets none (the sky, the
        street)."""
        return self._batched(lambda d: self._hit(origin, d), directions)

    def _hit(self, origin: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.edges) or not len(directions):
            return np.full((len(directions), 3), np.nan), np.full(len(directions), -1)
        x, y, z = origin
        t, s = self._crossings(x, y, directions[:, 0], directions[:, 1])
        e = self.edges
        with np.errstate(invalid="ignore"):  # a ray parallel to an edge: no crossing
            height = z + t * directions[:, 2:3]
        hit = (
            (t > 0.5) & (s >= 0) & (s <= 1) & (height >= e[:, 4][None]) & (height <= e[:, 5][None])
        )
        first = np.where(hit, t, np.inf)
        edge = first.argmin(axis=1)
        distance = first[np.arange(len(directions)), edge]
        found = np.isfinite(distance)
        reach = np.where(found, distance, 0.0)[:, None]
        points = np.where(found[:, None], origin[None] + reach * directions, np.nan)
        return points, np.where(found, edge, -1)

    def corners(self, x: float, y: float, z: float, metres: float) -> np.ndarray:
        """The free corners seen from (x, y, z) within `metres`, not hidden behind a nearer
        wall: one row each of bearing (radians clockwise from north), distance, base, top."""
        e = self.edges[self.free]
        dx, dy = e[:, 0] - x, e[:, 1] - y
        distance = np.hypot(dx, dy)
        close = (distance > 1) & (distance < metres)
        e, dx, dy, distance = e[close], dx[close], dy[close], distance[close]
        if not len(e):
            return np.empty((0, 4))
        nearest = self._batched(
            lambda ux, uy: self._nearest_at(x, y, z, ux, uy), dx / distance, dy / distance
        )
        seen = distance <= nearest + 0.3
        return np.column_stack([np.arctan2(dx, dy), distance, e[:, 4], e[:, 5]])[seen]

    def _nearest_at(
        self, x: float, y: float, z: float, dx: np.ndarray, dy: np.ndarray
    ) -> np.ndarray:
        """Along each horizontal direction, how far the nearest wall at height z stands."""
        t, s = self._crossings(x, y, dx, dy)
        at_eye = (self.edges[:, 4] <= z) & (self.edges[:, 5] >= z)
        crossed = (t > 0.5) & (s >= 0) & (s <= 1) & at_eye[None]
        return np.where(crossed, t, np.inf).min(axis=1, initial=np.inf)
