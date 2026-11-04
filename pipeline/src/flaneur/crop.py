"""Automatic crop: find the photographic print on its mount.

Museum scans show the print on a card mount, with handwriting, stamps and an inventory
number around it. The mount is one fairly uniform colour that reaches the image's edges, so
it is read from the image's border; a row or column belongs to the print when most of its
pixels are not of the mount's colour (a line of handwriting is mostly mount). A stereoscopic
view is two such blocks side by side, separated by mount, of about the same width: one of
them is enough. When the border is not uniform, the print reaches the edge: nothing to crop.

That border can also mix a scanner rim, a card cut askew and handwriting. The print is then
found from within: where the photo is textured (the card is smooth; handwriting, stamps and
labels are thin or small), grown over its pale parts (a sky, a pale wall: unlike the card's
colour) up to its own edge, a long straight line. Straight lines also frame each view of a
stereoscopic pair, however dark or rounded its corners.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

ANALYSIS_SIDE = 400  # the image is scaled to this size for the analysis
BORDER = 0.03  # share of the width and height read as the image's border
# A side of the border is mount when its grey levels deviate from their median by less than
# this (median absolute deviation: a line of handwriting or a stamp does not count).
UNIFORM_SIDE = 12.0
# There is a mount when at least this many sides are mount, of colours this close.
MOUNT_SIDES = 2
SAME_COLOUR = 30.0
# A pixel is mount when its colour is this close to the mount's (distance in RGB) and it is
# about as smooth as the mount: a pale print can share the mount's grey, not its texture.
MOUNT_TOLERANCE = 22.0
WINDOW = 7  # side of the square, in pixels, over which texture is measured
SMOOTH = 2.5  # times the mount's own texture
# A row or column belongs to the print when at least this share of it is not mount.
PRINT_SHARE = 0.3
# A row (or column) with less print than this is bare card: a pale sky inside a print never
# empties a whole one.
BARE = 0.02
LINE = 0.1  # a band thinner than this share of the height, beyond bare card, is a caption
# Runs of print rows (or columns) this close, as a share of the image, are one print.
MAX_GAP = 0.12
# Layers peeled at most (scanner background, card, print), and a layer that keeps more than
# this share of what it peels is the last one (nothing more around the print).
MAX_LAYERS = 3
THIN_LAYER = 0.97
INSET = 0.01  # shaved off a layer's edges, as a share of the image, before peeling inside it
# A crop keeping more than this share of the image is not worth proposing.
MAX_KEPT_AREA = 0.92
# A block narrower or lower than this share of the image is not a print.
MIN_PRINT_SIDE = 0.2
# Two blocks are a stereoscopic pair when their widths differ by less than this share.
STEREO_WIDTH_TOLERANCE = 0.15
# Margin kept around the print, as a share of the image (the print's own edge stays visible).
MARGIN = 0.005

# A stereoscopic plate may have no mount: its two views then reach the scan's edges. It is
# told by its shape and its content: a wide image whose two halves are the same picture, a
# few centimetres apart (correlation of the halves at least STEREO_MATCH, for an offset
# between STEREO_OFFSET of the width).
STEREO_ASPECT = 1.6
STEREO_OFFSET = (0.4, 0.6)
STEREO_MATCH = 0.6
SEPARATOR_STD = 8.0  # a column this even, between the two views, is the plate's edge

# The print found from within, and the views of a pair framed by straight lines.
FRAME_SIDE = 600  # the image is scaled to this size to find straight lines and texture
TILT_DEG = 3.0  # a line this near the horizontal or the vertical may be a print's edge
MIN_LINE = 0.08  # segments at least this share of the image's shorter side
LINE_GAP = 0.01  # joined across gaps this short, as a share of its longer side
SAME_LINE = 0.012  # segments this close, as a share of the image, are one line
COVERED = 0.6  # a frame's side runs along a line over at least this share of it
TEXTURE_WINDOW = 5  # pixels: texture is the spread of grey levels over such squares
TEXTURE = 4.0  # grey levels of spread, and at least 1.5 times the image's median spread
THIN = 0.02  # textured strokes thinner than this share of the image are handwriting
PART = 0.01  # a textured patch of at least this share of the image is part of the photo
INSIDE_SIDES = 3  # the textured part stands off the scan's edges on this many sides
OFF_EDGE = 0.03  # by at least this share of the image
RING = 0.03  # the card's colour is read in a ring this wide around the textured part
CARD_RGB = 18.0  # a pixel this close to it (distance in RGB) is card
TO_EDGE = 0.03  # a print's edge, this near the side found (share of the image)
# A card is cut this deep into the scan on at least DEEP_CUTS sides, and along a print's
# edge on one at least: a pale sky or street on no mount is cut only on one or two, along
# nothing.
DEEP = 0.05
DEEP_CUTS = 3
STEREO_HOLD = 0.3  # each view of a pair holds at least this share of the photo's texture
STEREO_LEFT_OUT = 0.15  # and both, all of it but this share
# Bounds on the work (frames grow as the fourth power of the lines): a scan of the pilot
# has at most 30 lines each way; a facade's grid of windows could have hundreds.
MOST_LINES = 30  # the longest, each way
MOST_FRAMES = 200  # the largest, tried as views of a pair
LEFT_PART = 0.6  # a plate's left view ends before this share of its width
TALL_VIEW = 0.8  # and holds at least this share of its height
TOUCHING_PX = 2  # the right view starts no more than this left of the left one's end

Crop = tuple[float, float, float, float]  # left, top, right, bottom, fractions of the image
Box = tuple[int, int, int, int]  # left, top, right, bottom, in pixels
Line = tuple[float, np.ndarray]  # where it runs, and which pixels along it it covers


def _texture(grey: np.ndarray) -> np.ndarray:
    """Local standard deviation of grey levels over WINDOW × WINDOW squares (box sums)."""
    pad = WINDOW // 2
    padded = np.pad(grey, pad, mode="edge")

    def box_mean(a: np.ndarray) -> np.ndarray:
        s = np.pad(np.cumsum(np.cumsum(a, axis=0), axis=1), ((1, 0), (1, 0)))
        w = WINDOW
        return (s[w:, w:] - s[:-w, w:] - s[w:, :-w] + s[:-w, :-w]) / w**2

    mean = box_mean(padded)
    return np.sqrt(np.maximum(box_mean(padded**2) - mean**2, 0))


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True, as (start, end) with end excluded."""
    edges = np.diff(np.concatenate([[0], flags.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _peel(rgb: np.ndarray) -> Crop | None:
    """One layer: the block inside this image's mount, or None when its edges are not a
    uniform mount. The block may be nearly the whole image (a thin layer)."""
    grey = rgb.mean(axis=2)
    height, width = grey.shape
    bh, bw = max(1, int(height * BORDER)), max(1, int(width * BORDER))
    edges = [
        (slice(0, bh), slice(None)),
        (slice(-bh, None), slice(None)),
        (slice(None), slice(0, bw)),
        (slice(None), slice(-bw, None)),
    ]
    colour = {i: float(np.median(grey[e])) for i, e in enumerate(edges)}
    spread = {i: float(np.median(np.abs(grey[e] - colour[i]))) for i, e in enumerate(edges)}
    uniform = [i for i in spread if spread[i] <= UNIFORM_SIDE]
    # The sides of one colour are the outer layer; a scan can cut the card on two sides and
    # show the scanner's background on the other two: the smoother layer is peeled first.
    groups = [[j for j in uniform if abs(colour[j] - colour[i]) <= SAME_COLOUR] for i in uniform]
    groups = [g for g in groups if len(g) >= MOUNT_SIDES]
    if not groups:
        return None  # no mount: the print reaches the edges
    outer = min(groups, key=lambda g: np.mean([spread[i] for i in g]))
    mount_edges = [edges[i] for i in outer]
    mount_rgb = np.median(np.concatenate([rgb[e].reshape(-1, 3) for e in mount_edges]), axis=0)
    texture = _texture(grey)
    mount_texture = float(np.median(np.concatenate([texture[e].ravel() for e in mount_edges])))
    not_mount = (np.linalg.norm(rgb - mount_rgb, axis=2) > MOUNT_TOLERANCE) | (
        texture > SMOOTH * max(mount_texture, 1.0)
    )

    def blocks(share: np.ndarray, size: int, join_gaps: bool) -> list[tuple[int, int]]:
        """Runs of rows (or columns) that are mostly print, keeping those big enough to be a
        print. Rows are joined across thin gaps (a pale sky inside the print), unless bare card
        lies in between and one side is a thin band (a caption); columns are not, since the
        mount between two columns of print may separate a stereoscopic pair."""
        joined: list[tuple[int, int]] = []
        for start, end in _runs(share >= PRINT_SHARE):
            if (
                join_gaps
                and joined
                and start - joined[-1][1] <= MAX_GAP * size
                and (
                    share[joined[-1][1] : start].min() >= BARE
                    or min(end - start, joined[-1][1] - joined[-1][0]) >= LINE * size
                )
            ):
                joined[-1] = (joined[-1][0], end)
            else:
                joined.append((start, end))
        return [r for r in joined if r[1] - r[0] >= MIN_PRINT_SIDE * size]

    rows = blocks(not_mount.mean(axis=1), height, join_gaps=True)
    if not rows:
        return None
    top, bottom = max(rows, key=lambda run: run[1] - run[0])
    columns = sorted(
        blocks(share := not_mount[top:bottom].mean(axis=0), width, join_gaps=False),
        key=lambda run: run[1] - run[0],
        reverse=True,
    )
    if not columns:
        return None
    left, right = columns[0]
    if len(columns) > 1:
        other = columns[1]
        widths = right - left, other[1] - other[0]
        between = share[min(right, other[1]) : max(left, other[0])]
        if between.size and between.min() >= BARE:
            left, right = min(left, other[0]), max(right, other[1])  # a pale street between
        elif abs(widths[0] - widths[1]) <= STEREO_WIDTH_TOLERANCE * max(widths):
            left, right = min((left, right), other)  # a stereoscopic pair: the left view
    return left / width, top / height, right / width, bottom / height


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """How alike two pictures of the same size are (1: the same, up to light and contrast)."""
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) or 1.0))


def _left_view(grey: np.ndarray) -> int | None:
    """The right edge, in columns, of the left view of a stereoscopic plate; None when the
    image is not one."""
    height, width = grey.shape
    if width < STEREO_ASPECT * height:
        return None

    def match(offset: int) -> float:
        return _correlation(grey[:, : width - offset], grey[:, offset:])

    low, high = (round(f * width) for f in STEREO_OFFSET)
    offset = max(range(low, high + 1), key=match)
    if match(offset) < STEREO_MATCH:
        return None
    right = offset
    columns = grey.std(axis=0)
    while right > 0 and columns[right - 1] < SEPARATOR_STD:
        right -= 1  # the white gap and the black edge between the views
    return right


def _peeled(image: Image.Image) -> Crop | None:
    """The print's frame, peeled from the outside (for a stereoscopic pair, the left view), or
    None. A scan can have several layers (the scanner's white background, then the card, then
    the print): they are peeled one after the other."""
    small = image.copy()
    small.thumbnail((ANALYSIS_SIDE, ANALYSIS_SIDE))
    rgb = np.asarray(small.convert("RGB"), dtype=float)
    height, width = rgb.shape[:2]
    crop = (0.0, 0.0, 1.0, 1.0)
    for layer in range(MAX_LAYERS):
        # A layer's edges keep a sliver of the layer around it: shaved before peeling. The
        # shave stays when nothing more is found inside: it also takes back the texture
        # window's spill past the print's edge (WINDOW // 2 pixels, about INSET).
        shave = INSET if layer else 0.0
        crop = (crop[0] + shave, crop[1] + shave, crop[2] - shave, crop[3] - shave)
        x0, y0 = round(crop[0] * width), round(crop[1] * height)
        x1, y1 = round(crop[2] * width), round(crop[3] * height)
        inner = _peel(rgb[y0:y1, x0:x1])
        if inner is None or (inner[2] - inner[0]) * (inner[3] - inner[1]) > THIN_LAYER:
            break
        w, h = crop[2] - crop[0], crop[3] - crop[1]
        crop = (
            crop[0] + inner[0] * w,
            crop[1] + inner[1] * h,
            crop[0] + inner[2] * w,
            crop[1] + inner[3] * h,
        )
    kept = (crop[2] - crop[0]) * (crop[3] - crop[1])
    if kept > MAX_KEPT_AREA:
        # no mount: a stereoscopic plate is cut to its left view, and nothing more (its
        # white sky would pass for a card)
        left_view = _left_view(rgb.mean(axis=2))
        return (0.0, 0.0, left_view / width, 1.0) if left_view else None
    return (
        max(0.0, crop[0] - MARGIN),
        max(0.0, crop[1] - MARGIN),
        min(1.0, crop[2] + MARGIN),
        min(1.0, crop[3] + MARGIN),
    )


def _lines(edges: np.ndarray) -> tuple[list[Line], list[Line]]:
    """The long horizontal and vertical straight lines of an edge map: segments near one
    another are one line."""
    h, w = edges.shape
    found = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720,
        threshold=30,
        minLineLength=int(MIN_LINE * min(h, w)),
        maxLineGap=int(LINE_GAP * max(h, w)),
    )
    across, down = [], []
    for x1, y1, x2, y2 in [] if found is None else found.reshape(-1, 4):
        tilt = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        if tilt <= TILT_DEG:
            across.append(((y1 + y2) / 2, min(x1, x2), max(x1, x2)))
        elif tilt >= 90 - TILT_DEG:
            down.append(((x1 + x2) / 2, min(y1, y2), max(y1, y2)))

    def grouped(segments: list, size: int, length: int) -> list[Line]:
        groups: list[tuple[list[float], np.ndarray]] = []
        for at, start, end in sorted(segments):
            if not groups or at - groups[-1][0][0] > SAME_LINE * size:  # no chain of edges
                groups.append(([], np.zeros(length, bool)))
            groups[-1][0].append(at)
            groups[-1][1][int(start) : int(end) + 1] = True
        longest = sorted(groups, key=lambda g: g[1].sum(), reverse=True)[:MOST_LINES]
        return sorted((float(np.median(at)), covers) for at, covers in longest)

    return grouped(across, h, w), grouped(down, w, h)


def _spread(grey: np.ndarray) -> np.ndarray:
    """How much grey levels vary around each pixel (over TEXTURE_WINDOW squares)."""
    g = grey.astype(np.float32)
    window = (TEXTURE_WINDOW, TEXTURE_WINDOW)
    mean = cv2.blur(g, window)
    return np.sqrt(np.maximum(cv2.blur(g * g, window) - mean**2, 0))


def _textured(spread: np.ndarray) -> np.ndarray:
    """Where the photo is, from the spread of its grey levels (`_spread`): textured patches big
    enough to be part of it (handwriting, thin, and stamps, small, are left out)."""
    textured = (spread > max(TEXTURE, 1.5 * float(np.median(spread)))).astype(np.uint8)
    k = max(3, int(THIN * max(spread.shape)))
    textured = cv2.morphologyEx(textured, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, label, stats, _ = cv2.connectedComponentsWithStats(textured)
    parts = [c for c in range(1, n) if stats[c, cv2.CC_STAT_AREA] >= PART * spread.size]
    return np.isin(label, parts)


def _framed(rows: list[Line], columns: list[Line], h: int, w: int) -> list[Box]:
    """The frames whose four sides run along lines (or the scan's edges)."""
    rows = [(0.0, np.ones(w, bool)), *rows, (float(h), np.ones(w, bool))]
    columns = [(0.0, np.ones(h, bool)), *columns, (float(w), np.ones(h, bool))]
    frames = []
    for top, along_top in rows:
        for bottom, along_bottom in rows:
            if bottom - top < MIN_PRINT_SIDE * h:
                continue
            t, b = int(top), int(bottom)
            sides = [int(at) for at, covers in columns if covers[t:b].mean() >= COVERED]
            frames += [
                (left, t, right, b)
                for left in sides
                for right in sides
                if right - left >= MIN_PRINT_SIDE * w
                and along_top[left:right].mean() >= COVERED
                and along_bottom[left:right].mean() >= COVERED
            ]
    return frames


def _alike(grey: np.ndarray, a: Box, b: Box) -> bool:
    """Whether two frames hold the same picture."""
    pa, pb = (
        cv2.resize(grey[top:bottom, left:right], (64, 64)) for left, top, right, bottom in (a, b)
    )
    return _correlation(pa, pb) >= STEREO_MATCH


def _stereo(grey: np.ndarray, frames: list[Box], photo: np.ndarray) -> Box | None:
    """The left view of a stereoscopic pair: of two framed views side by side (level, touching
    at most), alike in size (STEREO_WIDTH_TOLERANCE, for heights too) and in what they show,
    holding the photo between them (all but STEREO_LEFT_OUT of it), the largest."""
    held = np.pad(photo.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    total = held[-1, -1]
    holding = {}
    for f in frames:
        left, top, right, bottom = f
        holds = held[bottom, right] - held[top, right] - held[bottom, left] + held[top, left]
        if holds >= STEREO_HOLD * total:
            holding[f] = holds
    largest = sorted(holding, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]), reverse=True)
    largest = largest[:MOST_FRAMES]
    for a in largest:
        wa, ha = a[2] - a[0], a[3] - a[1]
        for b in largest:
            if (
                b[0] >= a[2] - TOUCHING_PX
                and abs(a[1] - b[1]) <= STEREO_WIDTH_TOLERANCE * ha  # side by side: level
                and abs(wa - (b[2] - b[0])) <= STEREO_WIDTH_TOLERANCE * wa
                and abs(ha - (b[3] - b[1])) <= STEREO_WIDTH_TOLERANCE * ha
                and holding[a] + holding[b] >= (1 - STEREO_LEFT_OUT) * total
                and _alike(grey, a, b)
            ):
                return a
    return None


def _on_card(
    rgb: np.ndarray, photo: np.ndarray, rows: list[Line], columns: list[Line]
) -> Box | None:
    """The print on its card, found from within: the textured part, grown over the print's
    pale parts (rows and columns not mostly of the card's colour), each side then moved to
    the print's edge nearby; None when that is not a card cut away (see DEEP)."""
    h, w = photo.shape
    ys, xs = np.nonzero(photo)
    if not len(xs):
        return None
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    off = [left, top, w - right, h - bottom] > OFF_EDGE * np.array([w, h, w, h])
    if off.sum() < INSIDE_SIDES:
        return None
    ring = np.zeros((h, w), bool)
    dx, dy = int(RING * w), int(RING * h)
    ring[max(0, top - dy) : bottom + dy, max(0, left - dx) : right + dx] = True
    ring[top:bottom, left:right] = False
    if not ring.any():
        return None  # too small a scan to read the card around the print
    print_ = np.linalg.norm(rgb - np.median(rgb[ring], axis=0), axis=2) > CARD_RGB

    def grown(at: int, step: int, end: int, line) -> int:
        """`at` moved by `step` while the row or column beyond it is still mostly print."""
        while at != end and line(at if step > 0 else at - 1).mean() >= PRINT_SHARE:
            at += step
        return at

    left = grown(left, -1, 0, lambda x: print_[top:bottom, x])
    right = grown(right, 1, w, lambda x: print_[top:bottom, x])
    top = grown(top, -1, 0, lambda y: print_[y, left:right])
    bottom = grown(bottom, 1, h, lambda y: print_[y, left:right])
    frame, cuts, along_edges = [], 0, 0
    for at, lines, (start, end), size in (
        (left, columns, (top, bottom), w),
        (top, rows, (left, right), h),
        (right, columns, (top, bottom), w),
        (bottom, rows, (left, right), h),
    ):
        near = [
            (abs(line - at), int(line))
            for line, covers in lines
            if abs(line - at) <= TO_EDGE * size and covers[start:end].mean() >= COVERED / 2
        ]
        frame.append(min(near)[1] if near else at)
        if min(at, size - at) > DEEP * size:
            cuts += 1
            along_edges += bool(near)
    return tuple(frame) if cuts >= DEEP_CUTS and along_edges else None


def detect(image: Image.Image) -> Crop | None:
    """The print's frame on its mount (for a stereoscopic pair, the left view), or None when
    there is no mount worth removing: a stereoscopic pair framed by straight lines, else the
    print peeled from the outside, else found on its card from within."""
    return _framed_print(image, _Scan.of(image))


def examine(image: Image.Image) -> tuple[Crop | None, list[str]]:
    """The print's frame (as `detect` finds it), and why it may need a hand (see `_doubts`;
    none for a stereoscopic pair cut to its left view, the other view left out on purpose)."""
    scan = _Scan.of(image)
    frame = _framed_print(image, scan)
    return frame, [] if frame and _left_of_a_pair(scan, frame) else _doubts(scan, frame)


def _framed_print(image: Image.Image, scan: "_Scan") -> Crop | None:
    """`detect`, on the scan already looked at."""
    h, w = scan.grey.shape
    found = scan.pair()
    if found is None and (peeled := _peeled(image)):
        return peeled
    found = found or _on_card(scan.rgb, scan.photo, scan.rows, scan.columns)
    if found is None or (found[2] - found[0]) * (found[3] - found[1]) > MAX_KEPT_AREA * w * h:
        return None
    left, top, right, bottom = found
    return (  # the print's own edge kept, as the peeled frame keeps it
        max(0.0, left / w - MARGIN),
        max(0.0, top / h - MARGIN),
        min(1.0, right / w + MARGIN),
        min(1.0, bottom / h + MARGIN),
    )


def _left_of_a_pair(scan: "_Scan", frame: Crop) -> bool:
    """Whether `frame` is the left view of a stereoscopic plate, however it was found: the
    scan is a plate's shape (STEREO_ASPECT), and the frame its left part, of nearly its
    height (the two views differ by their parallax: how alike they are tells little)."""
    h, w = scan.grey.shape
    left, top, right, bottom = frame
    return w >= STEREO_ASPECT * h and right <= LEFT_PART and bottom - top >= TALL_VIEW


@dataclass(frozen=True)
class _Scan:
    """A scan as the print is looked for on it: at FRAME_SIDE, its straight lines and where
    the photo is textured."""

    rgb: np.ndarray  # float32
    grey: np.ndarray
    rows: list[Line]
    columns: list[Line]
    spread: np.ndarray  # see _spread
    photo: np.ndarray

    @classmethod
    def of(cls, image: Image.Image) -> "_Scan":
        small = image.copy()
        small.thumbnail((FRAME_SIDE, FRAME_SIDE))  # before converting: the copy is small
        rgb = np.asarray(small.convert("RGB"))
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rows, columns = _lines(cv2.Canny(cv2.GaussianBlur(grey, (0, 0), 1.5), 10, 30))
        spread = _spread(grey)
        return cls(rgb.astype(np.float32), grey, rows, columns, spread, _textured(spread))

    def pair(self) -> Box | None:
        """The left view, when the scan is a stereoscopic pair."""
        if not self.photo.any():
            return None
        return _stereo(self.grey, _framed(self.rows, self.columns, *self.grey.shape), self.photo)


# What makes a crop doubtful, to be checked by hand (measured on 43 scans rated by eye: 6 of
# its 12 bad crops raised a doubt, 2 of its 31 good ones did).
SIDE_BAND = 0.04  # a crop's side is read on a band this deep, as a share of the crop
BARE_BAND = 0.15  # a band with less texture than this share is bare paper
GOES_ON = 0.35  # a band beyond the crop with this share of the photo: the print is cut
PRINTS = 0.03  # a patch of photo this large (share of the scan) may be a print of its own
PRINTS_APART = 0.015  # patches nearer than this share of the scan are one print
SIDES = ("gauche", "haut", "droite", "bas")


def _doubts(scan: _Scan, frame: Crop | None) -> list[str]:
    """Why the crop `frame` (None: the whole scan) may be wrong, in words for the review:
    several prints on the page; card inside the frame; the print cut."""
    h, w = scan.grey.shape
    left, top, right, bottom = frame or (0.0, 0.0, 1.0, 1.0)
    x0, y0, x1, y1 = round(left * w), round(top * h), round(right * w), round(bottom * h)
    found = []

    k = round(PRINTS_APART * max(h, w))
    closed = cv2.morphologyEx(
        scan.photo.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    patches = [stats[c, :4] for c in range(1, n) if stats[c, cv2.CC_STAT_AREA] >= PRINTS * h * w]
    kept = [
        (x, y, pw, ph)
        for x, y, pw, ph in patches
        if x >= x0 - 3 and y >= y0 - 3 and x + pw <= x1 + 3 and y + ph <= y1 + 3
    ]
    if (len(patches) > len(kept) and len(patches) >= 2) or len(kept) >= 3:
        found.append("plusieurs tirages sur la page")
    if frame is None:
        return found

    textured = scan.spread > max(TEXTURE, 1.5 * float(np.median(scan.spread)))
    around = np.zeros((h, w), bool)  # the ring just outside the frame, as _on_card reads it
    ring_x, ring_y = int(RING * w), int(RING * h)
    around[max(0, y0 - ring_y) : y1 + ring_y, max(0, x0 - ring_x) : x1 + ring_x] = True
    around[y0:y1, x0:x1] = False
    card = np.median(scan.rgb[around], axis=0) if around.any() else None
    dx, dy = max(2, round(SIDE_BAND * (x1 - x0))), max(2, round(SIDE_BAND * (y1 - y0)))
    bare, cut = [], []
    for side, inner, outer in zip(
        SIDES,
        (np.s_[y0:y1, x0 : x0 + dx], np.s_[y0 : y0 + dy, x0:x1])
        + (np.s_[y0:y1, x1 - dx : x1], np.s_[y1 - dy : y1, x0:x1]),
        (np.s_[y0:y1, max(0, x0 - dx) : x0], np.s_[max(0, y0 - dy) : y0, x0:x1])
        + (np.s_[y0:y1, x1 : x1 + dx], np.s_[y1 : y1 + dy, x0:x1]),
        strict=True,
    ):
        colour = np.median(scan.rgb[inner].reshape(-1, 3), axis=0)
        if (
            card is not None
            and textured[inner].mean() < BARE_BAND
            and np.linalg.norm(colour - card) < CARD_RGB
        ):
            bare.append(side)
        if scan.photo[outer].size and scan.photo[outer].mean() > GOES_ON:
            cut.append(side)
    if bare:
        found.append("carton dans le cadre : " + ", ".join(bare))
    if cut:
        found.append("tirage coupé : " + ", ".join(cut))
    return found


def apply(image: Image.Image, frame: Crop) -> Image.Image:
    """The part of the image inside a frame given as fractions of it."""
    width, height = image.size
    left, top, right, bottom = frame
    return image.crop((left * width, top * height, right * width, bottom * height))
