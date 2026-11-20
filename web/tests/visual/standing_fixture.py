"""Makes what tests/visual/walk3d.mjs checks a photo laid on the walls against: a checkerboard
taken by a camera standing 2 m off a leaning panorama of the pilot, on a wall across the street,
and the pipeline's own drawing of it (registration.composite_standing) in a view; written to the
folder given (nothing in the site's data). Run with the pipeline's environment:
    cd pipeline && uv run python ../web/tests/visual/standing_fixture.py <folder> [fov]"""

import json
import shutil
import sys
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from flaneur import cli, db, facades, match, panoramas, registration

OPENING_FOV = 72.0  # see alignment_fixture.py
out = Path(sys.argv[1])
fov = float(sys.argv[2]) if len(sys.argv) > 2 else OPENING_FOV
out.mkdir(parents=True, exist_ok=True)
with closing(db.connect(cli.DB_PATH)) as conn:
    pictures = cli._lined_up(panoramas.load(conn))
p = next(q for q in pictures if q.id == "cf6f0374-23bf-4944-95ac-399b90d77007")  # leans 7°, 8°
axes = match.camera_axes(p.azimuth or 0.0, p.pitch, p.roll)
equirect = cv2.imread(str(cli.PANORAMAS / f"{p.id}.jpg"))
shutil.copy(cli.PANORAMAS / f"{p.id}.jpg", out / "panorama.jpg")

board = (np.indices((600, 800)).sum(0) // 50 % 2 * 255).astype(np.uint8)
board = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
board[:, :, 2] = 255  # red and white
cv2.imwrite(str(out / "board.png"), board)

heading = 140.0
ahead = np.array([np.sin(np.radians(heading)), np.cos(np.radians(heading)), 0.0])
right = np.array([ahead[1], -ahead[0], 0.0])
# a facade across the street 12 m ahead, 30 m wide, its top 15 m over the panorama; the street
# 2.5 m under it
walls = [
    facades.Wall(
        tuple((12 * ahead - 15 * right)[:2]), tuple((12 * ahead + 15 * right)[:2]), 15.0, "fitted"
    )
]
scene = facades.Scene(p.id, (0.0, 0.0, 0.0), -2.5, tuple(walls), (), 8)
# the photographer 2 m right of the panorama and 1 m back, 0.8 m lower, facing the facade a
# little to the left of straight on
position = 2.0 * right - 1.0 * ahead + [0.0, 0.0, -0.8]
target = 12 * ahead - 1.0 * right + [0.0, 0.0, 2.0]
z = (target - position) / np.linalg.norm(target - position)
x = np.cross(z, [0.0, 0.0, 1.0])
x /= np.linalg.norm(x)
y = np.cross(z, x)
turn = cv2.Rodrigues(np.array([x, y, z]))[0].ravel()
s = registration.Standing(tuple(position), tuple(turn), 1200.0, (0.02, -0.04), (600, 800))

side = 640
seen = registration.ViewOf(heading=heading, pitch=0.0, fov=fov)  # as "En 3D" opens it
view = match.perspective(equirect, axes, seen.heading, seen.fov, side, seen.pitch)
metres = facades.in_view(scene, seen, side).astype(np.float32)
drawn = registration.composite_standing(view, board, s, np.zeros(3), seen, metres)
cv2.imwrite(str(out / "expected.png"), drawn)
# the flat fit the page opens on (where "En 3D" faces), through points of the board on the wall
spread = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.5, 0.5], [0.3, 0.6]]) * [
    800,
    600,
]
rays = s.rays(spread)
points = []
for ray in rays:
    t = facades.cast(facades.Scene(p.id, tuple(position), -2.5, tuple(walls), (), 8), ray[None])[0][
        0
    ]
    points.append(position + t * ray)
points = np.array(points)
directions = points / np.linalg.norm(points, axis=1, keepdims=True)
view_px = registration.view_pixels(directions @ axes.T, seen, axes, side)
r, _ = registration.fit(spread, view_px, (600, 800), side, p.id, seen, axes)
json.dump(
    {
        "panorama": p.id,
        "lon": p.lon,
        "lat": p.lat,
        "heading": seen.heading,
        "registration": registration.to_json(r),
        "standing": {
            **registration.standing_to_json(s),
            "position": [round(v, 3) for v in position],
        },
        "walls": [[round(v, 2) for v in (*w.a, *w.b, w.top)] for w in walls],
        "ground": -2.5,
    },
    (out / "fixture.json").open("w"),
    default=list,
)
print(out, int(((drawn[:, :, 2] > 200) & (drawn[:, :, 0] < 90)).sum()), "red pixels drawn")
