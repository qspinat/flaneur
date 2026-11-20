"""Makes what tests/visual/walk3d.mjs checks a photo laid on the walls against: a checkerboard
taken by a camera standing 2 m off a leaning panorama of the pilot, on a wall across the street,
and the pipeline's own drawing of it (registration.composite_standing) in a view; written to the
folder given (nothing in the site's data). Run with the pipeline's environment:
    cd pipeline && uv run python ../web/tests/visual/standing_fixture.py <folder> [fov]"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from fixture_common import OPENING_FOV, pilot

from flaneur import facades, match, registration, scene

GROUND = -2.5  # the street, from the panorama
out = Path(sys.argv[1])
fov = float(sys.argv[2]) if len(sys.argv) > 2 else OPENING_FOV
p, axes, equirect, board = pilot(out)

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
around = facades.Scene(p.id, (0.0, 0.0, 0.0), GROUND, tuple(walls), (), 8)
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
metres = facades.in_view(around, seen, side).astype(np.float32)
drawn = registration.composite_standing(view, board, s, np.zeros(3), seen, metres)
cv2.imwrite(str(out / "expected.png"), drawn)
# the flat fit the page opens on (where "En 3D" faces), through points of the board on the wall
spread = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.5, 0.5], [0.3, 0.6]]) * [
    800,
    600,
]
rays = s.rays(spread)
points = position + facades.cast(replace(around, origin=tuple(position)), rays)[0][:, None] * rays
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
        "walls": scene.walls_shown(around),
        "ground": GROUND,
    },
    (out / "fixture.json").open("w"),
)
print(out, int(((drawn[:, :, 2] > 200) & (drawn[:, :, 0] < 90)).sum()), "red pixels drawn")
