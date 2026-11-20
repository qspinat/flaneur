"""Makes what tests/visual/walk3d.mjs checks the 3D walk against: a checkerboard registered on a
leaning panorama of the pilot, and the pipeline's own drawing of it in a view; written to the
folder given (nothing in the site's data). Run with the pipeline's environment:
    cd pipeline && uv run python ../web/tests/visual/alignment_fixture.py <folder> [fov]
`fov` defaults to the page's opening view (OPENING_FOV); another fails the check though the page
is right."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from fixture_common import OPENING_FOV, pilot

from flaneur import match, registration

out = Path(sys.argv[1])
fov = float(sys.argv[2]) if len(sys.argv) > 2 else OPENING_FOV
p, axes, equirect, board = pilot(out)
h = np.array([[0.45, 0.04, 0.22], [-0.03, 0.4, 0.3], [0.0002, 0.0003, 1.0]])
spread = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9], [0.5, 0.4], [0.3, 0.7]])
side = 640
r, _ = registration.fit(
    spread * [800, 600],
    cv2.perspectiveTransform(spread[None], h)[0] * side,
    (600, 800),
    side,
    p.id,
    registration.ViewOf(heading=140.0),
    axes,
)
seen = registration.ViewOf(heading=r.view.heading, pitch=0.0, fov=fov)  # as "En 3D" opens it
view = match.perspective(equirect, axes, seen.heading, seen.fov, side, seen.pitch)
cv2.imwrite(str(out / "expected.png"), registration.composite(view, board, r, seen, axes))
json.dump(
    {
        "panorama": p.id,
        "lon": p.lon,
        "lat": p.lat,
        "heading": seen.heading,
        "registration": registration.to_json(r),
    },
    (out / "fixture.json").open("w"),
)
print(out)
