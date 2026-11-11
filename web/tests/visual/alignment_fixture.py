"""Makes what tests/visual/walk3d.mjs checks the 3D walk against: a checkerboard registered on a
leaning panorama of the pilot, and the pipeline's own drawing of it in a view; written to the
folder given (nothing in the site's data). Run with the pipeline's environment:
    cd pipeline && uv run python ../web/tests/visual/alignment_fixture.py <folder> [fov]
The view is drawn as the 3D walk opens it: `fov` must be the page's own (72°, see OPENING_FOV);
another draws the board at another size and the check fails though the page is right."""

import json
import shutil
import sys
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from flaneur import cli, db, match, panoramas, registration

# The 3D walk's opening view: Photo Sphere Viewer's defaultZoomLvl 30 (web/src/sphere/main3d.ts)
# between its default widest 90° and narrowest 30°: 90 - 0.30 * (90 - 30).
OPENING_FOV = 72.0
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
h = np.array([[0.45, 0.04, 0.22], [-0.03, 0.4, 0.3], [0.0002, 0.0003, 1.0]])
spread = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9], [0.5, 0.4], [0.3, 0.7]])
side = 640
r, _ = registration.fit(
    spread * [800, 600], cv2.perspectiveTransform(spread[None], h)[0] * side, (600, 800), side,
    p.id, registration.ViewOf(heading=140.0), axes,
)
seen = registration.ViewOf(heading=r.view.heading, pitch=0.0, fov=fov)  # as "En 3D" opens it
view = match.perspective(equirect, axes, seen.heading, seen.fov, side, seen.pitch)
cv2.imwrite(str(out / "expected.png"), registration.composite(view, board, r, seen, axes))
json.dump(
    {"panorama": p.id, "lon": p.lon, "lat": p.lat, "heading": seen.heading,
     "registration": registration.to_json(r)},
    (out / "fixture.json").open("w"),
)
print(out)
