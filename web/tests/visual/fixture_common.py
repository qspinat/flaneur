"""What the 3D walk's checks in a browser share (alignment_fixture.py, standing_fixture.py): the
pilot's leaning panorama, copied, and the red and white board laid in it, written to `out`."""

import shutil
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from flaneur import cli, db, match, panoramas

# The 3D walk's opening view: Photo Sphere Viewer's defaultZoomLvl 30 (web/src/sphere/main3d.ts)
# between its default widest 90° and narrowest 30°: 90 - 0.30 * (90 - 30).
OPENING_FOV = 72.0
LEANING = "cf6f0374-23bf-4944-95ac-399b90d77007"  # leans 7°, 8°


def pilot(out: Path):
    """The leaning panorama (as lined up), its axes and picture, and the board (600 × 800)."""
    out.mkdir(parents=True, exist_ok=True)
    with closing(db.connect(cli.DB_PATH)) as conn:
        pictures = cli._lined_up(panoramas.load(conn))
    p = next(q for q in pictures if q.id == LEANING)
    axes = match.camera_axes(p.azimuth or 0.0, p.pitch, p.roll)
    equirect = cv2.imread(str(cli.PANORAMAS / f"{p.id}.jpg"))
    shutil.copy(cli.PANORAMAS / f"{p.id}.jpg", out / "panorama.jpg")
    board = (np.indices((600, 800)).sum(0) // 50 % 2 * 255).astype(np.uint8)
    board = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    board[:, :, 2] = 255  # red and white
    cv2.imwrite(str(out / "board.png"), board)
    return p, axes, equirect, board
