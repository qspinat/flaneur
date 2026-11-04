"""The landmark marking tool: a local page where the user marks, on each photo as the matcher
sees it (cropped), the landmarks it shows: which one, its column and its top. Written to
data/overrides/sights.jsonl (versioned with the code, see landmarks.rows). Local use only."""

import json
import math
from collections.abc import Callable, Sequence
from http.server import HTTPServer
from pathlib import Path
from urllib.parse import unquote

from flaneur import landmarks, localpage
from flaneur.sources import write_text

PAGE = Path(__file__).with_name("marking_page.html")


def by_photo(path: Path) -> dict[str, list[dict]]:
    """The rows of the marks file, by photo."""
    found: dict[str, list[dict]] = {}
    for row in landmarks.rows(path):
        found.setdefault(row["photo_id"], []).append(row)
    return found


def checked(rows: list[dict]) -> bool:
    """Whether the user saved a photo's marks (or said it shows none)."""
    return all(row.get("checked") for row in rows)


def _write(path: Path, rows: list[dict]) -> None:
    write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def save(path: Path, photo_id: str, image: str, marks: list[dict]) -> list[dict]:
    """A photo's marks, replacing those it had (none: checked, it shows no landmark)."""
    if not isinstance(marks, list):
        raise ValueError("les repères sont une liste")
    base = {"photo_id": photo_id, "image": image, "checked": True}
    new = [{**base, **landmarks.mark(m)} for m in marks] or [{**base, "none": True}]
    _write(path, [row for row in landmarks.rows(path) if row["photo_id"] != photo_id] + new)
    return new


def propose(path: Path, photo_id: str, image: str, marks: list[dict]) -> bool:
    """Marks proposed for a photo, for the user to check (not `checked`), replacing those
    proposed before (none: they go); never over the user's own. Whether they were written."""
    if checked(rows := by_photo(path).get(photo_id, [])) and rows:
        return False
    kept = [row for row in landmarks.rows(path) if row["photo_id"] != photo_id]
    _write(
        path, kept + [{"photo_id": photo_id, "image": image, **landmarks.mark(m)} for m in marks]
    )
    return True


def offered(place: tuple[float, float], years: Sequence[int | None]) -> list[str]:
    """The landmarks to offer for a photo: those standing in its `years`, nearest its `place`
    first (the page puts the others apart)."""
    standing = [key for key, lm in landmarks.LANDMARKS.items() if lm.stands(years)]
    return sorted(standing, key=lambda key: math.dist(landmarks.LANDMARKS[key].at, place))


def queue(photos: list[dict], sights_path: Path, image_of: Callable[[str], Path]) -> list[dict]:
    """The photos to mark (`photos`: id, title, years and place, Lambert-93; their image
    exported): those with marks proposed, to check, first; then those not yet looked
    at; then those the user saved (marks, or none); each with its marks, whether they were
    made on its image as it is now, and the landmarks to offer (see offered)."""
    rows = by_photo(sights_path)
    items = []
    for photo in photos:
        marked = rows.get(photo["id"], [])
        state = "new" if not marked else "done" if checked(marked) else "check"
        now = landmarks.image_hash(image_of(photo["id"])) if marked else None
        items.append(
            {
                **photo,
                "offer": offered(photo["place"], photo["years"]),
                "state": state,
                "stale": any(r["image"] != now for r in marked),
                "marks": [r for r in marked if not r.get("none")],
            }
        )
    order = {"check": 0, "new": 1, "done": 2}
    return sorted(items, key=lambda item: order[item["state"]])


def serve(
    photos: list[dict], sights_path: Path, image_of: Callable[[str], Path], port: int
) -> HTTPServer:
    ids = {photo["id"] for photo in photos}
    catalogue = {
        "landmarks": [[key, landmark.name] for key, landmark in landmarks.LANDMARKS.items()],
        "spires": list(landmarks.SPIRES),
    }

    class Handler(localpage.Handler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
            path = unquote(self.path.split("?")[0])  # a photo's id: its colon comes quoted
            if path == "/":
                self._send(200, localpage.page(PAGE), "text/html; charset=utf-8")
            elif path == "/api/queue":
                try:
                    self._json({**catalogue, "photos": queue(photos, sights_path, image_of)})
                except ValueError as error:  # the marks file written wrong: said, not dropped
                    self._send(500, str(error).encode(), "text/plain; charset=utf-8")
            elif path.startswith("/image/") and (photo_id := path.removeprefix("/image/")) in ids:
                self._send(200, image_of(photo_id).read_bytes(), "image/webp")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/marks":
                self._send(404, b"not found", "text/plain")
                return
            try:
                body = self._body()
                if body["photo_id"] not in ids:
                    raise ValueError(f"photo inconnue : {body['photo_id']}")
                image = landmarks.image_hash(image_of(body["photo_id"]))
                self._json(save(sights_path, body["photo_id"], image, body["marks"]))
            except (ValueError, KeyError, TypeError) as error:
                self._send(400, str(error).encode(), "text/plain; charset=utf-8")

    return localpage.server(Handler, port)
