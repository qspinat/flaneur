"""The confirmation page: a local page where the user says yes or no to each photo placed
(against today's panoramas, or chained from other old photos), seeing it beside today's true
view framed as the camera found, with its outlines drawn on that view. Verdicts are appended to
data/overrides/anchors.jsonl (versioned with the code): a "no" withdraws the photo, and the
next chaining places again, without it, the photos it helped place. A "near" (the right
buildings, the camera off) withdraws it too, kept for its camera to be set right later.
Local use only."""

from collections.abc import Callable
from http.server import HTTPServer
from pathlib import Path
from urllib.parse import unquote

from flaneur import localpage
from flaneur.sources import append_row, seconds_taken

PAGE = Path(__file__).with_name("confirming_page.html")
VERDICTS = ("yes", "near", "no")


def record(
    path: Path, photo_id: str, verdict: str, inputs: str, seconds: float, camera: dict | None = None
) -> dict:
    """Append the user's verdict on a photo placed from `inputs` (see cli._inputs), with the
    camera judged (`camera`: x, y, heading): recorded so that a verdict can later be carried
    over to a new placement that hardly moves the camera (not done yet)."""
    if verdict not in VERDICTS:
        raise ValueError(f"un verdict est l'un de {VERDICTS}")
    row = {
        "photo_id": photo_id,
        "verdict": verdict,
        "inputs": inputs,
        "seconds": seconds_taken(seconds),
    }
    if camera is not None:
        row["camera"] = camera
    return append_row(path, row)


def queue(placed: list[dict], verdicts: dict[str, dict]) -> list[dict]:
    """The photos placed (`placed`: id, title, round, inputs, `doubt` if the others disagree
    with it, `by` the matcher that found it, why to `check` it closely…), those the user has not
    judged for how they are placed now first, those in doubt or to check first among them,
    round by round; each with its verdict (None: none for this placement)."""
    items = []
    for photo in placed:
        last = verdicts.get(photo["id"])
        verdict = last["verdict"] if last and last["inputs"] == photo["inputs"] else None
        items.append({**photo, "verdict": verdict})
    return sorted(
        items,
        key=lambda item: (
            item["verdict"] is not None,
            "doubt" not in item and "check" not in item,
            item["round"],
        ),
    )


def serve(
    placed: list[dict],
    path: Path,
    verdicts: Callable[[], dict[str, dict]],
    sheet: Callable[[str], bytes],
    port: int,
) -> HTTPServer:
    """`sheet(photo_id)`: its row of pictures (JPEG): the photo, today's real view framed as
    its camera, the photo's outlines on that view."""
    by_id = {photo["id"]: photo for photo in placed}

    class Handler(localpage.Handler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
            route = unquote(self.path.split("?")[0])
            if route == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/queue":
                self._json(queue(placed, verdicts()))
            elif (
                route.startswith("/sheet/") and (photo_id := route.removeprefix("/sheet/")) in by_id
            ):
                self._send(200, sheet(photo_id), "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/verdict":
                self._send(404, b"not found", "text/plain")
                return
            try:
                body = self._body()
                photo = by_id.get(body.get("photo_id"))
                if photo is None:
                    raise ValueError(f"photo inconnue : {body.get('photo_id')}")
                self._json(
                    record(
                        path,
                        photo["id"],
                        body.get("verdict"),
                        photo["inputs"],
                        body.get("seconds", 0),
                        photo.get("camera"),
                    )
                )
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                self._send(400, str(error).encode(), "text/plain; charset=utf-8")

    return localpage.server(Handler, port)
