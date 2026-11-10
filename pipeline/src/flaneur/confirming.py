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
# Why a placement is wrong, said with a "no" (the guards learn from them): a reproduction of
# the photo in today's street (a panel, a poster), another street, not a street view, the
# image mirrored, another reason.
REASONS = ("reproduction", "other_street", "not_street", "mirrored", "other")
FRAMES = ("old", "today", "outlines")  # the photo, today's view framed alike, the outlines on it


def record(
    path: Path,
    photo_id: str,
    verdict: str,
    inputs: str,
    seconds: float,
    camera: dict | None = None,
    reason: str | None = None,
) -> dict:
    """Append the user's verdict on a photo placed from `inputs` (see cli._inputs), with the
    camera judged (`camera`: x, y, heading): recorded so that a verdict can later be carried
    over to a new placement that hardly moves the camera (not done yet); a "no" with its
    `reason` when one was given (see REASONS)."""
    if verdict not in VERDICTS:
        raise ValueError(f"un verdict est l'un de {VERDICTS}")
    if reason is not None and (verdict != "no" or reason not in REASONS):
        raise ValueError(f"une raison va avec un « non » et est l'une de {REASONS}")
    row = {
        "photo_id": photo_id,
        "verdict": verdict,
        "inputs": inputs,
        "seconds": seconds_taken(seconds),
    }
    if camera is not None:
        row["camera"] = camera
    if reason is not None:
        row["reason"] = reason
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
    frame: Callable[[str, str], bytes | None],
    port: int,
) -> HTTPServer:
    """`frame(photo_id, part)`: one of its pictures (JPEG), all of one size: the photo
    ("old"), today's real view framed as its camera ("today"), the photo's outlines on that
    view ("outlines"); None when they cannot be drawn."""
    by_id = {photo["id"]: photo for photo in placed}

    class Handler(localpage.Handler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
            route = unquote(self.path.split("?")[0])
            parts = route.strip("/").split("/")
            if route == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/queue":
                self._json(queue(placed, verdicts()))
            elif (
                parts[0] == "frame"
                and len(parts) == 3
                and parts[1] in by_id
                and parts[2] in FRAMES
                and (jpeg := frame(parts[1], parts[2]))
            ):
                self._send(200, jpeg, "image/jpeg")
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
                        body.get("reason"),
                    )
                )
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                self._send(400, str(error).encode(), "text/plain; charset=utf-8")

    return localpage.server(Handler, port)
