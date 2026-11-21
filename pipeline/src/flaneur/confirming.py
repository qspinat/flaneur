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
from urllib.parse import parse_qs, unquote

from flaneur import localpage
from flaneur.sources import append_row, number, rows, seconds_taken

PAGE = Path(__file__).with_name("confirming_page.html")
VERDICTS = ("yes", "near", "no")
# Why a placement is wrong, said with a "no" (the guards learn from them): a reproduction of
# the photo in today's street (a panel, a poster), another street, not a street view, the
# image mirrored, another reason.
REASONS = ("reproduction", "other_street", "not_street", "mirrored", "other")
# the photo, today's view framed alike, the outlines on it, and so again where today's view sees
# what they show (the parallax of a panorama standing elsewhere set aside)
FRAMES = ("old", "today", "outlines", "parallax")


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
    camera judged (`camera`: x, y, heading, fov, horizon), which a verdict follows to a
    placement that hardly moves it (see cli._Judged); a "no" with its `reason` when one was
    given (see REASONS)."""
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


Point = tuple[float, float]


def mark_pair(path: Path, photo_id: str, inputs: str, point: object, wrong: object) -> dict:
    """Append the user's mark on a pair shown (`point`: its point's pixel on the photo, as the
    placement made from `inputs` keeps it — a mark follows its detail, whatever rows a new
    match gives it): wrong (the two details are not the same thing), or that mark taken back."""
    if not (isinstance(point, list) and len(point) == 2 and all(number(v) for v in point)):
        raise ValueError("une paire est désignée par le pixel de son point")
    if not isinstance(wrong, bool):
        raise ValueError("une paire est marquée fausse ou ne l'est plus (vrai ou faux)")
    return append_row(
        path, {"photo_id": photo_id, "inputs": inputs, "point": point, "wrong": wrong}
    )


def marked(path: Path) -> dict[tuple[str, str], set[Point]]:
    """The points whose pairs are marked wrong, by placement (photo, inputs): the last mark on
    each holds."""
    last = {(r["photo_id"], r["inputs"], tuple(r["point"])): r["wrong"] for r in rows(path)}
    found: dict[tuple[str, str], set[Point]] = {}
    for (photo_id, inputs, point), wrong in last.items():
        if wrong:
            found.setdefault((photo_id, inputs), set()).add(point)
    return found


def with_marks(scene: dict, wrong: set[Point]) -> dict:
    """A scene's pairs, each saying whether it is marked wrong (`wrong`: their points)."""
    if scene.get("pairs") is None:
        return scene
    return {
        **scene,
        "pairs": [{**p, "wrong": tuple(p["point"]) in wrong} for p in scene["pairs"]],
    }


def queue(placed: list[dict], verdicts: dict[str, dict]) -> list[dict]:
    """The photos placed (`placed`: id, title, round, inputs, `doubt` if the others disagree
    with it, `by` the matcher that found it, why to `check` it closely…), those the user has not
    judged for how they are placed now first, those in doubt or to check first among them,
    round by round; each with its verdict (`verdicts`: the one that holds for each placement as
    it is, see cli._Judged; none: not judged)."""
    items = []
    for photo in placed:
        now = verdicts.get(photo["id"]) or {}
        items.append({**photo, "verdict": now.get("verdict"), "reason": now.get("reason")})
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
    marks: Path,
    verdicts: Callable[[], dict[str, dict]],
    frame: Callable[..., bytes | None],
    scene: Callable[..., dict | None],
    port: int,
) -> HTTPServer:
    """`frame(photo_id, part)`: one of its pictures (JPEG), all of one size: the photo
    ("old"), today's real view framed as its camera ("today"), the photo's outlines on that
    view ("outlines"), and drawn where that view sees what they show ("parallax"); None when they
    cannot be drawn. `scene(photo_id)`: what the page draws over them (the pairs its camera was
    found from, the plan around it); None when there is none. Asked with `?from=source`, both
    are given `source=True`: today's view from the panorama the camera was found from, not the
    one next to it. The pairs it gives say which the user marked wrong (`marks`, read anew:
    see mark_pair)."""
    by_id = {photo["id"]: photo for photo in placed}

    def marks_on(photo: dict) -> set[Point]:
        try:
            return marked(marks).get((photo["id"], photo["inputs"]), set())
        except (ValueError, KeyError, TypeError):  # the file written wrong: none shown
            return set()

    def pair_shown(photo: dict, index: object) -> dict:
        """The pair the page showed as `index` (in today's view from either panorama)."""
        for source in (False, True):
            for pair in (scene(photo["id"], source=source) or {}).get("pairs") or []:
                if pair["index"] == index:
                    return pair
        raise ValueError(f"paire inconnue : {index}")

    writes = {  # a verdict on the photo, or a mark on one of its pairs (not a verdict)
        "/api/verdict": lambda photo, body: record(
            path,
            photo["id"],
            body.get("verdict"),
            photo["inputs"],
            body.get("seconds", 0),
            photo.get("camera"),
            body.get("reason"),
        ),
        "/api/pair": lambda photo, body: mark_pair(
            marks,
            photo["id"],
            photo["inputs"],
            pair_shown(photo, body.get("index"))["point"],
            body.get("wrong"),
        ),
    }

    class Handler(localpage.Handler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
            path, _, query = self.path.partition("?")
            route = unquote(path)
            source = parse_qs(query).get("from") == ["source"]
            parts = [unquote(p) for p in path.strip("/").split("/")]  # ids may hold "/"
            if route == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/queue":
                self._json(queue(placed, verdicts()))
            elif route == "/api/reasons":
                self._json(list(REASONS))
            elif (
                parts[0] == "frame"
                and len(parts) == 3
                and parts[1] in by_id
                and parts[2] in FRAMES
                and (jpeg := frame(parts[1], parts[2], source=source))
            ):
                self._send(200, jpeg, "image/jpeg")
            elif (
                parts[0] == "scene"
                and len(parts) == 2
                and parts[1] in by_id
                and (found := scene(parts[1], source=source)) is not None
            ):
                self._json(with_marks(found, marks_on(by_id[parts[1]])))
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in writes:
                self._send(404, b"not found", "text/plain")
                return
            try:
                body = self._body()
                photo = by_id.get(body.get("photo_id"))
                if photo is None:
                    raise ValueError(f"photo inconnue : {body.get('photo_id')}")
                self._json(writes[self.path](photo, body))
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                self._send(400, str(error).encode(), "text/plain; charset=utf-8")

    return localpage.server(Handler, port)
