"""The placing page: a local page where the user places each photo, on the map (where the
photographer stood, which way they looked), then in its nearest panorama (the proposal checked,
or points clicked in pairs), or says that no panorama sees it. Answers go to
data/overrides/placings.jsonl (see placings). Local use only (see localpage)."""

import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from http.server import HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import cv2
import numpy as np
from shapely.geometry import Point

from flaneur import localpage, match, placings, queueing, registration
from flaneur.gazetteer import to_degrees, to_metres
from flaneur.registration import ViewOf

PAGE = Path(__file__).with_name("placing_page.html")
NEAR_M = queueing.NEAR_M  # the panoramas offered around a photo's spot (where proposals are made)
SIDE = match.VIEW_SIDE
KEPT_PICTURES = 8  # 360° pictures kept decoded (6 MB each)


@dataclass
class Sources:
    photos: dict[str, dict]  # id: title, years, lon, lat, heading, image (its file), inputs
    spheres: list[match.Panorama]  # levelled
    placings: Path
    # (photo, panorama): {registration, carried_from} or None: proposed, not carried
    proposal: Callable[[str, str], dict | None]
    # the photos to come first, with why, in order (asked after each answer)
    ahead: Callable[[], dict[str, str]] = dict
    graph: dict[str, set[str]] = field(default_factory=dict)  # old neighbours, by photo
    # the photos proposed from the placed ones: {registration, carried_from}, by photo
    carried: Callable[[], dict[str, dict]] = dict


class Around:
    """The panoramas near a spot, found at once among all of them."""

    def __init__(self, spheres: list[match.Panorama]):
        self.spheres = spheres
        self.at = np.array([(p.x, p.y) for p in spheres]).reshape(-1, 2)

    def near(
        self, x: float, y: float, radius: float = NEAR_M
    ) -> list[tuple[match.Panorama, float]]:
        """Those within `radius` metres of (x, y), nearest first, with how far."""
        metres = np.hypot(*(self.at - (x, y)).T)
        return [
            (self.spheres[i], float(metres[i])) for i in np.argsort(metres) if metres[i] <= radius
        ]


def settings() -> dict:
    """What the page shows the user by the server's rules."""
    return {
        "min_points": registration.MIN_POINTS,
        "view_step": 360 // len(match.VIEW_HEADINGS),
        "near_m": NEAR_M,
    }


def _metres(lon: float, lat: float) -> tuple[float, float]:
    here = to_metres(Point(lon, lat))
    return here.x, here.y


def _shown(photo: dict) -> dict:
    """A photo as the page gets it (not its file nor its image's hash)."""
    return {k: v for k, v in photo.items() if k not in ("image", "inputs")}


def carried(sources: Sources) -> dict[str, dict]:
    """The photos proposed from those placed; none when making them failed (the page goes on)."""
    try:
        return sources.carried()
    except Exception:  # a pair or a feature kept wrong: the queue and the other proposals go on
        traceback.print_exc(file=sys.stderr)
        return {}


def next_photo(sources: Sources, nearest: dict[str, float], where: dict) -> dict:
    """The next photo to place, in the order of queueing.order (`nearest`: metres to the
    nearest panorama, `where`: x, y, by photo), how many are left, and the photos placed (to
    reopen from the map)."""
    held, skipped, answers = placings.state(sources.placings)
    # placed on an image the photo no longer has (cropped again): to see again, first
    stale = {
        i
        for i, r in held.items()
        if r["answer"] in placings.PLACED
        and r["inputs"] != sources.photos.get(i, {}).get("inputs", "")
    }
    held = {i: r for i, r in held.items() if i not in stale}
    found = queueing.order(
        list(sources.photos),
        where=where,
        held={
            i: {"answer": r["answer"], **dict(zip("xy", _metres(r["lon"], r["lat"]), strict=True))}
            for i, r in held.items()
        },
        skipped=skipped,
        ahead={**{i: "revoir" for i in sorted(stale)}, **sources.ahead()},
        graph=sources.graph,
        nearest_m=nearest,
        answered=sum(i in sources.photos for i in answers),
    )
    head = found and {"id": found[0][0], **_shown(sources.photos[found[0][0]]), "why": found[0][1]}
    proposed = head and carried(sources).get(head["id"])
    if proposed:  # opened where its source was placed, on the panorama it was carried onto
        head.update(
            registration=registration.to_json(proposed["registration"]),
            answer="fits",
            carried_from=proposed["carried_from"],
            **{k: proposed[k] for k in ("lon", "lat", "heading")},
        )
    placed = [
        {"id": i, "answer": r["answer"], "lon": r["lon"], "lat": r["lat"]}
        for i, r in held.items()
        if i in sources.photos
    ]
    return {"photo": head or None, "left": len(found), "placed": placed}


def reopened(sources: Sources, photo_id: str) -> dict:
    """A photo answered, as the page shows it again: its answer's place and registration."""
    row = placings.holding(sources.placings)[photo_id]
    same_image = row["inputs"] == sources.photos[photo_id].get("inputs", "")
    return {
        "id": photo_id,
        **_shown(sources.photos[photo_id]),
        **{k: row[k] for k in ("lon", "lat", "heading", "answer", "carried_from", "flipped")},
        "registration": row["registration"] if same_image else None,  # made on another image
    }


def _jpeg(image: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def serve(sources: Sources, port: int) -> HTTPServer:
    spheres = {p.id: p for p in sources.spheres}
    around = Around(sources.spheres)
    where = {i: _metres(p["lon"], p["lat"]) for i, p in sources.photos.items()}
    nearest = {}
    for photo_id, (x, y) in where.items():
        found = around.near(x, y)
        nearest[photo_id] = found[0][1] if found else np.inf
    proposals: dict[tuple[str, str], dict | None] = {}  # made once per photo and panorama
    matching = threading.Lock()  # one proposal at a time (the matcher and its caches)

    def near(lon: float, lat: float) -> list[dict]:
        here = to_metres(Point(lon, lat))
        found = []
        for p, metres in around.near(here.x, here.y):
            where = to_degrees(Point(p.x, p.y))
            found.append({"id": p.id, "lon": where.x, "lat": where.y, "metres": round(metres, 1)})
        return found

    def proposal(photo_id: str, panorama: str) -> dict | None:
        """Carried from a photo placed (as the answers stand now), else made once; none when
        making it failed (the user places by points)."""
        found = carried(sources).get(photo_id)
        if found and found["registration"].panorama == panorama:
            return {
                "registration": registration.to_json(found["registration"]),
                "carried_from": found["carried_from"],
            }
        with matching:
            if (photo_id, panorama) not in proposals:
                try:
                    found = sources.proposal(photo_id, panorama)
                except Exception:  # the matcher, its models, memory: the page goes on
                    traceback.print_exc(file=sys.stderr)
                    found = None
                proposals[photo_id, panorama] = found and {
                    "registration": registration.to_json(found["registration"]),
                    "carried_from": found["carried_from"],
                }
            return proposals[photo_id, panorama]

    @lru_cache(maxsize=KEPT_PICTURES)
    def picture(panorama: str) -> np.ndarray:
        return cv2.imread(str(spheres[panorama].path))

    def photo_image(photo_id: str, flipped: bool) -> np.ndarray:
        image = cv2.imread(str(sources.photos[photo_id]["image"]))
        return image[:, ::-1].copy() if flipped else image

    def cut(panorama: str, view: ViewOf) -> np.ndarray:
        sphere = spheres[panorama]
        return match.perspective(
            picture(panorama), sphere.axes, view.heading, view.fov, SIDE, view.pitch
        )

    def known(body: dict) -> tuple[str, match.Panorama | None]:
        photo_id = body.get("photo_id")
        if photo_id not in sources.photos:
            raise ValueError(f"photo inconnue : {photo_id}")
        panorama = body.get("panorama")
        if panorama is not None and panorama not in spheres:
            raise ValueError(f"panorama inconnu : {panorama}")
        return photo_id, spheres.get(panorama)

    class Handler(localpage.Handler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
            url = urlsplit(self.path)
            route, ask = unquote(url.path), parse_qs(url.query)
            parts = route.strip("/").split("/")
            try:
                if route == "/":
                    self._send(200, localpage.page(PAGE), "text/html; charset=utf-8")
                elif route == "/api/queue":
                    self._json({"settings": settings(), **next_photo(sources, nearest, where)})
                elif parts[:2] == ["api", "reopen"] and len(parts) == 3:
                    self._json(reopened(sources, parts[2]))
                elif route == "/api/near":
                    self._json(near(float(ask["lon"][0]), float(ask["lat"][0])))
                elif parts[:2] == ["api", "proposal"] and len(parts) == 4:
                    self._json(proposal(parts[2], parts[3]))
                elif parts[0] == "photo" and len(parts) == 2 and parts[1] in sources.photos:
                    flipped = ask.get("flipped", ["0"])[0] == "1"
                    self._send(200, _jpeg(photo_image(parts[1], flipped)), "image/jpeg")
                elif parts[0] == "view" and len(parts) == 2 and parts[1] in spheres:
                    view = ViewOf(
                        float(ask["heading"][0]),
                        float(ask.get("pitch", [match.VIEW_PITCH])[0]),
                        float(ask.get("fov", [match.VIEW_FOV])[0]),
                    )
                    self._send(200, _jpeg(cut(parts[1], view)), "image/jpeg")
                else:
                    self._send(404, b"not found", "text/plain")
            except (KeyError, ValueError) as error:
                self._send(400, str(error).encode(), "text/plain; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._body()
                photo_id, sphere = known(body)
                if self.path == "/api/answer":
                    found = body.get("registration")
                    found = found and registration.from_json(found)
                    if found and found.panorama not in spheres:
                        raise ValueError(f"panorama inconnu : {found.panorama}")
                    self._json(
                        placings.record(
                            sources.placings,
                            photo_id,
                            body.get("answer"),
                            body.get("seconds"),
                            lon=body.get("lon"),
                            lat=body.get("lat"),
                            heading=body.get("heading"),
                            registration=found,
                            flipped=body.get("flipped", False),
                            carried_from=body.get("carried_from"),
                            inputs=sources.photos[photo_id].get("inputs", ""),
                        )
                    )
                elif self.path == "/api/fit":
                    if sphere is None:
                        raise ValueError("le panorama où poser la photo manque")
                    image = photo_image(photo_id, bool(body.get("flipped")))
                    fitted, _ = registration.fit(
                        np.array(body["photo_px"], float),
                        np.array(body["view_px"], float),
                        image.shape[:2],
                        SIDE,
                        sphere.id,
                        ViewOf(**body["view"]),
                        sphere.axes,
                    )
                    self._json(registration.to_json(fitted))
                elif self.path == "/api/composite":
                    laid = registration.from_json(body["registration"])
                    image = photo_image(photo_id, bool(body.get("flipped")))
                    axes = spheres[laid.panorama].axes
                    view = cut(laid.panorama, laid.view)
                    self._send(
                        200,
                        _jpeg(registration.composite(view, image, laid, laid.view, axes)),
                        "image/jpeg",
                    )
                else:
                    self._send(404, b"not found", "text/plain")
            except (ValueError, KeyError, TypeError) as error:
                self._send(400, str(error).encode(), "text/plain; charset=utf-8")

    return localpage.server(Handler, port, threaded=True)
