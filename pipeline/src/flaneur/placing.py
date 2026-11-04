"""The placing page: a local page where the user places each photo, on the map (where the
photographer stood, which way they looked), then in its nearest panorama (the proposal checked,
or points clicked in pairs), or says that no panorama sees it. Answers go to
data/overrides/placings.jsonl (see placings). Local use only (see localpage)."""

import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from http.server import HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import cv2
import numpy as np
from shapely.geometry import Point

from flaneur import localpage, match, placings, registration
from flaneur.gazetteer import to_degrees, to_metres
from flaneur.registration import ViewOf

PAGE = Path(__file__).with_name("placing_page.html")
NEAR_M = 40  # the panoramas offered around a photo's spot (and where proposals are made)
SIDE = match.VIEW_SIDE
KEPT_PICTURES = 8  # 360° pictures kept decoded (6 MB each)


@dataclass
class Sources:
    photos: dict[str, dict]  # id: title, years, lon, lat, heading, image (its file), inputs
    spheres: list[match.Panorama]  # levelled
    placings: Path
    # (photo, panorama): {registration, carried_from} or None
    proposal: Callable[[str, str], dict | None]


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


def queue(sources: Sources, nearest: dict[str, float]) -> list[dict]:
    """The photos not answered yet, those with a panorama nearest first (`nearest`: metres, by
    photo), then those skipped."""
    held, skipped = placings.holding(sources.placings), placings.skipped(sources.placings)
    items = [
        {"id": photo_id, **{k: v for k, v in photo.items() if k != "image"}}
        for photo_id, photo in sources.photos.items()
        if photo_id not in held
    ]
    return sorted(items, key=lambda item: (item["id"] in skipped, nearest[item["id"]]))


def _jpeg(image: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def serve(sources: Sources, port: int) -> HTTPServer:
    spheres = {p.id: p for p in sources.spheres}
    around = Around(sources.spheres)
    nearest = {}
    for photo_id, photo in sources.photos.items():
        here = to_metres(Point(photo["lon"], photo["lat"]))
        found = around.near(here.x, here.y)
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
        """The proposal, made once; none when making it failed (the user places by points)."""
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
                    self._json({"settings": settings(), "photos": queue(sources, nearest)})
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
