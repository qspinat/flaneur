"""The placing page: each photo on the map, then in its panorama."""

import dataclasses
import json

import cv2
import numpy as np
import pytest
from shapely.geometry import Point

from flaneur import depth, depthmaps, facades, match, placing, placings, registration
from flaneur.gazetteer import to_degrees, to_metres
from flaneur.registration import ViewOf

SIDE = 640
H = np.array([[0.6, 0.05, 0.2], [-0.02, 0.55, 0.25], [0.0001, 0.0002, 1.0]])
SPREAD = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9], [0.5, 0.4], [0.3, 0.7]])


@pytest.fixture
def page(tmp_path):
    picture = tmp_path / "p1.jpg"
    cv2.imwrite(str(picture), np.random.default_rng(0).integers(0, 255, (256, 512, 3), np.uint8))
    photo = tmp_path / "pm_1.jpg"
    cv2.imwrite(str(photo), np.full((60, 80, 3), 200, np.uint8))
    sphere = match.Panorama("p1", 652_058.0, 6_861_530.0, 0.0, 198.0, picture, 6.9, -8.4)
    photos = {
        "pm:1": {
            "title": "Rue",
            "years": [1900],
            "lon": 2.3466,
            "lat": 48.852,
            "heading": 40.0,
            "image": photo,
        }
    }
    proposal = registration.fit(
        SPREAD * [80, 60],
        cv2.perspectiveTransform(SPREAD[None], H)[0] * SIDE,
        (60, 80),
        SIDE,
        "p1",
        ViewOf(heading=140.0),
        sphere.axes,
    )[0]
    sources = placing.Sources(
        photos,
        [sphere],
        tmp_path / "placings.jsonl",
        lambda *_: {"registration": proposal, "carried_from": None},
    )
    by_id = lambda i: next(p for p in sources.spheres if p.id == i)  # noqa: E731
    sources.pair = lambda home, other: depth.Pair(np.eye(3), placing._at(by_id(other)))  # as placed
    return sources, proposal


@pytest.fixture
def one(ask):
    """One request to the placing page, its server started and closed around it."""
    return lambda sources, method, path, body=None: ask(
        placing.serve(sources, 0), (method, path, body)
    )[0]


ANSWER = {
    "photo_id": "pm:1",
    "answer": "far",
    "seconds": 4.2,
    "lon": 2.3466,
    "lat": 48.852,
    "heading": 40.0,
}


def test_an_answer_is_recorded_with_its_seconds(page, one):
    sources, _ = page

    status, _ = one(sources, "POST", "/api/answer", ANSWER)

    assert status == 200
    assert placings.holding(sources.placings)["pm:1"]["seconds"] == 4.2


def test_a_photo_accepted_is_recorded_with_its_registration(page, one):
    sources, proposal = page
    body = {**ANSWER, "answer": "fits", "registration": registration.to_json(proposal)}

    status, _ = one(sources, "POST", "/api/answer", body)

    assert status == 200
    assert placings.registration_of(placings.holding(sources.placings)["pm:1"]) == proposal
    assert placings.holding(sources.placings)["pm:1"]["panorama_at"] == [
        652_058.0,
        6_861_530.0,
        198.0,
    ]


def test_a_fit_without_a_registration_is_refused_with_the_reason(page, one):
    sources, _ = page

    status, reason = one(sources, "POST", "/api/answer", {**ANSWER, "answer": "fits"})

    assert status == 400 and "enregistrement" in reason.decode()


def test_points_bunched_are_refused_with_the_reason(page, one):
    sources, _ = page
    bunched = (SPREAD * 0.05).tolist()
    body = {
        "photo_id": "pm:1",
        "panorama": "p1",
        "view": {"heading": 140.0},
        "photo_px": [[x * 80, y * 60] for x, y in bunched],
        "view_px": [[x * placing.SHOWN, y * placing.SHOWN] for x, y in bunched],
    }

    status, reason = one(sources, "POST", "/api/fit", body)

    assert status == 400 and "groupés" in reason.decode()


def test_points_clicked_give_the_registration_back(page, one):
    sources, proposal = page
    seen = cv2.perspectiveTransform(SPREAD[None], H)[0]
    body = {
        "photo_id": "pm:1",
        "panorama": "p1",
        "view": {"heading": 140.0},
        "photo_px": (SPREAD * [80, 60]).tolist(),
        "view_px": (seen * placing.SHOWN).tolist(),  # clicked in the view as shown
    }

    status, answer = one(sources, "POST", "/api/fit", body)

    assert status == 200
    found = json.loads(answer)["registration"]
    assert np.allclose(registration.from_json(found).matrix, proposal.matrix)


def test_the_proposal_and_the_panoramas_near_are_given(page, one):
    sources, proposal = page

    _, near = one(sources, "GET", "/api/near?lon=2.3466&lat=48.852")
    _, given = one(sources, "GET", "/api/proposal/pm:1/p1")

    assert [p["id"] for p in json.loads(near)] == ["p1"]
    assert registration.from_json(json.loads(given)["registration"]) == proposal


def test_a_view_of_a_panorama_is_served_as_a_picture(page, one):
    sources, _ = page

    status, jpeg = one(sources, "GET", "/view/p1?heading=140&pitch=5&fov=75")

    assert status == 200
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1).shape == (placing.SHOWN, placing.SHOWN, 3)


def test_a_view_is_cut_from_the_panoramas_picture_at_full_size_when_there_is_one(page, one):
    sources, _ = page
    sources.full = lambda panorama: np.zeros((1024, 2048, 3), np.uint8)  # black, unlike its own

    _, jpeg = one(sources, "GET", "/view/p1?heading=140&pitch=5&fov=75")

    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1).max() == 0


def test_a_larger_picture_not_had_is_asked_again_and_its_own_shown_meanwhile(
    page, ask, monkeypatch
):
    sources, _ = page
    monkeypatch.setattr(placing, "RETRY_S", 0.0)  # asked again at once
    had = iter([None, np.zeros((1024, 2048, 3), np.uint8)])  # offline, then back
    sources.full = lambda panorama: next(had)
    view = ("GET", "/view/p1?heading=140&pitch=5&fov=75", None)

    (_, first), (_, second) = ask(placing.serve(sources, 0), view, view)

    assert cv2.imdecode(np.frombuffer(first, np.uint8), 1).max() > 0  # its own
    assert cv2.imdecode(np.frombuffer(second, np.uint8), 1).max() == 0


def test_the_queue_leaves_out_photos_answered_and_puts_skipped_ones_last(page, tmp_path, one):
    sources, _ = page
    sources.photos["pm:2"] = {**sources.photos["pm:1"]}
    sources.photos["pm:3"] = {**sources.photos["pm:1"]}
    placings.record(sources.placings, "pm:1", "far", 1.0, lon=2.3, lat=48.8)
    placings.record(sources.placings, "pm:2", "skip", 1.0)

    _, queue = one(sources, "GET", "/api/queue")

    assert (json.loads(queue)["photo"]["id"], json.loads(queue)["left"]) == ("pm:3", 2)


def test_the_page_is_told_the_rules_it_shows():
    assert placing.settings() == {
        "min_points": registration.MIN_POINTS,
        "min_standing": registration.MIN_STANDING,
        "view_step": 30,
        "near_m": placing.NEAR_M,
    }


def test_a_proposal_is_made_once_per_photo_and_panorama(page, ask):
    sources, proposal = page
    made = []
    sources.proposal = lambda *key: (
        made.append(key) or {"registration": proposal, "carried_from": None}
    )

    answers = ask(
        placing.serve(sources, 0),
        ("GET", "/api/proposal/pm:1/p1"),
        ("GET", "/api/proposal/pm:1/p1"),
    )

    assert [status for status, _ in answers] == [200, 200] and made == [("pm:1", "p1")]


def test_the_panoramas_near_a_spot_are_those_within_reach_nearest_first():
    spheres = [
        match.Panorama(n, x, 0.0, 0.0, 0.0, None) for n, x in [("a", 30.0), ("b", 5.0), ("c", 60.0)]
    ]

    found = placing.Around(spheres).near(0.0, 0.0)

    assert [(p.id, round(m)) for p, m in found] == [("b", 5), ("a", 30)]


def test_a_proposal_that_fails_says_there_is_none(page, one):
    sources, _ = page

    def fails(*_):
        raise RuntimeError("out of memory")

    sources.proposal = fails

    status, given = one(sources, "GET", "/api/proposal/pm:1/p1")

    assert status == 200 and json.loads(given) is None


def test_an_answer_keeps_the_version_of_the_photo_it_was_made_on(page, one):
    sources, _ = page
    sources.photos["pm:1"]["inputs"] = "abc123"

    one(sources, "POST", "/api/answer", ANSWER)

    assert placings.holding(sources.placings)["pm:1"]["inputs"] == "abc123"


def test_a_registration_on_an_unknown_panorama_is_refused(page, one):
    sources, proposal = page
    elsewhere = {**registration.to_json(proposal), "panorama": "nowhere"}

    status, reason = one(
        sources, "POST", "/api/answer", {**ANSWER, "answer": "fits", "registration": elsewhere}
    )

    assert status == 400 and "panorama" in reason.decode()


def test_a_fit_without_a_panorama_is_refused_with_the_reason(page, one):
    sources, _ = page

    status, reason = one(sources, "POST", "/api/fit", {"photo_id": "pm:1", "view": {"heading": 0}})

    assert status == 400 and "panorama" in reason.decode()


def test_views_are_served_while_a_proposal_is_being_made(page):
    import http.client
    import threading

    sources, proposal = page
    release = threading.Event()
    sources.proposal = lambda *_: (
        release.wait(10) and {"registration": proposal, "carried_from": None}
    )
    server = placing.serve(sources, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:

        def get(path):
            c = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            c.request("GET", path)
            return c.getresponse().status

        slow = threading.Thread(target=get, args=("/api/proposal/pm:1/p1",))
        slow.start()
        assert get("/view/p1?heading=140") == 200  # not behind the proposal
        assert slow.is_alive()
        release.set()
        slow.join(10)
    finally:
        server.shutdown()
        server.server_close()


def test_the_next_photo_says_why_it_comes_and_photos_asked_ahead_come_first(page, one):
    sources, _ = page
    sources.photos["pm:2"] = {**sources.photos["pm:1"]}
    sources.ahead = lambda: {"pm:2": "revoir"}

    _, queue = one(sources, "GET", "/api/queue")

    assert (json.loads(queue)["photo"]["id"], json.loads(queue)["photo"]["why"]) == (
        "pm:2",
        "revoir",
    )


def test_a_photo_placed_can_be_reopened_with_its_answer(page, one):
    sources, proposal = page
    placings.record(
        sources.placings,
        "pm:1",
        "fits",
        3.0,
        lon=2.3467,
        lat=48.8521,
        heading=50.0,
        registration=proposal,
    )

    _, queue = one(sources, "GET", "/api/queue")
    _, again = one(sources, "GET", "/api/reopen/pm:1")

    assert [(p["id"], p["answer"]) for p in json.loads(queue)["placed"]] == [("pm:1", "fits")]
    reopened = json.loads(again)
    assert (reopened["lon"], reopened["heading"]) == (2.3467, 50.0)
    assert registration.from_json(reopened["registration"]) == proposal


def test_a_photo_reopened_keeps_how_it_was_placed_and_not_a_registration_on_another_image(
    page, one
):
    sources, proposal = page
    sources.photos["pm:1"]["inputs"] = "new"
    placings.record(
        sources.placings,
        "pm:1",
        "points",
        3.0,
        lon=2.3467,
        lat=48.8521,
        heading=50.0,
        registration=proposal,
        inputs="old",
    )

    _, again = one(sources, "GET", "/api/reopen/pm:1")

    reopened = json.loads(again)
    assert reopened["answer"] == "points" and reopened["registration"] is None


def test_a_photo_carried_from_a_placed_one_comes_with_its_proposal(page, one):
    sources, proposal = page
    sources.ahead = lambda: {"pm:1": "portée"}
    sources.carried = lambda: {
        "pm:1": {
            "registration": proposal,
            "carried_from": "pm:9",
            "lon": 2.3,
            "lat": 48.8,
            "heading": 0.0,
        }
    }

    _, queue = one(sources, "GET", "/api/queue")

    head = json.loads(queue)["photo"]
    assert (head["why"], head["carried_from"], head["answer"]) == ("portée", "pm:9", "fits")
    assert registration.from_json(head["registration"]) == proposal


def test_a_proposal_carried_later_is_shown_where_none_was_before(page):
    import http.client
    import threading

    sources, proposal = page
    sources.proposal = lambda *_: None  # none made on this panorama
    carried = {}
    sources.carried = lambda: carried
    server = placing.serve(sources, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def get():
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        connection.request("GET", "/api/proposal/pm:1/p1")
        return json.loads(connection.getresponse().read())

    try:
        before = get()
        carried["pm:1"] = {
            "registration": proposal,
            "carried_from": "pm:9",
            "lon": 2.3,
            "lat": 48.8,
            "heading": 0.0,
        }  # a photo placed meanwhile
        after = get()
    finally:
        server.shutdown()
        server.server_close()

    assert before is None and after["carried_from"] == "pm:9"


def test_a_photo_carried_opens_where_its_source_was_placed(page, one):
    sources, proposal = page
    sources.ahead = lambda: {"pm:1": "portée"}
    sources.carried = lambda: {
        "pm:1": {
            "registration": proposal,
            "carried_from": "pm:9",
            "lon": 2.3467,
            "lat": 48.8521,
            "heading": 40.0,
        }
    }

    _, queue = one(sources, "GET", "/api/queue")

    head = json.loads(queue)["photo"]
    assert (head["lon"], head["lat"], head["heading"]) == (2.3467, 48.8521, 40.0)


def test_carried_proposals_that_fail_leave_the_queue_going(page, one):
    sources, _ = page

    def fails():
        raise IndexError("a pair beyond the features kept")

    sources.carried = fails

    status, queue = one(sources, "GET", "/api/queue")

    assert status == 200 and json.loads(queue)["photo"]["id"] == "pm:1"


def test_a_photo_placed_on_an_image_it_no_longer_has_comes_back_to_be_seen_again(page, one):
    sources, proposal = page
    sources.photos["pm:1"]["inputs"] = "recropped"
    placings.record(
        sources.placings,
        "pm:1",
        "fits",
        3.0,
        lon=2.3467,
        lat=48.8521,
        heading=50.0,
        registration=proposal,
        inputs="before",
    )

    _, queue = one(sources, "GET", "/api/queue")

    head = json.loads(queue)["photo"]
    assert (head["id"], head["why"]) == ("pm:1", "revoir")


def test_a_larger_picture_not_had_is_not_asked_again_at_once(page, ask):
    sources, _ = page
    asked = []
    sources.full = lambda panorama: asked.append(panorama)  # never had: None
    view = ("GET", "/view/p1?heading=140&pitch=5&fov=75", None)

    ask(placing.serve(sources, 0), view, view, view)

    assert asked == ["p1"]


def test_a_picture_not_a_whole_sphere_is_not_taken_for_one(page, one):
    sources, _ = page
    sources.full = lambda panorama: np.zeros((800, 2048, 3), np.uint8)  # not 2:1: cut off

    _, jpeg = one(sources, "GET", "/view/p1?heading=140&pitch=5&fov=75")

    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1).max() > 0  # its own


# --- parallax: the old camera where the photographer stood


def _facing(heading: float) -> np.ndarray:
    """From the world to a camera facing `heading`, level (x right, y down, z ahead)."""
    h = np.radians(heading)
    return np.array([[np.cos(h), -np.sin(h), 0.0], [0.0, 0.0, -1.0], [np.sin(h), np.cos(h), 0.0]])


DETAILS = [  # along the view at 140°, across it, up: on its two facades and its end
    (8, -4, -1),
    (12, 4, 1),
    (15, -4, 2),
    (15, 4, -1.5),
    (30, -2, 4),
    (30, 3, 0),
    (12, -4, 3),
    (25, 4, -2),
]
AHEAD = np.array([np.sin(np.radians(140)), np.cos(np.radians(140)), 0.0])  # the view's way
RIGHT = np.array([np.cos(np.radians(140)), -np.sin(np.radians(140)), 0.0])


def _street(sphere: match.Panorama, open_from: float | None = None, walls: bool = True):
    """An old camera standing 2 m from the panorama, facing its view at 140°, the details of a
    street it saw (in the world, as clicked in the photo, 80 × 60, and in the view shown), and
    the scene of today's walls they lie on: two facades 8 m apart and the street's end at 30 m,
    drawn by hand (the left facade open from `open_from` to 4 m further; none, `walls` False)."""
    home = np.array([sphere.x, sphere.y, sphere.z])
    camera = registration.Standing(
        tuple(home + [1.5, 1.3, -1.0]),
        tuple(cv2.Rodrigues(_facing(140.0))[0].ravel()),
        70.0,
        (0.0, 0.0),
        (60, 80),
    )
    world = np.array([home + d * AHEAD + x * RIGHT + [0.0, 0.0, z] for d, x, z in DETAILS])
    rays = (world - home) / np.linalg.norm(world - home, axis=1, keepdims=True)
    view_px = registration.view_pixels(
        rays @ sphere.axes.T, ViewOf(140.0), sphere.axes, placing.SHOWN
    )

    def wall(d0, x0, d1, x1, made="fitted"):
        a, b = (home + d * AHEAD + x * RIGHT for d, x in ((d0, x0), (d1, x1)))
        return facades.Wall(tuple(a[:2]), tuple(b[:2]), sphere.z + 12.0, made)

    left = [wall(0, -4, 40, -4)]
    if open_from is not None:
        left = [wall(0, -4, open_from, -4), wall(open_from + 4, -4, 40, -4)]
    found = [*left, wall(0, 4, 40, 4), wall(30, -6, 30, 6, "drawn")] if walls else []
    scene = facades.Scene(
        sphere.id, tuple(home), sphere.z - 2.5, tuple(found), ("p2",), len(DETAILS)
    )
    return camera, world, camera.project(world), view_px, scene


def _parallax_body(sphere, photo_px, view_px, pin_off=(1.0, -1.0)):
    camera_at = to_degrees(Point(sphere.x + 1.5 + pin_off[0], sphere.y + 1.3 + pin_off[1]))
    return {
        "photo_id": "pm:1",
        "panorama": "p1",
        "view": {"heading": 140.0, "pitch": 5, "fov": 75},
        "photo_px": photo_px.tolist(),
        "view_px": view_px.tolist(),
        "lon": camera_at.x,
        "lat": camera_at.y,
    }


def test_a_fit_gives_each_pairs_error(page, one):
    sources, _ = page
    seen = cv2.perspectiveTransform(SPREAD[None], H)[0]
    body = {
        "photo_id": "pm:1",
        "panorama": "p1",
        "view": {"heading": 140.0},
        "photo_px": (SPREAD * [80, 60]).tolist(),
        "view_px": (seen * placing.SHOWN).tolist(),
    }

    status, answer = one(sources, "POST", "/api/fit", body)

    found = json.loads(answer)
    assert status == 200 and len(found["errors"]) == len(found["held_out"]) == len(SPREAD)
    assert registration.from_json(found["registration"]).panorama == "p1"


def test_parallax_gives_where_the_photographer_stood(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    camera, _, photo_px, view_px, scene = _street(sphere)
    sources.scene = lambda panorama, chosen: scene

    status, answer = one(
        sources, "POST", "/api/parallax", _parallax_body(sphere, photo_px, view_px)
    )

    found = json.loads(answer)
    stood = to_metres(Point(found["lon"], found["lat"]))
    assert status == 200 and all(found["used"])
    assert np.hypot(stood.x - camera.position[0], stood.y - camera.position[1]) < 0.2
    assert found["height_above_ground"] == pytest.approx(1.5, abs=0.2)
    assert found["source"][:5] == ["wall", "wall", "wall", "wall", "drawn"]  # the end: drawn
    assert found["neighbours"] == ["p2"]
    assert len(found["held_out"]["parallax"]) == len(photo_px)
    assert registration.from_json(found["registration"]).panorama == "p1"  # the flat fit too
    assert found["heading"] == pytest.approx(140.0, abs=0.5)  # the way its camera looked


def test_parallax_without_neighbours_says_why(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, _ = _street(sphere)

    status, reason = one(
        sources, "POST", "/api/parallax", _parallax_body(sphere, photo_px, view_px)
    )

    assert status == 400 and "voisin" in reason.decode()


def test_parallax_uses_the_scene_of_the_panorama_and_the_neighbours_chosen(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, scene = _street(sphere)
    asked = []
    sources.scene = lambda panorama, chosen: asked.append((panorama, chosen)) or scene
    sources.spheres += [
        match.Panorama(n, sphere.x + 5, sphere.y, 0.0, 0.0, sphere.path) for n in ("p2", "p3")
    ]
    body = _parallax_body(sphere, photo_px, view_px)

    one(sources, "POST", "/api/parallax", body)
    one(sources, "POST", "/api/parallax", {**body, "neighbours": ["p3", "p2"]})
    one(sources, "POST", "/api/parallax", {**body, "neighbours": []})  # none: walls drawn only

    assert asked == [("p1", None), ("p1", ("p2", "p3")), ("p1", ())]


def test_pairs_on_the_street_or_meeting_nothing_say_so(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, scene = _street(sphere, open_from=10)  # (12, -4, 3) seen through
    sources.scene = lambda panorama, chosen: scene
    down = AHEAD + [0.0, 0.0, -0.3]
    down /= np.linalg.norm(down)
    on_the_street = registration.view_pixels(
        down[None] @ sphere.axes.T, ViewOf(140.0), sphere.axes, placing.SHOWN
    )
    body = _parallax_body(sphere, photo_px, view_px)

    status, answer = one(sources, "POST", "/api/parallax", body)

    found = json.loads(answer)
    assert status == 200
    assert (found["source"][6], found["metres"][6], found["used"][6]) == (None, None, False)
    assert placing.depth_at(scene, sphere, on_the_street, ViewOf(140.0))[1] == ["ground"]


def test_a_point_meeting_nothing_takes_the_model_s_depth_where_it_is_given(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, scene = _street(sphere, open_from=10)  # (12, -4, 3) seen through
    sources.scene = lambda panorama, chosen: scene
    sources.fill = depthmaps.fills(
        lambda panorama: np.ones((len(match.VIEW_HEADINGS), 40, 40), np.float32), lambda _: None
    )

    status, answer = one(
        sources, "POST", "/api/parallax", _parallax_body(sphere, photo_px, view_px)
    )

    found = json.loads(answer)
    assert status == 200
    assert found["source"][6] == "model" and found["metres"][6] > 0
    assert found["source"][:5] == ["wall", "wall", "wall", "wall", "drawn"]  # walls first


def test_the_model_fills_rays_meeting_no_wall_the_street_too():
    sphere = match.Panorama("p1", 0.0, 0.0, 0.0, 0.0, None)
    _, _, _, _, scene = _street(sphere, open_from=10)
    rays = np.array([AHEAD, -AHEAD + [0.0, 0.0, 0.5], AHEAD + [0.0, 0.0, -0.3]])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)  # the street's end; up behind; down

    metres, source = placing._depth(scene, rays, lambda r: np.full(len(r), 7.0))

    assert source[0] == "drawn" and metres[0] != 7.0
    assert (source[1], metres[1]) == ("model", 7.0)
    assert (source[2], metres[2]) == ("model", 7.0)
    assert placing._depth(scene, rays)[1][2] == "ground"  # without the model: the street


def test_maps_are_scaled_to_the_scene_s_walls():
    sphere = match.Panorama("p1", 0.0, 0.0, 0.0, 0.0, None)
    _, _, _, _, scene = _street(sphere)
    grid = depthmaps.grid_rays(40)
    true = facades.cast(scene, grid.reshape(-1, 3))[0].reshape(grid.shape[:3])
    maps = np.where(np.isfinite(true), true / 3, 1.0)  # the street's shape, a third too near
    down_the_street = AHEAD[None]

    fill = depthmaps.Filler(scene, maps)

    assert fill(down_the_street)[0] == pytest.approx(
        placing._depth(scene, down_the_street)[0][0], rel=0.05
    )


def test_maps_are_no_use_without_walls_to_scale_them():
    sphere = match.Panorama("p1", 0.0, 0.0, 0.0, 0.0, None)
    _, _, _, _, scene = _street(sphere, walls=False)

    fill = depthmaps.Filler(scene, np.ones((len(match.VIEW_HEADINGS), 40, 40)))

    assert np.isnan(fill(AHEAD[None])).all()


def test_parallax_on_a_panorama_without_walls_says_why(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, scene = _street(sphere, walls=False)
    sources.scene = lambda panorama, chosen: scene
    body = _parallax_body(sphere, photo_px, view_px)

    status, reason = one(sources, "POST", "/api/parallax", body)

    assert status == 400 and "aucun mur" in reason.decode()


def test_a_photo_laid_by_its_standing_camera_is_drawn(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    camera, _, _, _, scene = _street(sphere)
    sources.scene = lambda panorama, chosen: scene
    body = {
        "photo_id": "pm:1",
        "panorama": "p1",
        "view": {"heading": 140.0, "pitch": 5, "fov": 75},
        "standing": registration.standing_to_json(camera),
    }

    status, jpeg = one(sources, "POST", "/api/composite", body)

    assert status == 200 and cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1) is not None


def test_an_answer_in_parallax_keeps_the_camera_its_pairs_give(page, one):
    sources, proposal = page
    sphere = sources.spheres[0]
    camera, _, photo_px, view_px, scene = _street(sphere)
    sources.scene = lambda panorama, chosen: scene
    pin = _parallax_body(sphere, photo_px, view_px)
    body = {
        **ANSWER,
        "answer": "points",
        "registration": registration.to_json(proposal),
        "lon": pin["lon"],
        "lat": pin["lat"],
        "parallax": {"lon": pin["lon"], "lat": pin["lat"]},
        "pairs": {"photo_px": photo_px.tolist(), "view_px": view_px.tolist(), "view": pin["view"]},
    }

    status, _ = one(sources, "POST", "/api/answer", body)

    row = placings.holding(sources.placings)["pm:1"]
    assert status == 200 and row["fit"] == "parallax" and row["pairs"]["side"] == placing.SHOWN
    assert np.allclose(np.linalg.norm(row["pairs"]["own"], axis=1), 1.0)
    kept = placings.standing_of(row)  # made again from the pairs saved, not taken from the page
    assert np.allclose(kept.position, camera.position, atol=0.2)
    assert row["standing"]["source"][0] == "wall"
    assert row["standing"]["neighbours"] == ["p2"]
    assert row["standing"]["chosen"] is None  # picked by their geometry: replayed so


def test_an_answer_by_points_keeps_its_pairs_without_parallax(page, one):
    sources, proposal = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, _ = _street(sphere)
    body = {
        **ANSWER,
        "answer": "points",
        "registration": registration.to_json(proposal),
        "pairs": {
            "photo_px": photo_px.tolist(),
            "view_px": view_px.tolist(),
            "view": {"heading": 140.0, "pitch": 5, "fov": 75},
        },
    }

    status, _ = one(sources, "POST", "/api/answer", body)

    row = placings.holding(sources.placings)["pm:1"]
    assert status == 200 and row["fit"] == "flat" and placings.standing_of(row) is None


def test_the_walls_of_a_panorama_are_made_ahead_when_asked(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    _, _, _, _, scene = _street(sphere)
    asked = []
    sources.scene = lambda panorama, chosen: asked.append((panorama, chosen)) or scene
    sources.segments = lambda x, y: [((x, y), (x + 10.0, y))]
    body = {"photo_id": "pm:1", "panorama": "p1", "heading": 140.0}

    status, answer = one(sources, "POST", "/api/depth", body)

    found = json.loads(answer)
    assert status == 200 and found["details"] == len(DETAILS) and found["neighbours"] == ["p2"]
    assert [w["made"] for w in found["walls"]] == ["fitted", "fitted", "drawn"]
    at = to_metres(Point(*found["walls"][0]["a"]))
    assert np.allclose([at.x, at.y], scene.walls[0].a, atol=0.01)  # sent in degrees
    assert len(found["segments"]) == 1 and asked == [("p1", None)]


def test_a_panorama_without_walls_says_why_when_its_walls_are_asked(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    sources.scene = lambda panorama, chosen: _street(sphere, walls=False)[4]
    body = {"photo_id": "pm:1", "panorama": "p1", "heading": 140.0}

    status, answer = one(sources, "POST", "/api/depth", body)

    found = json.loads(answer)
    assert status == 200 and found["walls"] == [] and "aucun mur" in found["why"]


def test_no_depth_ahead_without_neighbours_says_why(page, one):
    sources, _ = page

    status, reason = one(
        sources, "POST", "/api/depth", {"photo_id": "pm:1", "panorama": "p1", "heading": 0}
    )

    assert status == 400 and "voisin" in reason.decode()


def test_depth_that_fails_to_be_made_says_why(page, one):
    sources, _ = page

    def broken(panorama, chosen):
        raise RuntimeError("le graphique a manqué de mémoire")

    sources.scene = broken
    body = {"photo_id": "pm:1", "panorama": "p1", "heading": 140.0}

    status, reason = one(sources, "POST", "/api/depth", body)

    assert status == 400 and "mémoire" in reason.decode()


def test_an_answer_in_parallax_refits_from_the_pin_it_started_from(page, one):
    sources, proposal = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, scene = _street(sphere)
    sources.scene = lambda panorama, chosen: scene
    asked = _parallax_body(sphere, photo_px, view_px)
    _, shown = one(sources, "POST", "/api/parallax", asked)
    shown = json.loads(shown)
    body = {
        **ANSWER,
        "answer": "points",
        "registration": registration.to_json(proposal),
        "lon": shown["lon"],
        "lat": shown["lat"],  # the position accepted, the pin moved there
        "parallax": {"lon": asked["lon"], "lat": asked["lat"]},  # where the fit started
        "pairs": {
            "photo_px": asked["photo_px"],
            "view_px": asked["view_px"],
            "view": asked["view"],
        },
    }

    status, _ = one(sources, "POST", "/api/answer", body)

    kept = placings.standing_of(placings.holding(sources.placings)["pm:1"])
    assert status == 200 and kept == registration.standing_from_json(shown["standing"])


def test_a_composite_with_no_standing_camera_is_the_flat_one(page, one):
    sources, proposal = page
    body = {"photo_id": "pm:1", "registration": registration.to_json(proposal), "standing": None}

    status, _ = one(sources, "POST", "/api/composite", body)

    assert status == 200


def test_a_photo_reopened_says_which_fit_it_kept(page, one):
    sources, proposal = page
    placings.record(
        sources.placings,
        "pm:1",
        "points",
        3.0,
        lon=2.3466,
        lat=48.852,
        heading=40.0,
        registration=proposal,
    )

    _, found = one(sources, "GET", "/api/reopen/pm:1")

    assert json.loads(found)["fit"] is None  # placed before pairs were kept


def _saved(sphere, photo_px, view_px, proposal, start=None):
    """An answer by points as saved: its pairs, its place, in parallax the pin it started from."""
    pin = _parallax_body(sphere, photo_px, view_px)
    pairs = placing.kept_pairs(
        {"photo_px": photo_px.tolist(), "view_px": view_px.tolist(), "view": pin["view"]},
        sphere,
        {},
    )
    row = {
        "photo_id": "pm:1",
        "lon": pin["lon"],
        "lat": pin["lat"],
        "pairs": pairs,
        "registration": registration.to_json(proposal),
        "fit": "flat",
        "standing": None,
    }
    if start:
        row.update(fit="parallax", standing={"start": start}, lon=0.0, lat=0.0)  # place accepted
    return row, pin


def test_an_answer_s_two_fits_are_compared_from_its_saved_pairs(page):
    sources, proposal = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, scene = _street(sphere)
    row, _ = _saved(sphere, photo_px, view_px, proposal)

    found = placing.compared(row, sphere, (60, 80), scene, {})

    assert found["pairs"] == found["with_depth"] == len(photo_px)
    assert found["refused"] == {"flat": 0, "parallax": 0} and found["fit"] == "flat"
    assert found["parallax"] < found["flat"]  # a street in depth, seen from 2 m off
    assert found["moved_m"] == pytest.approx(np.hypot(1.0, 1.0), abs=0.3)  # from the pin


def test_both_fits_are_compared_on_the_same_pairs_those_with_depth(page):
    sources, proposal = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, scene = _street(sphere, open_from=10)  # (12, -4, 3) seen through
    row, _ = _saved(sphere, photo_px, view_px, proposal)

    found = placing.compared(row, sphere, (60, 80), scene, {})

    assert found["with_depth"] == len(photo_px) - 1


def test_a_photo_whose_parallax_fit_is_refused_counts_as_the_worst(page):
    sources, proposal = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, scene = _street(sphere, walls=False)  # three on the street: few
    row, _ = _saved(sphere, photo_px, view_px, proposal)

    found = placing.compared(row, sphere, (60, 80), scene, {})

    assert np.isinf(found["parallax"])
    assert found["moved_m"] is None
    assert "6" in found["why"]


def test_an_answer_kept_in_parallax_is_compared_from_the_pin_its_fit_started_from(page):
    sources, proposal = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, scene = _street(sphere)
    pin = _parallax_body(sphere, photo_px, view_px)
    row, _ = _saved(
        sphere, photo_px, view_px, proposal, start={"lon": pin["lon"], "lat": pin["lat"]}
    )

    found = placing.compared(row, sphere, (60, 80), scene, {})

    assert found["fit"] == "parallax" and found["moved_m"] == pytest.approx(
        np.hypot(1.0, 1.0), abs=0.3
    )


def test_an_answer_saved_again_on_the_same_fit_keeps_its_pairs_and_camera(page, one):
    sources, proposal = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, scene = _street(sphere)
    sources.scene = lambda panorama, chosen: scene
    pin = _parallax_body(sphere, photo_px, view_px)
    first = {
        **ANSWER,
        "answer": "points",
        "registration": registration.to_json(proposal),
        "lon": pin["lon"],
        "lat": pin["lat"],
        "parallax": {"lon": pin["lon"], "lat": pin["lat"]},
        "pairs": {"photo_px": photo_px.tolist(), "view_px": view_px.tolist(), "view": pin["view"]},
    }
    one(sources, "POST", "/api/answer", first)
    reopened = {
        **ANSWER,
        "answer": "points",
        "registration": registration.to_json(proposal),
        "heading": 120.0,
    }  # its cone turned, nothing clicked again

    status, _ = one(sources, "POST", "/api/answer", reopened)

    row = placings.holding(sources.placings)["pm:1"]
    assert status == 200 and row["heading"] == 120.0
    assert row["pairs"] is not None and row["fit"] == "parallax" and placings.standing_of(row)


# --- the hand choices: neighbours, walls, a pair's depth by a second click


def _beside(sources, sphere) -> match.Panorama:
    """A neighbour 3 m east of the panorama, known to the page."""
    other = match.Panorama("p2", sphere.x + 3.0, sphere.y, sphere.z, 198.0, sphere.path)
    sources.spheres.append(other)
    return other


def _seen_from(other: match.Panorama, point: np.ndarray) -> list[float]:
    """Where a point of today falls in the neighbour's view facing as the view (140°)."""
    ray = point - [other.x, other.y, other.z]
    ray /= np.linalg.norm(ray)
    return registration.view_pixels(
        ray[None] @ other.axes.T, ViewOf(140.0), other.axes, placing.SHOWN
    )[0].tolist()


def test_panoramas_near_say_which_colmap_placed(page, one):
    sources, _ = page
    sources.fixed = {"p1"}

    _, found = one(sources, "GET", "/api/near?lon=2.3466&lat=48.852")

    assert [p["fixed"] for p in json.loads(found)] == [True]


def _wall(sphere, d0: float, x0: float, d1: float, x1: float) -> dict:
    home = np.array([sphere.x, sphere.y])
    a, b = (
        to_degrees(Point(*(home + d * AHEAD[:2] + x * RIGHT[:2]))) for d, x in ((d0, x0), (d1, x1))
    )
    return {"a": [a.x, a.y], "b": [b.x, b.y]}


def test_walls_drawn_and_removed_are_saved_for_the_panorama(page, one, tmp_path):
    sources, _ = page
    sphere = sources.spheres[0]
    scene = _street(sphere)[4]
    sources.scene = lambda panorama, chosen: scene
    sources.walls = tmp_path / "walls.jsonl"
    body = {
        "photo_id": "pm:1",
        "panorama": "p1",
        "heading": 0.0,
        "drawn": [_wall(sphere, 0, -3.8, 40, -3.8) | {"top": 15.0}, _wall(sphere, 50, -5, 50, 5)],
        "removed": [_wall(sphere, 0, 4, 40, 4)],
    }

    status, answer = one(sources, "POST", "/api/walls", body)

    drawn, removed, _ = facades.saved_at(sources.walls, "p1")
    assert status == 200
    assert "walls" in json.loads(answer)
    assert [w.top for w in drawn] == [15.0, None]  # its top when given; else made with the scene
    assert len(removed) == 1


def test_walls_drawn_at_other_panoramas_say_so(page, one, tmp_path):
    sources, _ = page
    sphere = sources.spheres[0]
    drawn = facades.Wall((sphere.x - 4, sphere.y), (sphere.x - 4, sphere.y + 20), None, "drawn")
    scene = dataclasses.replace(_street(sphere)[4], walls=(dataclasses.replace(drawn, at="p9"),))
    sources.scene = lambda panorama, chosen: scene
    body = {"photo_id": "pm:1", "panorama": "p1", "heading": 140.0}

    _, answer = one(sources, "POST", "/api/depth", body)

    assert [w["at"] for w in json.loads(answer)["walls"]] == ["p9"]  # sent back by p9 only


def _second(sphere, other, i: int) -> dict:
    _, world, _, view_px, _ = _street(sphere)
    return {
        "photo_id": "pm:1",
        "panorama": "p1",
        "view": {"heading": 140.0, "pitch": 5, "fov": 75},
        "view_px": view_px[i].tolist(),
        "neighbour": other.id,
        "neighbour_px": _seen_from(other, world[i]),
    }


def test_a_second_click_gives_a_pair_its_depth(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    other = _beside(sources, sphere)
    _, world, _, _, _ = _street(sphere)

    status, answer = one(sources, "POST", "/api/second", _second(sphere, other, 6))

    expected = np.linalg.norm(world[6] - [sphere.x, sphere.y, sphere.z])
    assert status == 200 and json.loads(answer)["metres"] == pytest.approx(expected, rel=0.02)


def test_a_second_click_whose_rays_do_not_meet_is_refused(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    other = _beside(sources, sphere)
    body = _second(sphere, other, 6)
    body["neighbour_px"] = [body["neighbour_px"][0], body["neighbour_px"][1] + 200]

    status, reason = one(sources, "POST", "/api/second", body)

    assert status == 400 and "ne se rencontrent pas" in reason.decode()
    assert "m l'un de l'autre" in reason.decode()  # by how much: a panorama off, or not the same


def test_the_line_where_a_detail_may_be_in_the_neighbour_passes_through_it(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    other = _beside(sources, sphere)
    body = _second(sphere, other, 6)
    at = np.array(body.pop("neighbour_px"))

    status, answer = one(sources, "POST", "/api/epipolar", body)

    line = np.array(json.loads(answer)["line"])
    assert status == 200 and len(line) > 10
    assert np.min(np.linalg.norm(line - at, axis=1)) < 3  # sampled finely near it


def test_a_view_tinted_by_its_depth_is_drawn(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    scene = _street(sphere)[4]
    sources.scene = lambda panorama, chosen: scene
    body = {"photo_id": "pm:1", "panorama": "p1", "view": {"heading": 140.0, "pitch": 5, "fov": 75}}

    status, jpeg = one(sources, "POST", "/api/depthview", body)

    assert status == 200 and cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1) is not None


def _with_second(sphere, other, i: int) -> dict:
    """A parallax body whose pair i has a second click in the neighbour."""
    _, world, photo_px, view_px, _ = _street(sphere)
    body = _parallax_body(sphere, photo_px, view_px)
    seconds = [None] * len(photo_px)
    seconds[i] = {"neighbour": other.id, "px": _seen_from(other, world[i])}
    return body | {"second_clicks": seconds}


def test_a_pair_s_second_click_gives_its_depth_in_the_parallax_fit(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    other = _beside(sources, sphere)
    scene = _street(sphere, open_from=10)[4]  # pair 6 meets no wall
    sources.scene = lambda panorama, chosen: scene

    status, answer = one(sources, "POST", "/api/parallax", _with_second(sphere, other, 6))

    found = json.loads(answer)
    assert status == 200 and found["source"][6] == "click" and found["used"][6]


def test_an_answer_keeps_its_second_clicks(page, one):
    sources, proposal = page
    sphere = sources.spheres[0]
    other = _beside(sources, sphere)
    scene = _street(sphere, open_from=10)[4]
    sources.scene = lambda panorama, chosen: scene
    asked = _with_second(sphere, other, 6)
    body = {
        **ANSWER,
        "answer": "points",
        "registration": registration.to_json(proposal),
        "lon": asked["lon"],
        "lat": asked["lat"],
        "parallax": {"lon": asked["lon"], "lat": asked["lat"]},
        "pairs": {k: asked[k] for k in ("photo_px", "view_px", "view", "second_clicks")},
    }

    status, _ = one(sources, "POST", "/api/answer", body)

    row = placings.holding(sources.placings)["pm:1"]
    second = row["pairs"]["second_clicks"][6]
    assert status == 200 and second["neighbour"] == "p2" and len(second["own"]) == 3
    assert row["standing"]["source"][6] == "click"


def test_a_panorama_s_own_walls_come_back_as_saved_however_far(page, one, tmp_path):
    sources, _ = page
    sphere = sources.spheres[0]
    scene = _street(sphere)[4]
    sources.scene = lambda panorama, chosen: scene
    sources.walls = tmp_path / "walls.jsonl"
    far = facades.Wall((sphere.x + 35, sphere.y), (sphere.x + 35, sphere.y + 10), None, "drawn")
    facades.record(
        sources.walls, "p1", [far], [((sphere.x, sphere.y + 5), (sphere.x + 5, sphere.y + 5))]
    )
    asked = {"photo_id": "pm:1", "panorama": "p1", "heading": 140.0}

    _, answer = one(sources, "POST", "/api/depth", asked)
    mine = json.loads(answer)["mine"]
    one(sources, "POST", "/api/walls", asked | mine)  # sent back as it came

    assert [w["top"] for w in mine["drawn"]] == [None]  # as saved: made with the scene
    assert facades.saved_at(sources.walls, "p1")[0][0].top is None
    assert len(facades.saved_at(sources.walls, "p1")[1]) == 1


def test_second_clicks_not_as_pairs_are_refused_with_why(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    sources.scene = lambda panorama, chosen: _street(sphere)[4]
    _, _, photo_px, view_px, _ = _street(sphere)
    body = _parallax_body(sphere, photo_px, view_px) | {"second_clicks": [1] * len(photo_px)}

    status, reason = one(sources, "POST", "/api/parallax", body)

    assert status == 400 and "second clic" in reason.decode()


def test_a_second_click_saved_whose_rays_no_longer_meet_leaves_its_pair_to_the_walls(page):
    sources, proposal = page
    sphere = sources.spheres[0]
    other = _beside(sources, sphere)
    _, _, photo_px, view_px, scene = _street(sphere)
    row, _ = _saved(sphere, photo_px, view_px, proposal)
    row["pairs"]["second_clicks"] = [None] * len(photo_px)
    row["pairs"]["second_clicks"][0] = {"neighbour": "p2", "own": [0.0, 0.0, 1.0]}  # straight up

    found = placing.compared(row, sphere, (60, 80), scene, {"p2": other})

    assert found["with_depth"] == len(photo_px)  # its wall's depth, not the photo dropped


def test_walls_shown_say_which_are_the_panorama_s_own(page, one, tmp_path):
    sources, _ = page
    sphere = sources.spheres[0]
    sources.walls = tmp_path / "walls.jsonl"
    mine = facades.Wall((sphere.x + 4, sphere.y), (sphere.x + 4, sphere.y + 20), None, "drawn")
    facades.record(sources.walls, "p1", [mine], [])
    street = _street(sphere)[4]
    shown = dataclasses.replace(mine, top=20.0, at="p1")
    scene = dataclasses.replace(street, walls=street.walls[:1] + (shown,))
    sources.scene = lambda panorama, chosen: scene

    _, answer = one(
        sources, "POST", "/api/depth", {"photo_id": "pm:1", "panorama": "p1", "heading": 0}
    )

    assert [w["mine"] for w in json.loads(answer)["walls"]] == [None, 0]  # fitted; drawn here


def test_the_neighbour_shown_beside_is_the_one_giving_most_where_the_view_looks(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    _beside(sources, sphere)
    north = match.Panorama("p3", sphere.x, sphere.y + 6.0, sphere.z, 198.0, sphere.path)
    sources.spheres.append(north)
    scene = dataclasses.replace(_street(sphere)[4], neighbours=("p2", "p3"))
    asked = []
    sources.scene = lambda panorama, chosen: scene
    sources.aimed = lambda panorama, neighbours, heading, fov: asked.append(heading) or ["p3", "p2"]

    _, found = one(
        sources, "POST", "/api/depth", {"photo_id": "pm:1", "panorama": "p1", "heading": 90.0}
    )

    assert json.loads(found)["best"] == "p3" and asked == [90.0]


def test_an_answer_in_parallax_with_no_neighbour_chosen_is_saved(page, one):
    sources, proposal = page
    sphere = sources.spheres[0]
    camera, _, photo_px, view_px, scene = _street(sphere)
    drawn_only = dataclasses.replace(scene, neighbours=(), chosen=())  # walls drawn by hand only
    sources.scene = lambda panorama, chosen: drawn_only
    pin = _parallax_body(sphere, photo_px, view_px)
    body = {
        **ANSWER,
        "answer": "points",
        "registration": registration.to_json(proposal),
        "lon": pin["lon"],
        "lat": pin["lat"],
        "neighbours": [],
        "parallax": {"lon": pin["lon"], "lat": pin["lat"]},
        "pairs": {"photo_px": photo_px.tolist(), "view_px": view_px.tolist(), "view": pin["view"]},
    }

    status, reason = one(sources, "POST", "/api/answer", body)

    assert status == 200, reason
    assert placings.holding(sources.placings)["pm:1"]["standing"]["chosen"] == []


def test_a_wall_removed_elsewhere_stays_removed_when_its_panorama_saves_again(page, one, tmp_path):
    sources, _ = page
    sphere = sources.spheres[0]
    sources.scene = lambda panorama, chosen: _street(sphere)[4]
    sources.walls = tmp_path / "walls.jsonl"
    theirs = facades.Wall((sphere.x + 4, sphere.y), (sphere.x + 4, sphere.y + 20), None, "drawn")
    facades.record(sources.walls, "p1", [theirs], [])  # drawn at p1
    facades.record(sources.walls, "p9", [], [(theirs.a, theirs.b)])  # then removed at p9
    asked = {"photo_id": "pm:1", "panorama": "p1", "heading": 140.0}
    _, answer = one(sources, "POST", "/api/depth", asked)
    mine = json.loads(answer)["mine"]  # p1's own walls, as the page holds them
    mine["drawn"].append(_wall(sphere, 0, -6, 20, -6))  # another drawn there

    one(sources, "POST", "/api/walls", asked | mine)

    drawn, _ = facades.saved_near(sources.walls, sphere.x, sphere.y)
    assert not [w for w in drawn if abs(w.a[0] - theirs.a[0]) < 0.1]  # still removed
    assert len(drawn) == 1  # the new one


def test_a_second_click_from_a_neighbour_in_line_with_the_detail_says_so(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    _, world, _, _, _ = _street(sphere)
    ahead = (world[4] - [sphere.x, sphere.y, sphere.z])[:2]  # the detail 30 m up the view
    spot = np.array([sphere.x, sphere.y]) - 3 * ahead / np.linalg.norm(ahead)
    behind = match.Panorama("p2", *spot, sphere.z, 198.0, sphere.path)
    sources.spheres.append(behind)  # 3 m behind the panorama, on the detail's line
    body = _second(sphere, behind, 4)

    status, reason = one(sources, "POST", "/api/second", body)

    assert status == 400 and "dans l'axe" in reason.decode()


def test_the_walls_are_drawn_in_the_view_their_foot_under_their_top(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    sources.scene = lambda panorama, chosen: _street(sphere)[4]
    body = {"photo_id": "pm:1", "panorama": "p1", "view": {"heading": 140.0, "pitch": 5, "fov": 75}}

    status, answer = one(sources, "POST", "/api/wallsview", body)

    walls = json.loads(answer)["walls"]
    end = next(w for w in walls if w["made"] == "drawn")  # the street's end, across the view
    assert status == 200 and all(p is not None for p in end["foot"] + end["top"])
    assert all(f[1] > t[1] for f, t in zip(end["foot"], end["top"], strict=True))  # foot lower


def test_the_neighbours_offered_are_measured_from_the_panorama(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    sources.scene = lambda panorama, chosen: _street(sphere)[4]
    sources.fixed = {"p2"}
    _beside(sources, sphere)  # 3 m east of the panorama
    far = match.Panorama("p3", sphere.x + 30.0, sphere.y, sphere.z, 0.0, sphere.path)
    sources.spheres.append(far)
    body = {"photo_id": "pm:1", "panorama": "p1", "heading": 140.0}

    _, answer = one(sources, "POST", "/api/depth", body)

    found = json.loads(answer)["candidates"]
    beside = placing._degrees(sphere.x + 3.0, sphere.y)
    assert [(c["id"], c["metres"], c["fixed"]) for c in found] == [("p2", 3.0, True)]
    assert found[0]["at"] == pytest.approx(beside)  # where it stands, to show it on the map


def test_a_second_click_whose_rays_cross_behind_a_panorama_says_so(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    other = _beside(sources, sphere)
    body = _second(sphere, other, 0)  # a detail on the left facade
    body["neighbour_px"] = [20, body["neighbour_px"][1]]  # far left: the two rays part ahead

    status, reason = one(sources, "POST", "/api/second", body)

    assert status == 400 and "derrière" in reason.decode()


def test_a_second_click_in_a_neighbour_that_cannot_be_set_right_is_refused(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    other = _beside(sources, sphere)
    sources.pair = lambda home, neighbour: None

    status, reason = one(sources, "POST", "/api/second", _second(sphere, other, 6))

    assert status == 400 and "ne s'accorde pas" in reason.decode()


def test_walls_asked_without_a_heading_show_their_first_neighbour(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    _beside(sources, sphere)
    scene = dataclasses.replace(_street(sphere)[4], neighbours=("p2",))
    sources.scene = lambda panorama, chosen: scene

    status, found = one(sources, "POST", "/api/depth", {"photo_id": "pm:1", "panorama": "p1"})

    assert status == 200 and json.loads(found)["best"] == "p2"


def test_walls_shown_give_the_street_s_level(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    scene = _street(sphere)[4]
    sources.scene = lambda panorama, chosen: scene

    _, answer = one(
        sources, "POST", "/api/depth", {"photo_id": "pm:1", "panorama": "p1", "heading": 0}
    )

    assert json.loads(answer)["ground"] == scene.ground  # a wall's height shown over it


def test_a_wall_drawn_higher_than_any_is_refused(page, one, tmp_path):
    sources, _ = page
    sphere = sources.spheres[0]
    sources.scene = lambda panorama, chosen: _street(sphere)[4]
    sources.walls = tmp_path / "walls.jsonl"
    ground = sphere.z - placing.PANORAMA_HEIGHT_M
    too_high = ground + placing.HIGHEST_WALL_M + 1
    body = {"photo_id": "pm:1", "panorama": "p1", "heading": 0.0, "removed": []}
    body["drawn"] = [_wall(sphere, 0, -3.8, 40, -3.8) | {"top": too_high}]

    status, _ = one(sources, "POST", "/api/walls", body)

    assert status == 400 and not sources.walls.exists()
