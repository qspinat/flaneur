"""The placing page: each photo on the map, then in its panorama."""

import json

import cv2
import numpy as np
import pytest
from shapely.geometry import Point

from flaneur import depth, match, placing, placings, registration
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


def _street(sphere: match.Panorama):
    """An old camera standing 2 m from the panorama, facing its view at 140°, and the details
    of a street it saw: in the world, as clicked in the photo (80 × 60) and in the view shown."""
    home = np.array([sphere.x, sphere.y, sphere.z])
    camera = registration.Standing(
        tuple(home + [1.5, 1.3, -1.0]),
        tuple(cv2.Rodrigues(_facing(140.0))[0].ravel()),
        70.0,
        (0.0, 0.0),
        (60, 80),
    )
    ahead = np.array([np.sin(np.radians(140)), np.cos(np.radians(140)), 0.0])
    right = np.array([np.cos(np.radians(140)), -np.sin(np.radians(140)), 0.0])
    world = np.array(
        [
            home + d * ahead + x * right + [0.0, 0.0, z]
            for d, x, z in [
                (8, -3, -1),
                (10, 2.5, 1),
                (15, -4, 2),
                (15, 4, -1.5),
                (30, -2, 4),
                (30, 5, 0),
                (12, 0, 3),
                (25, 1, -2),
            ]
        ]
    )
    rays = (world - home) / np.linalg.norm(world - home, axis=1, keepdims=True)
    view_px = registration.view_pixels(
        rays @ sphere.axes.T, ViewOf(140.0), sphere.axes, placing.SHOWN
    )
    depths = depth.Depths(
        "p1",
        rays,
        np.linalg.norm(world - home, axis=1),
        np.array(["p2"] * len(world)),
        np.ones(len(world), bool),
    )
    return camera, world, camera.project(world), view_px, depths


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
    camera, _, photo_px, view_px, depths = _street(sphere)
    sources.depths = lambda panorama, heading: depths

    status, answer = one(
        sources, "POST", "/api/parallax", _parallax_body(sphere, photo_px, view_px)
    )

    found = json.loads(answer)
    stood = to_metres(Point(found["lon"], found["lat"]))
    assert status == 200 and all(found["used"])
    assert np.hypot(stood.x - camera.position[0], stood.y - camera.position[1]) < 0.2
    assert found["height_above_ground"] == pytest.approx(1.5, abs=0.2)
    assert found["from"][0] == ["p2"] and len(found["held_out"]["parallax"]) == len(photo_px)
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


def test_parallax_uses_the_depths_of_the_panorama_and_the_way_asked(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, depths = _street(sphere)
    asked = []
    sources.depths = lambda panorama, heading: asked.append((panorama, heading)) or depths

    one(sources, "POST", "/api/parallax", _parallax_body(sphere, photo_px, view_px))

    assert asked == [("p1", 140.0)]


def test_a_photo_laid_by_its_standing_camera_is_drawn(page, one):
    sources, _ = page
    sphere = sources.spheres[0]
    camera, _, _, _, depths = _street(sphere)
    sources.depths = lambda panorama, heading: depths
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
    camera, _, photo_px, view_px, depths = _street(sphere)
    sources.depths = lambda panorama, heading: depths
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
    assert row["standing"]["from"][0] == ["p2"]


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


def test_the_depth_of_a_view_is_made_ahead_when_asked(page, one):
    sources, _ = page
    _, _, _, _, depths = _street(sources.spheres[0])
    asked = []
    sources.depths = lambda panorama, heading: asked.append((panorama, heading)) or depths
    body = {"photo_id": "pm:1", "panorama": "p1", "heading": 140.0}

    status, answer = one(sources, "POST", "/api/depth", body)

    assert status == 200 and json.loads(answer)["details"] == len(depths.metres)
    assert asked == [("p1", 140.0)]


def test_no_depth_ahead_without_neighbours_says_why(page, one):
    sources, _ = page

    status, reason = one(
        sources, "POST", "/api/depth", {"photo_id": "pm:1", "panorama": "p1", "heading": 0}
    )

    assert status == 400 and "voisin" in reason.decode()


def test_depth_that_fails_to_be_made_says_why(page, one):
    sources, _ = page

    def broken(panorama, heading):
        raise RuntimeError("le graphique a manqué de mémoire")

    sources.depths = broken
    body = {"photo_id": "pm:1", "panorama": "p1", "heading": 140.0}

    status, reason = one(sources, "POST", "/api/depth", body)

    assert status == 400 and "mémoire" in reason.decode()


def test_an_answer_in_parallax_refits_from_the_pin_it_started_from(page, one):
    sources, proposal = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, depths = _street(sphere)
    sources.depths = lambda panorama, heading: depths
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
        {"photo_px": photo_px.tolist(), "view_px": view_px.tolist(), "view": pin["view"]}, sphere
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
    _, _, photo_px, view_px, depths = _street(sphere)
    row, _ = _saved(sphere, photo_px, view_px, proposal)

    found = placing.compared(row, sphere, (60, 80), depths)

    assert found["pairs"] == found["with_depth"] == len(photo_px)
    assert found["refused"] == {"flat": 0, "parallax": 0} and found["fit"] == "flat"
    assert found["parallax"] < found["flat"]  # a street in depth, seen from 2 m off
    assert found["moved_m"] == pytest.approx(np.hypot(1.0, 1.0), abs=0.3)  # from the pin


def test_both_fits_are_compared_on_the_same_pairs_those_with_depth(page):
    sources, proposal = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, depths = _street(sphere)
    keep = np.arange(len(depths.metres)) != 0  # the first detail: no depth
    depths = depth.Depths(
        depths.panorama,
        depths.directions[keep],
        depths.metres[keep],
        depths.seen_from[keep],
        depths.fixed[keep],
    )
    row, _ = _saved(sphere, photo_px, view_px, proposal)

    found = placing.compared(row, sphere, (60, 80), depths)

    assert found["with_depth"] == len(photo_px) - 1


def test_a_photo_whose_parallax_fit_is_refused_counts_as_the_worst(page):
    sources, proposal = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, depths = _street(sphere)
    few = np.arange(len(depths.metres)) < 3  # three details with a depth: under four
    depths = depth.Depths(
        depths.panorama,
        depths.directions[few],
        depths.metres[few],
        depths.seen_from[few],
        depths.fixed[few],
    )
    row, _ = _saved(sphere, photo_px, view_px, proposal)

    found = placing.compared(row, sphere, (60, 80), depths)

    assert np.isinf(found["parallax"]) and found["moved_m"] is None and "4" in found["why"]


def test_an_answer_kept_in_parallax_is_compared_from_the_pin_its_fit_started_from(page):
    sources, proposal = page
    sphere = sources.spheres[0]
    _, _, photo_px, view_px, depths = _street(sphere)
    pin = _parallax_body(sphere, photo_px, view_px)
    row, _ = _saved(
        sphere, photo_px, view_px, proposal, start={"lon": pin["lon"], "lat": pin["lat"]}
    )

    found = placing.compared(row, sphere, (60, 80), depths)

    assert found["fit"] == "parallax" and found["moved_m"] == pytest.approx(
        np.hypot(1.0, 1.0), abs=0.3
    )
