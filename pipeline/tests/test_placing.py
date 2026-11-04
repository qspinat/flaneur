"""The placing page: each photo on the map, then in its panorama."""

import json

import cv2
import numpy as np
import pytest

from flaneur import match, placing, placings, registration
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
    bunched = (SPREAD * 0.15).tolist()
    body = {
        "photo_id": "pm:1",
        "panorama": "p1",
        "view": {"heading": 140.0},
        "photo_px": [[x * 80, y * 60] for x, y in bunched],
        "view_px": [[x * SIDE, y * SIDE] for x, y in bunched],
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
        "view_px": (seen * SIDE).tolist(),
    }

    status, answer = one(sources, "POST", "/api/fit", body)

    assert status == 200
    assert np.allclose(registration.from_json(json.loads(answer)).matrix, proposal.matrix)


def test_the_proposal_and_the_panoramas_near_are_given(page, one):
    sources, proposal = page

    _, near = one(sources, "GET", "/api/near?lon=2.3466&lat=48.852")
    _, given = one(sources, "GET", "/api/proposal/pm:1/p1")

    assert [p["id"] for p in json.loads(near)] == ["p1"]
    assert registration.from_json(json.loads(given)["registration"]) == proposal


def test_a_view_of_a_panorama_is_served_as_a_picture(page, one):
    sources, _ = page

    status, jpeg = one(sources, "GET", "/view/p1?heading=140&pitch=5&fov=75")

    assert status == 200 and cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1).shape == (640, 640, 3)


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
