"""The local pages answer only themselves: not a request naming another host, not a body
another website could send without asking first."""

import http.client
import json
import threading

from flaneur import confirming


def _confirming(tmp_path, photo_id="pm:1"):
    placed = [
        {
            "id": photo_id,
            "title": "a",
            "inputs": "abc",
            "round": 0,
            "camera": {"x": 1.0, "y": 2.0, "heading": 3.0},
        }
    ]
    frames = {"old": b"O", "today": b"T", "outlines": b"L", "parallax": b"P"}
    scenes = {photo_id: {"pairs": [], "plan": {}}}
    server = confirming.serve(
        placed,
        tmp_path / "anchors.jsonl",
        dict,
        lambda _, part, source=False: b"S" + frames[part] if source else frames[part],
        lambda k, source=False: {**scenes[k], "source": True} if source else scenes.get(k),
        0,
    )
    threading.Thread(target=server.handle_request, daemon=True).start()
    return server


def _ask(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()
        server.server_close()


def test_a_request_naming_another_host_is_refused(tmp_path):
    server = _confirming(tmp_path)

    status, _ = _ask(server, "GET", "/api/queue", headers={"Host": "attacker.example"})

    assert status == 403


def test_a_body_not_sent_as_json_is_refused(tmp_path):
    server = _confirming(tmp_path)
    body = json.dumps({"photo_id": "pm:1", "verdict": "no", "seconds": 1})

    status, _ = _ask(server, "POST", "/api/verdict", body, {"Content-Type": "text/plain"})

    assert status == 400
    assert not (tmp_path / "anchors.jsonl").exists()


def test_a_verdict_sent_as_json_is_kept_with_the_camera_judged(tmp_path):
    server = _confirming(tmp_path)
    body = json.dumps({"photo_id": "pm:1", "verdict": "yes", "seconds": 1})

    status, _ = _ask(server, "POST", "/api/verdict", body, {"Content-Type": "application/json"})

    assert status == 200
    row = json.loads((tmp_path / "anchors.jsonl").read_text())
    assert (row["verdict"], row["camera"]) == ("yes", {"x": 1.0, "y": 2.0, "heading": 3.0})


def test_the_photo_today_s_view_and_the_outlines_are_served_apart_and_nothing_else(tmp_path):
    assert _ask(_confirming(tmp_path), "GET", "/frame/pm:1/today") == (200, b"T")
    assert _ask(_confirming(tmp_path), "GET", "/frame/pm:1/old") == (200, b"O")
    assert _ask(_confirming(tmp_path), "GET", "/frame/pm:1/sky")[0] == 404
    assert _ask(_confirming(tmp_path), "GET", "/frame/pm:2/old")[0] == 404


def test_a_no_sent_with_its_reason_is_kept_with_it(tmp_path):
    server = _confirming(tmp_path)
    body = json.dumps({"photo_id": "pm:1", "verdict": "no", "reason": "other_street", "seconds": 1})

    status, _ = _ask(server, "POST", "/api/verdict", body, {"Content-Type": "application/json"})

    assert status == 200
    assert json.loads((tmp_path / "anchors.jsonl").read_text())["reason"] == "other_street"


def test_a_photo_whose_id_has_a_slash_is_served_too(tmp_path):
    server = _confirming(tmp_path, photo_id="src:ark/1")

    assert _ask(server, "GET", "/frame/src%3Aark%2F1/old") == (200, b"O")


def test_the_page_is_told_the_reasons_a_no_can_give(tmp_path):
    status, body = _ask(_confirming(tmp_path), "GET", "/api/reasons")

    assert (status, json.loads(body)) == (200, list(confirming.REASONS))


def test_a_photo_s_scene_and_its_outlines_corrected_are_served(tmp_path):
    status, body = _ask(_confirming(tmp_path), "GET", "/scene/pm:1")
    assert status == 200 and json.loads(body) == {"pairs": [], "plan": {}}

    status, body = _ask(_confirming(tmp_path), "GET", "/frame/pm:1/parallax")
    assert status == 200 and body == b"P"

    status, _ = _ask(_confirming(tmp_path), "GET", "/scene/pm:2")  # not on the page
    assert status == 404


def test_today_s_view_from_the_camera_s_source_is_served_when_asked(tmp_path):
    assert _ask(_confirming(tmp_path), "GET", "/frame/pm:1/today?from=source") == (200, b"ST")
    assert _ask(_confirming(tmp_path), "GET", "/frame/pm:1/today") == (200, b"T")
    scene = _ask(_confirming(tmp_path), "GET", "/scene/pm:1?from=source")[1]
    assert json.loads(scene)["source"] is True
