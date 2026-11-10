"""The local pages answer only themselves: not a request naming another host, not a body
another website could send without asking first."""

import http.client
import json
import threading

from flaneur import confirming


def _confirming(tmp_path):
    placed = [
        {
            "id": "pm:1",
            "title": "a",
            "inputs": "abc",
            "round": 0,
            "camera": {"x": 1.0, "y": 2.0, "heading": 3.0},
        }
    ]
    frames = {"old": b"O", "today": b"T", "outlines": b"L"}
    server = confirming.serve(
        placed, tmp_path / "anchors.jsonl", dict, lambda _, part: frames[part], 0
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
