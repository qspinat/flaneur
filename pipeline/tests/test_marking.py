"""The landmark marking tool: marks checked, written per photo, and queued."""

import json

import pytest

from flaneur import landmarks, localpage, marking, review


def test_a_photos_marks_replace_those_it_had(tmp_path):
    path = tmp_path / "sights.jsonl"
    marking.save(path, "pm:1", "a1", [{"landmarks": ["notre-dame-tours"], "column": 0.3}])
    marking.save(path, "pm:2", "b2", [{"other": "Panthéon", "column": 0.6, "top": 0.1}])

    marking.save(path, "pm:1", "a1", [{"landmarks": ["saint-severin-tour"], "column": 0.45}])

    rows = landmarks.rows(path)
    assert [(r["photo_id"], r.get("landmarks"), r["column"]) for r in rows] == [
        ("pm:2", None, 0.6),
        ("pm:1", ["saint-severin-tour"], 0.45),
    ]
    assert list(landmarks.marked(path)) == ["pm:1"]  # another landmark: not for the matcher


def test_a_photo_checked_with_no_landmark_is_kept_as_such(tmp_path):
    path = tmp_path / "sights.jsonl"

    marking.save(path, "pm:1", "a1", [])

    assert landmarks.rows(path) == [
        {"photo_id": "pm:1", "image": "a1", "checked": True, "none": True}
    ]
    assert landmarks.marked(path) == {}


@pytest.mark.parametrize(
    "mark",
    [
        {"landmarks": ["notre-dame-tour"], "column": 0.3},
        {"landmarks": [], "column": 0.3},
        {"landmarks": ["notre-dame-tours"], "column": 1.3},
        {"landmarks": ["notre-dame-tours"], "column": 0.3, "top": -0.1},
        {"other": " ", "column": 0.3},
    ],
)
def test_a_mark_written_wrong_is_refused(tmp_path, mark):
    with pytest.raises(ValueError):
        marking.save(tmp_path / "sights.jsonl", "pm:1", "a1", [mark])


def test_the_queue_puts_proposed_marks_first_and_tells_marks_made_on_another_crop(tmp_path):
    images = {}
    for name in ("pm:1", "pm:2", "pm:3", "pm:4"):
        images[name] = tmp_path / f"{name[-1]}.webp"
        images[name].write_bytes(name.encode())
    path = tmp_path / "sights.jsonl"
    proposed = {"photo_id": "pm:2", "image": "recropped", "landmarks": ["notre-dame-tours"]}
    path.write_text(json.dumps({**proposed, "column": 0.3}) + "\n")  # not saved by the user
    marking.save(path, "pm:3", landmarks.image_hash(images["pm:3"]), [])
    tower = {"landmarks": ["saint-severin-tour"], "column": 0.4}
    marking.save(path, "pm:4", landmarks.image_hash(images["pm:4"]), [tower])
    photos = [{"id": name, "title": name, "years": [], "place": [0, 0]} for name in images]

    queue = marking.queue(photos, path, images.get)

    assert [(p["id"], p["state"], p["stale"]) for p in queue] == [
        ("pm:2", "check", True),
        ("pm:1", "new", False),
        ("pm:3", "done", False),
        ("pm:4", "done", False),
    ]
    assert json.dumps(queue)  # served as is


def test_the_queue_offers_the_landmarks_standing_in_a_photos_years(tmp_path):
    image = tmp_path / "1.webp"
    image.write_bytes(b"webp")
    severin = list(landmarks.LANDMARKS["saint-severin-tour"].at)
    photos = [{"id": "pm:1", "title": "a", "years": [1852, 1852], "place": severin}]

    [item] = marking.queue(photos, tmp_path / "sights.jsonl", lambda _: image)

    assert item["offer"][0] == "saint-severin-tour"  # nearest first
    assert "sainte-chapelle-fleche" not in item["offer"]  # raised in 1853
    assert "notre-dame-tours" in item["offer"]


def test_the_page_serves_a_photo_asked_for_by_its_quoted_id(tmp_path):
    import threading
    import urllib.request

    image = tmp_path / "1.webp"
    image.write_bytes(b"webp")
    server = marking.serve(
        [{"id": "pm:1", "title": "a", "years": [], "place": [0, 0]}],
        tmp_path / "sights.jsonl",
        lambda _: image,
        0,
    )
    threading.Thread(target=server.handle_request, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/image/pm%3A1"
        assert urllib.request.urlopen(url, timeout=5).read() == b"webp"
    finally:
        server.server_close()


@pytest.mark.parametrize("mark", [0.5, {"landmarks": ["notre-dame-tours"]}])
def test_a_mark_not_an_object_or_not_placed_is_refused(tmp_path, mark):
    with pytest.raises(ValueError):
        marking.save(tmp_path / "sights.jsonl", "pm:1", "a1", [mark])


def test_marks_proposed_replace_those_proposed_never_the_users(tmp_path):
    path = tmp_path / "sights.jsonl"
    marking.save(path, "pm:1", "a1", [])  # the user saw it: nothing to propose
    tower = {"landmarks": ["saint-severin-tour"], "column": 0.4}
    towers = {"landmarks": ["notre-dame-tours"], "column": 0.7}

    assert not marking.propose(path, "pm:1", "a1", [tower])
    assert marking.propose(path, "pm:2", "b2", [tower])
    assert marking.propose(path, "pm:2", "b2", [towers])  # looked for again: replaced
    assert marking.propose(path, "pm:3", "c3", [tower])
    assert marking.propose(path, "pm:3", "c3", [])  # nothing found again: they go

    rows = landmarks.rows(path)
    assert [(r["photo_id"], r.get("checked"), r.get("landmarks")) for r in rows] == [
        ("pm:1", True, None),
        ("pm:2", None, ["notre-dame-tours"]),
    ]


@pytest.mark.parametrize("page", [marking.PAGE, review.PAGE])
def test_the_local_pages_are_served_with_the_shared_zoom(page):
    served = localpage.page(page).decode()

    assert "function zoomable(" in served
    assert localpage.INCLUDE not in served


def test_a_page_not_asking_for_the_shared_zoom_is_refused(tmp_path):
    page = tmp_path / "page.html"
    page.write_text("<script src='zoom.js'></script>")

    with pytest.raises(ValueError, match="zoom"):
        localpage.page(page)
