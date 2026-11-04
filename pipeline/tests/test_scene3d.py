"""The 3D walk's scene: today's panoramas, and the photos placed in them."""

from flaneur import scene
from flaneur.export import WalkPhoto
from flaneur.panoramax import Picture

REG = {
    "panorama": "p1",
    "m": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "view": {"heading": 30.0, "pitch": 5.0, "fov": 75.0},
}


def _picture(n: str, lon: float) -> Picture:
    return Picture(
        n,
        lon,
        48.852,
        120.0,
        True,
        "2024-05-01",
        "t",
        f"https://pano/{n}.jpg",
        producer="Jean Dupont",
        roll=2.0,
        pitch=-1.0,
    )


def _photo(n: str) -> WalkPhoto:
    return WalkPhoto(
        n,
        "Rue",
        None,
        [1900, 1900],
        "Eugène Atget",
        "Paris Musées, CC0",
        None,
        2.3466,
        48.852,
        40.0,
        ("Rue", "Galande"),
        5,
        "u",
        None,
    )


def _row(**kw):
    return {"answer": "fits", "registration": REG, "inputs": "now", "flipped": False, **kw}


def _build(shown, copies=None):
    pictures = [_picture("p1", 2.3466), _picture("p2", 2.3467), _picture("p3", 2.3500)]
    walk = [_photo("pm:1"), _photo("pm:2")]
    return scene.build3d(pictures, shown, copies or {}, walk)


def test_the_photos_placed_are_shown_on_their_home_then_their_copies():
    copies = {"pm:1": [{"registration": {**REG, "panorama": "p2"}, "strength": 0.5}]}

    built, _ = _build({"pm:1": _row()}, copies)

    assert [p["id"] for p in built["photos"]] == ["pm:1"]
    assert built["photos"][0]["shown"] == [
        {"panorama": "p1", "registration": REG, "strength": 1.0},
        {"panorama": "p2", "registration": {**REG, "panorama": "p2"}, "strength": 0.5},
    ]


def test_panoramas_holding_a_photo_are_copied_others_are_not():
    built, copied = _build({"pm:1": _row()})

    images = {p["id"]: p["image"] for p in built["panoramas"]}
    assert copied == {"p1"}
    assert images == {
        "p1": "panoramas/p1.jpg",
        "p2": "https://pano/p2.jpg",
        "p3": "https://pano/p3.jpg",
    }


def test_panoramas_come_with_what_levels_them_their_credit_and_the_next_ones_by_index():
    built, _ = _build({})

    p1, p2 = built["panoramas"][:2]
    assert (p1["azimuth"], p1["roll"], p1["pitch"]) == (120.0, 2.0, -1.0)
    assert "Jean Dupont" in built["credits"][p1["credit"]]
    assert [link[0] for link in p1["links"]] == [1]  # p2, 7 m east; p3 is 250 m away
    assert [link[0] for link in p2["links"]] == [0]


def test_a_flipped_photo_is_shown_flipped_and_its_credit_says_so():
    built, _ = _build({"pm:1": _row(flipped=True)})

    photo = built["photos"][0]
    assert photo["flipped"] is True and "image retournée" in photo["credit"]


def test_a_photo_on_a_panorama_no_longer_there_is_left_out_and_a_copy_too():
    gone = {**REG, "panorama": "gone"}
    copies = {"pm:1": [{"registration": gone, "strength": 0.5}]}

    built, copied = _build({"pm:1": _row(), "pm:2": _row(registration=gone)}, copies)

    assert [p["id"] for p in built["photos"]] == ["pm:1"]
    assert [w["panorama"] for w in built["photos"][0]["shown"]] == ["p1"] and copied == {"p1"}
