"""The 3D walk's scene: today's panoramas, and the photos placed in them."""

from flaneur import facades, scene
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
    placed = {"registration": REG, "inputs": "now", "flipped": False, "status": "checked"}
    return {
        **placed,
        "unchecked_area": False,
        "proposed_marks": False,
        "unsure_sources": False,
        **kw,
    }


def _build(shown, copies=None, walls_of=lambda panorama: None):
    pictures = [_picture("p1", 2.3466), _picture("p2", 2.3467), _picture("p3", 2.3500)]
    walk = [_photo("pm:1"), _photo("pm:2")]
    return scene.build3d(pictures, shown, copies or {}, walk, walls_of)


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


def test_a_photo_says_how_sure_its_placement_is():
    row = _row(status="automatic", unchecked_area=True, proposed_marks=False, unsure_sources=True)

    built, _ = _build({"pm:1": row})

    photo = built["photos"][0]
    assert (
        photo["status"],
        photo["unchecked_area"],
        photo["proposed_marks"],
        photo["unsure_sources"],
    ) == ("automatic", True, False, True)


def test_the_walls_around_a_photo_standing_off_its_panorama_go_with_that_panorama_alone():
    standing = {**REG, "standing": {"position": [1.0, 0.5, 0.2]}}
    copies = {"pm:1": [{"registration": {**REG, "panorama": "p2"}, "strength": 0.5}]}
    wall = facades.Wall((652010.123, 6861525.0), (652019.0, 6861525.0), 47.456, "fitted")
    around = facades.Scene("p1", (652000.0, 6861520.0, 35.0), 32.5, (wall,), (), 8)
    asked = []

    built, _ = _build(
        {"pm:1": _row(registration=standing)},
        copies,
        lambda panorama: asked.append(panorama) or around,
    )

    p1, p2, _ = built["panoramas"]
    assert p1["walls"] == [[10.12, 5.0, 19.0, 5.0, 12.46]]  # east, north, top: metres off it
    assert p1["ground"] == -2.5
    assert "walls" not in p2 and asked == ["p1"]  # a copy keeps its matrix


def test_a_photo_resting_on_panoramas_nothing_checks_keeps_its_matrix():
    standing = {**REG, "standing": {"position": [1.0, 0.5, 0.2]}}
    asked = []

    built, _ = _build(
        {"pm:1": _row(registration=standing, unsure_sources=True)},
        walls_of=lambda panorama: asked.append(panorama),
    )

    assert built["photos"][0]["shown"][0]["registration"] == REG and asked == []
