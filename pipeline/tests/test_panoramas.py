import pytest
from shapely.geometry import Point

from flaneur import db, panoramas
from flaneur.gazetteer import to_degrees
from flaneur.panoramax import Picture


def picture(id: str, x: float, y: float, sequence: str = "s", rank: int = 0) -> Picture:
    """A 360° picture at (x, y) Lambert-93 metres."""
    where = to_degrees(Point(x, y))
    return Picture(
        id=id,
        lon=where.x,
        lat=where.y,
        azimuth=0,
        panoramic=True,
        taken="2025-01-01",
        thumb_url="",
        sd_url="",
        sequence=sequence,
        rank=rank,
    )


@pytest.mark.parametrize(
    ("azimuth", "travel", "doubtful"),
    [(0, 0, False), (182, 0, False), (45, 0, True), (355, 10, False), (100, 280, False)],
)
def test_heading_doubtful(azimuth, travel, doubtful):
    assert panoramas.heading_doubtful(azimuth, travel) is doubtful


def test_travel_bearing_follows_the_sequence():
    north = [picture(str(i), 652000, 6861000 + 5 * i, rank=i) for i in range(3)]

    travel = panoramas.travel_bearing(north)

    assert travel["1"] == pytest.approx(0, abs=0.5)
    assert set(travel) == {"0", "1", "2"}


def test_links_keep_the_nearest_per_sector():
    here = picture("here", 652000, 6861000)
    others = [
        picture("north_near", 652000, 6861006),
        picture("north_far", 652000, 6861012),
        picture("east", 652009, 6861000),
        picture("too_close", 651999.5, 6860999.5),
        picture("too_far", 651980, 6861000),
    ]

    found = panoramas.links([here, *others])["here"]

    assert {target for target, _, _ in found} == {"north_near", "east"}
    bearing = {target: b for target, b, _ in found}
    assert bearing["east"] == pytest.approx(90, abs=0.5)


def test_saved_panoramas_load_back(tmp_path):
    conn = db.connect(tmp_path / "f.sqlite")
    here = picture("here", 652000, 6861000)

    panoramas.save(conn, [here])

    assert panoramas.load(conn) == [here]
