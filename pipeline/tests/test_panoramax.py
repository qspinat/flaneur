"""How a Panoramax picture is credited."""

import httpx
import pytest

from flaneur import panoramax


def _picture(producer):
    return panoramax.Picture("p1", 2.3, 48.8, 0.0, True, "2024-01-01", "t", "s", producer=producer)


@pytest.mark.parametrize(
    ("producer", "credit"),
    [
        ("Immergis pour la Ville de Paris", "Panoramax, Immergis pour la Ville de Paris"),
        ("zach@rando360.com", "Panoramax, zach"),  # an address, for harvesters: not shown
        ("Jean Dupont <jean.dupont@example.fr>", "Panoramax, Jean Dupont"),
        ("Jean Dupont (jean.dupont@example.fr)", "Panoramax, Jean Dupont"),
        (None, "Panoramax"),
    ],
)
def test_a_picture_is_credited_to_its_producer_never_to_an_address(producer, credit):
    assert panoramax.credit_of(_picture(producer)) == credit


def test_a_picture_looked_up_gives_its_original_full_size():
    feature = {
        "id": "p1",
        "geometry": {"coordinates": [2.3, 48.8]},
        "properties": {"datetime": "2024-01-01T10:00:00Z"},
        "assets": {
            "thumb": {"href": "https://pano.example/derivates/p1/thumb.jpg"},
            "sd": {"href": "https://pano.example/derivates/p1/sd.jpg"},
            "hd": {"href": "https://pano.example/images/p1.jpg"},  # each instance its own way
        },
    }
    server = httpx.MockTransport(lambda request: httpx.Response(200, json={"features": [feature]}))

    found = panoramax.by_id(httpx.Client(transport=server), "p1")

    assert found.hd_url == "https://pano.example/images/p1.jpg"
