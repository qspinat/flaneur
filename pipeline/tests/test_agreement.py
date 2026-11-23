"""Volunteers' panoramas turned to agree with their neighbours, on built scenes."""

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from flaneur import agreement, depth, match
from flaneur.panoramax import Picture


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _street() -> np.ndarray:
    """Details on two walls 8 m apart (x = ±4) along a street (north, y)."""
    return np.array(
        [[x, y, z] for x in (-4.0, 4.0) for y in np.arange(2.0, 40.0, 1.0) for z in (-1.0, 3.0)]
    )


def _pair(b: np.ndarray, rotvec_deg) -> tuple[np.ndarray, np.ndarray]:
    """A panorama at the origin and a neighbour at `b` whose orientation is off by a rotation:
    their rays to the street's details, as each records them."""
    street = _street()
    turn = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    return _unit(street), _unit(street - b) @ turn  # recorded = turn⁻¹ · true


def test_a_neighbour_s_turn_is_measured_from_their_pair():
    b = np.array([0.0, 6.0, 0.0])
    rays_a, rays_b = _pair(b, (0.4, -0.8, 1.5))

    r = agreement.measured(np.zeros(3), b, rays_a, rays_b)

    np.testing.assert_allclose(r, (0.4, -0.8, 1.5), atol=0.01)


def test_a_neighbour_too_near_says_nothing():
    b = np.array([0.0, depth.NEAR_M[0] - 0.5, 0.0])
    rays_a, rays_b = _pair(b, (0.4, -0.8, 1.5))

    assert agreement.measured(np.zeros(3), b, rays_a, rays_b) is None


def test_a_pair_set_right_by_a_large_turn_says_nothing():
    b = np.array([0.0, 6.0, 0.0])
    rays_a, rays_b = _pair(b, (0.0, 0.0, agreement.MOST_TURNED_DEG + 2.0))

    assert agreement.measured(np.zeros(3), b, rays_a, rays_b) is None


def _graph(outlier: bool = False):
    """Eight panoramas, two of the survey (right as they stand), all linked; each link says
    how far the second turns against the first (a little noise; one link, `outlier`, wildly
    wrong); and the turn that sets each right."""
    rng = np.random.default_rng(0)
    ids = [f"p{k}" for k in range(8)]
    truth = {i: rng.normal(0.0, 1.0, 3) for i in ids}
    held = {"p0", "p1"}
    truth.update({i: np.zeros(3) for i in held})
    links = [
        (i, j, truth[j] - truth[i] + rng.normal(0.0, 0.05, 3))
        for n, i in enumerate(ids)
        for j in ids[n + 1 :]
    ]
    if outlier:
        i, j, r = links[5]
        links[5] = (i, j, r + [20.0, -15.0, 10.0])
    return links, held, truth


def test_each_volunteer_takes_the_turn_its_links_agree_on():
    links, held, truth = _graph()

    found = agreement.averaged(links, held)

    assert found.keys() == truth.keys() - held  # the survey's held where they stand
    for i, c in found.items():
        np.testing.assert_allclose(c, truth[i], atol=0.1)


def test_one_link_wildly_wrong_does_not_drag_the_others():
    links, held, truth = _graph(outlier=True)

    found = agreement.averaged(links, held)

    for i, c in found.items():  # its pull bounded (SCALE_DEG): a plain average moves them 4°
        np.testing.assert_allclose(c, truth[i], atol=0.25)


def test_no_link_turns_nothing():
    assert agreement.averaged([], {"p0"}) == {}


def test_a_camera_s_angles_are_read_back_from_its_axes():
    rng = np.random.default_rng(0)
    angles = np.column_stack(
        [rng.uniform(0, 360, 200), rng.uniform(-15, 15, 200), rng.uniform(-15, 15, 200)]
    )

    found = np.array([agreement.camera_angles(match.camera_axes(*a)) for a in angles])

    np.testing.assert_allclose(found, angles, atol=1e-9)


def _picture(azimuth=40.0, pitch=2.0, roll=-3.0) -> Picture:
    return Picture("v", 2.3466, 48.852, azimuth, True, "2024-01-01", "", "", roll=roll, pitch=pitch)


def test_a_panorama_turned_faces_and_leans_as_its_axes_turned():
    p, rotvec = _picture(), np.array([0.7, -1.2, 2.5])

    turned = agreement.corrected(p, rotvec)

    expected = match.camera_axes(p.azimuth, p.pitch, p.roll) @ (
        Rotation.from_rotvec(np.radians(rotvec)).as_matrix().T
    )
    np.testing.assert_allclose(
        match.camera_axes(turned.azimuth, turned.pitch, turned.roll), expected, atol=1e-12
    )
    assert (turned.lon, turned.lat) == (p.lon, p.lat)


def test_a_correction_is_applied_only_to_the_panorama_it_was_computed_against():
    p = _picture()
    kept = {"v": agreement.Agreed((0.0, 0.0, 1.0), (40.0, 2.0, -3.0))}

    (now,) = agreement.applied([p], kept)
    (moved,) = agreement.applied([replace(p, roll=-3.05)], kept)  # levelled otherwise since
    (other,) = agreement.applied([replace(p, id="w")], kept)

    assert now == agreement.corrected(p, np.array([0.0, 0.0, 1.0]))
    assert moved == replace(p, roll=-3.05) and other == replace(p, id="w")


def test_corrections_are_read_as_they_were_kept(tmp_path):
    path = tmp_path / "agreed.json"
    found = {"v": np.array([0.1, -0.2, 0.3]), "w": np.array([1.0, 0.0, -1.5])}
    against = {"v": (40.0, 2.0, -3.0), "w": (350.5, -1.0, 0.25)}

    agreement.save(path, found, against)

    assert agreement.load(path) == {
        "v": agreement.Agreed((0.1, -0.2, 0.3), (40.0, 2.0, -3.0)),
        "w": agreement.Agreed((1.0, 0.0, -1.5), (350.5, -1.0, 0.25)),
    }


def test_corrections_computed_otherwise_are_not_read(tmp_path, monkeypatch):
    path = tmp_path / "agreed.json"
    agreement.save(path, {"v": np.array([0.1, -0.2, 0.3])}, {"v": (40.0, 2.0, -3.0)})

    monkeypatch.setattr(agreement, "PULL", 0.02)

    assert agreement.load(path) == {} and agreement.load(tmp_path / "none.json") == {}
