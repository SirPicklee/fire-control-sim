import numpy as np
import pytest

from fcsim.frames import aer_to_enu, enu_to_aer


@pytest.mark.parametrize(
    "az_deg, el_deg, expected",
    [
        (0.0, 0.0, [0.0, 100.0, 0.0]),     # North
        (90.0, 0.0, [100.0, 0.0, 0.0]),    # East
        (180.0, 0.0, [0.0, -100.0, 0.0]),  # South
        (270.0, 0.0, [-100.0, 0.0, 0.0]),  # West
        (0.0, 90.0, [0.0, 0.0, 100.0]),    # Zenith
    ],
)
def test_cardinal_directions(az_deg, el_deg, expected):
    enu = aer_to_enu(100.0, np.deg2rad(az_deg), np.deg2rad(el_deg))
    np.testing.assert_allclose(enu, expected, atol=1e-9)


def test_west_azimuth_is_positive():
    _, az, _ = enu_to_aer([-50.0, 0.0, 0.0])
    assert az == pytest.approx(np.deg2rad(270.0))


def test_round_trip_aer_enu_aer():
    rng = np.random.default_rng(0)
    r = rng.uniform(10.0, 2000.0, 1000)
    az = rng.uniform(0.0, 2 * np.pi, 1000)
    el = rng.uniform(np.deg2rad(-80), np.deg2rad(80), 1000)  # avoid zenith

    r2, az2, el2 = enu_to_aer(aer_to_enu(r, az, el))
    np.testing.assert_allclose(r2, r, rtol=1e-12)
    np.testing.assert_allclose(az2, az, atol=1e-10)
    np.testing.assert_allclose(el2, el, atol=1e-10)


def test_round_trip_enu_aer_enu():
    rng = np.random.default_rng(1)
    enu = rng.uniform(-1000.0, 1000.0, (1000, 3))
    np.testing.assert_allclose(aer_to_enu(*enu_to_aer(enu)), enu, atol=1e-9)


def test_range_equals_vector_norm():
    rng = np.random.default_rng(2)
    enu = rng.normal(0.0, 500.0, (200, 3))
    r, _, _ = enu_to_aer(enu)
    np.testing.assert_allclose(r, np.linalg.norm(enu, axis=1))