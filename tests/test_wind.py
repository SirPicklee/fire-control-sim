import numpy as np
import pytest

from fcsim.wind import GustModel, Wind, WindProfile


def test_speed_at_reference_height():
    p = WindProfile(speed_ref=5.0, pivot_height=2.0)
    assert p.speed(8.0) == pytest.approx(5.0)          # 2 + 8 = 10 m


def test_power_law_increase_with_height():
    p = WindProfile(speed_ref=5.0, pivot_height=0.0)
    assert p.speed(80.0) == pytest.approx(5.0 * 8.0 ** (1 / 7))


def test_north_wind_blows_south():
    p = WindProfile(speed_ref=5.0, from_az=0.0, pivot_height=10.0)
    np.testing.assert_allclose(p.velocity(0.0), [0.0, -5.0, 0.0], atol=1e-12)


def test_west_wind_blows_east():
    p = WindProfile(speed_ref=5.0, from_az=1.5 * np.pi, pivot_height=10.0)
    np.testing.assert_allclose(p.velocity(0.0), [5.0, 0.0, 0.0], atol=1e-12)


def test_below_ground_is_clipped_not_nan():
    s = WindProfile(speed_ref=5.0).speed(-10.0)
    assert np.isfinite(s) and s > 0


def test_gust_statistics():
    gm = GustModel(sigma=1.5, sigma_up=0.5, tau=2.0)
    dt = 0.05
    t = np.arange(0.0, 10_000.0, dt)
    g = gm.simulate(t, np.random.default_rng(0))
    assert g[:, 0].std() == pytest.approx(1.5, rel=0.05)
    assert g[:, 2].std() == pytest.approx(0.5, rel=0.05)
    lag1 = np.corrcoef(g[:-1, 0], g[1:, 0])[0, 1]
    assert lag1 == pytest.approx(np.exp(-dt / 2.0), abs=0.01)


def test_gusts_reproducible_with_seed():
    a = Wind(gust=GustModel(), rng=np.random.default_rng(42)).velocity(3.3, 50.0)
    b = Wind(gust=GustModel(), rng=np.random.default_rng(42)).velocity(3.3, 50.0)
    np.testing.assert_array_equal(a, b)


def test_wind_without_gust_equals_profile():
    p = WindProfile(speed_ref=4.0, from_az=0.3)
    np.testing.assert_allclose(Wind(p).velocity(1.0, 30.0), p.velocity(30.0))