import numpy as np
import pytest

from fcsim.atmosphere import (
    Atmosphere, air_density, density_altitude, isa_pressure,
    isa_temperature, speed_of_sound,
)


def test_sea_level_standard_values():
    assert isa_temperature(0.0) == pytest.approx(288.15)
    assert isa_pressure(0.0) == pytest.approx(101_325.0)
    assert air_density(0.0) == pytest.approx(1.225, rel=1e-3)
    assert speed_of_sound(0.0) == pytest.approx(340.29, rel=1e-3)


def test_tropopause_values():
    assert isa_temperature(11_000.0) == pytest.approx(216.65)
    assert isa_pressure(11_000.0) == pytest.approx(22_632.0, rel=1e-3)
    assert air_density(11_000.0) == pytest.approx(0.3639, rel=1e-3)


def test_density_at_1000_m():
    assert air_density(1000.0) == pytest.approx(1.1117, rel=1e-3)


def test_hot_day_is_thinner_and_faster():
    ratio = air_density(0.0, dT=20.0) / air_density(0.0)
    assert ratio == pytest.approx(288.15 / 308.15)
    assert speed_of_sound(0.0, dT=20.0) > speed_of_sound(0.0)


def test_density_altitude_round_trip():
    h = np.array([0.0, 1000.0, 5000.0])
    np.testing.assert_allclose(density_altitude(air_density(h)), h, atol=1e-6)


def test_hot_day_ankara_density_altitude():
    atm = Atmosphere(site_alt=938.0, dT=20.0)
    assert 1500.0 < density_altitude(atm.density()) < 1800.0


def test_altitude_out_of_range_rejected():
    with pytest.raises(ValueError):
        air_density(12_000.0)