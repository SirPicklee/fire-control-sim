import numpy as np
import pytest

from fcsim.ballistics import OMEGA_EARTH, coriolis_accel, fly
from fcsim.projectile import m80_ball
from fcsim.wind import Wind, WindProfile

P = m80_ball()
LAT_ANKARA = np.deg2rad(40.0)


def uniform_crosswind(speed):
    # alpha = 0 -> same speed at every height; from West -> blows East (+x)
    return Wind(WindProfile(speed_ref=speed, from_az=1.5 * np.pi, alpha=0.0))


def test_drift_matches_didion_lag_rule():
    R, W = 500.0, 5.0
    t, p, _ = fly(P, 0.0, 0.0, wind=uniform_crosswind(W), max_range=R + 30).at_horizontal(R)
    didion = W * (t - R / P.v0)
    assert p[0] == pytest.approx(didion, rel=0.03)


def test_drift_is_linear_in_wind_speed():
    R = 500.0
    d5 = fly(P, 0.0, 0.0, wind=uniform_crosswind(5.0), max_range=R + 30).at_horizontal(R)[1][0]
    d10 = fly(P, 0.0, 0.0, wind=uniform_crosswind(10.0), max_range=R + 30).at_horizontal(R)[1][0]
    assert d10 / d5 == pytest.approx(2.0, rel=0.02)


def test_coriolis_is_perpendicular_to_velocity():
    v = np.array([120.0, 700.0, 40.0])
    assert np.dot(coriolis_accel(v, LAT_ANKARA), v) == pytest.approx(0.0, abs=1e-12)


def test_northward_fire_deflects_right_in_north_hemisphere():
    R = 1000.0
    _, p, _ = fly(P, 0.0, 0.0, vacuum=True, latitude=LAT_ANKARA,
                  max_range=R + 30).at_horizontal(R)
    expected = OMEGA_EARTH * np.sin(LAT_ANKARA) * R**2 / P.v0    # vacuum, flat fire
    assert p[0] > 0.0
    assert p[0] == pytest.approx(expected, rel=0.02)


def test_eastward_fire_rises_eotvos():
    common = dict(vacuum=True, t_max=1.0)
    z_cor = fly(P, np.pi / 2, 0.0, latitude=LAT_ANKARA, **common).pos[-1, 2]
    z_ref = fly(P, np.pi / 2, 0.0, **common).pos[-1, 2]
    assert z_cor > z_ref


def test_no_horizontal_coriolis_for_northward_fire_at_equator():
    _, p, _ = fly(P, 0.0, 0.0, vacuum=True, latitude=0.0,
                  max_range=1030.0).at_horizontal(1000.0)
    assert abs(p[0]) < 1e-3