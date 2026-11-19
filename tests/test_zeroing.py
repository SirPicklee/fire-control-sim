import numpy as np
import pytest
from scipy.optimize import brentq

from fcsim.atmosphere import Atmosphere
from fcsim.ballistics import G, firing_table, fly, solve_elevation
from fcsim.projectile import m80_ball

P = m80_ball()


def test_vacuum_zero_matches_analytic():
    # Level target, no sight offset: el = 0.5 * asin(g R / v^2)
    R = 500.0

    def miss(e):
        _, p, _ = fly(P, 0.0, e, vacuum=True, max_range=R + 50).at_horizontal(R)
        return p[2]

    el = brentq(miss, 0.0, 0.05, xtol=1e-10)
    assert el == pytest.approx(0.5 * np.arcsin(G * R / P.v0**2), abs=1e-7)


def test_solved_elevation_hits_target_height():
    tgt = np.array([0.0, 300.0, 10.0])
    el = solve_elevation(P, tgt)
    _, p, _ = fly(P, 0.0, el, max_range=360.0).at_horizontal(300.0)
    assert p[2] == pytest.approx(10.0, abs=1e-3)


def test_path_is_zero_at_zero_range_and_minus_sight_height_at_muzzle():
    _, tab = firing_table(P, [1e-3, 300.0], zero_range=300.0, sight_height=0.30)
    assert tab["path"][0] == pytest.approx(-0.30, abs=1e-3)
    assert tab["path"][1] == pytest.approx(0.0, abs=1e-4)


def test_bullet_above_sight_line_between_crossings():
    _, tab = firing_table(P, [200.0], zero_range=300.0, sight_height=0.30)
    assert tab["path"][0] > 0.0


def test_beyond_zero_needs_positive_correction():
    _, tab = firing_table(P, [500.0], zero_range=300.0, sight_height=0.30)
    assert tab["path"][0] < 0.0
    assert tab["correction_mrad"][0] > 0.0


def test_at_horizontal_distance_is_exact():
    tr = fly(P, 0.7, 0.01, max_range=450.0)
    _, p, _ = tr.at_horizontal(400.0)
    assert np.hypot(p[0], p[1]) == pytest.approx(400.0, abs=1e-6)


def test_hot_day_needs_less_elevation():
    tgt = [0.0, 800.0, 0.0]
    el_cold = solve_elevation(P, tgt, atm=Atmosphere(site_alt=938.0, dT=-19.0))
    el_hot = solve_elevation(P, tgt, atm=Atmosphere(site_alt=938.0, dT=26.0))
    assert el_hot < el_cold