import numpy as np
import pytest

from fcsim.ballistics import G, fly
from fcsim.projectile import m80_ball
from fcsim.wind import Wind, WindProfile

P = m80_ball()


def test_vacuum_matches_analytic_parabola():
    v0, el = 500.0, 0.3
    tr = fly(P, 0.0, el, v0=v0, vacuum=True, t_max=1.0)
    y = v0 * np.cos(el) * tr.t
    z = v0 * np.sin(el) * tr.t - 0.5 * G * tr.t**2
    np.testing.assert_allclose(tr.pos[:, 1], y, atol=1e-9)
    np.testing.assert_allclose(tr.pos[:, 2], z, atol=1e-9)


def test_vacuum_apex_height():
    v0, el = 100.0, np.deg2rad(30.0)
    tr = fly(P, 0.0, el, v0=v0, vacuum=True, t_max=6.0)
    apex = (v0 * np.sin(el)) ** 2 / (2 * G)
    assert tr.pos[:, 2].max() == pytest.approx(apex, abs=1e-3)


def test_vacuum_conserves_energy():
    tr = fly(P, 0.5, 0.2, vacuum=True, t_max=2.0)
    e = 0.5 * tr.speed**2 + G * tr.pos[:, 2]
    np.testing.assert_allclose(e, e[0], rtol=1e-9)


def test_drag_slows_projectile():
    # Direction check only; magnitude is validated against a firing table in Step 6.
    air = fly(P, 0.0, 0.0, t_max=0.5)
    vac = fly(P, 0.0, 0.0, t_max=0.5, vacuum=True)
    assert vac.speed[-1] - air.speed[-1] > 100.0     # drag clearly removes speed
    assert air.pos[-1, 1] < vac.pos[-1, 1]           # and shortens the path


def test_time_step_convergence():
    a = fly(P, 0.0, 0.002, dt=2e-3, max_range=520.0).at_range(500.0)
    b = fly(P, 0.0, 0.002, dt=1e-3, max_range=520.0).at_range(500.0)
    np.testing.assert_allclose(a[1], b[1], atol=1e-3)     # < 1 mm


def test_tailwind_carries_further_than_headwind():
    tail = Wind(WindProfile(speed_ref=10.0, from_az=np.pi))   # blows north
    head = Wind(WindProfile(speed_ref=10.0, from_az=0.0))     # blows south
    y_tail = fly(P, 0.0, 0.0, wind=tail, t_max=1.0).pos[-1, 1]
    y_head = fly(P, 0.0, 0.0, wind=head, t_max=1.0).pos[-1, 1]
    assert y_tail > y_head


def test_west_crosswind_drifts_east():
    cross = Wind(WindProfile(speed_ref=5.0, from_az=1.5 * np.pi))
    _, pos, _ = fly(P, 0.0, 0.0, wind=cross, max_range=520.0).at_range(500.0)
    assert pos[0] > 0.0


def test_at_range_interpolates_to_requested_range():
    tr = fly(P, 0.0, 0.01, max_range=320.0)
    _, pos, _ = tr.at_range(300.0)
    assert np.linalg.norm(pos) == pytest.approx(300.0, abs=1e-6)


def test_azimuth_east_flies_east():
    tr = fly(P, np.pi / 2, 0.0, t_max=0.2)
    assert tr.pos[-1, 0] > 100.0
    assert abs(tr.pos[-1, 1]) < 1e-6