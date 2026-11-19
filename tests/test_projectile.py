import numpy as np
import pytest

from fcsim.projectile import (
    G7_CD, G7_MACH, LB_IN2_TO_KG_M2, Projectile, m80_ball,
)

A_SL = 340.29   # speed of sound, sea level
RHO = 1.225


def test_cd_interpolates_table():
    p = Projectile("ref", 0.01, 0.008, 800.0, G7_MACH, G7_CD, form_factor=1.0)
    assert p.cd(1.0) == pytest.approx(0.380)
    assert p.cd(0.975) == pytest.approx((0.164 + 0.380) / 2)


def test_drag_opposes_velocity():
    p = m80_ball()
    v = np.array([100.0, 700.0, 50.0])
    a = p.drag_accel(v, RHO, A_SL)
    assert np.dot(a, v) < 0
    np.testing.assert_allclose(np.cross(a, v), 0.0, atol=1e-6)


def test_drag_scales_with_speed_squared_at_constant_cd():
    p = m80_ball()   # Cd is constant below Mach 0.7
    a1 = np.linalg.norm(p.drag_accel([0, 100.0, 0], RHO, A_SL))
    a2 = np.linalg.norm(p.drag_accel([0, 200.0, 0], RHO, A_SL))
    assert a2 / a1 == pytest.approx(4.0)


def test_drag_proportional_to_density():
    p = m80_ball()
    a1 = p.drag_accel([0, 800.0, 0], 1.0, A_SL)
    a2 = p.drag_accel([0, 800.0, 0], 0.5, A_SL)
    np.testing.assert_allclose(a2, 0.5 * a1)


def test_zero_velocity_gives_zero_drag():
    a = m80_ball().drag_accel([0.0, 0.0, 0.0], RHO, A_SL)
    np.testing.assert_array_equal(a, np.zeros(3))


def test_bc_formula_cd_area_over_mass():
    bc = 0.200
    p = Projectile.from_g7_bc("x", 9.53e-3, 7.82e-3, 840.0, bc)
    lhs = p.cd(2.0) * p.area / p.mass
    rhs = np.pi * 0.298 / (4.0 * bc * LB_IN2_TO_KG_M2)
    assert lhs == pytest.approx(rhs)


def test_m80_initial_deceleration_is_plausible():
    p = m80_ball()
    a0 = np.linalg.norm(p.drag_accel([0.0, p.v0, 0.0], RHO, A_SL))
    assert 400.0 < a0 < 1000.0   # m/s^2, roughly 1 m/s lost per metre