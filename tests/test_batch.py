import numpy as np
import pytest

from fcsim.ballistics import G, fly, fly_batch
from fcsim.projectile import m80_ball

P = m80_ball()


def test_single_projectile_matches_scalar_solver():
    b = fly_batch(P, 0.3, 0.02, t_end=0.5)
    s = fly(P, 0.3, 0.02, t_max=0.5)
    np.testing.assert_allclose(b.pos[-1, 0], s.pos[-1], atol=1e-6)
    np.testing.assert_allclose(b.vel[-1, 0], s.vel[-1], atol=1e-6)


def test_each_projectile_matches_its_own_scalar_run():
    az = np.array([0.0, 0.5, 1.0])
    el = np.array([0.0, 0.01, -0.01])
    b = fly_batch(P, az, el, t_end=0.4)
    for k in range(3):
        s = fly(P, az[k], el[k], t_max=0.4)
        np.testing.assert_allclose(b.pos[-1, k], s.pos[-1], atol=1e-6)


def test_batch_at_horizontal_matches_scalar():
    b = fly_batch(P, [0.0, 0.2], [0.0, 0.0], t_end=0.6)
    t_b, p_b, _ = b.at_horizontal(300.0)
    for k, az in enumerate([0.0, 0.2]):
        t_s, p_s, _ = fly(P, az, 0.0, max_range=330.0).at_horizontal(300.0)
        assert t_b[k] == pytest.approx(t_s, abs=1e-6)
        np.testing.assert_allclose(p_b[k], p_s, atol=1e-5)


def test_unreached_distance_gives_nan():
    b = fly_batch(P, [0.0, 0.0], [0.0, 0.0], v0=[825.0, 100.0], t_end=0.3)
    t, pos, _ = b.at_horizontal(200.0)
    assert np.isfinite(t[0]) and np.isnan(t[1])
    assert np.all(np.isnan(pos[1]))


def test_vacuum_batch_matches_analytic():
    v0, el = np.array([300.0, 500.0]), np.array([0.2, 0.1])
    b = fly_batch(P, 0.0, el, v0=v0, vacuum=True, t_end=1.0)
    z = v0 * np.sin(el) * 1.0 - 0.5 * G * 1.0**2
    np.testing.assert_allclose(b.pos[-1, :, 2], z, atol=1e-9)