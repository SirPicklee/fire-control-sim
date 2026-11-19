import numpy as np
import pytest

from fcsim.calibration import fit_drag_correction, speed_and_tof, with_drag_correction
from fcsim.projectile import m80_ball

P = m80_ball()


def test_unit_factors_leave_drag_unchanged():
    p = with_drag_correction(P, [1.0, 2.0], [1.0, 1.0])
    for m in (0.6, 1.0, 1.05, 1.7, 2.4):
        assert p.cd(m) == pytest.approx(P.cd(m))


def test_constant_factor_scales_drag():
    p = with_drag_correction(P, [1.0, 2.0], [1.1, 1.1])
    assert p.cd(1.5) == pytest.approx(1.1 * P.cd(1.5))


def test_factor_is_clamped_outside_nodes():
    p = with_drag_correction(P, [1.2, 2.0], [1.3, 1.0])
    assert p.cd(0.8) == pytest.approx(1.3 * P.cd(0.8))
    assert p.cd(3.0) == pytest.approx(1.0 * P.cd(3.0))


def test_speed_decreases_and_tof_increases_with_range():
    v, t = speed_and_tof(P, [100.0, 300.0, 500.0])
    assert np.all(np.diff(v) < 0)
    assert np.all(np.diff(t) > 0)


def test_fit_recovers_known_correction():
    nodes = np.array([1.2, 1.6, 2.2])
    truth = np.array([1.10, 1.04, 1.00])
    ranges = np.arange(100.0, 701.0, 100.0)            # Mach stays above ~1.15
    v_ref, _ = speed_and_tof(with_drag_correction(P, nodes, truth), ranges)
    fit = fit_drag_correction(P, ranges, v_ref, nodes)
    np.testing.assert_allclose(fit.factors, truth, atol=0.02)
    assert fit.rms_rel < 1e-4