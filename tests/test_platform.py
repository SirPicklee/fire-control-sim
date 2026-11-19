import numpy as np
import pytest

from fcsim.frames import aer_to_enu, enu_to_aer
from fcsim.platform import (
    Mount, apply_boresight_error, barrel_angles, line_miss_distance,
    parallax_angle,
)


def test_no_offsets_barrel_equals_sensor():
    az_b, el_b = barrel_angles(Mount(), 500.0, 1.0, 0.2)
    assert az_b == pytest.approx(1.0)
    assert el_b == pytest.approx(0.2)


def test_parallax_angle_is_offset_over_range():
    m = Mount(sensor_offset=[0.0, 0.0, 0.3])
    tgt = aer_to_enu(500.0, 0.0, 0.0)
    assert parallax_angle(m, tgt) == pytest.approx(0.3 / 500.0, rel=1e-3)


def test_true_range_removes_parallax():
    m = Mount(sensor_offset=[0.2, 0.0, 0.3])
    tgt = np.array([100.0, 400.0, 50.0])
    rng, az, el = enu_to_aer(tgt - m.sensor_pos)        # perfect sensor + LRF
    az_b, el_b = barrel_angles(m, rng, az, el)
    assert line_miss_distance(m.muzzle_pos, az_b, el_b, tgt) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("R", [50.0, 1000.0])
def test_parallel_mount_miss_equals_offset(R):
    m = Mount(sensor_offset=[0.0, 0.0, 0.3])
    tgt = np.array([0.0, R, 0.3])                       # straight ahead of sensor
    miss = line_miss_distance(m.muzzle_pos, 0.0, 0.0, tgt)  # barrel parallel to sensor
    assert miss == pytest.approx(0.3)


def test_converged_mount_zero_at_convergence_range_only():
    m = Mount(sensor_offset=[0.0, 0.0, 0.3])
    az_b, el_b = barrel_angles(m, 100.0, 0.0, 0.0)      # converged at 100 m
    at_100 = line_miss_distance(m.muzzle_pos, az_b, el_b, [0.0, 100.0, 0.3])
    at_200 = line_miss_distance(m.muzzle_pos, az_b, el_b, [0.0, 200.0, 0.3])
    assert at_100 == pytest.approx(0.0, abs=1e-9)
    assert at_200 == pytest.approx(0.3, rel=1e-3)       # d * |1 - R/R0|


def test_boresight_error_miss_grows_with_range():
    az_b, el_b = apply_boresight_error(0.0, 0.0, 0.5e-3, 0.0)
    miss = line_miss_distance(np.zeros(3), az_b, el_b, [0.0, 1000.0, 0.0])
    assert miss == pytest.approx(1000.0 * np.sin(0.5e-3), rel=1e-6)


def test_offset_rotates_with_heading():
    m = Mount(sensor_offset=[0.0, 1.0, 0.0], heading=np.pi / 2)
    np.testing.assert_allclose(m.sensor_pos, [1.0, 0.0, 0.0], atol=1e-12)