import numpy as np
import pytest

from fcsim.frames import (
    aer_to_enu, body_to_enu, body_to_enu_matrix, enu_to_aer,
    enu_to_body, gimbal_angles, rot_x, rot_y, rot_z,
)

D = np.deg2rad


@pytest.mark.parametrize("rot", [rot_x, rot_y, rot_z])
def test_rotation_is_orthonormal(rot):
    for a in np.random.default_rng(0).uniform(-np.pi, np.pi, 20):
        R = rot(a)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)


@pytest.mark.parametrize("rot", [rot_x, rot_y, rot_z])
def test_inverse_is_negative_angle(rot):
    np.testing.assert_allclose(rot(0.7) @ rot(-0.7), np.eye(3), atol=1e-12)


def test_heading_east_points_forward_to_east():
    C = body_to_enu_matrix(D(90), 0.0, 0.0)
    np.testing.assert_allclose(body_to_enu([0, 1, 0], C), [1, 0, 0], atol=1e-12)


def test_pitch_up_raises_boresight():
    C = body_to_enu_matrix(0.0, D(30), 0.0)
    _, _, el = enu_to_aer(body_to_enu([0, 1, 0], C))
    assert el == pytest.approx(D(30))


def test_roll_right_lowers_right_axis():
    C = body_to_enu_matrix(0.0, 0.0, D(90))
    np.testing.assert_allclose(body_to_enu([1, 0, 0], C), [0, 0, -1], atol=1e-12)


def test_round_trip_enu_body_enu():
    rng = np.random.default_rng(1)
    C = body_to_enu_matrix(*rng.uniform(-1, 1, 3))
    v = rng.normal(0, 100, (500, 3))
    np.testing.assert_allclose(body_to_enu(enu_to_body(v, C), C), v, atol=1e-9)


def test_level_platform_gimbal_equals_world_angles():
    tgt = aer_to_enu(500.0, D(120), D(15))
    az_b, el_b = gimbal_angles(tgt, 0.0, 0.0, 0.0)
    assert az_b == pytest.approx(D(120))
    assert el_b == pytest.approx(D(15))


def test_heading_offsets_gimbal_azimuth():
    tgt = aer_to_enu(500.0, D(45), D(10))
    az_b, el_b = gimbal_angles(tgt, D(45), 0.0, 0.0)
    assert az_b == pytest.approx(0.0, abs=1e-9)
    assert el_b == pytest.approx(D(10))


def test_pitched_platform_sees_target_on_boresight():
    tgt = aer_to_enu(300.0, 0.0, D(20))
    az_b, el_b = gimbal_angles(tgt, 0.0, D(20), 0.0)
    assert el_b == pytest.approx(0.0, abs=1e-9)