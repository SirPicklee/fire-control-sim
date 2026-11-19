import dataclasses

import numpy as np
import pytest

from fcsim.messages import (
    EOMeasurement, LRFMeasurement, RadarDetection, TargetTruth, Track,
)

R3 = np.diag([4.0, 1e-4, 1e-4])


def test_target_truth_defaults_acc_zero():
    tt = TargetTruth(0.0, [1, 2, 3], [4, 5, 6])
    np.testing.assert_array_equal(tt.acc, np.zeros(3))


def test_vector_shape_is_checked():
    with pytest.raises(ValueError, match="pos"):
        TargetTruth(0.0, [1, 2], [0, 0, 0])


def test_radar_detection_valid():
    det = RadarDetection(1.0, 1500.0, 0.5, 0.1, R3)
    assert det.R.shape == (3, 3)


@pytest.mark.parametrize("az, el", [(-0.1, 0.0), (2 * np.pi, 0.0), (0.0, 2.0)])
def test_angles_out_of_range_rejected(az, el):
    with pytest.raises(ValueError):
        RadarDetection(0.0, 100.0, az, el, R3)


def test_covariance_must_be_symmetric():
    with pytest.raises(ValueError, match="symmetric"):
        EOMeasurement(0.0, 0.1, 0.1, [[1e-8, 1e-9], [0.0, 1e-8]])


def test_covariance_must_be_psd():
    with pytest.raises(ValueError, match="semi-definite"):
        EOMeasurement(0.0, 0.1, 0.1, [[1e-8, 0.0], [0.0, -1e-8]])


def test_lrf_dropout_has_nan_range():
    m = LRFMeasurement.dropout(t=2.0, sigma=0.5)
    assert not m.valid and np.isnan(m.rng)


def test_lrf_valid_requires_positive_range():
    with pytest.raises(ValueError):
        LRFMeasurement(0.0, -5.0, 0.5)


def test_track_pos_vel_views():
    trk = Track(0.0, 1, [1, 2, 3, 4, 5, 6], np.eye(6))
    np.testing.assert_array_equal(trk.pos, [1, 2, 3])
    np.testing.assert_array_equal(trk.vel, [4, 5, 6])


def test_track_status_checked():
    with pytest.raises(ValueError, match="status"):
        Track(0.0, 1, np.zeros(6), np.eye(6), status="alive")


def test_messages_are_frozen():
    det = RadarDetection(1.0, 1500.0, 0.5, 0.1, R3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        det.rng = 10.0