import numpy as np
import pytest

from fcsim.messages import RadarDetection, TargetTruth
from fcsim.radar import (
    RadarParams, SearchRadar, measurement_sigma, pd_swerling1, snr,
)

P = RadarParams()


def truth_at(r, az=0.3, el=0.05):
    pos = r * np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    return TargetTruth(0.0, pos, np.zeros(3))


def test_snr_follows_fourth_power_law():
    assert snr(P, P.ref_range, P.ref_rcs) == pytest.approx(P.snr_ref)
    assert snr(P, 2 * P.ref_range, P.ref_rcs) == pytest.approx(P.snr_ref / 16)
    assert snr(P, P.ref_range, 2 * P.ref_rcs) == pytest.approx(2 * P.snr_ref)


def test_pd_at_reference_equals_spec():
    assert pd_swerling1(P.snr_ref, P.pfa) == pytest.approx(P.ref_pd)


def test_simulated_pd_matches_swerling1_formula():
    radar = SearchRadar(RadarParams(n_cells=0.0))
    r = 3500.0
    expected = pd_swerling1(snr(P, r, 0.01), P.pfa)
    rng = np.random.default_rng(0)
    hits = sum(radar.scan(0.0, truth_at(r), rng)[1].any() for _ in range(4000))
    assert hits / 4000 == pytest.approx(expected, abs=0.03)


@pytest.mark.parametrize("r", [20.0, 6000.0])
def test_no_detection_outside_coverage(r):
    radar = SearchRadar(RadarParams(n_cells=0.0))
    dets, _ = radar.scan(0.0, truth_at(r), np.random.default_rng(1))
    assert dets == []


def test_false_alarm_rate():
    params = RadarParams(n_cells=3e6)                     # mean 3 per scan
    radar = SearchRadar(params)
    rng = np.random.default_rng(2)
    n = [len(radar.scan(0.0, None, rng)[0]) for _ in range(2000)]
    assert np.mean(n) == pytest.approx(3.0, rel=0.05)


def test_measurement_noise_matches_predicted_sigma():
    radar = SearchRadar(RadarParams(n_cells=0.0))
    rng = np.random.default_rng(3)
    r_true = 500.0
    dets = []
    for _ in range(3000):
        d, _ = radar.scan(0.0, truth_at(r_true), rng)
        dets.extend(d)                  # Swerling 1: rare fades give empty scans
    assert len(dets) > 2950             # Pd ~ 0.9996 at 500 m
    err = np.array([d.rng - r_true for d in dets])
    pred = np.sqrt(np.mean([d.R[0, 0] for d in dets]))
    assert err.std() == pytest.approx(pred, rel=0.1)


def test_detections_are_valid_messages_and_azimuth_wraps():
    radar = SearchRadar(RadarParams(n_cells=0.0))
    rng = np.random.default_rng(4)
    for _ in range(200):
        dets, flags = radar.scan(0.0, truth_at(800.0, az=1e-4), rng)
        for d in dets:
            assert isinstance(d, RadarDetection)
            assert 0.0 <= d.az < 2 * np.pi
        assert flags.all()


def test_sr01_detection_at_1500m_by_analysis():
    # SR-01: 0.01 m^2 at >= 1500 m with Pd >= 0.9 and Pfa <= 1e-4
    assert P.pfa <= 1e-4
    assert pd_swerling1(snr(P, 1500.0, 0.01), P.pfa) >= 0.9


def test_high_snr_hits_accuracy_floor():
    sr, sa, se = measurement_sigma(P, 1e9)
    assert sr == pytest.approx(P.range_floor, rel=1e-3)
    assert sa == pytest.approx(P.az_floor, rel=1e-3)