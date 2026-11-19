import numpy as np
import pytest

from fcsim.lrf import LaserRangefinder, LRFParams, lrf_pd, lrf_snr
from fcsim.messages import EOMeasurement, LRFMeasurement
from fcsim.timing import Latency, MeasurementBuffer, SensorClock

P = LRFParams()
R2 = np.diag([1e-8, 1e-8])


def test_snr_is_inverse_square_then_inverse_fourth():
    near = lrf_snr(P, 100.0) / lrf_snr(P, 200.0)          # beam smaller than drone
    far = lrf_snr(P, 1000.0) / lrf_snr(P, 2000.0)         # beam larger than drone
    assert near == pytest.approx(4.0, rel=1e-6)
    assert far == pytest.approx(16.0, rel=1e-6)


def test_pd_at_reference_equals_spec():
    assert lrf_pd(P, P.ref_range) == pytest.approx(P.ref_pd)


def test_large_pointing_error_causes_dropouts():
    lrf = LaserRangefinder(P)
    rng = np.random.default_rng(0)
    pos = np.array([0.0, 1000.0, 50.0])
    good = np.mean([lrf.measure(0.0, pos, rng, 0.02e-3).valid for _ in range(1000)])
    bad = np.mean([lrf.measure(0.0, pos, rng, 2e-3).valid for _ in range(1000)])
    assert good > 0.9 and bad < 0.3


def test_range_noise_matches_reported_sigma():
    lrf = LaserRangefinder(P)
    rng = np.random.default_rng(1)
    pos = np.array([0.0, 300.0, 20.0])
    R = np.linalg.norm(pos)
    ms = [m for m in (lrf.measure(0.0, pos, rng, 0.0) for _ in range(3000)) if m.valid]
    err = np.array([m.rng - R for m in ms])
    assert err.std() == pytest.approx(np.sqrt(np.mean([m.sigma**2 for m in ms])), rel=0.1)


def test_dropout_message_and_out_of_range():
    lrf = LaserRangefinder(P)
    m = lrf.measure(1.0, np.array([0.0, 9000.0, 0.0]), np.random.default_rng(2))
    assert isinstance(m, LRFMeasurement) and not m.valid and np.isnan(m.rng)


def test_clock_times_with_phase():
    t = SensorClock(rate=5.0, phase=0.03).times(1.0)
    np.testing.assert_allclose(t, [0.03, 0.23, 0.43, 0.63, 0.83])


def test_latency_arrival_is_never_before_validity():
    lat = Latency(mean=0.05, std=0.1)
    rng = np.random.default_rng(3)
    arr = np.array([lat.arrival(1.0, rng) for _ in range(2000)])
    assert np.all(arr >= 1.0)
    assert Latency(0.2).arrival(1.0) == pytest.approx(1.2)


def test_buffer_releases_in_validity_order_and_flags_oosm():
    buf = MeasurementBuffer()
    a = EOMeasurement(1.00, 0.1, 0.1, R2)
    b = EOMeasurement(1.10, 0.1, 0.1, R2)
    c = EOMeasurement(0.95, 0.1, 0.1, R2)            # old, slow sensor
    buf.push(1.05, a)
    buf.push(1.12, b)
    buf.push(1.30, c)
    first = buf.pop_ready(1.2)
    assert [m.t for m, _ in first] == [1.00, 1.10]
    assert not any(o for _, o in first)
    second = buf.pop_ready(1.4)
    assert second[0][0].t == 0.95 and second[0][1] is True
    assert buf.n_oosm == 1 and len(buf) == 0


def test_buffer_sorts_simultaneous_arrivals_by_validity():
    buf = MeasurementBuffer()
    buf.push(2.0, EOMeasurement(1.9, 0.1, 0.1, R2))
    buf.push(2.0, EOMeasurement(1.7, 0.1, 0.1, R2))
    assert [m.t for m, _ in buf.pop_ready(2.0)] == [1.7, 1.9]