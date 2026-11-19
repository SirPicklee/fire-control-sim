import numpy as np
import pytest

from fcsim.messages import TargetTruth
from fcsim.target import Jink, Straight, Turn, Weave, crossing, inbound, simulate


def test_straight_flight_is_exact():
    tr = simulate([Straight(10.0)], pos0=[0, 0, 50], speed0=12.0, heading0=np.pi / 2)
    np.testing.assert_allclose(tr.pos[-1], [120.0, 0.0, 50.0], atol=1e-9)
    np.testing.assert_allclose(tr.acc, 0.0, atol=1e-12)


def test_full_circle_closes_and_has_constant_radius():
    s, w = 15.0, 0.5
    tr = simulate([Turn(2 * np.pi / w, w)], pos0=[0, 0, 0], speed0=s, heading0=0.0)
    np.testing.assert_allclose(tr.pos[-1], tr.pos[0], atol=1e-9)
    center = np.array([s / w, 0.0, 0.0])          # right of a North-flying start
    np.testing.assert_allclose(np.linalg.norm(tr.pos - center, axis=1), s / w, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(tr.acc[1:], axis=1), s * w, rtol=1e-9)


def test_positive_rate_turns_right():
    s, w = 10.0, 0.5
    tr = simulate([Turn(np.pi / 2 / w, w)], pos0=[0, 0, 0], speed0=s, heading0=0.0)
    np.testing.assert_allclose(tr.vel[-1], [s, 0.0, 0.0], atol=1e-9)    # North -> East


def test_acceleration_leg_changes_speed():
    tr = simulate([Straight(4.0, accel=2.0)], pos0=[0, 0, 0], speed0=5.0, heading0=0.0)
    assert tr.speed[-1] == pytest.approx(13.0)
    assert tr.pos[-1, 1] == pytest.approx(5.0 * 4.0 + 0.5 * 2.0 * 16.0)


def test_climb_rate_changes_height():
    tr = simulate([Straight(5.0, climb_rate=3.0)], pos0=[0, 0, 20], speed0=10.0, heading0=0.0)
    assert tr.pos[-1, 2] == pytest.approx(35.0)


def test_jink_is_reproducible_and_respects_g_limit():
    kw = dict(pos0=[0, 0, 50], speed0=20.0, heading0=0.0, a_lat_max=10.0)
    a = simulate([Jink(20.0, max_rate=2.0, switch_mean=0.8)], rng=np.random.default_rng(3), **kw)
    b = simulate([Jink(20.0, max_rate=2.0, switch_mean=0.8)], rng=np.random.default_rng(3), **kw)
    np.testing.assert_array_equal(a.pos, b.pos)
    assert np.linalg.norm(a.acc, axis=1).max() <= 10.0 + 1e-9


def test_jink_without_rng_fails():
    with pytest.raises(ValueError):
        simulate([Jink(5.0, 1.0)], pos0=[0, 0, 0], speed0=10.0, heading0=0.0)


def test_weave_returns_to_original_heading_after_full_period():
    tr = simulate([Weave(4.0, amplitude=0.6, period=4.0)], pos0=[0, 0, 0], speed0=10.0, heading0=0.0)
    np.testing.assert_allclose(tr.vel[-1], [0.0, 10.0, 0.0], atol=1e-6)


def test_inbound_flies_toward_weapon():
    tr = inbound(800.0, np.deg2rad(30), 60.0, 15.0, [Straight(20.0)])
    assert tr.slant_range[-1] < tr.slant_range[0] - 290.0
    horiz = tr.pos[0, :2]
    assert np.dot(tr.vel[0, :2], horiz) == pytest.approx(-15.0 * np.linalg.norm(horiz))


def test_crossing_closest_approach_equals_offset():
    tr = crossing(100.0, 40.0, 15.0, 300.0, [Straight(40.0)])
    i = np.argmin(np.hypot(tr.pos[:, 0], tr.pos[:, 1]))
    assert np.hypot(tr.pos[i, 0], tr.pos[i, 1]) == pytest.approx(100.0, abs=0.2)


def test_legs_join_continuously_and_state_at_returns_truth():
    tr = simulate([Straight(2.0), Turn(2.0, 0.4), Straight(2.0)],
                  pos0=[0, 0, 0], speed0=10.0, heading0=0.0)
    step = np.linalg.norm(np.diff(tr.pos, axis=0), axis=1)
    assert step.max() < 10.0 * 0.01 + 1e-9                 # no jumps
    st = tr.state_at(3.0)
    assert isinstance(st, TargetTruth)
    assert np.linalg.norm(st.vel) == pytest.approx(10.0, rel=1e-3)