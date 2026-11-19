import numpy as np
import pytest

from fcsim.lethality import (
    Component, DroneVulnerability, expected_pellets_in_disc, mean_kill_prob,
    pk_from_pellets, pk_poisson, small_quadcopter,
)

DRONE = small_quadcopter(beta=0.0)   # hard thresholds for the step-function tests
Q_ALL = sum(c.area * c.p_kill for c in DRONE.components) / DRONE.disc_area


def test_weak_pellet_cannot_kill():
    assert DRONE.kill_prob_per_pellet(1.0) == 0.0


def test_strong_pellet_uses_all_components():
    assert DRONE.kill_prob_per_pellet(100.0) == pytest.approx(Q_ALL)


def test_threshold_is_inclusive_and_stepwise():
    q_low = DRONE.kill_prob_per_pellet(2.9)     # propellers only
    q_high = DRONE.kill_prob_per_pellet(3.0)    # + flight controller
    assert q_high > q_low > 0.0


def test_single_pellet_inside_disc():
    pk = pk_from_pellets([[0.05, 0.0]], [100.0], DRONE)
    assert pk == pytest.approx(Q_ALL)


def test_pellets_outside_disc_and_nan_are_ignored():
    off = [[0.5, 0.0], [np.nan, np.nan], [0.0, 0.3]]
    assert pk_from_pellets(off, [100.0, 100.0, 100.0], DRONE) == 0.0


def test_more_pellets_more_kill():
    one = pk_from_pellets(np.zeros((1, 2)), [10.0], DRONE)
    ten = pk_from_pellets(np.zeros((10, 2)), [10.0] * 10, DRONE)
    assert ten > one


def test_poisson_matches_exact_for_many_pellets():
    n = 40
    exact = pk_from_pellets(np.zeros((n, 2)), [4.0] * n, DRONE)
    approx = pk_poisson(n, 4.0, DRONE)
    assert approx == pytest.approx(exact, rel=0.02)


@pytest.mark.parametrize("miss", [0.0, 0.3, 0.6])
def test_expected_pellets_in_disc_matches_sampling(miss):
    rng = np.random.default_rng(0)
    sigma, R, N, trials = 0.35, 0.175, 150, 4000
    pts = rng.normal(0.0, sigma, (trials, N, 2))
    pts[..., 0] -= miss
    counted = np.mean(np.sum(np.linalg.norm(pts, axis=-1) <= R, axis=1))
    assert expected_pellets_in_disc(N, sigma, miss, R) == pytest.approx(counted, rel=0.03)


def test_expected_pellets_decrease_with_miss():
    n = expected_pellets_in_disc(150, 0.35, np.array([0.0, 0.2, 0.5, 1.0]), 0.175)
    assert np.all(np.diff(n) < 0)


def test_invalid_component_area_rejected():
    with pytest.raises(ValueError):
        DroneVulnerability(components=(Component("huge", 1.0, 1.0, 0.5),))


def test_soft_threshold_is_half_at_e50():
    soft = small_quadcopter(beta=0.4)
    assert soft.damage_prob(2.0)[0] == pytest.approx(0.5)     # propellers, E50 = 2 J


def test_soft_q_is_monotonic_and_continuous():
    q = small_quadcopter(beta=0.4).kill_prob_per_pellet(np.linspace(0.5, 30.0, 600))
    assert np.all(np.diff(q) >= 0.0)
    assert np.max(np.diff(q)) < 0.01                          # no cliff


def test_tiny_beta_recovers_hard_threshold():
    e = np.array([1.0, 4.0, 8.0, 12.0, 20.0])                 # away from thresholds
    np.testing.assert_allclose(small_quadcopter(beta=1e-3).kill_prob_per_pellet(e),
                               DRONE.kill_prob_per_pellet(e), atol=1e-9)


def test_mean_q_differs_from_q_of_mean_near_threshold():
    energies = np.array([1.0, 3.0])                           # straddle 2 J
    assert mean_kill_prob(energies, DRONE) == pytest.approx(DRONE.kill_prob_per_pellet(3.0) / 2)
    assert DRONE.kill_prob_per_pellet(energies.mean()) != pytest.approx(mean_kill_prob(energies, DRONE))


def test_default_drone_is_soft():
    assert small_quadcopter().kill_prob_per_pellet(1.9) > 0.0