import numpy as np
import pytest

from fcsim.ballistics import fly_batch
from fcsim.projectile import m80_ball
from fcsim.shotgun import (
    CHOKE_PERCENT, PATTERN_DIST, PATTERN_RADIUS, bore_axes, choke_sigma, fire,
    fraction_within, pattern_offsets, pellet, r50, sample_pellets, shot_load,
)


def test_no4_lead_pellet_mass_and_count():
    load = shot_load("4", "lead", payload_g=32.0)
    assert load.pellet.mass == pytest.approx(0.213e-3, rel=0.02)
    assert 140 <= load.n_pellets <= 160


def test_sphere_cd_subsonic():
    assert pellet().cd(0.2) == pytest.approx(0.47, abs=0.02)


@pytest.mark.parametrize("choke", list(CHOKE_PERCENT))
def test_choke_sigma_reproduces_pattern_percentage(choke):
    rng = np.random.default_rng(0)
    az_i, el_i, _ = sample_pellets(shot_load(), choke, 0.0, 0.0, rng, n=40_000)
    dx = (az_i + np.pi) % (2 * np.pi) - np.pi
    r = np.hypot(dx, el_i) * PATTERN_DIST
    assert np.mean(r <= PATTERN_RADIUS) == pytest.approx(CHOKE_PERCENT[choke], abs=0.01)


def test_tighter_choke_has_smaller_sigma():
    assert choke_sigma("full") < choke_sigma("modified") < choke_sigma("cylinder")


def test_pellet_loses_speed_much_faster_than_bullet():
    p = pellet(v0=400.0)
    b = m80_ball()
    _, _, vp = fly_batch(p, 0.0, 0.0, t_end=0.3).at_horizontal(40.0)
    _, _, vb = fly_batch(b, 0.0, 0.0, t_end=0.3).at_horizontal(40.0)
    frac_p = np.linalg.norm(vp[0]) / p.v0
    frac_b = np.linalg.norm(vb[0]) / b.v0
    assert frac_p < 0.8 < frac_b


def test_denser_pellet_retains_more_speed():
    _, _, vs = fly_batch(pellet("4", "steel"), 0.0, 0.0, t_end=0.3).at_horizontal(40.0)
    _, _, vt = fly_batch(pellet("4", "tungsten"), 0.0, 0.0, t_end=0.3).at_horizontal(40.0)
    assert np.linalg.norm(vt[0]) > np.linalg.norm(vs[0])


def test_bore_axes_are_orthonormal():
    f, r, u = bore_axes(0.7, 0.3)
    M = np.stack([f, r, u])
    np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)
    assert u[2] > 0


def test_pattern_grows_with_distance_and_choke():
    load = shot_load()
    def size(choke, d, seed):
        tr = fire(load, choke, 0.0, 0.0, np.random.default_rng(seed), t_end=0.4)
        _, pos, _ = tr.at_horizontal(d)
        return r50(pattern_offsets(pos, 0.0, 0.0))
    assert size("modified", 40.0, 1) > size("modified", 20.0, 1)
    assert size("full", 30.0, 2) < size("cylinder", 30.0, 2)


def test_fire_is_reproducible_with_seed():
    load = shot_load()
    a = fire(load, "modified", 0.0, 0.0, np.random.default_rng(5), t_end=0.1).pos[-1]
    b = fire(load, "modified", 0.0, 0.0, np.random.default_rng(5), t_end=0.1).pos[-1]
    np.testing.assert_array_equal(a, b)