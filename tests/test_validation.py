"""Model validation against external reference data (VAL-xx requirements)."""

import numpy as np

from fcsim.atmosphere import Atmosphere
from fcsim.calibration import speed_and_tof
from fcsim.projectile import m80_ball, m80_ball_g7
from fcsim.ballistics import fly_batch
from fcsim.shotgun import pellet

# RWS 7.62x51 NATO Ball (M80) 9.5 g, DMR table: range [m], speed [m/s], TOF [ms]
# https://www.rws-technology.com/en/armed-forces-law-enforcement/product/762x51-nato-ball-m80-146gr-2426829
RWS_M80 = np.array([
    [100, 754, 127], [200, 686, 266], [300, 621, 419], [400, 559, 589],
    [500, 501, 778], [600, 444, 990], [700, 389, 1230], [800, 342, 1505],
], dtype=float)


def _max_errors(proj):
    v, t = speed_and_tof(proj, RWS_M80[:, 0], atm=Atmosphere())
    v_err = np.abs(v - RWS_M80[:, 1]) / RWS_M80[:, 1]
    t_err = np.abs(t * 1e3 - RWS_M80[:, 2]) / RWS_M80[:, 2]
    return v_err.max(), t_err.max()


def test_val01_calibrated_m80_matches_manufacturer():
    v_max, t_max = _max_errors(m80_ball())
    assert v_max <= 0.02, f"speed error {v_max:.2%}"
    assert t_max <= 0.02, f"TOF error {t_max:.2%}"


def test_uncalibrated_g7_does_not_meet_val01():
    # Documents why calibration is needed (exp04): constant G7 BC fails at 700-800 m.
    v_max, _ = _max_errors(m80_ball_g7())
    assert v_max > 0.02



# Brass Fetcher chronograph at 50 yd: (size key, measured % velocity lost)
# https://brassfetcher.com/Shotguns/Shotguns.html
BRASS_FETCHER_BUCK = [("4B", 36.0), ("00", 26.0), ("000", 30.0)]


def test_val02_buckshot_loss_bracketed_by_model():
    ft, d = 0.3048, 50 * 0.9144
    v0 = np.linspace(1200.0, 1450.0, 11) * ft
    for size, measured in BRASS_FETCHER_BUCK:
        p = pellet(size, "lead", 400.0)
        _, _, v = fly_batch(p, np.zeros(v0.size), 0.0, v0=v0, t_end=0.4).at_horizontal(d)
        loss = (1.0 - np.linalg.norm(v, axis=1) / v0) * 100.0
        assert loss.min() - 5.0 <= measured <= loss.max() + 5.0, f"{size}: {measured} %"