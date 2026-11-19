"""exp07 - VAL-02: sphere drag model vs measured buckshot velocity loss.

Reference: Brass Fetcher chronograph at 50 yd (brassfetcher.com/Shotguns/Shotguns.html):
#4 buck ~36 %, 00 buck ~26 %, 000 buck ~30 % velocity lost. Muzzle velocities
not stated -> model run over 1200-1450 ft/s; measured loss must lie within the
model band +/- 5 points.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fcsim.atmosphere import Atmosphere
from fcsim.ballistics import fly_batch
from fcsim.shotgun import pellet

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

FT, YD = 0.3048, 0.9144
D = 50 * YD
TOL_PTS = 5.0
RHO = float(Atmosphere().density())
LOADS = [("#4 buck", "4B", 36.0), ("00 buck", "00", 26.0), ("000 buck", "000", 30.0)]
V_FPS = np.linspace(1150.0, 1500.0, 36)
BAND = (V_FPS >= 1200.0) & (V_FPS <= 1450.0)


def loss_curve(size):
    p = pellet(size, "lead", 400.0)
    v0 = V_FPS * FT
    _, _, v = fly_batch(p, np.zeros(v0.size), 0.0, v0=v0, t_end=0.4).at_horizontal(D)
    v_end = np.linalg.norm(v, axis=1)
    loss = (1.0 - v_end / v0) * 100.0
    cd_eff = 2.0 * p.mass * (-np.log(v_end / v0) / D) / (RHO * p.area)
    return p, loss, cd_eff


print("exp07 - VAL-02: velocity loss at 50 yd, model band 1200-1450 ft/s vs measured\n")
print(f"{'load':<10}{'m[g]':>7}{'meas%':>7}{'model%':>15}{'Cd_eff mod':>12}"
      f"{'Cd_eff meas*':>14}  check")
curves, all_ok = {}, True
for name, size, meas in LOADS:
    p, loss, cd_eff = loss_curve(size)
    curves[name] = loss
    lo, hi = loss[BAND].min(), loss[BAND].max()
    ok = (lo - TOL_PTS) <= meas <= (hi + TOL_PTS)
    all_ok &= ok
    i1325 = np.argmin(np.abs(V_FPS - 1325.0))
    cd_meas = 2.0 * p.mass * (-np.log(1.0 - meas / 100.0) / D) / (RHO * p.area)
    print(f"{name:<10}{p.mass*1e3:7.2f}{meas:7.1f}{lo:7.1f} -{hi:5.1f}"
          f"{cd_eff[i1325]:12.2f}{cd_meas:14.2f}  {'PASS' if ok else 'FAIL'}")
print("* effective Cd implied by the measured loss (exponential decay, sea-level ISA)")
print(f"\nVAL-02 (band +/- {TOL_PTS:.0f} points): {'PASS' if all_ok else 'FAIL'}")

fig, ax = plt.subplots(figsize=(7.5, 4.8))
for (name, _, meas), c in zip(LOADS, ("tab:blue", "tab:orange", "tab:green")):
    ax.plot(V_FPS, curves[name], color=c, lw=2, label=f"model {name}")
    ax.axhline(meas, color=c, ls="--", lw=1)
    ax.axhspan(meas - TOL_PTS, meas + TOL_PTS, color=c, alpha=0.08)
ax.axvspan(1200, 1450, color="gray", alpha=0.08, label="assumed muzzle velocity band")
ax.set_xlabel("muzzle velocity [ft/s]")
ax.set_ylabel("velocity lost at 50 yd [%]")
ax.set_title("VAL-02: model (solid) vs Brass Fetcher measurement (dashed ±5)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
out = RESULTS / "exp07_val02_buckshot.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")