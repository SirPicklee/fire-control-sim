"""exp04 - Validate the ballistic model against manufacturer data, then
build the system firing table with wind drift and Coriolis.

Reference: RWS 7.62x51 NATO Ball (M80) 9.5 g, DMR table (H&K G27 A1)
https://www.rws-technology.com/en/armed-forces-law-enforcement/product/762x51-nato-ball-m80-146gr-2426829
Acceptance (VAL-01): speed and TOF within 2 % of reference up to 800 m.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fcsim.atmosphere import Atmosphere
from fcsim.ballistics import firing_table, fly
from fcsim.projectile import Projectile
from fcsim.wind import Wind, WindProfile

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

# Reference: range [m], speed [m/s], time of flight [ms]
RWS = np.array([
    [100, 754, 127], [200, 686, 266], [300, 621, 419], [400, 559, 589],
    [500, 501, 778], [600, 444, 990], [700, 389, 1230], [800, 342, 1505],
    [900, 310, 1814], [1000, 288, 2149], [1100, 270, 2508], [1200, 254, 2891],
], dtype=float)

# Model inputs taken from the same source: 9.5 g, 825 m/s, G7 BC 0.215
P = Projectile.from_g7_bc("RWS M80", mass=9.5e-3, diameter=7.82e-3, v0=825.0, bc_lb_in2=0.215)
ATM = Atmosphere()          # sea-level ISA (standard conditions)
TOL = 0.02                  # VAL-01
LAT = np.deg2rad(40.0)      # Ankara

# --- 1) Validation ------------------------------------------------------------
tr = fly(P, 0.0, 0.0, atm=ATM, max_range=1250.0, t_max=4.0, z_min=-200.0)
v_mod, t_mod = [], []
for R in RWS[:, 0]:
    t, _, v = tr.at_horizontal(R)
    v_mod.append(np.linalg.norm(v))
    t_mod.append(t * 1e3)
v_mod, t_mod = np.array(v_mod), np.array(t_mod)
v_err = (v_mod - RWS[:, 1]) / RWS[:, 1]
t_err = (t_mod - RWS[:, 2]) / RWS[:, 2]

print("exp04 (1) - model vs RWS M80 reference (sea-level ISA)\n")
print(f"{'R[m]':>5}{'v_ref':>7}{'v_mod':>7}{'err%':>7}{'TOF_ref':>9}{'TOF_mod':>9}{'err%':>7}  check")
for i, R in enumerate(RWS[:, 0]):
    ok = abs(v_err[i]) <= TOL and abs(t_err[i]) <= TOL
    tag = ("PASS" if ok else "FAIL") if R <= 800 else "info"
    print(f"{R:5.0f}{RWS[i, 1]:7.0f}{v_mod[i]:7.0f}{v_err[i]*100:7.2f}"
          f"{RWS[i, 2]:9.0f}{t_mod[i]:9.0f}{t_err[i]*100:7.2f}  {tag}")
sup = RWS[:, 0] <= 800
print(f"\nVAL-01 (<= 800 m, |err| <= {TOL*100:.0f} %): "
      f"{'PASS' if np.all(np.abs(v_err[sup]) <= TOL) and np.all(np.abs(t_err[sup]) <= TOL) else 'FAIL'}"
      f"   max |v err| {np.abs(v_err[sup]).max()*100:.2f} %, max |TOF err| {np.abs(t_err[sup]).max()*100:.2f} %")

# --- 2) System firing table ---------------------------------------------------
ranges = np.arange(100.0, 801.0, 100.0)
W = 5.0
cross = Wind(WindProfile(speed_ref=W, from_az=1.5 * np.pi, alpha=0.0))   # uniform, blows East

el0, tab = firing_table(P, ranges, zero_range=300.0, sight_height=0.30, atm=ATM)
_, tab_w = firing_table(P, ranges, zero_range=300.0, sight_height=0.30, atm=ATM, wind=cross)
_, tab_c = firing_table(P, ranges, zero_range=300.0, sight_height=0.30, atm=ATM, latitude=LAT)
didion = W * (tab["tof"] - ranges / P.v0)

print(f"\nexp04 (2) - firing table: zero 300 m, sight 0.30 m above bore, "
      f"zero elevation {el0*1e3:.2f} mrad")
print(f"{'R[m]':>5}{'TOF[s]':>8}{'v':>6}{'E[J]':>7}{'path[cm]':>10}{'corr[mrad]':>11}"
      f"{'wind5[cm]':>10}{'Didion':>8}{'wind[mrad]':>11}{'Cor[cm]':>8}")
for i, R in enumerate(ranges):
    print(f"{R:5.0f}{tab['tof'][i]:8.3f}{tab['speed'][i]:6.0f}{tab['energy'][i]:7.0f}"
          f"{tab['path'][i]*100:10.1f}{tab['correction_mrad'][i]:11.2f}"
          f"{tab_w['drift'][i]*100:10.1f}{didion[i]*100:8.1f}"
          f"{tab_w['drift'][i]/R*1e3:11.2f}{tab_c['drift'][i]*100:8.1f}")
print("wind5 = 5 m/s full-value crosswind; Cor = Coriolis, firing North at 40° N")

# --- Figure -------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))

axs[0].plot(RWS[:, 0], RWS[:, 1], "ko", label="RWS reference")
axs[0].plot(tr.at_horizontal(10.0)[0] * 0 + np.hypot(tr.pos[:, 0], tr.pos[:, 1]),
            tr.speed, lw=2, label="fcsim model")
axs[0].axvspan(800, 1250, color="tab:red", alpha=0.08, label="transonic / subsonic")
axs[0].set_xlim(0, 1250)
axs[0].set_xlabel("range [m]")
axs[0].set_ylabel("speed [m/s]")
axs[0].set_title("Velocity: model vs manufacturer")
axs[0].grid(alpha=0.3)
axs[0].legend()

axs[1].plot(RWS[:, 0], v_err * 100, "o-", label="speed")
axs[1].plot(RWS[:, 0], t_err * 100, "s-", label="time of flight")
axs[1].axhspan(-TOL * 100, TOL * 100, color="tab:green", alpha=0.15, label="VAL-01 ±2 %")
axs[1].axvline(800, color="tab:red", ls=":")
axs[1].set_xlabel("range [m]")
axs[1].set_ylabel("model error [%]")
axs[1].set_title("Model error")
axs[1].grid(alpha=0.3)
axs[1].legend()

axs[2].plot(ranges, tab["path"] * 100, "o-", label="path vs sight line (zero 300 m)")
axs[2].plot(ranges, tab_w["drift"] * 100, "s-", label="drift, 5 m/s crosswind")
axs[2].plot(ranges, tab_c["drift"] * 100 * 10, "^-", label="Coriolis drift ×10")
axs[2].axhline(0, color="k", lw=0.8)
axs[2].set_xlabel("range [m]")
axs[2].set_ylabel("[cm]")
axs[2].set_title("System firing table")
axs[2].grid(alpha=0.3)
axs[2].legend(fontsize=8)

fig.tight_layout()
out = RESULTS / "exp04_validation_firing_table.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")