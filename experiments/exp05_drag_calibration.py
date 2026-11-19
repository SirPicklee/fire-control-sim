"""exp05 - Calibrate a Mach-dependent drag correction to RWS M80 data.

Fit on speed only; time of flight is held out as independent validation.
Cross-validation: fit on even ranges, test on odd ranges.
Reference: RWS 7.62x51 NATO Ball (M80) 9.5 g, DMR table
https://www.rws-technology.com/en/armed-forces-law-enforcement/product/762x51-nato-ball-m80-146gr-2426829
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fcsim.atmosphere import Atmosphere
from fcsim.calibration import fit_drag_correction, speed_and_tof
from fcsim.projectile import Projectile

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

RWS = np.array([
    [100, 754, 127], [200, 686, 266], [300, 621, 419], [400, 559, 589],
    [500, 501, 778], [600, 444, 990], [700, 389, 1230], [800, 342, 1505],
    [900, 310, 1814], [1000, 288, 2149], [1100, 270, 2508], [1200, 254, 2891],
], dtype=float)
R, V_REF, T_REF = RWS[:, 0], RWS[:, 1], RWS[:, 2] / 1e3

BASE = Projectile.from_g7_bc("RWS M80", mass=9.5e-3, diameter=7.82e-3, v0=825.0, bc_lb_in2=0.215)
ATM = Atmosphere()
NODES = np.array([0.8, 1.0, 1.2, 1.6, 2.4])
TOL = 0.02
train = (R % 200 == 0)          # 200, 400, ..., 1200
test = ~train                   # 100, 300, ..., 1100

# --- Fits ---------------------------------------------------------------------
print("Fitting (cross-validation, even ranges only)...")
fit_cv = fit_drag_correction(BASE, R[train], V_REF[train], NODES, atm=ATM)
print("Fitting (all ranges)...")
fit_all = fit_drag_correction(BASE, R, V_REF, NODES, atm=ATM)

print("\nexp05 (1) - fitted drag correction k(M)")
print(f"{'Mach':>6}{'k (CV)':>9}{'k (all)':>9}")
for m, a, b in zip(NODES, fit_cv.factors, fit_all.factors):
    print(f"{m:6.2f}{a:9.3f}{b:9.3f}")
print(f"RMS speed residual: CV {fit_cv.rms_rel*100:.2f} %, all {fit_all.rms_rel*100:.2f} %  "
      f"({fit_cv.n_eval} + {fit_all.n_eval} model evaluations)")

# --- Evaluate -----------------------------------------------------------------
v_b, t_b = speed_and_tof(BASE, R, atm=ATM)
v_cv, t_cv = speed_and_tof(fit_cv.projectile, R, atm=ATM)
v_c, t_c = speed_and_tof(fit_all.projectile, R, atm=ATM)
err = lambda mod, ref: (mod - ref) / ref * 100

print("\nexp05 (2) - errors [%]: base G7 / cross-validated / final")
print(f"{'R[m]':>5}{'set':>6}{'v_base':>8}{'v_CV':>7}{'v_cal':>7}{'TOF_base':>10}{'TOF_CV':>8}{'TOF_cal':>9}")
for i in range(R.size):
    print(f"{R[i]:5.0f}{'train' if train[i] else 'TEST':>6}"
          f"{err(v_b, V_REF)[i]:8.2f}{err(v_cv, V_REF)[i]:7.2f}{err(v_c, V_REF)[i]:7.2f}"
          f"{err(t_b, T_REF)[i]:10.2f}{err(t_cv, T_REF)[i]:8.2f}{err(t_c, T_REF)[i]:9.2f}")

cv_test_max = np.abs(err(v_cv, V_REF)[test]).max()
print(f"\nCross-validation: max |speed error| on unseen ranges = {cv_test_max:.2f} %")
print(f"Held-out TOF (final model): max |error| = {np.abs(err(t_c, T_REF)).max():.2f} % (all ranges)")

sup = R <= 800
val_ok = (np.all(np.abs(err(v_c, V_REF)[sup]) <= TOL * 100)
          and np.all(np.abs(err(t_c, T_REF)[sup]) <= TOL * 100))
print(f"VAL-01 (<= 800 m, |err| <= 2 %): {'PASS' if val_ok else 'FAIL'}  "
      f"(base model: max |v err| {np.abs(err(v_b, V_REF)[sup]).max():.2f} %)")

print("\nPaste into projectile.py (Step 6f):")
print(f"M80_MACH_NODES = np.array({np.round(NODES, 3).tolist()})")
print(f"M80_DRAG_FACTORS = np.array({np.round(fit_all.factors, 4).tolist()})")

# --- Figure -------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))

mg = np.linspace(0.6, 2.6, 400)
axs[0].plot(mg, BASE.cd(mg), "k--", label="G7 × form factor (base)")
axs[0].plot(mg, fit_all.projectile.cd(mg), lw=2, label="calibrated")
axs[0].plot(NODES, fit_all.projectile.cd(NODES), "o", color="tab:orange", label="nodes")
for m in V_REF / float(ATM.sound_speed()):
    axs[0].axvline(m, color="gray", lw=0.5, alpha=0.5)
axs[0].set_xlabel("Mach")
axs[0].set_ylabel("Cd")
axs[0].set_title("Drag curve (grey lines: Mach of data points)")
axs[0].grid(alpha=0.3)
axs[0].legend()

axs[1].plot(R, err(v_b, V_REF), "k--o", label="base")
axs[1].plot(R[train], err(v_cv, V_REF)[train], "s", color="tab:blue", label="CV fit, train")
axs[1].plot(R[test], err(v_cv, V_REF)[test], "D", color="tab:red", label="CV fit, TEST (unseen)")
axs[1].plot(R, err(v_c, V_REF), "-", color="tab:green", label="final")
axs[1].axhspan(-2, 2, color="tab:green", alpha=0.1)
axs[1].set_xlabel("range [m]")
axs[1].set_ylabel("speed error [%]")
axs[1].set_title("Speed (fitted quantity)")
axs[1].grid(alpha=0.3)
axs[1].legend(fontsize=8)

axs[2].plot(R, err(t_b, T_REF), "k--o", label="base")
axs[2].plot(R, err(t_c, T_REF), "-o", color="tab:green", label="final (TOF never fitted)")
axs[2].axhspan(-2, 2, color="tab:green", alpha=0.1, label="VAL-01 ±2 %")
axs[2].axvline(800, color="tab:red", ls=":")
axs[2].set_xlabel("range [m]")
axs[2].set_ylabel("time-of-flight error [%]")
axs[2].set_title("Time of flight (independent validation)")
axs[2].grid(alpha=0.3)
axs[2].legend(fontsize=8)

fig.tight_layout()
out = RESULTS / "exp05_drag_calibration.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")