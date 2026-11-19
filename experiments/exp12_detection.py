"""exp12 - Detector performance vs range: Pd, accuracy, bias, consistency (SR-03).

SR-03: EO angle error <= 0.2 mrad RMS for targets >= 4 px wide (R <~ 810 m).
Errors are relative to the drone's geometric centre. Boresight jitter of
+/-3 px randomises the sub-pixel phase.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fcsim.detection import detect
from fcsim.frames import aer_to_enu
from fcsim.thermal_scene import CameraParams, DroneSignature, ThermalCamera

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

CAM = CameraParams()
SIG = DroneSignature()
WIDTH_M = SIG.span + 2 * SIG.motor_r
SR03 = 0.2e-3
N = 150
EL = np.deg2rad(3.0)
RANGES = np.array([50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000], dtype=float)
rng = np.random.default_rng(12)


def run(gamma):
    cam = ThermalCamera(CAM, SIG, rng=np.random.default_rng(0), gamma=gamma)
    out = []
    for R in RANGES:
        pos = aer_to_enu(R, 0.0, EL)
        e_az, e_el, c_az, c_el, npx, hit = [], [], [], [], [], 0
        for _ in range(N):
            jx, jy = rng.uniform(-3.0, 3.0, 2) * CAM.ifov
            fr = cam.frame(0.0, pos, jx, EL + jy, rng)
            m = detect(fr)
            if m is None:
                continue
            da = ((m.az + np.pi) % (2 * np.pi) - np.pi) * np.cos(EL)
            de = m.el - EL
            if np.hypot(da, de) > 3 * CAM.ifov:          # detected something else
                continue
            hit += 1
            e_az.append(da)
            e_el.append(de)
            c_az.append(da**2 / (m.R[0, 0] * np.cos(EL) ** 2))
            c_el.append(de**2 / m.R[1, 1])
            npx.append(m.n_pix)
        e_az, e_el = np.array(e_az), np.array(e_el)
        ok = hit > 0
        out.append(dict(
            R=R, pd=hit / N, width=WIDTH_M / (R * CAM.ifov),
            rms_az=np.sqrt(np.mean(e_az**2)) if ok else np.nan,
            rms_el=np.sqrt(np.mean(e_el**2)) if ok else np.nan,
            bias_el=np.mean(e_el) if ok else np.nan,
            std_el=np.std(e_el) if ok else np.nan,
            cons_az=np.mean(c_az) if ok else np.nan,
            cons_el=np.mean(c_el) if ok else np.nan,
            npx=np.mean(npx) if ok else np.nan))
    return out


clear = run(0.2e-3)
haze = run(1.0e-3)

print(f"exp12 - detector vs range ({N} frames each, errors in mrad, clear air)\n")
print(f"{'R[m]':>6}{'width':>7}{'Pd':>6}{'Pd haze':>8}{'rms az':>8}{'rms el':>8}"
      f"{'bias el':>8}{'std el':>8}{'cons az':>8}{'cons el':>8}{'n_pix':>7}")
for c, h in zip(clear, haze):
    print(f"{c['R']:6.0f}{c['width']:7.1f}{c['pd']:6.2f}{h['pd']:8.2f}"
          f"{c['rms_az']*1e3:8.3f}{c['rms_el']*1e3:8.3f}{c['bias_el']*1e3:8.3f}"
          f"{c['std_el']*1e3:8.3f}{c['cons_az']:8.2f}{c['cons_el']:8.2f}{c['npx']:7.1f}")
print("cons = mean (error / reported sigma)^2, should be ~1")

sel = [c for c in clear if c["width"] >= 4.0]
worst = max(max(c["rms_az"], c["rms_el"]) for c in sel)
print(f"\nSR-03 (targets >= 4 px, R <= {sel[-1]['R']:.0f} m): worst per-axis RMS "
      f"{worst*1e3:.3f} mrad -> {'PASS' if worst <= SR03 else 'FAIL'} (limit {SR03*1e3:.1f} mrad)")
worst_noise = max(max(c["rms_az"], c["std_el"]) for c in sel)
print(f"            without the elevation bias (noise only): {worst_noise*1e3:.3f} mrad")

# --- Figure ------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
R = RANGES
axs[0].plot(R, [c["pd"] for c in clear], "o-", lw=2, label="clear")
axs[0].plot(R, [h["pd"] for h in haze], "s-", lw=2, label="haze")
axs[0].set_xscale("log")
axs[0].set_xlabel("range [m]")
axs[0].set_ylabel("detection probability per frame")
axs[0].set_title("EO detection")
axs[0].grid(alpha=0.3, which="both")
axs[0].legend()

axs[1].loglog(R, [c["rms_az"] * 1e3 for c in clear], "o-", lw=2, label="azimuth RMS")
axs[1].loglog(R, [c["rms_el"] * 1e3 for c in clear], "s-", lw=2, label="elevation RMS")
axs[1].loglog(R, [abs(c["bias_el"]) * 1e3 for c in clear], "k:", label="|elevation bias|")
axs[1].axhline(SR03 * 1e3, color="tab:red", ls="--", label="SR-03 0.2 mrad")
axs[1].axvline(WIDTH_M / (4 * CAM.ifov), color="gray", ls=":", label="4 px")
axs[1].set_xlabel("range [m]")
axs[1].set_ylabel("angle error [mrad]")
axs[1].set_title("Accuracy (clear air)")
axs[1].grid(alpha=0.3, which="both")
axs[1].legend(fontsize=8)

axs[2].semilogx(R, [c["cons_az"] for c in clear], "o-", lw=2, label="azimuth")
axs[2].semilogx(R, [c["cons_el"] for c in clear], "s-", lw=2, label="elevation")
axs[2].axhline(1.0, color="k", ls="--", label="ideal = 1")
axs[2].set_yscale("log")
axs[2].set_xlabel("range [m]")
axs[2].set_ylabel("mean (error / reported σ)²")
axs[2].set_title("Is the reported R honest?")
axs[2].grid(alpha=0.3, which="both")
axs[2].legend()

fig.tight_layout()
out = RESULTS / "exp12_detection.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")