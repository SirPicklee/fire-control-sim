"""exp11 - What does the drone look like to the thermal camera?

Noisy frames from 50 m to 2000 m, peak SNR, pixels on target, and the
effect of haze. Noise per pixel: sqrt(NETD^2 + FPN^2) (Project B values).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fcsim.frames import aer_to_enu
from fcsim.thermal_scene import (
    CameraParams, DroneSignature, ThermalCamera, atmospheric_transmission,
    render_signature,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

CAM = CameraParams()
SIG = DroneSignature()
NOISE = float(np.hypot(CAM.netd, CAM.fpn))
GAMMA_CLEAR, GAMMA_HAZE = 0.2e-3, 1.0e-3
EL = np.deg2rad(3.0)
RANGES = [50.0, 150.0, 300.0, 700.0, 1000.0, 2000.0]

camera = ThermalCamera(CAM, SIG, rng=np.random.default_rng(0), gamma=GAMMA_CLEAR)
rng = np.random.default_rng(11)


def metrics(R, gamma):
    clean = render_signature(SIG, R, CAM, (64.3, 63.8), gamma)
    peak = clean.max()
    n5 = int(np.sum(clean > 5 * NOISE))
    return peak, peak / NOISE, n5


# --- Table -----------------------------------------------------------------------
print(f"exp11 - drone vs sky, pixel noise {NOISE*1e3:.0f} mK, IFOV {CAM.ifov*1e3:.2f} mrad\n")
print(f"{'R[m]':>6}{'cm/px':>7}{'width px':>9}{'tau':>6}{'peak dT':>8}{'SNR':>7}"
      f"{'px>5s':>7}{'SNR haze':>9}{'px>5s haze':>11}")
for R in [50, 100, 150, 300, 500, 700, 1000, 1500, 2000, 3000]:
    peak, snr_c, n_c = metrics(R, GAMMA_CLEAR)
    _, snr_h, n_h = metrics(R, GAMMA_HAZE)
    width = (SIG.span + 2 * SIG.motor_r) / (R * CAM.ifov)
    print(f"{R:6.0f}{R*CAM.ifov*100:7.1f}{width:9.1f}{atmospheric_transmission(R, GAMMA_CLEAR):6.2f}"
          f"{peak:8.2f}{snr_c:7.0f}{n_c:7d}{snr_h:9.0f}{n_h:11d}")
print("peak dT in K above sky; SNR = peak / pixel noise; px>5s = pixels above 5 sigma")

# --- Figure ------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 8))
for i, R in enumerate(RANGES):
    pos = aer_to_enu(R, 0.0, EL)
    fr = camera.frame(0.0, pos, 0.2 * CAM.ifov, EL - 0.3 * CAM.ifov, rng)
    width = (SIG.span + 2 * SIG.motor_r) / (R * CAM.ifov)
    h = int(max(10, 0.8 * width))
    cx, cy = (int(v) for v in fr.truth_px)
    crop = fr.image[cy - h:cy + h, cx - h:cx + h] - CAM.sky_temp
    ax = fig.add_subplot(2, len(RANGES), i + 1)
    im = ax.imshow(crop, cmap="inferno", interpolation="nearest")
    ax.set_title(f"{R:.0f} m ({2*h} px)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, label="ΔT [K]")

r_plot = np.geomspace(50.0, 4000.0, 60)
snr_clear = [metrics(r, GAMMA_CLEAR)[1] for r in r_plot]
snr_haze = [metrics(r, GAMMA_HAZE)[1] for r in r_plot]
npx_clear = [metrics(r, GAMMA_CLEAR)[2] for r in r_plot]

ax1 = fig.add_subplot(2, 2, 3)
ax1.loglog(r_plot, snr_clear, lw=2, label="clear (γ 0.2 /km)")
ax1.loglog(r_plot, snr_haze, lw=2, label="haze (γ 1.0 /km)")
ax1.axhline(5, color="tab:red", ls="--", label="5σ detection")
ax1.set_xlabel("range [m]")
ax1.set_ylabel("peak pixel SNR")
ax1.set_title("Peak SNR vs range")
ax1.grid(alpha=0.3, which="both")
ax1.legend()

ax2 = fig.add_subplot(2, 2, 4)
ax2.loglog(r_plot, np.maximum(npx_clear, 0.5), lw=2, label="pixels > 5σ (clear)")
ax2.loglog(r_plot, (SIG.span + 2 * SIG.motor_r) / (r_plot * CAM.ifov), "k--", label="drone width [px]")
ax2.axhline(4, color="tab:red", ls=":", label="SR-03: 4 px")
ax2.set_xlabel("range [m]")
ax2.set_ylabel("pixels")
ax2.set_title("Pixels on target")
ax2.grid(alpha=0.3, which="both")
ax2.legend()

fig.tight_layout()
out = RESULTS / "exp11_thermal_scene.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")