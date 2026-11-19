"""exp01 - Parallax vs boresight error: which dominates at which range?

Sensor mounted 0.30 m above the muzzle. Compares the straight-line miss
(no ballistics) of five pointing strategies against a rough shotgun
pattern radius and the drone size.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fcsim.frames import aer_to_enu
from fcsim.platform import (
    Mount, apply_boresight_error, barrel_angles, line_miss_distance,
    parallax_angle,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

# --- Parameters -------------------------------------------------------------
OFFSET_UP = 0.30            # m, sensor above muzzle
CONVERGENCE_R = 100.0       # m, fixed convergence range
LRF_ERROR = 1.0             # m, range error used for correction
BORESIGHT = 0.5e-3          # rad, angular misalignment
SR03_ANGLE = 0.2e-3         # rad, EO angle accuracy requirement (SR-03)
PATTERN_HALF_ANGLE = 10e-3  # rad, rough 12 ga pattern; refined in Step 7
DRONE_HALF_SPAN = 0.175     # m, AS-01

mount = Mount(sensor_offset=[0.0, 0.0, OFFSET_UP])
R = np.geomspace(20.0, 1000.0, 300)
tgt = mount.sensor_pos + aer_to_enu(R, 0.0, 0.0)   # straight ahead of sensor

# --- Strategies -------------------------------------------------------------
par_angle = parallax_angle(mount, tgt)

miss_parallel = line_miss_distance(mount.muzzle_pos, 0.0, 0.0, tgt)

az_c, el_c = barrel_angles(mount, CONVERGENCE_R, 0.0, 0.0)
miss_converged = line_miss_distance(mount.muzzle_pos, az_c, el_c, tgt)

az_l, el_l = barrel_angles(mount, R + LRF_ERROR, 0.0, 0.0)
miss_lrf = line_miss_distance(mount.muzzle_pos, az_l, el_l, tgt)

az_t, el_t = barrel_angles(mount, R, 0.0, 0.0)
az_b, el_b = apply_boresight_error(az_t, el_t, BORESIGHT, 0.0)
miss_boresight = line_miss_distance(mount.muzzle_pos, az_b, el_b, tgt)

pattern_radius = PATTERN_HALF_ANGLE * R

# --- Table ------------------------------------------------------------------
print("exp01 - straight-line miss [cm] (sensor 30 cm above muzzle)\n")
print(f"{'R [m]':>7} {'parallax':>9} {'parallel':>9} {'conv100':>8} "
      f"{'LRF+1m':>7} {'bore0.5':>8} {'pattern':>8}")
print(f"{'':>7} {'[mrad]':>9}")
for r_show in (25, 50, 100, 200, 500, 1000):
    i = np.argmin(np.abs(R - r_show))
    print(f"{r_show:7.0f} {par_angle[i]*1e3:9.3f} {miss_parallel[i]*100:9.1f} "
          f"{miss_converged[i]*100:8.1f} {miss_lrf[i]*100:7.2f} "
          f"{miss_boresight[i]*100:8.1f} {pattern_radius[i]*100:8.1f}")

print(f"\nParallax angle > SR-03 ({SR03_ANGLE*1e3:.1f} mrad) up to "
      f"{OFFSET_UP / SR03_ANGLE:.0f} m")
print(f"Boresight miss = parallel-mount miss at {OFFSET_UP / BORESIGHT:.0f} m")
print(f"Boresight miss > drone half-span beyond {DRONE_HALF_SPAN / BORESIGHT:.0f} m")

# --- Figure -----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

ax1.loglog(R, par_angle * 1e3, lw=2, label="parallax angle (0.30 m offset)")
ax1.axhline(SR03_ANGLE * 1e3, color="k", ls="--", label="SR-03 angle accuracy")
ax1.axhline(BORESIGHT * 1e3, color="tab:red", ls=":", label="boresight error 0.5 mrad")
ax1.set_xlabel("range [m]")
ax1.set_ylabel("angle [mrad]")
ax1.set_title("Angular error vs range")
ax1.grid(True, which="both", alpha=0.3)
ax1.legend()

ax2.loglog(R, miss_parallel * 100, lw=2, label="parallel mount")
ax2.loglog(R, np.maximum(miss_converged, 1e-4) * 100, lw=2, label="converged at 100 m")
ax2.loglog(R, miss_lrf * 100, lw=2, label="LRF-corrected (1 m error)")
ax2.loglog(R, miss_boresight * 100, lw=2, color="tab:red", label="boresight 0.5 mrad")
ax2.loglog(R, pattern_radius * 100, "k--", label="shot pattern radius (rough)")
ax2.axhline(DRONE_HALF_SPAN * 100, color="gray", ls=":", label="drone half-span")
ax2.set_ylim(0.1, 2000)
ax2.set_xlabel("range [m]")
ax2.set_ylabel("miss distance [cm]")
ax2.set_title("Straight-line miss (no ballistics)")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(fontsize=8)

fig.tight_layout()
out = RESULTS / "exp01_parallax_boresight.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")