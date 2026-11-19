"""exp03 - What does air do to a 7.62 mm bullet?

M80 fired horizontally (el = 0, no zeroing yet). Compares vacuum vs air,
finds the transonic range, and shows the Ankara winter vs summer drop
difference that follows from the 17 % density change found in exp02.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fcsim.atmosphere import Atmosphere, isa_temperature
from fcsim.ballistics import fly
from fcsim.projectile import m80_ball

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

P = m80_ball()
R_MAX = 1000.0
RANGES = (100, 200, 300, 500, 700, 1000)


def atm_from_celsius(site_alt, t_celsius):
    dT = (t_celsius + 273.15) - isa_temperature(site_alt)
    return Atmosphere(site_alt=site_alt, dT=float(dT))


def shoot(atm=None, vacuum=False):
    return fly(P, 0.0, 0.0, atm=atm, vacuum=vacuum, max_range=R_MAX + 20.0, t_max=4.0)


isa = Atmosphere()
winter = atm_from_celsius(938.0, -10.0)
summer = atm_from_celsius(938.0, 35.0)

tr_vac = shoot(vacuum=True)
tr_air = shoot(isa)
tr_win = shoot(winter)
tr_sum = shoot(summer)

# --- Table ------------------------------------------------------------------
print(f"exp03 - {P.name}, v0 = {P.v0:.0f} m/s, fired horizontally, sea-level ISA\n")
print(f"{'R[m]':>5}{'TOF':>7}{'TOFvac':>7}{'v[m/s]':>8}{'Mach':>6}{'E[J]':>7}"
      f"{'drop':>7}{'dropVac':>8}{'winter':>8}{'summer':>8}{'diff':>6}")
print(f"{'':>5}{'[s]':>7}{'[s]':>7}{'':>8}{'':>6}{'':>7}"
      f"{'[cm]':>7}{'[cm]':>8}{'[cm]':>8}{'[cm]':>8}{'[cm]':>6}")
for R in RANGES:
    t, p, v = tr_air.at_range(R)
    tv, pv, _ = tr_vac.at_range(R)
    _, pw, _ = tr_win.at_range(R)
    _, ps, _ = tr_sum.at_range(R)
    speed = np.linalg.norm(v)
    mach = speed / float(isa.sound_speed(p[2]))
    energy = 0.5 * P.mass * speed**2
    print(f"{R:5d}{t:7.3f}{tv:7.3f}{speed:8.0f}{mach:6.2f}{energy:7.0f}"
          f"{-p[2]*100:7.0f}{-pv[2]*100:8.0f}{-pw[2]*100:8.0f}{-ps[2]*100:8.0f}"
          f"{(ps[2]-pw[2])*100:6.0f}")

# Transonic crossing
mach_air = tr_air.speed / isa.sound_speed(tr_air.pos[:, 2])
for m_lim in (1.2, 1.0):
    i = np.argmax(mach_air < m_lim)
    if mach_air[i] < m_lim:
        print(f"\nMach < {m_lim}: at {tr_air.slant_range[i]:.0f} m, t = {tr_air.t[i]:.2f} s", end="")
    else:
        print(f"\nMach stays above {m_lim} out to {R_MAX:.0f} m", end="")
print()

# --- Figure -----------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(13, 9))

for tr, lab, st in ((tr_vac, "vacuum", "k--"), (tr_air, "air, sea-level ISA", "-"),
                    (tr_win, "Ankara winter (-10 °C)", "-"), (tr_sum, "Ankara summer (35 °C)", "-")):
    axs[0, 0].plot(tr.pos[:, 1], tr.pos[:, 2], st, lw=1.8, label=lab)
axs[0, 0].set_xlim(0, R_MAX)
axs[0, 0].set_xlabel("downrange [m]")
axs[0, 0].set_ylabel("height relative to muzzle [m]")
axs[0, 0].set_title("Trajectory (fired horizontally)")
axs[0, 0].grid(alpha=0.3)
axs[0, 0].legend()

axs[0, 1].plot(tr_air.slant_range, tr_air.speed, lw=2, label="bullet speed")
axs[0, 1].axhline(float(isa.sound_speed()), color="tab:red", ls="--", label="speed of sound")
axs[0, 1].axhspan(0.8 * float(isa.sound_speed()), 1.2 * float(isa.sound_speed()),
                  color="tab:red", alpha=0.1, label="transonic (M 0.8–1.2)")
axs[0, 1].set_xlim(0, R_MAX)
axs[0, 1].set_xlabel("range [m]")
axs[0, 1].set_ylabel("speed [m/s]")
axs[0, 1].set_title("Velocity decay")
axs[0, 1].grid(alpha=0.3)
axs[0, 1].legend()

axs[1, 0].plot(tr_air.slant_range, tr_air.t, lw=2, label="air")
axs[1, 0].plot(tr_vac.slant_range, tr_vac.t, "k--", label="vacuum")
axs[1, 0].set_xlim(0, R_MAX)
axs[1, 0].set_ylim(0, None)
axs[1, 0].set_xlabel("range [m]")
axs[1, 0].set_ylabel("time of flight [s]")
axs[1, 0].set_title("Time of flight")
axs[1, 0].grid(alpha=0.3)
axs[1, 0].legend()

r_grid = np.arange(50.0, R_MAX + 1.0, 10.0)
diff = [(tr_sum.at_range(r)[1][2] - tr_win.at_range(r)[1][2]) * 100 for r in r_grid]
axs[1, 1].plot(r_grid, diff, lw=2, color="tab:purple")
axs[1, 1].axhline(17.5, color="gray", ls=":", label="drone half-span (17.5 cm)")
axs[1, 1].set_xlabel("range [m]")
axs[1, 1].set_ylabel("summer minus winter impact height [cm]")
axs[1, 1].set_title("Ankara: same aim, summer vs winter")
axs[1, 1].grid(alpha=0.3)
axs[1, 1].legend()

fig.tight_layout()
out = RESULTS / "exp03_exterior_ballistics.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")