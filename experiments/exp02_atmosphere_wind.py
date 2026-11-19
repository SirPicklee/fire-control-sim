"""exp02 - How much do atmosphere and wind change the firing problem?

1) Air density / speed of sound for real sites and seasons (drag scales with rho).
2) Mean wind profile vs height (power law).
3) Gust variability over a typical time of flight.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fcsim.atmosphere import RHO0, Atmosphere, density_altitude, isa_temperature
from fcsim.wind import GustModel, Wind, WindProfile

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

BULLET_V = 830.0     # m/s, 7.62 mm muzzle velocity
SHOT_V = 400.0       # m/s, 12 ga muzzle velocity


def atm_from_celsius(site_alt, t_celsius):
    dT = (t_celsius + 273.15) - isa_temperature(site_alt)
    return Atmosphere(site_alt=site_alt, dT=float(dT))


# --- 1) Sites and seasons ---------------------------------------------------
cases = [
    ("Sea level, ISA", 0.0, 15.0),
    ("Antalya summer", 30.0, 40.0),
    ("Ankara winter", 938.0, -10.0),
    ("Ankara summer", 938.0, 35.0),
    ("Erzurum winter", 1850.0, -25.0),
    ("Erzurum summer", 1850.0, 30.0),
]

print("exp02 (1) - air at the weapon site\n")
print(f"{'case':<17}{'alt':>6}{'T[C]':>6}{'rho':>8}{'drag':>7}"
      f"{'a':>7}{'M_bul':>7}{'M_shot':>7}{'DA[m]':>7}")
for name, alt, tc in cases:
    atm = atm_from_celsius(alt, tc)
    rho = float(atm.density())
    a = float(atm.sound_speed())
    print(f"{name:<17}{alt:6.0f}{tc:6.0f}{rho:8.3f}{rho / RHO0:7.2f}"
          f"{a:7.1f}{BULLET_V / a:7.2f}{SHOT_V / a:7.2f}{density_altitude(rho):7.0f}")
print("drag = density relative to sea-level ISA (same speed, same Cd)")

# --- 2) Wind profile --------------------------------------------------------
z = np.linspace(-2.0, 150.0, 300)
prof_open = WindProfile(speed_ref=5.0, alpha=1 / 7)
prof_rough = WindProfile(speed_ref=5.0, alpha=0.25)

print("\nexp02 (2) - mean wind speed [m/s], 5 m/s at 10 m")
for zz in (0.0, 8.0, 48.0, 98.0, 148.0):
    print(f"  height {zz + 2:5.0f} m : open {float(prof_open.speed(zz)):5.2f}"
          f"   suburban {float(prof_rough.speed(zz)):5.2f}")

# --- 3) Gusts ---------------------------------------------------------------
gust = GustModel(sigma=1.5, sigma_up=0.5, tau=2.0)
rng = np.random.default_rng(2)
wind = Wind(WindProfile(speed_ref=5.0, from_az=1.5 * np.pi), gust=gust,
            duration=3600.0, dt=0.05, rng=rng)
t = np.arange(0.0, 3600.0, 0.05)
cross = wind.velocity(t, 8.0)[:, 0]          # east component at 10 m

print("\nexp02 (3) - crosswind change over a time gap (sigma 1.5 m/s, tau 2 s)")
for gap in (0.1, 0.25, 0.5, 1.0, 2.0):
    k = int(round(gap / 0.05))
    emp = np.std(cross[k:] - cross[:-k])
    theo = gust.sigma * np.sqrt(2.0 * (1.0 - np.exp(-gap / gust.tau)))
    print(f"  dt = {gap:4.2f} s : std {emp:4.2f} m/s (theory {theo:4.2f})")

# --- Figure -----------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))

tc_grid = np.linspace(-30, 45, 100)
for alt, label in ((0.0, "sea level"), (938.0, "Ankara 938 m"), (1850.0, "Erzurum 1850 m")):
    rho = [float(atm_from_celsius(alt, tc).density()) for tc in tc_grid]
    axs[0].plot(tc_grid, np.array(rho) / RHO0 * 100, lw=2, label=label)
axs[0].axhline(100, color="k", lw=0.8)
axs[0].set_xlabel("air temperature [°C]")
axs[0].set_ylabel("density / sea-level ISA [%]")
axs[0].set_title("Air density (drag) by site and temperature")
axs[0].grid(alpha=0.3)
axs[0].legend()

axs[1].plot(prof_open.speed(z), z + 2, lw=2, label="open terrain (α = 1/7)")
axs[1].plot(prof_rough.speed(z), z + 2, lw=2, label="suburban (α = 0.25)")
axs[1].axhline(10, color="gray", ls=":", label="reference 10 m")
axs[1].set_xlabel("mean wind speed [m/s]")
axs[1].set_ylabel("height above ground [m]")
axs[1].set_title("Wind profile")
axs[1].grid(alpha=0.3)
axs[1].legend()

sel = t < 60.0
axs[2].plot(t[sel], cross[sel], lw=1.2, label="crosswind with gusts")
axs[2].axhline(5.0, color="k", ls="--", label="mean 5 m/s")
axs[2].axvspan(20.0, 20.5, color="tab:red", alpha=0.25, label="0.5 s time of flight")
axs[2].set_xlabel("time [s]")
axs[2].set_ylabel("east wind [m/s]")
axs[2].set_title("Gusts (σ = 1.5 m/s, τ = 2 s)")
axs[2].grid(alpha=0.3)
axs[2].legend()

fig.tight_layout()
out = RESULTS / "exp02_atmosphere_wind.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")