"""exp06 - Shotgun pellets: validation, pattern growth, energy, shot string.

Reference for validation (VAL-02, 5 %): gunnersden.com/shotgun-drop-factor
(1-1/8 oz lead, 1200 ft/s, per-pellet energy and time to 60 yards).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fcsim.ballistics import fly_batch
from fcsim.shotgun import fire, pattern_offsets, pellet, r50, shot_load

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

FT, YD, FTLB = 0.3048, 0.9144, 1.35582
DRONE_R = 0.175                     # m, AS-01 half-span
rng = np.random.default_rng(6)

# --- 1) Validation ---------------------------------------------------------------
REF_V0, REF_D, TOL = 1200 * FT, 60 * YD, 0.05
REF = {"4": (10.34, 3.01, 0.2140), "5": (8.21, 2.22, 0.2193), "6": (6.19, 1.54, 0.2255),
       "7.5": (3.97, 0.85, 0.2375), "8": (3.37, 0.68, 0.2423), "9": (2.38, 0.42, 0.2538)}

print("exp06 (1) - lead pellets vs reference at 60 yd (1200 ft/s)\n")
print(f"{'size':>5}{'m_ref[g]':>9}{'m_mod':>7}{'v_ref':>7}{'v_mod':>7}{'err%':>7}"
      f"{'TOF_ref':>9}{'TOF_mod':>9}{'err%':>7}")
worst = 0.0
for s, (e0, e60, tof_ref) in REF.items():
    m_ref = 2 * e0 * FTLB / REF_V0**2
    v_ref = REF_V0 * np.sqrt(e60 / e0)
    p = pellet(s, "lead", REF_V0)
    t, _, v = fly_batch(p, 0.0, 0.0, t_end=0.5).at_horizontal(REF_D)
    v_mod, t_mod = np.linalg.norm(v[0]), t[0]
    ev, et = (v_mod - v_ref) / v_ref, (t_mod - tof_ref) / tof_ref
    worst = max(worst, abs(ev), abs(et))
    print(f"{s:>5}{m_ref*1e3:9.3f}{p.mass*1e3:7.3f}{v_ref:7.0f}{v_mod:7.0f}{ev*100:7.2f}"
          f"{tof_ref:9.4f}{t_mod:9.4f}{et*100:7.2f}")
print(f"\nVAL-02 (|err| <= {TOL*100:.0f} %): {'PASS' if worst <= TOL else 'FAIL'}  "
      f"(worst {worst*100:.2f} %)")

# --- 2) Pattern growth: #4 lead, 32 g, 400 m/s -----------------------------------
LOAD = shot_load("4", "lead", payload_g=32.0, v0=400.0)
DIST = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0])
N_SHOTS = 5
CHOKES = ("cylinder", "modified", "full")
res = {c: {"r50": np.zeros((N_SHOTS, DIST.size)), "hits": np.zeros((N_SHOTS, DIST.size))}
       for c in CHOKES}
speed = np.zeros((N_SHOTS, DIST.size))
string = np.zeros((N_SHOTS, DIST.size))
example = None

for c in CHOKES:
    for k in range(N_SHOTS):
        tr = fire(LOAD, c, 0.0, 0.0, rng, t_end=1.2)
        for j, d in enumerate(DIST):
            t, pos, vel = tr.at_horizontal(d)
            off = pattern_offsets(pos, 0.0, 0.0)
            res[c]["r50"][k, j] = r50(off)
            res[c]["hits"][k, j] = np.sum(np.linalg.norm(off, axis=1) <= DRONE_R)
            if c == "modified":
                spd = np.linalg.norm(vel, axis=1)
                speed[k, j] = np.nanmean(spd)
                string[k, j] = (np.nanpercentile(t, 95) - np.nanpercentile(t, 5)) * np.nanmean(spd)
                if k == 0 and d == 40.0:
                    example = off

print(f"\nexp06 (2) - {LOAD.name}, mean of {N_SHOTS} shots, perfectly aimed")
print(f"{'d[m]':>5}{'R50 cyl':>9}{'R50 mod':>9}{'R50 full':>9}{'hits mod':>9}"
      f"{'hits full':>10}{'v[m/s]':>8}{'E[J]':>6}{'string[m]':>10}")
for j, d in enumerate(DIST):
    v_m = speed[:, j].mean()
    print(f"{d:5.0f}{res['cylinder']['r50'][:, j].mean()*100:9.1f}"
          f"{res['modified']['r50'][:, j].mean()*100:9.1f}{res['full']['r50'][:, j].mean()*100:9.1f}"
          f"{res['modified']['hits'][:, j].mean():9.1f}{res['full']['hits'][:, j].mean():10.1f}"
          f"{v_m:8.0f}{0.5*LOAD.pellet.mass*v_m**2:6.1f}{string[:, j].mean():10.2f}")
print("R50 in cm; hits = pellets inside a 17.5 cm-radius drone disc at pattern centre")

# --- 3) Load comparison ----------------------------------------------------------
LOADS = [("#7.5 lead", shot_load("7.5", "lead")), ("#4 lead", shot_load("4", "lead")),
         ("#4 steel", shot_load("4", "steel")), ("#4 tungsten", shot_load("4", "tungsten")),
         ("BB lead", shot_load("BB", "lead"))]
d_grid = np.arange(5.0, 151.0, 5.0)
energy_curves = {}
print("\nexp06 (3) - 32 g loads at 400 m/s, modified choke")
print(f"{'load':<13}{'pellets':>8}{'m[g]':>7}{'E25[J]':>8}{'E50[J]':>8}{'E100[J]':>8}{'hits50':>8}")
for name, ld in LOADS:
    tr1 = fly_batch(ld.pellet, 0.0, 0.0, t_end=1.5)
    E = []
    for d in d_grid:
        _, _, v = tr1.at_horizontal(d)
        E.append(0.5 * ld.pellet.mass * np.sum(v[0] ** 2))
    energy_curves[name] = np.array(E)
    hits = []
    for _ in range(3):
        _, pos, _ = fire(ld, "modified", 0.0, 0.0, rng, t_end=0.6).at_horizontal(50.0)
        hits.append(np.sum(np.linalg.norm(pattern_offsets(pos, 0.0, 0.0), axis=1) <= DRONE_R))
    Ei = lambda d: np.interp(d, d_grid, energy_curves[name])
    print(f"{name:<13}{ld.n_pellets:8d}{ld.pellet.mass*1e3:7.3f}{Ei(25):8.1f}{Ei(50):8.1f}"
          f"{Ei(100):8.1f}{np.mean(hits):8.1f}")

# --- Figure ----------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))

axs[0].scatter(example[:, 0] * 100, example[:, 1] * 100, s=6)
axs[0].add_patch(plt.Circle((0, 0), DRONE_R * 100, fill=False, color="tab:red", lw=2, label="drone (r 17.5 cm)"))
axs[0].add_patch(plt.Circle((0, 0), 38.1, fill=False, color="k", ls="--", label="30-inch circle"))
axs[0].set_aspect("equal")
axs[0].set_xlim(-80, 80)
axs[0].set_ylim(-80, 80)
axs[0].set_xlabel("right [cm]")
axs[0].set_ylabel("up [cm]")
axs[0].set_title("One shot at 40 m (#4 lead, modified)")
axs[0].legend(fontsize=8, loc="upper right")

for c in CHOKES:
    axs[1].plot(DIST, res[c]["r50"].mean(0) * 100, "o-", label=f"R50 {c}")
axs[1].axhline(DRONE_R * 100, color="tab:red", ls=":", label="drone half-span")
ax2 = axs[1].twinx()
ax2.plot(DIST, res["modified"]["hits"].mean(0), "k--s", label="hits on drone (modified)")
ax2.set_ylabel("pellets on drone disc")
axs[1].set_xlabel("distance [m]")
axs[1].set_ylabel("R50 [cm]")
axs[1].set_title("Pattern growth and pellets on target")
axs[1].grid(alpha=0.3)
axs[1].legend(fontsize=8, loc="upper left")
ax2.legend(fontsize=8, loc="center right")

for name, E in energy_curves.items():
    axs[2].semilogy(d_grid, E, lw=2, label=name)
axs[2].set_xlabel("distance [m]")
axs[2].set_ylabel("energy per pellet [J]")
axs[2].set_title("Pellet energy (threshold set in Step 8)")
axs[2].grid(alpha=0.3, which="both")
axs[2].legend(fontsize=8)

fig.tight_layout()
out = RESULTS / "exp06_shotgun_pattern.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")