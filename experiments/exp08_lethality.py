"""exp08 - Which load kills a small drone, and out to what range?

Exact Pk from simulated pellets (positions + energies, drone at pattern
centre) vs the fast Poisson-Gaussian approximation; then Pk vs aim error.
Vulnerability: lethality.small_quadcopter() with log-normal fragility
(beta = 0.4). Values are assumptions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fcsim.lethality import (
    expected_pellets_in_disc, mean_kill_prob, pk_from_pellets, pk_poisson_q,
    small_quadcopter,
)
from fcsim.shotgun import fire, pattern_offsets, r50, shot_load

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

DRONE = small_quadcopter()            # beta = 0.4 (soft thresholds)
CHOKE = "modified"
N_SHOTS = 6
DIST = np.arange(10.0, 121.0, 10.0)
R50_TO_SIGMA = 1.0 / np.sqrt(2.0 * np.log(2.0))
rng = np.random.default_rng(8)

LOADS = [("#7.5 lead", shot_load("7.5", "lead")), ("#4 lead", shot_load("4", "lead")),
         ("#4 steel", shot_load("4", "steel")), ("#4 tungsten", shot_load("4", "tungsten")),
         ("BB lead", shot_load("BB", "lead"))]

pk_exact = {n: np.zeros((N_SHOTS, DIST.size)) for n, _ in LOADS}
pk_fast = {n: np.zeros((N_SHOTS, DIST.size)) for n, _ in LOADS}
sigma_d = {n: np.zeros(DIST.size) for n, _ in LOADS}
e_mean = {n: np.zeros(DIST.size) for n, _ in LOADS}
q_mean = {n: np.zeros(DIST.size) for n, _ in LOADS}

for name, ld in LOADS:
    for k in range(N_SHOTS):
        tr = fire(ld, CHOKE, 0.0, 0.0, rng, t_end=1.6)
        for j, d in enumerate(DIST):
            _, pos, vel = tr.at_horizontal(d)
            off = pattern_offsets(pos, 0.0, 0.0)
            energy = 0.5 * ld.pellet.mass * np.sum(vel**2, axis=1)
            pk_exact[name][k, j] = pk_from_pellets(off, energy, DRONE)
            sig = r50(off) * R50_TO_SIGMA
            e_bar = np.nanmean(energy)
            q_bar = mean_kill_prob(energy, DRONE)
            n_in = expected_pellets_in_disc(ld.n_pellets, sig, 0.0, DRONE.radius)
            pk_fast[name][k, j] = pk_poisson_q(n_in, q_bar)
            if k == 0:
                sigma_d[name][j], e_mean[name][j], q_mean[name][j] = sig, e_bar, q_bar

# --- 1) Pk vs distance --------------------------------------------------------------
print(f"exp08 (1) - Pk of one shot, perfectly aimed ({CHOKE} choke, mean of {N_SHOTS} shots)\n")
print(f"{'d[m]':>5}" + "".join(f"{n:>13}" for n, _ in LOADS))
for j, d in enumerate(DIST):
    print(f"{d:5.0f}" + "".join(f"{pk_exact[n][:, j].mean():13.2f}" for n, _ in LOADS))

# --- 2) Fast model check -----------------------------------------------------------
print("\nexp08 (2) - exact (pellet-by-pellet) vs fast Poisson-Gaussian Pk")
print(f"{'load':<13}{'max |diff|':>11}{'mean diff':>11}")
for n, _ in LOADS:
    diff = pk_fast[n].mean(0) - pk_exact[n].mean(0)
    print(f"{n:<13}{np.abs(diff).max():11.3f}{diff.mean():11.3f}")

# --- 3) Effective range ------------------------------------------------------------
def max_range(pk_curve, level):
    ok = np.nonzero(pk_curve >= level)[0]
    return DIST[ok[-1]] if ok.size else 0.0

print("\nexp08 (3) - effective range (last distance where mean Pk >= level)")
print(f"{'load':<13}{'Pk>=0.5':>9}{'Pk>=0.8':>9}")
for n, _ in LOADS:
    m = pk_exact[n].mean(0)
    print(f"{n:<13}{max_range(m, 0.5):9.0f}{max_range(m, 0.8):9.0f}")

# --- 4) Pk vs aim error at 50 m (fast model) ---------------------------------------
j50 = int(np.argmin(np.abs(DIST - 50.0)))
miss = np.linspace(0.0, 1.5, 151)
print("\nexp08 (4) - Pk vs aim error at 50 m (fast model)")
print(f"{'load':<13}{'Pk(0)':>7}{'miss for Pk/2 [cm]':>20}")
aim_curves = {}
for n, ld in LOADS:
    n_in = expected_pellets_in_disc(ld.n_pellets, sigma_d[n][j50], miss, DRONE.radius)
    pk = pk_poisson_q(n_in, q_mean[n][j50])
    aim_curves[n] = pk
    half = miss[np.argmax(pk <= pk[0] / 2)] if pk[0] > 0 else np.nan
    print(f"{n:<13}{pk[0]:7.2f}{half*100:20.0f}")

# --- Figure -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))

for n, _ in LOADS:
    axs[0].plot(DIST, pk_exact[n].mean(0), "o-", lw=2, label=n)
axs[0].plot(DIST, pk_fast["#4 lead"].mean(0), "k--", label="#4 lead, fast model")
axs[0].axhline(0.8, color="gray", ls=":")
axs[0].axhline(0.5, color="gray", ls=":")
axs[0].set_xlabel("distance [m]")
axs[0].set_ylabel("Pk (one shot, perfect aim)")
axs[0].set_title("Kill probability vs distance")
axs[0].grid(alpha=0.3)
axs[0].legend(fontsize=8)

for n, _ in LOADS:
    axs[1].semilogy(DIST, e_mean[n], lw=2, label=n)
for c in DRONE.components:
    axs[1].axhline(c.e_min, color="gray", ls=":", lw=0.8)
    axs[1].text(DIST[-1], c.e_min, f" {c.name}", fontsize=7, va="center")
axs[1].set_xlabel("distance [m]")
axs[1].set_ylabel("mean pellet energy [J]")
axs[1].set_title("Pellet energy vs component E50")
axs[1].grid(alpha=0.3, which="both")
axs[1].legend(fontsize=8, loc="lower left")

for n, _ in LOADS:
    axs[2].plot(miss * 100, aim_curves[n], lw=2, label=n)
axs[2].axvline(DRONE.radius * 100, color="tab:red", ls=":", label="drone half-span")
axs[2].set_xlabel("aim error at 50 m [cm]")
axs[2].set_ylabel("Pk")
axs[2].set_title("Pk vs aim error at 50 m")
axs[2].grid(alpha=0.3)
axs[2].legend(fontsize=8)

fig.tight_layout()
out = RESULTS / "exp08_lethality.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")