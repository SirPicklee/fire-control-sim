"""exp10 - What does the search radar give us against an inbound small drone?

Pd vs range and RCS, measurement accuracy, first-detection and 3-of-4
confirmation ranges (Monte Carlo), cumulative detection, false alarms.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fcsim.radar import RadarParams, SearchRadar, measurement_sigma, pd_swerling1, snr
from fcsim.target import Straight, inbound

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

P = RadarParams()
SPEED = 15.0
RCS = {"small 0.003": 0.003, "nominal 0.01": 0.01, "large 0.03": 0.03}
rng = np.random.default_rng(10)

# --- 1) Pd and accuracy vs range -----------------------------------------------------
R_TAB = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], dtype=float)
print("exp10 (1) - single-scan Pd and accuracy (accuracy at mean SNR, RCS 0.01 m^2)\n")
print(f"{'R[m]':>6}" + "".join(f"{'Pd ' + k.split()[0]:>10}" for k in RCS)
      + f"{'sR[m]':>8}{'saz[deg]':>9}{'sel[deg]':>9}{'cross[m]':>9}")
for r in R_TAB:
    sr, sa, se = measurement_sigma(P, snr(P, r, 0.01))
    print(f"{r:6.0f}" + "".join(f"{pd_swerling1(snr(P, r, s), P.pfa):10.3f}" for s in RCS.values())
          + f"{sr:8.2f}{np.rad2deg(sa):9.2f}{np.rad2deg(se):9.2f}{r * sa:9.1f}")

# SR-01 by simulation
radar_sr01 = SearchRadar(RadarParams(n_cells=0.0), rcs=0.01)
tr0 = inbound(1500.0, 0.5, 0.0, 0.0, [Straight(1.0)])
hits = sum(radar_sr01.scan(0.0, tr0.state_at(0.0), rng)[1].any() for _ in range(2000))
print(f"\nSR-01 by simulation: Pd(1500 m, 0.01 m^2) = {hits / 2000:.3f} (requirement >= 0.9)")

# --- 2) Cumulative detection, analytic -----------------------------------------------
def p_cumulative(r, rcs):
    rj = np.arange(r, P.max_range + 1e-9, SPEED * P.revisit)
    return 1.0 - np.prod(1.0 - pd_swerling1(snr(P, rj, rcs), P.pfa))

# --- 3) Inbound Monte Carlo: first detection and 3-of-4 confirmation ----------------
N_RUNS = 200
track = inbound(4500.0, 0.5, 60.0, SPEED, [Straight(290.0)])
print(f"\nexp10 (2) - inbound drone, {SPEED:.0f} m/s, {N_RUNS} runs, revisit {P.revisit:.0f} s")
print(f"{'RCS':<14}{'R_cum90':>9}{'first med':>10}{'first p10':>10}"
      f"{'conf med':>9}{'conf p10':>9}")
confirm = {}
for name, rcs in RCS.items():
    radar = SearchRadar(RadarParams(n_cells=0.0), rcs=rcs)
    first, conf = [], []
    for _ in range(N_RUNS):
        scans = radar.run(track, 0.0, track.t[-1], rng)
        det = np.array([f.any() for _, _, f in scans])
        ranges = np.array([float(np.linalg.norm(track.state_at(t).pos)) for t, _, _ in scans])
        i_first = np.argmax(det) if det.any() else None
        win = np.convolve(det.astype(int), np.ones(4, dtype=int), mode="full")[:det.size]
        i_conf = np.argmax(win >= 3) if (win >= 3).any() else None
        first.append(ranges[i_first] if i_first is not None else np.nan)
        conf.append(ranges[i_conf] if i_conf is not None else np.nan)
    first, conf = np.array(first), np.array(conf)
    confirm[name] = conf
    r_grid = np.arange(4500.0, 100.0, -10.0)
    r_cum90 = r_grid[np.argmax([p_cumulative(r, rcs) >= 0.9 for r in r_grid])]
    print(f"{name:<14}{r_cum90:9.0f}{np.nanmedian(first):10.0f}{np.nanpercentile(first, 10):10.0f}"
          f"{np.nanmedian(conf):9.0f}{np.nanpercentile(conf, 10):9.0f}")
print("ranges in m; p10 = 10th percentile (worst 10 % of runs detect later than this)")

# --- 4) False alarms ----------------------------------------------------------------
fa_per_scan = P.pfa * P.n_cells
sr, sa, se = measurement_sigma(P, snr(P, 1000.0, 0.01))
cells_gate = (6 * sr / P.range_res) * (6 * sa / P.beam_az) * (6 * se / P.beam_el)
print(f"\nexp10 (3) - false alarms: {fa_per_scan:.1f} per scan over the whole volume")
print(f"3-sigma gate around a 1000 m track: {cells_gate:.0f} cells -> "
      f"{P.pfa * cells_gate:.1e} false alarms per scan inside the gate")

# --- Figure -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
r_plot = np.linspace(200.0, 5000.0, 400)
for name, rcs in RCS.items():
    l, = axs[0].plot(r_plot, pd_swerling1(snr(P, r_plot, rcs), P.pfa), lw=2, label=f"single scan, {name}")
    axs[0].plot(r_plot, [p_cumulative(r, rcs) for r in r_plot], "--", color=l.get_color(), lw=1.2)
axs[0].axhline(0.9, color="gray", ls=":")
axs[0].axvline(1500, color="tab:red", ls=":", label="SR-01 1500 m")
axs[0].set_xlabel("range [m]")
axs[0].set_ylabel("detection probability")
axs[0].set_title("Pd (solid) and cumulative inbound (dashed)")
axs[0].grid(alpha=0.3)
axs[0].legend(fontsize=7)

sr, sa, se = measurement_sigma(P, snr(P, r_plot, 0.01))
axs[1].plot(r_plot, sr, lw=2, label="range σ")
axs[1].plot(r_plot, r_plot * sa, lw=2, label="cross-range σ (azimuth)")
axs[1].plot(r_plot, r_plot * se, lw=2, label="vertical σ (elevation)")
axs[1].set_xlabel("range [m]")
axs[1].set_ylabel("position error [m]")
axs[1].set_title("Radar accuracy (RCS 0.01 m², mean SNR)")
axs[1].grid(alpha=0.3)
axs[1].legend(fontsize=8)

bins = np.arange(0.0, 4600.0, 100.0)
for name, conf in confirm.items():
    axs[2].hist(conf[np.isfinite(conf)], bins=bins, alpha=0.5, label=name)
axs[2].set_xlabel("range at 3-of-4 confirmation [m]")
axs[2].set_ylabel("runs")
axs[2].set_title("Track confirmation range")
axs[2].legend(fontsize=8)

fig.tight_layout()
out = RESULTS / "exp10_radar.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")