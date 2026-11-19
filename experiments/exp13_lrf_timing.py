"""exp13 - LRF against a small drone, and the multi-sensor timeline.

1) LRF valid-return probability vs range for several pointing accuracies.
2) Gaps between valid ranges (the tracker must coast through them).
3) Measurement age and out-of-sequence measurements (OOSM) when radar,
   camera and LRF with different rates and latencies feed one processor.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fcsim.lrf import LaserRangefinder, LRFParams, lrf_pd
from fcsim.messages import EOMeasurement, LRFMeasurement, RadarDetection
from fcsim.timing import Latency, MeasurementBuffer, SensorClock

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

P = LRFParams()
lrf = LaserRangefinder(P)
rng = np.random.default_rng(13)
POINTING = [0.02e-3, 0.1e-3, 0.3e-3, 1.0e-3]
RANGES = np.array([100, 300, 500, 1000, 1500, 2000, 3000, 4000], dtype=float)
N = 2000

# --- 1) Valid-return probability -----------------------------------------------------
print("exp13 (1) - LRF valid-return probability per pulse (small drone, 0.02 m^2)\n")
print(f"{'R[m]':>6}{'beam[cm]':>9}{'fill':>6}{'perfect':>8}"
      + "".join(f"{f'{p*1e3:.2f}mrad':>10}" for p in POINTING) + f"{'sigma[cm]':>10}")
pd_mc = {p: [] for p in POINTING}
for R in RANGES:
    pos = np.array([0.0, R, 0.0])
    row = []
    sig = []
    for p in POINTING:
        ms = [lrf.measure(0.0, pos, rng, p) for _ in range(N)]
        v = np.mean([m.valid for m in ms])
        pd_mc[p].append(v)
        row.append(v)
        sig += [m.sigma for m in ms if m.valid]
    print(f"{R:6.0f}{P.divergence*R*100:9.1f}{float(P.fill(R)):6.2f}{float(lrf_pd(P, R)):8.3f}"
          + "".join(f"{v:10.3f}" for v in row)
          + f"{np.sqrt(np.mean(np.square(sig)))*100 if sig else np.nan:10.1f}")

# --- 2) Gaps between valid ranges at 1000 m -------------------------------------------
print(f"\nexp13 (2) - gaps between valid ranges at 1000 m, {P.rate:.0f} Hz, 600 s")
print(f"{'pointing':>10}{'valid/s':>9}{'mean gap':>10}{'p99 gap':>9}{'max gap':>9}")
gap_hist = {}
t = np.arange(0.0, 600.0, 1.0 / P.rate)
for p in POINTING:
    valid = np.array([lrf.measure(tk, np.array([0.0, 1000.0, 0.0]), rng, p).valid for tk in t])
    tv = t[valid]
    gaps = np.diff(tv) if tv.size > 1 else np.array([np.inf])
    gap_hist[p] = gaps
    print(f"{p*1e3:8.2f} mr{valid.mean()*P.rate:9.2f}{gaps.mean():10.2f}"
          f"{np.percentile(gaps, 99):9.2f}{gaps.max():9.2f}")
print("gaps in seconds; the nominal gap is 1 / rate = 0.20 s")

# --- 3) Multi-sensor timeline -----------------------------------------------------------
R2, R3 = np.diag([1e-8, 1e-8]), np.diag([1.0, 1e-5, 1e-5])
SENSORS = {
    "radar": (SensorClock(1.0, 0.37), Latency(0.20, 0.05),
              lambda tk: RadarDetection(tk, 1000.0, 0.1, 0.05, R3)),
    "camera": (SensorClock(30.0, 0.0, 0.0), Latency(0.04, 0.01),
               lambda tk: EOMeasurement(tk, 0.1, 0.05, R2)),
    "lrf": (SensorClock(5.0, 0.11), Latency(0.01, 0.002),
            lambda tk: LRFMeasurement(tk, 1000.0, 0.03)),
}
T_SIM, DT_PROC = 60.0, 0.01
buf = MeasurementBuffer()
kind = {}
arrivals = {k: [] for k in SENSORS}
for name, (clock, lat, make) in SENSORS.items():
    for tk in clock.times(T_SIM):
        m = make(float(tk))
        a = lat.arrival(float(tk), rng)
        buf.push(a, m)
        kind[id(m)] = name
        arrivals[name].append((tk, a))

age = {k: [] for k in SENSORS}
oosm = {k: 0 for k in SENSORS}
count = {k: 0 for k in SENSORS}
for now in np.arange(0.0, T_SIM + 1.0, DT_PROC):
    for m, is_oosm in buf.pop_ready(now):
        k = kind[id(m)]
        count[k] += 1
        oosm[k] += int(is_oosm)
        age[k].append(now - m.t)

print(f"\nexp13 (3) - processor at {1/DT_PROC:.0f} Hz, {T_SIM:.0f} s")
print(f"{'sensor':<8}{'rate':>6}{'n':>6}{'age mean':>10}{'age p95':>9}{'OOSM':>7}{'OOSM %':>8}")
for k, (clock, _, _) in SENSORS.items():
    a = np.array(age[k]) * 1e3
    print(f"{k:<8}{clock.rate:6.0f}{count[k]:6d}{a.mean():10.0f}{np.percentile(a, 95):9.0f}"
          f"{oosm[k]:7d}{100 * oosm[k] / count[k]:8.1f}")
print("age = time between validity and use, in ms (includes the 10 ms processor tick)")

# --- Figure ------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
r_plot = np.geomspace(50.0, 5000.0, 200)
axs[0].semilogx(r_plot, lrf_pd(P, r_plot), "k--", lw=1.5, label="perfect pointing")
for p in POINTING:
    axs[0].semilogx(RANGES, pd_mc[p], "o-", lw=2, label=f"pointing {p*1e3:.2f} mrad")
axs[0].axvline(2 * P.target_radius / P.divergence, color="gray", ls=":", label="beam = drone")
axs[0].set_xlabel("range [m]")
axs[0].set_ylabel("valid return probability")
axs[0].set_title("LRF against a small drone")
axs[0].grid(alpha=0.3, which="both")
axs[0].legend(fontsize=8)

for p in POINTING:
    g = gap_hist[p]
    if np.isfinite(g).all():
        axs[1].hist(g, bins=np.arange(0.1, 3.0, 0.2), histtype="step", lw=2,
                    label=f"{p*1e3:.2f} mrad", density=True)
axs[1].set_yscale("log")
axs[1].set_xlabel("gap between valid ranges [s]")
axs[1].set_ylabel("density")
axs[1].set_title("LRF gaps at 1000 m (5 Hz)")
axs[1].grid(alpha=0.3)
axs[1].legend(fontsize=8)

colors = {"radar": "tab:red", "camera": "tab:blue", "lrf": "tab:green"}
for i, (k, pts) in enumerate(arrivals.items()):
    for tk, a in pts:
        if tk <= 2.0:
            axs[2].plot([tk, a], [i, i + 0.4], color=colors[k], lw=1)
            axs[2].plot(tk, i, "o", color=colors[k], ms=3)
axs[2].set_yticks([0.2, 1.2, 2.2])
axs[2].set_yticklabels(list(arrivals))
axs[2].set_xlim(0, 2.3)
axs[2].set_xlabel("time [s]  (dot = validity, line end = arrival)")
axs[2].set_title("Sensor timeline, first 2 s")
axs[2].grid(alpha=0.3, axis="x")

fig.tight_layout()
out = RESULTS / "exp13_lrf_timing.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")