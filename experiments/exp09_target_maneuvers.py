"""exp09 - How far does a maneuvering drone move while the shot is in flight?

Even with perfect knowledge of the current state, a maneuver during the
prediction horizon tau (time of flight + latency) causes an irreducible
miss. Compare constant-velocity (CV) and constant-acceleration (CA)
extrapolation for four maneuver types.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fcsim.ballistics import fly, fly_batch
from fcsim.projectile import m80_ball
from fcsim.shotgun import pellet
from fcsim.target import Jink, Straight, Turn, Weave, simulate

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

G = 9.80665
SPEED = 15.0
A_LAT_MAX = 1.5 * G
LATENCY = 0.1
DT = 0.01
SHOT_TOL = 0.55          # m, aim error that halves shotgun Pk at 50 m (exp08)
DRONE_HALF = 0.175       # m, rifle needs miss below this

# --- Operational prediction horizons from our own ballistics --------------------
t_shot = fly_batch(pellet("4", "lead", 400.0), 0.0, 0.0, t_end=0.5).at_horizontal(50.0)[0][0]
t_rifle = fly(m80_ball(), 0.0, 0.0, max_range=330.0).at_horizontal(300.0)[0]
TAU_SHOT, TAU_RIFLE = t_shot + LATENCY, t_rifle + LATENCY
print(f"Prediction horizon: shotgun 50 m tau = {TAU_SHOT:.3f} s, "
      f"rifle 300 m tau = {TAU_RIFLE:.3f} s (latency {LATENCY} s)\n")

# --- Scenarios -------------------------------------------------------------------
common = dict(pos0=[0.0, 0.0, 50.0], speed0=SPEED, heading0=0.0, dt=DT, a_lat_max=A_LAT_MAX)
SCEN = {
    "straight": [simulate([Straight(60.0)], **common)],
    "turn 0.76 g": [simulate([Turn(60.0, 0.5)], **common)],
    "weave": [simulate([Weave(60.0, amplitude=0.8, period=4.0)], **common)],
    "jink": [simulate([Jink(60.0, max_rate=1.5, switch_mean=1.0)],
                      rng=np.random.default_rng(s), **common) for s in range(5)],
}


def prediction_errors(tracks, tau, model):
    k = int(round(tau / DT))
    tau = k * DT                 # snap the horizon to the simulation grid
    errs = []
    for tr in tracks:
        p0, v0, a0 = tr.pos[10:-k], tr.vel[10:-k], tr.acc[10:-k]
        truth = tr.pos[10 + k:]
        pred = p0 + v0 * tau
        if model == "CA":
            pred = pred + 0.5 * a0 * tau**2
        errs.append(np.linalg.norm(pred - truth, axis=1))
    return np.concatenate(errs)


# --- Tables ------------------------------------------------------------------------
for label, tau in (("shotgun 50 m", TAU_SHOT), ("rifle 300 m", TAU_RIFLE)):
    print(f"exp09 - prediction error at tau = {tau:.3f} s ({label}) [cm]")
    print(f"{'scenario':<13}{'CV rms':>8}{'CV p95':>8}{'CA rms':>8}{'CA p95':>8}"
          f"{'a_rms[g]':>10}")
    for name, tracks in SCEN.items():
        cv = prediction_errors(tracks, tau, "CV")
        ca = prediction_errors(tracks, tau, "CA")
        a_rms = np.sqrt(np.mean(np.concatenate([np.sum(t.acc**2, 1) for t in tracks]))) / G
        print(f"{name:<13}{np.sqrt(np.mean(cv**2))*100:8.1f}{np.percentile(cv, 95)*100:8.1f}"
              f"{np.sqrt(np.mean(ca**2))*100:8.1f}{np.percentile(ca, 95)*100:8.1f}{a_rms:10.2f}")
    print(f"theory, steady 0.76 g turn, CV: 0.5 a tau^2 = {0.5*SPEED*0.5*tau**2*100:.1f} cm\n")

# --- Figure ------------------------------------------------------------------------
taus = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))

for i, (name, tracks) in enumerate(SCEN.items()):
    tr = tracks[0]
    sel = tr.t <= 30.0
    axs[0].plot(tr.pos[sel, 0] + 120 * i, tr.pos[sel, 1], lw=1.5, label=name)
axs[0].set_aspect("equal")
axs[0].set_xlabel("east [m] (offset per scenario)")
axs[0].set_ylabel("north [m]")
axs[0].set_title("30 s of flight, top view")
axs[0].grid(alpha=0.3)
axs[0].legend(fontsize=8)

for name, tracks in SCEN.items():
    p95 = [np.percentile(prediction_errors(tracks, t, "CV"), 95) for t in taus]
    axs[1].loglog(taus, np.maximum(p95, 1e-4), "o-", lw=2, label=name)
axs[1].loglog(taus, 0.5 * SPEED * 0.5 * taus**2, "k:", label="½ a τ² (0.76 g)")
axs[1].axhline(SHOT_TOL, color="tab:green", ls="--", label="shotgun tolerance 50 m")
axs[1].axhline(DRONE_HALF, color="tab:red", ls="--", label="drone half-span")
for t in (TAU_SHOT, TAU_RIFLE):
    axs[1].axvline(t, color="gray", lw=0.8)
axs[1].set_ylim(1e-3, 10)
axs[1].set_xlabel("prediction horizon τ [s]")
axs[1].set_ylabel("CV prediction error, 95th pct [m]")
axs[1].set_title("Constant-velocity extrapolation")
axs[1].grid(alpha=0.3, which="both")
axs[1].legend(fontsize=7)

for model, st in (("CV", "o-"), ("CA", "s--")):
    p95 = [np.percentile(prediction_errors(SCEN["jink"], t, model), 95) for t in taus]
    axs[2].loglog(taus, p95, st, lw=2, label=f"jink, {model}")
    p95t = [np.percentile(prediction_errors(SCEN["turn 0.76 g"], t, model), 95) for t in taus]
    axs[2].loglog(taus, np.maximum(p95t, 1e-4), st, lw=1, alpha=0.6, label=f"turn, {model}")
axs[2].axhline(SHOT_TOL, color="tab:green", ls="--")
axs[2].axhline(DRONE_HALF, color="tab:red", ls="--")
axs[2].set_ylim(1e-3, 10)
axs[2].set_xlabel("prediction horizon τ [s]")
axs[2].set_ylabel("95th pct error [m]")
axs[2].set_title("Does knowing acceleration help?")
axs[2].grid(alpha=0.3, which="both")
axs[2].legend(fontsize=8)

fig.tight_layout()
out = RESULTS / "exp09_target_maneuvers.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")