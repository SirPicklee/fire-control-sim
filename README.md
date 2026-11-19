
# fire-control-sim

Closed-loop **counter-UAS fire control simulation** in Python: radar cueing,
EO/thermal tracking, laser ranging, exterior ballistics, fire correction and
engagement Monte Carlo against small multirotor drones.

The project is built step by step against a written requirements set
([`docs/requirements.md`](docs/requirements.md)) and an interface control
document ([`docs/icd.md`](docs/icd.md)). Each modelling step comes with unit
tests and an experiment script that answers one engineering question and saves
a figure to `results/`.

> Status: **v0.1, work in progress.** Sensor, ballistics and lethality models
> are in place. Tracker (Kalman / IMM), gimbal control, fire control loop,
> fire correction and the full engagement / swarm Monte Carlo are planned
> (requirements steps 14–28) but not yet implemented.

---

## Quick start

Requires Python ≥ 3.11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"

pytest                                       # run the test suite
python experiments/exp03_exterior_ballistics.py   # run one experiment
```

Experiments write their plots to `results/expNN_*.png`.

---

## Repository layout

```
fire-control-sim/
├── src/fcsim/          # the simulation library
├── tests/              # pytest unit tests (one file per module/topic)
├── experiments/        # exp01…exp13: one question per script, plots to results/
├── results/            # generated figures (git-ignored)
├── docs/
│   ├── requirements.md # system / software / validation requirements
│   └── icd.md          # frames, units and message definitions
└── pyproject.toml
```

### Modules (`src/fcsim`)

| Module             | What it does |
|--------------------|--------------|
| `frames.py`        | ENU ↔ az/el/range, rotation matrices, body ↔ world, gimbal angles |
| `messages.py`      | Immutable, self-validating message types from the ICD (`TargetTruth`, `RadarDetection`, `EOMeasurement`, `LRFMeasurement`, `Track`, `FireSolution`, `MissObservation`) |
| `platform.py`      | Sensor/muzzle lever arms, parallax and boresight error |
| `atmosphere.py`    | ISA troposphere with temperature offset (density, speed of sound, density altitude) |
| `wind.py`          | Power-law wind profile + Gauss–Markov gusts |
| `projectile.py`    | Projectile definitions and Mach-dependent drag |
| `ballistics.py`    | 3-DOF point-mass trajectories (RK4), Coriolis, elevation solver, firing tables, batch flight |
| `calibration.py`   | Fits a Mach-dependent drag correction to measured speed-vs-range data |
| `shotgun.py`       | Pellet loads, chokes and pattern sampling |
| `lethality.py`     | Drone vulnerability model and kill probability (exact and Poisson approximation) |
| `target.py`        | Drone kinematics from maneuver legs (straight, turn, weave, jink) |
| `radar.py`         | Search radar: Swerling-1 detection, measurement noise, false alarms |
| `thermal_scene.py` | Synthetic thermal frames of a drone against the sky |
| `detection.py`     | Thermal blob detection and sub-pixel centroiding → `EOMeasurement` |
| `lrf.py`           | Laser rangefinder return probability vs range and pointing |
| `timing.py`        | Sensor clocks, latency and an out-of-sequence-aware measurement buffer |

### Experiments (`experiments/`)

| Script | Question |
|--------|----------|
| `exp01_parallax_boresight`      | Parallax vs boresight error — which dominates at which range? |
| `exp02_atmosphere_wind`         | How much do atmosphere and wind change the firing problem? |
| `exp03_exterior_ballistics`     | Effect of air on a rifle bullet; vacuum vs drag, seasonal differences |
| `exp04_validation_firing_table` | Model validation against manufacturer data (VAL-01) and firing table |
| `exp05_drag_calibration`        | Drag-correction fit with held-out validation |
| `exp06_shotgun_pattern`         | Pellet model check, pattern growth, shot string |
| `exp07_val02_buckshot`          | Sphere drag model vs measured chronograph data (VAL-02) |
| `exp08_lethality`               | Kill probability vs load, range and aim error |
| `exp09_target_maneuvers`        | Irreducible miss from target maneuvers during flight + latency |
| `exp10_radar`                   | Radar detection ranges, accuracy, false alarms |
| `exp11_thermal_scene`           | What the drone looks like to the thermal camera vs range and haze |
| `exp12_detection`               | Detector Pd, accuracy and consistency vs range (SR-03) |
| `exp13_lrf_timing`              | LRF performance and the multi-sensor timeline |

---

## Conventions

Defined in full in [`docs/icd.md`](docs/icd.md):

- **SI units and radians** everywhere internally; degrees only at I/O boundaries.
- **World frame:** ENU (x East, y North, z Up), origin at the weapon pivot.
- **Body frame:** x right, y forward (boresight), z up.
- **Azimuth** from North, clockwise, `[0, 2π)`; **elevation** above horizontal.
- Every measurement carries its **time of validity**, not arrival time.
- Every stochastic function takes an explicit `numpy.random.Generator`, so all
  Monte Carlo runs are reproducible from a seed.

---

## Tests

```bash
pytest -q
```

Current state: **178 passed, 1 failed.**

Known failure: `tests/test_rotations.py::test_heading_offsets_gimbal_azimuth`.
`gimbal_angles` returns `2π` instead of `0` when the target is exactly on the
platform heading — a floating-point wrap-around in the `[0, 2π)` normalisation.
Fix: wrap the result once more (e.g. `az % (2*np.pi)` followed by mapping
values ≈ `2π` to `0`), or compare angles modulo `2π` in the test.

---

