# fire-control-sim — System Requirements

Status: draft v0.1 · Values are design targets; they may be revised with
justification as models mature. Final compliance is recorded in the
verification matrix (Step 28).

Verification methods: **T** = test / simulation, **A** = analysis, **I** = inspection.

## Scenario assumptions

| ID    | Assumption                                                        |
|-------|-------------------------------------------------------------------|
| AS-01 | Threat: small multirotor UAS, ~0.35 m span, RCS ~0.01 m², 5–25 m/s |
| AS-02 | Platform: fixed tripod/tower; vehicle disturbances added in Step 15 |
| AS-03 | Primary effector: 12-gauge shotgun; comparison: 7.62 mm rifle      |
| AS-04 | Atmosphere: ISA ±20 °C, wind up to 10 m/s with gusts              |

## System requirements

| ID    | Requirement                                                                              | Method | Step   |
|-------|------------------------------------------------------------------------------------------|--------|--------|
| SR-01 | Radar shall detect the threat (AS-01) at ≥ 1500 m with Pd ≥ 0.9, Pfa ≤ 1e-4 per cell     | A, T   | 10     |
| SR-02 | Radar-to-EO handoff shall succeed with probability ≥ 0.95 within ≤ 2 s at 1000 m          | T      | 14     |
| SR-03 | EO angle measurement error shall be ≤ 0.2 mrad RMS for targets ≥ 4 pixels                 | T      | 12     |
| SR-04 | Fused track position error shall be ≤ 1.0 m RMS at 300 m (non-maneuvering, steady state)  | T      | 16–19  |
| SR-05 | Tracking filters shall be statistically consistent (NEES within 95 % chi-square bounds)    | T      | 16–18  |
| SR-06 | Maneuver (≥ 1 g turn) shall be detected by the IMM within ≤ 1.0 s                         | T      | 18     |
| SR-07 | Gimbal shall slew ≥ 60 °/s and settle within 0.3 s for a 10° step (1 mrad tolerance)      | T      | 15     |
| SR-08 | Total sensor-to-fire latency shall be ≤ 100 ms; residual miss after compensation ≤ 0.2 m  | A, T   | 13, 21 |
| SR-09 | Shotgun burst Pk ≥ 0.8 vs non-maneuvering 15 m/s target at 100 m                         | T      | 24     |
| SR-10 | Shotgun burst Pk ≥ 0.6 vs 1 g maneuvering target at 100 m                                 | T      | 24     |
| SR-11 | Fire correction shall reduce boresight bias by ≥ 70 % after 3 observed shots              | T      | 22     |
| SR-12 | Detection-to-first-shot reaction time shall be ≤ 5 s at 1000 m initial detection          | T      | 25     |
| SR-13 | System shall engage a 5-UAS swarm and neutralize ≥ 4 before the closest reaches 50 m      | T      | 26     |
| SR-14 | LRF shall be eye-safe (IEC 60825-1 Class 1)                                               | A      | 13     |

## Software requirements

| ID    | Requirement                                                              | Method | Step  |
|-------|--------------------------------------------------------------------------|--------|-------|
| SW-01 | Every module shall have unit tests; CI shall run on Python 3.11 and 3.12  | I, T   | all, 28 |
| SW-02 | All internal quantities shall be SI units and radians                     | I      | all   |
| SW-03 | All frames shall follow docs/icd.md (ENU world, x-right/y-fwd/z-up body)  | I, T   | 1, 2  |
| SW-04 | All Monte Carlo runs shall be reproducible from a fixed seed              | T      | 24    |

## Model validation requirements

| ID     | Requirement                                                                 | Method | Step |
|--------|-----------------------------------------------------------------------------|--------|------|
| VAL-01 | M80 speed and time of flight within 2 % of RWS manufacturer data, 100–800 m | T      | 6    |

| VAL-02 | Lead buckshot (#4 buck, 00, 000) 50-yd velocity loss predicted for v0 = 1200–1450 ft/s brackets Brass Fetcher chronograph data within ±5 points | T | 7 |

Note on VAL-02: the first reference (gunnersden.com, computed table) was rejected
in exp06 because it implies an effective sphere Cd ≈ 0.45 through transonic
flight, below the subcritical sphere minimum (~0.47). Replaced by measured
chronograph data (brassfetcher.com). Muzzle velocities of the measured loads are
not stated, hence a bracketing check rather than a point comparison.


## Deviations and waivers

**SR-03 — FAIL, waiver granted (Step 12).**
exp12: worst per-axis RMS 0.210 mrad at 50 m (limit 0.2 mrad). The error is
almost entirely a constant aimpoint bias: the hot-motor centroid lies 1.05 cm
above the drone's geometric centre at every range (0.210 mrad x 50 m =
0.105 mrad x 100 m = 1.05 cm). Noise-only error is 0.005 mrad.
Impact: 1 cm is 1/50 of the 50 m shot pattern and 1/17 of the drone
half-span; negligible for weapon effects. Mitigation: the detector reports
an aimpoint uncertainty of 3 % of the target size so that the tracker does not
trust the bias (Step 12c). Recommendation: express SR-03 at the target
(e.g. <= 2 cm) rather than as an angle, which is over-tight at short range.