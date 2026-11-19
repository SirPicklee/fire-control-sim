# fire-control-sim — Interface Control Document (ICD)

Status: draft v0.1 · Every module shall produce and consume data exactly as
defined here (SW-02, SW-03). Changes to this file require updating the
affected tests.

## 1. Global conventions

| Item        | Convention                                                                 |
|-------------|----------------------------------------------------------------------------|
| Units       | SI everywhere: m, s, kg, m/s, J, K, Pa. Angles in **radians**.            |
| Degrees     | Only at input/output boundaries (plots, printed tables, config files).     |
| Time        | Simulation time `t` in seconds, float, starting at 0.                      |
| Timestamps  | Each measurement carries its **time of validity** (when the physical       |
|             | quantity was sensed), not its arrival time. Latency = arrival − validity.  |
| World frame | ENU (x East, y North, z Up), origin at the weapon pivot point.             |
| Body frame  | Platform: x right, y forward (boresight), z up. `C_b2e = Rz(−ψ)Rx(θ)Ry(φ)`. |
| Azimuth     | From North, clockwise, [0, 2π).                                            |
| Elevation   | Above local horizontal, [−π/2, π/2].                                       |
| Covariance  | Same units as the quantity (m², rad², (m/s)²). Always symmetric PSD.       |
| Randomness  | Every stochastic function takes an explicit `numpy.random.Generator`.     |

## 2. Data flow

    TargetTruth ──► Radar ──► RadarDetection ──► Handoff ──► GimbalCommand
         │                                                        │
         ├──────► EO camera ──► EOMeasurement ─┐                  ▼
         └──────► LRF ───────► LRFMeasurement ─┼──► Tracker ──► Track
                                               │                  │
                       RadarDetection ─────────┘                  ▼
                                                           FireControl
                                                                  │
                                    MissObservation ◄── Shot ◄── FireSolution
                                           │
                                           └──► FireCorrection ──► bias estimate

## 3. Messages

### TargetTruth (target model → sensors, evaluation)
| Field   | Type      | Unit  | Notes                     |
|---------|-----------|-------|---------------------------|
| t       | float     | s     |                           |
| pos     | (3,)      | m     | ENU                       |
| vel     | (3,)      | m/s   | ENU                       |
| acc     | (3,)      | m/s²  | ENU                       |

### RadarDetection (radar → handoff, tracker) · ~1 Hz
| Field   | Type      | Unit  | Notes                                   |
|---------|-----------|-------|-----------------------------------------|
| t       | float     | s     | time of validity                        |
| rng     | float     | m     | slant range from weapon pivot           |
| az, el  | float     | rad   | world frame                             |
| R       | (3, 3)    | m², rad² | diag(σ_r², σ_az², σ_el²)             |
| snr     | float     | —     | linear                                  |

### EOMeasurement (thermal camera + detector → tracker) · 30–60 Hz
| Field   | Type      | Unit  | Notes                                   |
|---------|-----------|-------|-----------------------------------------|
| t       | float     | s     | time of validity (frame exposure)       |
| az, el  | float     | rad   | **world frame** (gimbal attitude applied) |
| R       | (2, 2)    | rad²  | from centroid SNR and target size       |
| n_pix   | int       | —     | target size on detector                 |
| snr     | float     | —     | peak SNR                                |

### LRFMeasurement (LRF → tracker) · 1–10 Hz
| Field   | Type      | Unit  | Notes                                   |
|---------|-----------|-------|-----------------------------------------|
| t       | float     | s     | time of validity (pulse emission)       |
| rng     | float     | m     | NaN if no return                        |
| sigma   | float     | m     |                                         |
| valid   | bool      | —     | False on dropout                        |

### Track (tracker → fire control, evaluation)
| Field   | Type      | Unit  | Notes                                   |
|---------|-----------|-------|-----------------------------------------|
| t       | float     | s     | state valid at t                        |
| id      | int       | —     |                                         |
| x       | (6,)      | m, m/s | [e, n, u, ve, vn, vu] ENU              |
| P       | (6, 6)    | SI    |                                         |
| mu      | (M,)      | —     | IMM model probabilities (optional)      |
| status  | str       | —     | "tentative" / "confirmed" / "lost"      |

### FireSolution (fire control → gimbal, shot)
| Field      | Type   | Unit | Notes                                     |
|------------|--------|------|-------------------------------------------|
| t_fire     | float  | s    | planned trigger time                      |
| az, el     | float  | rad  | barrel command, world frame               |
| tof        | float  | s    | predicted time of flight                  |
| intercept  | (3,)   | m    | predicted intercept point, ENU            |
| valid      | bool   | —    | False if no converged solution            |

### MissObservation (sensor → fire correction)
| Field   | Type      | Unit  | Notes                                       |
|---------|-----------|-------|---------------------------------------------|
| t       | float     | s     |                                             |
| miss    | (2,)      | m     | [horizontal, vertical] in plane ⟂ to LOS    |
| R       | (2, 2)    | m²    | observation uncertainty                     |

## 4. Module map

| Module               | Step  | Produces / role                          |
|----------------------|-------|------------------------------------------|
| frames.py            | 1     | Coordinate conversions, rotations        |
| messages.py          | 2     | Dataclasses for section 3                |
| platform.py          | 3     | Lever arms, boresight, parallax          |
| atmosphere.py        | 4     | ISA, wind, gusts                         |
| ballistics.py        | 5–6   | Point-mass trajectories, firing tables   |
| shotgun.py           | 7     | Pellet flight, pattern                   |
| lethality.py         | 8     | Vulnerable areas, Pk|hit                 |
| target.py            | 9     | TargetTruth trajectories                 |
| radar.py             | 10    | RadarDetection                           |
| thermal_scene.py     | 11    | Synthetic frames                         |
| detection.py         | 12    | EOMeasurement                            |
| lrf.py, timing.py    | 13    | LRFMeasurement, latency, buffering       |
| handoff.py           | 14    | Radar → EO cueing                        |
| gimbal.py            | 15    | Servo dynamics, jitter                   |
| kalman.py            | 16    | Linear KF, NEES/NIS                      |
| nonlinear.py         | 17    | EKF, UKF                                 |
| imm.py               | 18    | IMM                                      |
| fusion.py            | 19    | Asynchronous multi-sensor tracker        |
| fire_control.py      | 20–21 | FireSolution, latency compensation       |
| fire_correction.py   | 22    | Bias estimation from MissObservation     |
| error_budget.py      | 23    | Variance budget, Sobol indices           |
| engagement.py        | 24    | Monte Carlo Ph/Pk                        |
| policy.py, swarm.py  | 25–26 | Firing policy, target assignment         |
| viz.py               | 27    | Plots, animation                         |