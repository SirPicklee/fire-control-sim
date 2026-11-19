"""Laser rangefinder against a small drone (1550 nm Er:glass, from Project D).

Speckle makes the return power exponential from pulse to pulse, so the
single-pulse Pd has the Swerling-1 form Pd = Pfa ** (1 / (1 + SNR)).
SNR ~ fill(R) / R^2: while the beam footprint is smaller than the drone the
whole beam is intercepted (1/R^2); beyond that only a fraction (1/R^4).
A return requires the beam to touch the drone despite pointing error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .messages import LRFMeasurement


@dataclass(frozen=True)
class LRFParams:
    rate: float = 5.0                  # Hz
    divergence: float = 0.5e-3         # rad, full angle
    target_area: float = 0.02          # m^2, presented drone area
    ref_range: float = 2000.0          # m, ref_pd reached here with perfect pointing
    ref_pd: float = 0.9
    pfa: float = 1e-4                  # per range gate
    sigma_floor: float = 0.03          # m, timing floor (Project D, CFD)
    sigma_snr: float = 1.0             # m, noise term scaled by 1/sqrt(2 SNR)
    pointing_sigma: float = 0.1e-3     # rad per axis (gimbal jitter + track error)
    min_range: float = 20.0
    max_range: float = 8000.0

    @property
    def threshold(self) -> float:
        return -np.log(self.pfa)

    @property
    def target_radius(self) -> float:
        return float(np.sqrt(self.target_area / np.pi))

    def fill(self, rng_m):
        """Fraction of beam energy intercepted by a centred drone."""
        beam_area = np.pi * (self.divergence * np.asarray(rng_m, dtype=float) / 2) ** 2
        return np.minimum(1.0, self.target_area / beam_area)

    @property
    def snr_ref(self) -> float:
        return np.log(self.pfa) / np.log(self.ref_pd) - 1.0


def lrf_snr(p: LRFParams, rng_m):
    """Mean SNR with perfect pointing."""
    rng_m = np.asarray(rng_m, dtype=float)
    return p.snr_ref * (p.ref_range / rng_m) ** 2 * p.fill(rng_m) / p.fill(p.ref_range)


def lrf_pd(p: LRFParams, rng_m):
    """Single-pulse Pd with perfect pointing (speckle, Swerling-1 form)."""
    return p.pfa ** (1.0 / (1.0 + lrf_snr(p, rng_m)))


class LaserRangefinder:
    def __init__(self, params: LRFParams = LRFParams()):
        self.p = params

    def measure(self, t: float, target_pos, rng: np.random.Generator,
                pointing_sigma: float | None = None) -> LRFMeasurement:
        p = self.p
        ps = p.pointing_sigma if pointing_sigma is None else pointing_sigma
        R = float(np.linalg.norm(target_pos))
        if not p.min_range <= R <= p.max_range:
            return LRFMeasurement.dropout(t, p.sigma_floor)
        miss = float(np.hypot(*rng.normal(0.0, ps, 2)))
        if miss * R > p.target_radius + p.divergence * R / 2:
            return LRFMeasurement.dropout(t, p.sigma_floor)       # beam missed the drone
        power = rng.exponential(1.0 + float(lrf_snr(p, R)))
        if power <= p.threshold:
            return LRFMeasurement.dropout(t, p.sigma_floor)       # speckle fade / weak
        sigma = float(np.hypot(p.sigma_floor, p.sigma_snr / np.sqrt(2.0 * (power - 1.0))))
        return LRFMeasurement(t, R + sigma * rng.standard_normal(), sigma, True)