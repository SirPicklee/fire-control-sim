"""Surveillance radar: detection, measurement noise and false alarms.

Detection follows Swerling 1 (exponentially fluctuating target): received
power ~ Exp(mean 1 + SNR) is compared with T = -ln(Pfa), which gives
Pd = Pfa ** (1 / (1 + SNR)). SNR scales as RCS / R^4 from a reference
specification. Measurement noise shrinks with SNR down to a calibration
floor. The radar is co-located with the weapon pivot (ENU origin).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import enu_to_aer
from .messages import RadarDetection, TargetTruth

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class RadarParams:
    ref_range: float = 2000.0          # m, where ref_pd is reached for ref_rcs
    ref_rcs: float = 0.01              # m^2
    ref_pd: float = 0.9
    pfa: float = 1e-6                  # per resolution cell
    n_cells: float = 1e6               # resolution cells per scan
    revisit: float = 1.0               # s
    range_res: float = 1.5             # m
    beam_az: float = np.deg2rad(2.0)
    beam_el: float = np.deg2rad(4.0)
    k_mono: float = 1.6                # monopulse slope factor
    range_floor: float = 0.5           # m
    az_floor: float = np.deg2rad(0.2)
    el_floor: float = np.deg2rad(0.4)
    min_range: float = 50.0
    max_range: float = 5000.0
    el_min: float = np.deg2rad(-2.0)
    el_max: float = np.deg2rad(40.0)

    @property
    def threshold(self) -> float:
        """Detection threshold on normalised power (noise mean = 1)."""
        return -np.log(self.pfa)

    @property
    def snr_ref(self) -> float:
        """SNR giving ref_pd at pfa for a Swerling 1 target."""
        return np.log(self.pfa) / np.log(self.ref_pd) - 1.0


def snr(params: RadarParams, rng_m, rcs):
    """Mean SNR (linear) of a target of RCS `rcs` at range `rng_m`."""
    rng_m = np.asarray(rng_m, dtype=float)
    return params.snr_ref * (np.asarray(rcs) / params.ref_rcs) * (params.ref_range / rng_m) ** 4


def pd_swerling1(snr_lin, pfa):
    """Single-look detection probability for a Swerling 1 target."""
    return pfa ** (1.0 / (1.0 + np.asarray(snr_lin, dtype=float)))


def measurement_sigma(params: RadarParams, snr_lin):
    """(sigma_range [m], sigma_az [rad], sigma_el [rad]) at a given SNR."""
    s = np.sqrt(2.0 * np.maximum(np.asarray(snr_lin, dtype=float), 1e-9))
    sr = np.hypot(params.range_res / s, params.range_floor)
    sa = np.hypot(params.beam_az / (params.k_mono * s), params.az_floor)
    se = np.hypot(params.beam_el / (params.k_mono * s), params.el_floor)
    return sr, sa, se


class SearchRadar:
    def __init__(self, params: RadarParams = RadarParams(), rcs: float = 0.01):
        self.p = params
        self.rcs = rcs

    def _in_coverage(self, rng_m, el) -> bool:
        p = self.p
        return p.min_range <= rng_m <= p.max_range and p.el_min <= el <= p.el_max

    def _make(self, t, rng_m, az, el, snr_inst, rng):
        sr, sa, se = measurement_sigma(self.p, snr_inst)
        r_m = max(rng_m + sr * rng.standard_normal(), 1.0)
        az_m = float(np.mod(az + sa * rng.standard_normal(), TWO_PI))
        el_m = float(np.clip(el + se * rng.standard_normal(), -np.pi / 2, np.pi / 2))
        return RadarDetection(t, r_m, az_m, el_m, np.diag([sr**2, sa**2, se**2]), snr=float(snr_inst))

    def scan(self, t: float, truth: TargetTruth | None, rng: np.random.Generator):
        """One scan: list of RadarDetection and a bool array (True = real target)."""
        dets, flags = [], []
        T = self.p.threshold

        if truth is not None:
            r_true, az_true, el_true = (float(v) for v in enu_to_aer(truth.pos))
            if self._in_coverage(r_true, el_true):
                mean_snr = float(snr(self.p, r_true, self.rcs))
                power = rng.exponential(1.0 + mean_snr)
                if power > T:
                    dets.append(self._make(t, r_true, az_true, el_true, power - 1.0, rng))
                    flags.append(True)

        for _ in range(rng.poisson(self.p.pfa * self.p.n_cells)):
            r_f = rng.uniform(self.p.min_range, self.p.max_range)
            az_f = rng.uniform(0.0, TWO_PI)
            el_f = rng.uniform(self.p.el_min, self.p.el_max)
            power = T + rng.exponential(1.0)          # noise-only, given it crossed T
            dets.append(self._make(t, r_f, az_f, el_f, power - 1.0, rng))
            flags.append(False)

        return dets, np.array(flags, dtype=bool)

    def run(self, track, t_start: float, t_end: float, rng: np.random.Generator):
        """Scan a TargetTrack every revisit period. Returns [(t, dets, flags), ...]."""
        out = []
        t = t_start
        while t <= t_end + 1e-9:
            out.append((t, *self.scan(t, track.state_at(t), rng)))
            t += self.p.revisit
        return out