"""Message types exchanged between modules (see docs/icd.md, section 3).

All fields are SI units and radians. Messages validate themselves on
construction and are immutable (frozen) afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TWO_PI = 2.0 * np.pi
TRACK_STATUSES = ("tentative", "confirmed", "lost")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _vec(x, n: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    return arr


def _cov(P, n: int, name: str) -> np.ndarray:
    arr = np.asarray(P, dtype=float)
    if arr.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n}), got {arr.shape}")
    if not np.allclose(arr, arr.T, rtol=1e-9, atol=1e-15):
        raise ValueError(f"{name} must be symmetric")
    if np.linalg.eigvalsh(arr).min() < -1e-12 * max(1.0, np.abs(arr).max()):
        raise ValueError(f"{name} must be positive semi-definite")
    return arr


def _check_az(az: float) -> None:
    if not 0.0 <= az < TWO_PI:
        raise ValueError(f"azimuth must be in [0, 2pi), got {az}")


def _check_el(el: float) -> None:
    if not -np.pi / 2 <= el <= np.pi / 2:
        raise ValueError(f"elevation must be in [-pi/2, pi/2], got {el}")


def _set(obj, name, value) -> None:
    object.__setattr__(obj, name, value)   # needed inside frozen dataclasses


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class TargetTruth:
    t: float
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        _set(self, "pos", _vec(self.pos, 3, "pos"))
        _set(self, "vel", _vec(self.vel, 3, "vel"))
        _set(self, "acc", _vec(self.acc, 3, "acc"))


@dataclass(frozen=True, eq=False)
class RadarDetection:
    t: float
    rng: float
    az: float
    el: float
    R: np.ndarray
    snr: float = np.nan

    def __post_init__(self):
        if not self.rng > 0:
            raise ValueError("rng must be positive")
        _check_az(self.az)
        _check_el(self.el)
        _set(self, "R", _cov(self.R, 3, "R"))


@dataclass(frozen=True, eq=False)
class EOMeasurement:
    t: float
    az: float
    el: float
    R: np.ndarray
    n_pix: int = 0
    snr: float = np.nan

    def __post_init__(self):
        _check_az(self.az)
        _check_el(self.el)
        _set(self, "R", _cov(self.R, 2, "R"))


@dataclass(frozen=True, eq=False)
class LRFMeasurement:
    t: float
    rng: float
    sigma: float
    valid: bool = True

    def __post_init__(self):
        if self.valid and not (np.isfinite(self.rng) and self.rng > 0):
            raise ValueError("valid LRF measurement needs a positive range")
        if not self.valid and not np.isnan(self.rng):
            raise ValueError("dropout must have rng = NaN")

    @classmethod
    def dropout(cls, t: float, sigma: float) -> "LRFMeasurement":
        return cls(t=t, rng=np.nan, sigma=sigma, valid=False)


@dataclass(frozen=True, eq=False)
class Track:
    t: float
    id: int
    x: np.ndarray
    P: np.ndarray
    mu: np.ndarray | None = None
    status: str = "tentative"

    def __post_init__(self):
        _set(self, "x", _vec(self.x, 6, "x"))
        _set(self, "P", _cov(self.P, 6, "P"))
        if self.status not in TRACK_STATUSES:
            raise ValueError(f"status must be one of {TRACK_STATUSES}")

    @property
    def pos(self) -> np.ndarray:
        return self.x[:3]

    @property
    def vel(self) -> np.ndarray:
        return self.x[3:]


@dataclass(frozen=True, eq=False)
class FireSolution:
    t_fire: float
    az: float
    el: float
    tof: float
    intercept: np.ndarray
    valid: bool = True

    def __post_init__(self):
        _check_az(self.az)
        _check_el(self.el)
        _set(self, "intercept", _vec(self.intercept, 3, "intercept"))


@dataclass(frozen=True, eq=False)
class MissObservation:
    t: float
    miss: np.ndarray
    R: np.ndarray

    def __post_init__(self):
        _set(self, "miss", _vec(self.miss, 2, "miss"))
        _set(self, "R", _cov(self.R, 2, "R"))