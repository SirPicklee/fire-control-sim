"""Target kinematics: piecewise maneuver legs.

State: horizontal speed s, heading psi (clockwise from North, like azimuth)
and vertical speed vz. A turn rate omega > 0 turns right. Constant-rate
turns are integrated with the exact arc formula, so a full circle closes
to machine precision. Output velocities and accelerations are ENU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import aer_to_enu
from .messages import TargetTruth


# ---------------------------------------------------------------------------
# Maneuver legs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Straight:
    duration: float
    accel: float = 0.0                 # m/s^2 along the velocity
    climb_rate: float | None = None    # m/s; None keeps the current vz

    def turn_rates(self, n, h, rng):
        return np.zeros(n)


@dataclass(frozen=True)
class Turn:
    duration: float
    rate: float                        # rad/s, > 0 turns right
    accel: float = 0.0
    climb_rate: float | None = None

    def turn_rates(self, n, h, rng):
        return np.full(n, self.rate)


@dataclass(frozen=True)
class Weave:
    duration: float
    amplitude: float                   # rad/s, peak turn rate
    period: float                      # s
    accel: float = 0.0
    climb_rate: float | None = None

    def turn_rates(self, n, h, rng):
        tk = (np.arange(n) + 0.5) * h
        return self.amplitude * np.sin(2.0 * np.pi * tk / self.period)


@dataclass(frozen=True)
class Jink:
    duration: float
    max_rate: float                    # rad/s
    switch_mean: float = 1.0           # s, mean time between new turn commands
    accel: float = 0.0
    climb_rate: float | None = None

    def turn_rates(self, n, h, rng):
        if rng is None:
            raise ValueError("Jink needs an explicit rng (ICD: randomness)")
        rates = np.empty(n)
        k = 0
        while k < n:
            m = max(1, int(round(rng.exponential(self.switch_mean) / h)))
            rates[k:k + m] = rng.uniform(-self.max_rate, self.max_rate)
            k += m
        return rates


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class TargetTrack:
    t: np.ndarray        # (N,)
    pos: np.ndarray      # (N, 3) ENU
    vel: np.ndarray      # (N, 3) ENU
    acc: np.ndarray      # (N, 3) ENU

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.vel, axis=1)

    @property
    def slant_range(self) -> np.ndarray:
        return np.linalg.norm(self.pos, axis=1)

    def state_at(self, t: float) -> TargetTruth:
        """Linearly interpolated truth at time t."""
        lerp = lambda a: np.array([np.interp(t, self.t, a[:, i]) for i in range(3)])
        return TargetTruth(float(t), lerp(self.pos), lerp(self.vel), lerp(self.acc))


def _vel(s, psi, vz):
    return np.array([s * np.sin(psi), s * np.cos(psi), vz])


def simulate(legs, *, pos0, speed0: float, heading0: float, vz0: float = 0.0,
             dt: float = 0.01, a_lat_max: float | None = None,
             rng: np.random.Generator | None = None) -> TargetTrack:
    """Fly a sequence of maneuver legs and return the truth trajectory."""
    pos = np.asarray(pos0, dtype=float).copy()
    s, psi, vz, t = float(speed0), float(heading0), float(vz0), 0.0
    T, P, V, A = [t], [pos.copy()], [_vel(s, psi, vz)], [np.zeros(3)]

    for leg in legs:
        if leg.climb_rate is not None:
            vz = float(leg.climb_rate)
        n = max(1, int(round(leg.duration / dt)))
        h = leg.duration / n
        rates = leg.turn_rates(n, h, rng)
        for k in range(n):
            w = rates[k]
            if a_lat_max is not None and s > 0.0:
                w = float(np.clip(w, -a_lat_max / s, a_lat_max / s))
            s_new = max(s + leg.accel * h, 0.0)
            s_avg = 0.5 * (s + s_new)
            psi_new = psi + w * h
            if abs(w) > 1e-12:
                dx = s_avg / w * (np.cos(psi) - np.cos(psi_new))
                dy = s_avg / w * (np.sin(psi_new) - np.sin(psi))
            else:
                dx = s_avg * h * np.sin(psi)
                dy = s_avg * h * np.cos(psi)
            pos = pos + np.array([dx, dy, vz * h])
            s, psi, t = s_new, psi_new, t + h
            along = np.array([np.sin(psi), np.cos(psi), 0.0])
            right = np.array([np.cos(psi), -np.sin(psi), 0.0])
            T.append(t)
            P.append(pos.copy())
            V.append(_vel(s, psi, vz))
            A.append(leg.accel * along + s * w * right)

    return TargetTrack(np.array(T), np.array(P), np.array(V), np.array(A))


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------
def inbound(range0: float, az: float, height: float, speed: float, legs, **kw) -> TargetTrack:
    """Start at horizontal range `range0` along azimuth `az`, flying toward the weapon."""
    start = aer_to_enu(range0, az, 0.0) + np.array([0.0, 0.0, height])
    return simulate(legs, pos0=start, speed0=speed, heading0=np.mod(az + np.pi, 2 * np.pi), **kw)


def crossing(offset: float, height: float, speed: float, half_length: float, legs, **kw) -> TargetTrack:
    """Fly West -> East along y = offset; closest approach above x = 0."""
    start = np.array([-half_length, offset, height])
    return simulate(legs, pos0=start, speed0=speed, heading0=np.pi / 2, **kw)