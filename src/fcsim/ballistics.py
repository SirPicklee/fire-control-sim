"""Point-mass (3-DOF) exterior ballistics integrated with RK4.

State s = [x, y, z, vx, vy, vz] in ENU. Forces: gravity, aerodynamic drag
relative to the (possibly moving) air and, optionally, the Coriolis
acceleration of the rotating Earth. Validated against the exact vacuum
solution and the Didion lag rule (tests/).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .atmosphere import Atmosphere
from .frames import aer_to_enu
from .projectile import Projectile
from .wind import Wind

G = 9.80665
GRAVITY = np.array([0.0, 0.0, -G])
OMEGA_EARTH = 7.2921159e-5      # rad/s


def coriolis_accel(vel, latitude: float):
    """Coriolis acceleration -2 Omega x v in ENU at the given latitude (rad)."""
    omega = OMEGA_EARTH * np.array([0.0, np.cos(latitude), np.sin(latitude)])
    return -2.0 * np.cross(omega, np.asarray(vel, dtype=float))


@dataclass(frozen=True, eq=False)
class Trajectory:
    t: np.ndarray        # (N,)
    pos: np.ndarray      # (N, 3) ENU
    vel: np.ndarray      # (N, 3) ENU

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.vel, axis=1)

    @property
    def slant_range(self) -> np.ndarray:
        return np.linalg.norm(self.pos, axis=1)

    def _first_crossing(self, r: np.ndarray, R: float):
        idx = np.nonzero(r >= R)[0]
        if idx.size == 0 or idx[0] == 0:
            raise ValueError(f"distance {R} m not reached by this trajectory")
        i = idx[0]
        f = (R - r[i - 1]) / (r[i] - r[i - 1])
        lerp = lambda a: a[i - 1] + f * (a[i] - a[i - 1])
        return lerp(self.t), lerp(self.pos), lerp(self.vel)

    def at_range(self, R: float):
        """(t, pos, vel) where the slant range from the pivot first reaches R."""
        return self._first_crossing(self.slant_range, R)

    def at_horizontal(self, d: float, origin=(0.0, 0.0, 0.0)):
        """(t, pos, vel) where the horizontal distance from origin first reaches d."""
        o = np.asarray(origin, dtype=float)
        h = np.hypot(self.pos[:, 0] - o[0], self.pos[:, 1] - o[1])
        return self._first_crossing(h, d)


def fly(proj: Projectile, az: float, el: float, *,
        atm: Atmosphere | None = None, wind: Wind | None = None,
        muzzle_pos=(0.0, 0.0, 0.0), v0: float | None = None,
        latitude: float | None = None,
        dt: float = 1e-3, t_max: float = 5.0, max_range: float | None = None,
        z_min: float = -100.0, vacuum: bool = False) -> Trajectory:
    """Integrate one trajectory fired at world azimuth/elevation (rad).

    latitude (rad): if given, the Coriolis acceleration is included.
    """
    atm = atm if atm is not None else Atmosphere()
    v0 = proj.v0 if v0 is None else v0
    state = np.concatenate([np.asarray(muzzle_pos, dtype=float), aer_to_enu(v0, az, el)])

    def f(t, s):
        pos, vel = s[:3], s[3:]
        acc = GRAVITY.copy()
        if not vacuum:
            z = pos[2]
            w = wind.velocity(t, z) if wind is not None else 0.0
            acc += proj.drag_accel(vel - w, atm.density(z), atm.sound_speed(z))
        if latitude is not None:
            acc += coriolis_accel(vel, latitude)
        return np.concatenate([vel, acc])

    t = 0.0
    ts, states = [t], [state.copy()]
    while t < t_max:
        k1 = f(t, state)
        k2 = f(t + dt / 2, state + dt / 2 * k1)
        k3 = f(t + dt / 2, state + dt / 2 * k2)
        k4 = f(t + dt, state + dt * k3)
        state = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        ts.append(t)
        states.append(state.copy())
        if state[2] < z_min:
            break
        if max_range is not None and np.linalg.norm(state[:3]) >= max_range:
            break

    S = np.array(states)
    return Trajectory(np.array(ts), S[:, :3], S[:, 3:])


def solve_elevation(proj: Projectile, target, *, az: float | None = None,
                    atm: Atmosphere | None = None, wind: Wind | None = None,
                    muzzle_pos=(0.0, 0.0, 0.0), latitude: float | None = None,
                    dt: float = 1e-3, bracket=(-0.05, 0.15),
                    xtol: float = 1e-7) -> float:
    """Barrel elevation (rad) whose trajectory passes through the target height.

    The vertical miss is evaluated at the target's horizontal distance from
    the muzzle. The search bracket is relative to the line-of-sight elevation.
    """
    target = np.asarray(target, dtype=float)
    m = np.asarray(muzzle_pos, dtype=float)
    d = target - m
    if az is None:
        az = float(np.mod(np.arctan2(d[0], d[1]), 2.0 * np.pi))
    horiz = float(np.hypot(d[0], d[1]))
    el_los = float(np.arctan2(d[2], horiz))
    max_range = float(np.linalg.norm(target)) + 50.0

    def miss(el):
        tr = fly(proj, az, el, atm=atm, wind=wind, muzzle_pos=m, latitude=latitude,
                 dt=dt, max_range=max_range, t_max=6.0, z_min=target[2] - 500.0)
        _, p, _ = tr.at_horizontal(horiz, origin=m)
        return p[2] - target[2]

    return brentq(miss, el_los + bracket[0], el_los + bracket[1], xtol=xtol)


def firing_table(proj: Projectile, ranges, *, zero_range: float, sight_height: float,
                 atm: Atmosphere | None = None, wind: Wind | None = None,
                 latitude: float | None = None, dt: float = 1e-3):
    """Zero in still air, then tabulate the trajectory relative to the sight line.

    Returns (zero elevation [rad], dict of arrays). 'path' is the height of the
    bullet relative to the line of sight (negative = below); 'correction_mrad'
    is the hold needed to bring the impact back to the line of sight; 'drift'
    is the lateral (East) displacement caused by wind and/or Coriolis.
    """
    atm = atm if atm is not None else Atmosphere()
    ranges = np.asarray(ranges, dtype=float)
    el0 = solve_elevation(proj, [0.0, zero_range, sight_height], az=0.0, atm=atm, dt=dt)
    tr = fly(proj, 0.0, el0, atm=atm, wind=wind, latitude=latitude, dt=dt,
             max_range=float(ranges.max()) + 50.0, t_max=6.0)

    rows = []
    for R in ranges:
        t, p, v = tr.at_horizontal(R)
        speed = float(np.linalg.norm(v))
        path = p[2] - sight_height
        rows.append((t, speed, 0.5 * proj.mass * speed**2, path, -path / R * 1e3, p[0]))
    a = np.array(rows)
    table = {"range": ranges, "tof": a[:, 0], "speed": a[:, 1], "energy": a[:, 2],
             "path": a[:, 3], "correction_mrad": a[:, 4], "drift": a[:, 5]}
    return el0, table



# ---------------------------------------------------------------------------
# Vectorized flight of many projectiles (shot pellets, Monte Carlo)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class BatchTrajectory:
    t: np.ndarray        # (M,)
    pos: np.ndarray      # (M, N, 3) ENU
    vel: np.ndarray      # (M, N, 3) ENU

    def at_horizontal(self, d: float, origin=(0.0, 0.0, 0.0)):
        """Per projectile (t, pos, vel) where horizontal distance first reaches d.

        Returns arrays (N,), (N, 3), (N, 3); NaN where d is never reached.
        """
        o = np.asarray(origin, dtype=float)
        h = np.hypot(self.pos[..., 0] - o[0], self.pos[..., 1] - o[1])   # (M, N)
        n = np.arange(h.shape[1])
        reached = h >= d
        i = np.argmax(reached, axis=0)                     # first True (0 if none)
        ok = reached[i, n] & (i > 0)
        i = np.where(ok, i, 1)
        f = (d - h[i - 1, n]) / (h[i, n] - h[i - 1, n])
        t = self.t[i - 1] + f * (self.t[i] - self.t[i - 1])
        pos = self.pos[i - 1, n] + f[:, None] * (self.pos[i, n] - self.pos[i - 1, n])
        vel = self.vel[i - 1, n] + f[:, None] * (self.vel[i, n] - self.vel[i - 1, n])
        t[~ok], pos[~ok], vel[~ok] = np.nan, np.nan, np.nan
        return t, pos, vel


def fly_batch(proj: Projectile, az, el, *, v0=None,
              atm: Atmosphere | None = None, wind: Wind | None = None,
              muzzle_pos=(0.0, 0.0, 0.0), latitude: float | None = None,
              dt: float = 1e-3, t_end: float = 1.0,
              vacuum: bool = False) -> BatchTrajectory:
    """Integrate N projectiles in parallel with vectorized RK4 for t_end seconds.

    az, el and v0 may be scalars or arrays of shape (N,).
    """
    atm = atm if atm is not None else Atmosphere()
    az, el = np.broadcast_arrays(np.atleast_1d(np.asarray(az, dtype=float)),
                                 np.atleast_1d(np.asarray(el, dtype=float)))
    v0 = np.broadcast_to(np.asarray(proj.v0 if v0 is None else v0, dtype=float), az.shape)
    N = az.size
    pos0 = np.broadcast_to(np.asarray(muzzle_pos, dtype=float), (N, 3))
    state = np.concatenate([pos0, aer_to_enu(v0, az, el)], axis=1)    # (N, 6)

    def f(t, s):
        p, v = s[:, :3], s[:, 3:]
        acc = np.broadcast_to(GRAVITY, v.shape).copy()
        if not vacuum:
            z = p[:, 2]
            w = wind.velocity(t, z) if wind is not None else 0.0
            acc += proj.drag_accel(v - w, atm.density(z), atm.sound_speed(z))
        if latitude is not None:
            acc += coriolis_accel(v, latitude)
        return np.concatenate([v, acc], axis=1)

    n_steps = int(np.ceil(t_end / dt))
    out = np.empty((n_steps + 1, N, 6))
    out[0] = state
    t = 0.0
    for k in range(n_steps):
        k1 = f(t, state)
        k2 = f(t + dt / 2, state + dt / 2 * k1)
        k3 = f(t + dt / 2, state + dt / 2 * k2)
        k4 = f(t + dt, state + dt * k3)
        state = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        out[k + 1] = state
    return BatchTrajectory(np.arange(n_steps + 1) * dt, out[..., :3], out[..., 3:])