"""Wind model: power-law mean profile + Gauss-Markov gusts.

Direction convention (meteorological): from_az is the azimuth the wind
blows FROM, clockwise from North. A north wind (from_az = 0) blows toward
the South, i.e. velocity = [0, -v, 0] in ENU.

Simplification: gusts vary in time but are spatially uniform. This is
adequate for projectile flight times of a few seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindProfile:
    speed_ref: float = 0.0       # m/s at h_ref above ground
    from_az: float = 0.0         # rad, meteorological direction
    h_ref: float = 10.0          # m, reference height
    alpha: float = 1.0 / 7.0     # power-law exponent (open terrain)
    pivot_height: float = 2.0    # m, weapon pivot above ground
    h_min: float = 0.5           # m, clip to avoid h -> 0

    def speed(self, z_up):
        h = np.maximum(self.pivot_height + np.asarray(z_up, dtype=float), self.h_min)
        return self.speed_ref * (h / self.h_ref) ** self.alpha

    def velocity(self, z_up):
        """Mean wind velocity (..., 3) in ENU at height z_up above the pivot."""
        s = self.speed(z_up)
        e = -s * np.sin(self.from_az)
        n = -s * np.cos(self.from_az)
        return np.stack([e, n, np.zeros_like(s)], axis=-1)


@dataclass(frozen=True)
class GustModel:
    sigma: float = 1.0           # m/s, horizontal gust std per axis
    sigma_up: float = 0.5        # m/s, vertical gust std
    tau: float = 2.0             # s, correlation time

    def simulate(self, t, rng: np.random.Generator):
        """Gust velocities (N, 3) at increasing times t (N,)."""
        t = np.asarray(t, dtype=float)
        sig = np.array([self.sigma, self.sigma, self.sigma_up])
        g = np.empty((t.size, 3))
        g[0] = sig * rng.standard_normal(3)          # stationary start
        for k in range(1, t.size):
            phi = np.exp(-(t[k] - t[k - 1]) / self.tau)
            g[k] = phi * g[k - 1] + sig * np.sqrt(1.0 - phi**2) * rng.standard_normal(3)
        return g


class Wind:
    """Mean profile plus an optional pre-computed gust time series."""

    def __init__(self, profile: WindProfile = WindProfile(),
                 gust: GustModel | None = None, duration: float = 10.0,
                 dt: float = 0.05, rng: np.random.Generator | None = None):
        self.profile = profile
        if gust is None:
            self._t = None
        else:
            if rng is None:
                raise ValueError("gusts need an explicit rng (ICD: randomness)")
            self._t = np.arange(0.0, duration + dt, dt)
            self._g = gust.simulate(self._t, rng)

    def gust(self, t):
        if self._t is None:
            return np.zeros(np.shape(t) + (3,))
        t = np.asarray(t, dtype=float)
        return np.stack([np.interp(t, self._t, self._g[:, i]) for i in range(3)], axis=-1)

    def velocity(self, t, z_up):
        """Total wind velocity (..., 3) in ENU at time t and height z_up."""
        return self.profile.velocity(z_up) + self.gust(t)