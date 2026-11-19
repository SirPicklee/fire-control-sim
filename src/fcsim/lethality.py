"""Target vulnerability and kill probability.

The drone is a disc of radius R (presented to the shooter) containing
critical components. A pellet entering the disc hits component i with
probability A_i / A_disc (uniform placement assumption). The component is
damaged with a log-normal fragility probability Phi(ln(E / E50) / beta)
(a hard step if beta = 0), and a damaged component kills the drone with
probability p_kill_i.

Component values are engineering assumptions (no public test data);
their influence is quantified with Sobol indices in Step 23.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ncx2, norm


@dataclass(frozen=True)
class Component:
    name: str
    area: float        # m^2, presented area
    e_min: float       # J, median damage energy E50 (hard threshold if beta = 0)
    p_kill: float      # P(kill | component damaged)
    beta: float = 0.0  # log-normal fragility spread; 0 = step function


@dataclass(frozen=True, eq=False)
class DroneVulnerability:
    components: tuple
    radius: float = 0.175

    def __post_init__(self):
        if sum(c.area for c in self.components) > self.disc_area:
            raise ValueError("component areas exceed the drone disc area")
        for c in self.components:
            if not 0.0 <= c.p_kill <= 1.0:
                raise ValueError(f"{c.name}: p_kill must be in [0, 1]")
            if c.beta < 0.0:
                raise ValueError(f"{c.name}: beta must be >= 0")

    @property
    def disc_area(self) -> float:
        return np.pi * self.radius**2

    def damage_prob(self, energy):
        """P(damage | hit) for each component, shape (..., n_components).

        Log-normal fragility: Phi(ln(E / E50) / beta); a step if beta = 0.
        """
        e = np.asarray(energy, dtype=float)[..., None]
        e50 = np.array([c.e_min for c in self.components])
        beta = np.array([c.beta for c in self.components])
        hard = (e >= e50).astype(float)
        with np.errstate(divide="ignore"):
            soft = norm.cdf(np.log(e / e50) / np.where(beta > 0, beta, 1.0))
        return np.where(beta > 0, soft, hard)

    def kill_prob_per_pellet(self, energy):
        """q(E): kill probability of one pellet inside the disc."""
        area = np.array([c.area for c in self.components])
        p = np.array([c.p_kill for c in self.components])
        return np.sum(area / self.disc_area * p * self.damage_prob(energy), axis=-1)


def small_quadcopter(beta: float = 0.4) -> DroneVulnerability:
    """Mavic-class quadcopter, side aspect. Values are assumptions.

    beta: log-normal fragility spread for every component (0 = hard thresholds).
    """
    return DroneVulnerability(components=(
        Component("propellers", 0.010, 2.0, 0.6, beta),
        Component("motors", 0.003, 5.0, 0.9, beta),
        Component("battery", 0.006, 10.0, 0.5, beta),
        Component("flight_controller", 0.002, 3.0, 0.8, beta),
        Component("arms", 0.004, 15.0, 0.4, beta),
    ))


def pk_from_pellets(offsets, energies, drone: DroneVulnerability) -> float:
    """Exact Pk for given pellets: offsets (N, 2) from drone centre, energies (N,)."""
    offsets = np.asarray(offsets, dtype=float)
    energies = np.asarray(energies, dtype=float)
    r = np.linalg.norm(offsets, axis=-1)
    inside = np.isfinite(r) & (r <= drone.radius)
    q = drone.kill_prob_per_pellet(energies[inside])
    return float(1.0 - np.prod(1.0 - q))


def pk_poisson(n_expected, energy, drone: DroneVulnerability):
    """Poisson approximation: Pk = 1 - exp(-lambda * q(E))."""
    return 1.0 - np.exp(-np.asarray(n_expected) * drone.kill_prob_per_pellet(energy))


def expected_pellets_in_disc(n_pellets, sigma, miss, radius):
    """Mean pellets inside a disc of given radius whose centre is `miss` (m)
    from the centre of a circular Gaussian pattern with std sigma (m) per axis."""
    miss = np.asarray(miss, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    frac = ncx2.cdf(radius**2 / sigma**2, df=2, nc=miss**2 / sigma**2)
    return n_pellets * frac


def mean_kill_prob(energies, drone: DroneVulnerability) -> float:
    """Average q(E) over a sample of pellet energies (NaN ignored).

    Use this instead of q(mean E): q is non-linear, so the two differ (Jensen).
    """
    e = np.asarray(energies, dtype=float)
    e = e[np.isfinite(e)]
    return float(np.mean(drone.kill_prob_per_pellet(e))) if e.size else 0.0


def pk_poisson_q(n_expected, q_mean):
    """Poisson Pk from the mean per-pellet kill probability."""
    return 1.0 - np.exp(-np.asarray(n_expected) * q_mean)