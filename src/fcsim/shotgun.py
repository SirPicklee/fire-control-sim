"""Shotgun pellets, shot loads, chokes and patterns.

Pellets are spheres. Drag uses an approximate sphere Cd(Mach) curve for
Reynolds numbers ~1e4-1e5 (subcritical); compared with published
shotshell data in exp06.

Pattern model: each pellet leaves the muzzle with a random angular
deviation, 2-D Gaussian with std sigma (rad) set by the choke so that the
classic pattern percentage falls inside a 30-inch (0.762 m) circle at
40 yards (36.58 m). A muzzle-velocity spread creates the shot string.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atmosphere import Atmosphere
from .ballistics import BatchTrajectory, fly_batch
from .projectile import Projectile
from .wind import Wind

# Sphere drag coefficient vs Mach (approximate, subcritical Reynolds number)
SPHERE_MACH = np.array([0.0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0])
SPHERE_CD = np.array([0.47, 0.49, 0.53, 0.58, 0.66, 0.77, 0.89, 0.97, 1.00, 0.98, 0.95, 0.92])

DENSITY = {"lead": 11_340.0, "steel": 7_860.0, "tungsten": 18_000.0}     # kg/m^3
SHOT_DIAMETER_MM = {"9": 2.03, "8": 2.29, "7.5": 2.41, "6": 2.79, "5": 3.05,
                    "4": 3.30, "2": 3.81, "BB": 4.57,
                    "4B": 6.10, "1B": 7.62, "00": 8.38, "000": 9.14}   # B = buckshot

PATTERN_RADIUS = 0.381       # m, 30-inch circle
PATTERN_DIST = 36.576        # m, 40 yards
CHOKE_PERCENT = {"cylinder": 0.40, "improved_cylinder": 0.50, "modified": 0.60,
                 "improved_modified": 0.65, "full": 0.70}


def choke_sigma(choke: str) -> float:
    """Angular std (rad) per axis giving the choke's 40-yard pattern percentage."""
    p = CHOKE_PERCENT[choke]
    return PATTERN_RADIUS / (PATTERN_DIST * np.sqrt(-2.0 * np.log(1.0 - p)))


def pellet(size: str = "4", material: str = "lead", v0: float = 400.0) -> Projectile:
    d = SHOT_DIAMETER_MM[size] * 1e-3
    m = DENSITY[material] * np.pi * d**3 / 6.0
    return Projectile(f"#{size} {material}", m, d, v0, SPHERE_MACH, SPHERE_CD)


@dataclass(frozen=True, eq=False)
class ShotLoad:
    pellet: Projectile
    n_pellets: int
    v0_sigma_rel: float = 0.03

    @property
    def name(self) -> str:
        return f"{self.pellet.name} x{self.n_pellets} @ {self.pellet.v0:.0f} m/s"


def shot_load(size: str = "4", material: str = "lead", payload_g: float = 32.0,
              v0: float = 400.0, v0_sigma_rel: float = 0.03) -> ShotLoad:
    p = pellet(size, material, v0)
    return ShotLoad(p, int(round(payload_g * 1e-3 / p.mass)), v0_sigma_rel)


def sample_pellets(load: ShotLoad, choke: str, az: float, el: float,
                   rng: np.random.Generator, n: int | None = None):
    """Random launch az, el (rad) and speed (m/s) for each pellet."""
    n = load.n_pellets if n is None else n
    s = choke_sigma(choke)
    dx, dy = rng.normal(0.0, s, (2, n))
    el_i = el + dy
    az_i = np.mod(az + dx / np.cos(el), 2.0 * np.pi)
    v0_i = load.pellet.v0 * (1.0 + load.v0_sigma_rel * rng.standard_normal(n))
    return az_i, el_i, v0_i


def fire(load: ShotLoad, choke: str, az: float, el: float, rng: np.random.Generator, *,
         atm: Atmosphere | None = None, wind: Wind | None = None,
         muzzle_pos=(0.0, 0.0, 0.0), dt: float = 1e-3,
         t_end: float = 1.0) -> BatchTrajectory:
    """Fly every pellet of one shot."""
    az_i, el_i, v0_i = sample_pellets(load, choke, az, el, rng)
    return fly_batch(load.pellet, az_i, el_i, v0=v0_i, atm=atm, wind=wind,
                     muzzle_pos=muzzle_pos, dt=dt, t_end=t_end)


def bore_axes(az: float, el: float):
    """Unit vectors (forward, right, up) of the bore in ENU."""
    fwd = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    right = np.array([np.cos(az), -np.sin(az), 0.0])
    up = np.cross(right, fwd)
    return fwd, right, up


def pattern_offsets(pos, az: float, el: float) -> np.ndarray:
    """Pellet offsets (N, 2) [right, up] from the pattern centroid, ⟂ to the bore."""
    _, right, up = bore_axes(az, el)
    d = np.asarray(pos, dtype=float) - np.nanmean(pos, axis=0)
    return np.stack([d @ right, d @ up], axis=-1)


def r50(offsets) -> float:
    """Radius containing half of the pellets (m)."""
    return float(np.nanmedian(np.linalg.norm(offsets, axis=-1)))


def fraction_within(offsets, radius: float) -> float:
    r = np.linalg.norm(offsets, axis=-1)
    r = r[np.isfinite(r)]
    return float(np.mean(r <= radius))