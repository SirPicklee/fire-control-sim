"""Projectile models and aerodynamic drag.

Drag acceleration: a = -(rho * Cd(M) * A / (2 m)) * |v_rel| * v_rel,
where v_rel is the projectile velocity relative to the air.

Bullets use a standard drag function (G7) scaled by a form factor
derived from the ballistic coefficient. Table values are a coarse
approximation of the published G7 function; validated in Step 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LB_IN2_TO_KG_M2 = 703.0696   # 1 lb/in^2 in kg/m^2

# G7 standard drag function: Cd vs Mach (coarse approximation)
G7_MACH = np.array([0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05,
                    1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0])
G7_CD = np.array([0.120, 0.120, 0.120, 0.122, 0.126, 0.134, 0.164, 0.380, 0.404,
                  0.401, 0.388, 0.373, 0.346, 0.320, 0.298, 0.264, 0.236, 0.200])

# Mach-dependent drag correction for M80, calibrated to RWS data (exp05).
M80_MACH_NODES = np.array([0.8, 1.0, 1.2, 1.6, 2.4])
M80_DRAG_FACTORS = np.array([1.8489, 1.2623, 1.0117, 1.0356, 1.0162])

@dataclass(frozen=True, eq=False)
class Projectile:
    name: str
    mass: float              # kg
    diameter: float          # m
    v0: float                # m/s, muzzle velocity
    mach_table: np.ndarray
    cd_table: np.ndarray
    form_factor: float = 1.0

    @property
    def area(self) -> float:
        return np.pi * self.diameter**2 / 4.0

    def cd(self, mach):
        """Drag coefficient at the given Mach number(s)."""
        return self.form_factor * np.interp(mach, self.mach_table, self.cd_table)

    def drag_accel(self, v_rel, rho, sound_speed):
        """Drag acceleration (..., 3) for air-relative velocity v_rel (..., 3)."""
        v_rel = np.asarray(v_rel, dtype=float)
        speed = np.linalg.norm(v_rel, axis=-1, keepdims=True)
        mach = speed / np.asarray(sound_speed, dtype=float)[..., None]
        k = 0.5 * np.asarray(rho, dtype=float)[..., None] * self.cd(mach) * self.area / self.mass
        return -k * speed * v_rel

    @classmethod
    def from_g7_bc(cls, name, mass, diameter, v0, bc_lb_in2):
        """Bullet defined by its G7 ballistic coefficient [lb/in^2]."""
        sectional_density = mass / diameter**2            # kg/m^2
        form_factor = sectional_density / (bc_lb_in2 * LB_IN2_TO_KG_M2)
        return cls(name, mass, diameter, v0, G7_MACH, G7_CD, form_factor)


def with_drag_correction(proj: Projectile, mach_nodes, factors) -> Projectile:
    """New projectile whose drag curve is multiplied by k(M).

    k is piecewise linear between Mach nodes and constant outside them.
    """
    mach_nodes = np.asarray(mach_nodes, dtype=float)
    factors = np.asarray(factors, dtype=float)
    grid = np.union1d(proj.mach_table, mach_nodes)
    cd = proj.cd(grid) * np.interp(grid, mach_nodes, factors)
    return Projectile(f"{proj.name} (cal)", proj.mass, proj.diameter, proj.v0,
                      grid, cd, form_factor=1.0)


def m80_ball_g7() -> Projectile:
    """7.62x51 mm M80 with a constant G7 BC (uncalibrated reference)."""
    return Projectile.from_g7_bc("7.62 M80 (G7)", mass=9.5e-3, diameter=7.82e-3,
                                 v0=825.0, bc_lb_in2=0.215)


def m80_ball() -> Projectile:
    """7.62x51 mm M80, drag calibrated to RWS manufacturer data (exp05).

    Validated to 800 m (VAL-01: speed and TOF within 2 %). Beyond 800 m the
    transonic drag is less certain (see exp05 cross-validation).
    """
    return with_drag_correction(m80_ball_g7(), M80_MACH_NODES, M80_DRAG_FACTORS)