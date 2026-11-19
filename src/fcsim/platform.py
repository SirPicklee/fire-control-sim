"""Platform geometry: sensor / muzzle lever arms, parallax and boresight.

Lever arms are given in the platform body frame (x right, y forward, z up)
relative to the weapon pivot, which is the ENU origin (docs/icd.md).
Simplification: lever arms are fixed to the platform and do not rotate
with the gimbal; this is adequate for the small offsets studied here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import aer_to_enu, body_to_enu, body_to_enu_matrix, enu_to_aer


@dataclass(frozen=True, eq=False)
class Mount:
    sensor_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    muzzle_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heading: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    def __post_init__(self):
        for name in ("sensor_offset", "muzzle_offset"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != (3,):
                raise ValueError(f"{name} must have shape (3,)")
            object.__setattr__(self, name, arr)

    @property
    def C(self) -> np.ndarray:
        return body_to_enu_matrix(self.heading, self.pitch, self.roll)

    @property
    def sensor_pos(self) -> np.ndarray:
        return body_to_enu(self.sensor_offset, self.C)

    @property
    def muzzle_pos(self) -> np.ndarray:
        return body_to_enu(self.muzzle_offset, self.C)


def target_from_sensor(mount: Mount, rng, az, el):
    """ENU position of a point seen by the sensor at (rng, az, el)."""
    return mount.sensor_pos + aer_to_enu(rng, az, el)


def barrel_angles(mount: Mount, rng, az, el):
    """World az/el that point the muzzle at the point the sensor sees.

    With the true (LRF) range this removes parallax exactly.
    With a fixed assumed range it models a mount converged at that range.
    """
    _, az_b, el_b = enu_to_aer(target_from_sensor(mount, rng, az, el) - mount.muzzle_pos)
    return az_b, el_b


def parallax_angle(mount: Mount, target_enu):
    """Angle between sensor->target and muzzle->target lines (rad)."""
    u_s = np.asarray(target_enu, dtype=float) - mount.sensor_pos
    u_m = np.asarray(target_enu, dtype=float) - mount.muzzle_pos
    cross = np.linalg.norm(np.cross(u_s, u_m), axis=-1)
    dot = np.sum(u_s * u_m, axis=-1)
    return np.arctan2(cross, dot)


def line_miss_distance(origin, az, el, target_enu):
    """Perpendicular distance from target to a straight line (no ballistics)."""
    u = aer_to_enu(1.0, az, el)
    d = np.asarray(target_enu, dtype=float) - np.asarray(origin, dtype=float)
    return np.linalg.norm(np.cross(d, u), axis=-1)


def apply_boresight_error(az, el, d_az, d_el):
    """Add a small angular misalignment (rad) to a pointing direction."""
    return np.mod(az + d_az, 2.0 * np.pi), el + d_el