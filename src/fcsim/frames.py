"""Coordinate frames and conversions.

Conventions
-----------
ENU : local tangent frame, x = East, y = North, z = Up.
      Origin is the weapon position.
Azimuth   : from North, clockwise toward East, radians in [0, 2*pi).
Elevation : above the local horizontal, radians in [-pi/2, pi/2].
All functions are vectorized: inputs may be scalars or arrays.
"""

from __future__ import annotations

import numpy as np


def aer_to_enu(rng, az, el):
    """Spherical (range, azimuth, elevation) -> ENU position.

    Returns an array with shape (..., 3): [east, north, up].
    """
    rng = np.asarray(rng, dtype=float)
    az = np.asarray(az, dtype=float)
    el = np.asarray(el, dtype=float)

    horiz = rng * np.cos(el)          # projection on the ground plane
    east = horiz * np.sin(az)
    north = horiz * np.cos(az)
    up = rng * np.sin(el)
    return np.stack([east, north, up], axis=-1)


def enu_to_aer(enu):
    """ENU position (..., 3) -> (range, azimuth, elevation)."""
    enu = np.asarray(enu, dtype=float)
    east, north, up = enu[..., 0], enu[..., 1], enu[..., 2]

    horiz = np.hypot(east, north)
    rng = np.hypot(horiz, up)
    az = np.mod(np.arctan2(east, north), 2.0 * np.pi)
    el = np.arctan2(up, horiz)
    return rng, az, el

# ---------------------------------------------------------------------------
# Rotations and body frame
# ---------------------------------------------------------------------------
# Body frame: x = right, y = forward (boresight), z = up.
# With zero attitude and heading North, body axes coincide with ENU,
# so enu_to_aer() applied to a body-frame vector gives gimbal angles.


def rot_x(a):
    """Active rotation by angle a (rad) about the x axis."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


def rot_y(a):
    """Active rotation by angle a (rad) about the y axis."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def rot_z(a):
    """Active rotation by angle a (rad) about the z axis."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def body_to_enu_matrix(heading, pitch, roll):
    """Direction cosine matrix C such that v_enu = C @ v_body.

    heading : clockwise from North (rad)
    pitch   : nose up positive (rad)
    roll    : right side down positive (rad)
    """
    return rot_z(-heading) @ rot_x(pitch) @ rot_y(roll)


def enu_to_body(v_enu, C):
    """Express ENU vector(s) (..., 3) in the body frame."""
    return np.asarray(v_enu, dtype=float) @ C      # equals C.T @ v


def body_to_enu(v_body, C):
    """Express body-frame vector(s) (..., 3) in ENU."""
    return np.asarray(v_body, dtype=float) @ C.T   # equals C @ v


def gimbal_angles(target_enu, heading, pitch, roll):
    """Azimuth/elevation of a target as seen in the platform body frame."""
    C = body_to_enu_matrix(heading, pitch, roll)
    _, az_b, el_b = enu_to_aer(enu_to_body(target_enu, C))
    return az_b, el_b