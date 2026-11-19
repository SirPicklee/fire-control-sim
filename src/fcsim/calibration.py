"""Custom drag model calibration.

A Mach-dependent correction k(M) multiplies the projectile's drag curve:
    Cd_new(M) = Cd(M) * k(M)
k is piecewise linear between Mach nodes and held constant outside them.
The node values are fitted to measured speed-vs-range data by least
squares. Time of flight is NOT used in the fit so that it can serve as
an independent validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .atmosphere import Atmosphere
from .ballistics import fly
from .projectile import Projectile, with_drag_correction


def speed_and_tof(proj: Projectile, ranges, *, atm: Atmosphere | None = None,
                  dt: float = 2e-3):
    """Speed [m/s] and time of flight [s] at horizontal ranges (flat fire)."""
    ranges = np.asarray(ranges, dtype=float)
    tr = fly(proj, 0.0, 0.0, atm=atm, dt=dt, max_range=float(ranges.max()) + 30.0,
             t_max=6.0, z_min=-500.0)
    rows = []
    for R in ranges:
        t, _, v = tr.at_horizontal(R)
        rows.append((np.linalg.norm(v), t))
    out = np.array(rows)
    return out[:, 0], out[:, 1]


@dataclass(frozen=True, eq=False)
class DragFit:
    projectile: Projectile       # calibrated projectile
    mach_nodes: np.ndarray
    factors: np.ndarray          # fitted k at each node
    rms_rel: float               # RMS relative speed residual on fit data
    n_eval: int                  # number of model evaluations


def fit_drag_correction(proj: Projectile, ranges, v_ref, mach_nodes, *,
                        atm: Atmosphere | None = None, dt: float = 2e-3,
                        bounds=(0.5, 2.0)) -> DragFit:
    """Fit k(M) at mach_nodes so that model speed matches v_ref at ranges."""
    ranges = np.asarray(ranges, dtype=float)
    v_ref = np.asarray(v_ref, dtype=float)
    mach_nodes = np.asarray(mach_nodes, dtype=float)

    def residuals(k):
        v, _ = speed_and_tof(with_drag_correction(proj, mach_nodes, k), ranges, atm=atm, dt=dt)
        return (v - v_ref) / v_ref

    sol = least_squares(residuals, np.ones(mach_nodes.size), bounds=bounds, diff_step=1e-3)
    return DragFit(projectile=with_drag_correction(proj, mach_nodes, sol.x),
                   mach_nodes=mach_nodes, factors=sol.x,
                   rms_rel=float(np.sqrt(np.mean(sol.fun**2))), n_eval=int(sol.nfev))