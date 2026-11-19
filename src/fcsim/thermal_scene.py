"""Synthetic thermal frames of a small drone against the sky.

Temperature-difference (Delta-T) model derived from Project B
(thermal-pipeline): NETD 50 mK, residual FPN after NUC 17 mK, Gaussian PSF.
The drone is a side-view silhouette (body, arms, hot motors) rendered on an
oversampled grid, blurred by the PSF, integrated over each pixel, attenuated
by the atmosphere and corrupted by temporal noise and fixed-pattern noise.

Pixel convention: pixel (i, j) covers x in [j, j+1), y in [i, i+1); x right,
y down. The boresight is at (W/2, H/2). Small-angle mapping:
    x = W/2 + wrap(az - az0) * cos(el0) / ifov,   y = H/2 - (el - el0) / ifov
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .frames import enu_to_aer


@dataclass(frozen=True)
class CameraParams:
    width: int = 128                 # px, tracking window
    height: int = 128
    ifov: float = 0.12e-3            # rad per pixel
    netd: float = 0.050              # K, temporal noise (Project B)
    fpn: float = 0.017               # K, residual fixed pattern after NUC (Project B)
    psf_sigma_px: float = 0.6        # px
    frame_rate: float = 30.0         # Hz
    oversample: int = 8
    sky_temp: float = 255.0          # K, apparent sky temperature


@dataclass(frozen=True)
class DroneSignature:
    span: float = 0.35               # m, motor-to-motor
    body_w: float = 0.20
    body_h: float = 0.08
    arm_h: float = 0.012
    motor_r: float = 0.02
    dT_body: float = 12.0            # K above sky
    dT_arm: float = 8.0
    dT_motor: float = 30.0


@dataclass(frozen=True, eq=False)
class ThermalFrame:
    t: float
    image: np.ndarray                # (H, W) apparent temperature [K]
    az0: float                       # boresight azimuth [rad]
    el0: float                       # boresight elevation [rad]
    ifov: float
    truth_px: np.ndarray             # (2,) true drone centre [x, y] in pixels
    truth_range: float               # m


def atmospheric_transmission(rng_m, gamma: float = 0.2e-3):
    """LWIR transmission exp(-gamma R), gamma in 1/m (clear air ~0.2 /km)."""
    return np.exp(-gamma * np.asarray(rng_m, dtype=float))


def angles_to_pixel(cam: CameraParams, az, el, az0, el0):
    d_az = (np.asarray(az) - az0 + np.pi) % (2 * np.pi) - np.pi
    x = cam.width / 2 + d_az * np.cos(el0) / cam.ifov
    y = cam.height / 2 - (np.asarray(el) - el0) / cam.ifov
    return x, y


def _silhouette(sig: DroneSignature, u, v):
    """Delta-T on metric target-plane coordinates u (right), v (up)."""
    dT = np.zeros_like(u)
    body = (np.abs(u) <= sig.body_w / 2) & (np.abs(v) <= sig.body_h / 2)
    dT[body] = sig.dT_body
    v_arm = sig.body_h / 2 - sig.arm_h / 2
    arm = (np.abs(u) <= sig.span / 2) & (np.abs(v - v_arm) <= sig.arm_h / 2) & ~body
    dT[arm] = sig.dT_arm
    for side in (-1.0, 1.0):
        motor = (u - side * sig.span / 2) ** 2 + (v - v_arm) ** 2 <= sig.motor_r**2
        dT[motor] = sig.dT_motor
    return dT


def render_signature(sig: DroneSignature, rng_m: float, cam: CameraParams,
                     center_px, gamma: float = 0.2e-3) -> np.ndarray:
    """Noise-free Delta-T image (H, W) of the drone centred at center_px = (x, y)."""
    H, W, os_ = cam.height, cam.width, cam.oversample
    img = np.zeros((H, W))
    mpp = rng_m * cam.ifov                              # metres per pixel at target
    half = 0.5 * sig.span / mpp + 4.0 * cam.psf_sigma_px + 2.0
    cx, cy = center_px
    x0, x1 = int(np.floor(cx - half)), int(np.ceil(cx + half))
    y0, y1 = int(np.floor(cy - half)), int(np.ceil(cy + half))
    if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
        return img                                       # outside the window

    xs = x0 + (np.arange((x1 - x0) * os_) + 0.5) / os_
    ys = y0 + (np.arange((y1 - y0) * os_) + 0.5) / os_
    X, Y = np.meshgrid(xs, ys)
    fine = _silhouette(sig, (X - cx) * mpp, -(Y - cy) * mpp)
    fine = gaussian_filter(fine, cam.psf_sigma_px * os_, mode="constant")
    patch = fine.reshape(y1 - y0, os_, x1 - x0, os_).mean(axis=(1, 3))
    patch *= atmospheric_transmission(rng_m, gamma)

    ya, yb, xa, xb = max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)
    img[ya:yb, xa:xb] = patch[ya - y0:yb - y0, xa - x0:xb - x0]
    return img


class ThermalCamera:
    """Tracking-window camera co-located with the weapon pivot."""

    def __init__(self, params: CameraParams = CameraParams(),
                 signature: DroneSignature = DroneSignature(),
                 rng: np.random.Generator | None = None, gamma: float = 0.2e-3):
        if rng is None:
            raise ValueError("ThermalCamera needs an rng for its fixed pattern noise")
        self.p, self.sig, self.gamma = params, signature, gamma
        self.fpn_map = rng.normal(0.0, params.fpn, (params.height, params.width))

    def frame(self, t: float, target_pos, az0: float, el0: float,
              rng: np.random.Generator) -> ThermalFrame:
        r, az, el = (float(v) for v in enu_to_aer(target_pos))
        cx, cy = angles_to_pixel(self.p, az, el, az0, el0)
        signal = render_signature(self.sig, r, self.p, (cx, cy), self.gamma)
        noise = rng.normal(0.0, self.p.netd, signal.shape)
        image = self.p.sky_temp + signal + self.fpn_map + noise
        return ThermalFrame(t, image, az0, el0, self.p.ifov, np.array([cx, cy]), r)