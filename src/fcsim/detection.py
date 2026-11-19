"""Thermal target detection and sub-pixel centroiding -> EOMeasurement.

Pipeline: robust background (median) and noise (1.4826 * MAD), threshold
at k*sigma, 8-connected components, pick the strongest blob (or the one
nearest a predicted position), dilate it, intensity-weighted centroid,
centroid variance from error propagation, pixel -> angle conversion.

Reported variance = noise propagation + floor^2 + (shape_frac * size)^2.
The shape term covers the aimpoint bias of extended targets: the hot-spot
centroid is not the geometric centre (exp12: ~3 % of the target size).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .messages import EOMeasurement
from .thermal_scene import ThermalFrame

EIGHT = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class DetectorParams:
    k_sigma: float = 5.0           # detection threshold
    dilate: int = 2                # px added around the blob for the centroid window
    floor_px: float = 0.05         # px, centroid error floor (quantisation, residual)
    shape_frac: float = 0.03       # aimpoint uncertainty as a fraction of target size
    gate_px: float = 20.0          # px, max distance from a predicted position


@dataclass(frozen=True)
class PixelDetection:
    x: float                       # centroid [px]
    y: float
    sx: float                      # 1-sigma centroid uncertainty [px]
    sy: float
    n_pix: int                     # pixels above threshold in the blob
    size_px: int                   # largest blob extent [px]
    snr: float                     # peak SNR
    noise: float                   # estimated pixel noise [K]


def detect_px(image: np.ndarray, params: DetectorParams = DetectorParams(),
              predicted_px=None) -> PixelDetection | None:
    bg = np.median(image)
    sigma = 1.4826 * np.median(np.abs(image - bg))
    s = image - bg
    labels, n = ndimage.label(s > params.k_sigma * sigma, structure=EIGHT)
    if n == 0:
        return None
    ids = np.arange(1, n + 1)
    sums = np.asarray(ndimage.sum(s, labels, ids))

    if predicted_px is None:
        k = ids[np.argmax(sums)]
    else:
        com = np.array(ndimage.center_of_mass(s, labels, ids))[:, ::-1] + 0.5   # (x, y)
        dist = np.linalg.norm(com - np.asarray(predicted_px), axis=1)
        if dist.min() > params.gate_px:
            return None
        k = ids[np.argmin(dist)]

    blob = labels == k
    by, bx = np.nonzero(blob)
    size = int(max(bx.max() - bx.min(), by.max() - by.min()) + 1)
    window = ndimage.binary_dilation(blob, structure=EIGHT, iterations=params.dilate) \
        if params.dilate > 0 else blob
    ys, xs = np.nonzero(window)
    w = s[ys, xs]
    total = w.sum()
    if total <= 0:
        return None
    xc = float(np.sum(w * (xs + 0.5)) / total)
    yc = float(np.sum(w * (ys + 0.5)) / total)
    extra = params.floor_px**2 + (params.shape_frac * size) ** 2
    var_x = sigma**2 * np.sum((xs + 0.5 - xc) ** 2) / total**2 + extra
    var_y = sigma**2 * np.sum((ys + 0.5 - yc) ** 2) / total**2 + extra
    return PixelDetection(xc, yc, float(np.sqrt(var_x)), float(np.sqrt(var_y)),
                          int(blob.sum()), size, float(s[blob].max() / sigma), float(sigma))


def pixel_to_angles(x, y, frame: ThermalFrame):
    H, W = frame.image.shape
    az = np.mod(frame.az0 + (x - W / 2) * frame.ifov / np.cos(frame.el0), 2 * np.pi)
    el = np.clip(frame.el0 - (y - H / 2) * frame.ifov, -np.pi / 2, np.pi / 2)
    return float(az), float(el)


def detect(frame: ThermalFrame, params: DetectorParams = DetectorParams(),
           predicted_px=None) -> EOMeasurement | None:
    """Detect the drone in a thermal frame and return an ICD EOMeasurement."""
    d = detect_px(frame.image, params, predicted_px)
    if d is None:
        return None
    az, el = pixel_to_angles(d.x, d.y, frame)
    s_az = d.sx * frame.ifov / np.cos(frame.el0)
    s_el = d.sy * frame.ifov
    return EOMeasurement(frame.t, az, el, np.diag([s_az**2, s_el**2]),
                         n_pix=d.n_pix, snr=d.snr)