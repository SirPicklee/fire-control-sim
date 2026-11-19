import numpy as np
import pytest

from fcsim.frames import aer_to_enu
from fcsim.thermal_scene import (
    CameraParams, DroneSignature, ThermalCamera, angles_to_pixel,
    atmospheric_transmission, render_signature,
)

CAM = CameraParams()
SIG = DroneSignature()


def _x_centroid(cx, cam):
    img = render_signature(SIG, 300.0, cam, (cx, 64.0), gamma=0.0)
    xs = np.arange(cam.width) + 0.5
    return np.sum(img.sum(axis=0) * xs) / img.sum()


def test_transmission_is_one_at_zero_and_decreasing():
    t = atmospheric_transmission(np.array([0.0, 500.0, 2000.0]))
    assert t[0] == pytest.approx(1.0)
    assert np.all(np.diff(t) < 0)


def test_drone_width_in_pixels_scales_with_range():
    sharp = CameraParams(psf_sigma_px=1e-3)
    R = 500.0
    img = render_signature(SIG, R, sharp, (64.0, 64.0), gamma=0.0)
    cols = np.nonzero(img.max(axis=0) > 0.5 * SIG.dT_arm)[0]
    expected = (SIG.span + 2 * SIG.motor_r) / (R * CAM.ifov)
    assert cols.size == pytest.approx(expected, abs=1.5)


def test_psf_blur_conserves_integrated_signal():
    sharp = render_signature(SIG, 400.0, CameraParams(psf_sigma_px=1e-3), (64.3, 63.7), gamma=0.0)
    blur = render_signature(SIG, 400.0, CAM, (64.3, 63.7), gamma=0.0)
    assert blur.sum() == pytest.approx(sharp.sum(), rel=1e-3)


def test_horizontal_centroid_follows_subpixel_position():
    # Oversampling on a 1/os grid quantises edges; error must stay well below
    # SR-03 (0.2 mrad ~ 1.7 px) and shrink as oversampling increases.
    for cx in (64.0, 64.25, 64.5, 64.8, 65.37):
        assert _x_centroid(cx, CAM) == pytest.approx(cx, abs=0.05)


def test_centroid_quantisation_error_shrinks_with_oversampling():
    cx = 64.8                                       # not a multiple of 1/8
    err8 = abs(_x_centroid(cx, CameraParams(oversample=8)) - cx)
    err32 = abs(_x_centroid(cx, CameraParams(oversample=32)) - cx)
    assert err32 < err8
    assert err32 < 0.01


def test_contrast_drops_with_range():
    near = render_signature(SIG, 500.0, CAM, (64.0, 64.0)).max()
    far = render_signature(SIG, 2000.0, CAM, (64.0, 64.0)).max()
    assert far < near


def test_noise_statistics_and_fixed_pattern():
    cam = ThermalCamera(CAM, SIG, rng=np.random.default_rng(0))
    far_away = aer_to_enu(1000.0, np.pi, 0.0)           # behind the camera
    rng = np.random.default_rng(1)
    f1 = cam.frame(0.0, far_away, 0.0, 0.0, rng).image
    f2 = cam.frame(0.1, far_away, 0.0, 0.0, rng).image
    assert f1.std() == pytest.approx(np.hypot(CAM.netd, CAM.fpn), rel=0.05)
    assert (f1 - f2).std() == pytest.approx(np.sqrt(2) * CAM.netd, rel=0.05)   # FPN cancels


def test_target_outside_window_leaves_only_background():
    img = render_signature(SIG, 500.0, CAM, (-100.0, 64.0))
    assert np.all(img == 0.0)


def test_pixel_mapping_of_boresight_and_offset():
    x, y = angles_to_pixel(CAM, 0.0, 0.0, 0.0, 0.0)
    assert (x, y) == pytest.approx((64.0, 64.0))
    x, y = angles_to_pixel(CAM, 10 * CAM.ifov, -5 * CAM.ifov, 0.0, 0.0)
    assert (x, y) == pytest.approx((74.0, 69.0))


def test_camera_requires_rng():
    with pytest.raises(ValueError):
        ThermalCamera(CAM, SIG)