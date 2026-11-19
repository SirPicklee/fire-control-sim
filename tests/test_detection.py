import numpy as np
import pytest

from fcsim.detection import DetectorParams, detect, detect_px
from fcsim.frames import aer_to_enu
from fcsim.messages import EOMeasurement
from fcsim.thermal_scene import CameraParams, DroneSignature, ThermalCamera, render_signature

CAM = CameraParams()
SIG = DroneSignature()


def make_camera(seed=0, **kw):
    return ThermalCamera(CameraParams(**kw), SIG, rng=np.random.default_rng(seed))


def test_centroid_finds_subpixel_position_on_clean_image():
    quiet = CameraParams(netd=1e-4, fpn=1e-4)
    for cx in (60.2, 64.5, 70.85):
        img = quiet.sky_temp + render_signature(SIG, 600.0, quiet, (cx, 64.0)) \
            + np.random.default_rng(0).normal(0, 1e-4, (128, 128))
        d = detect_px(img)
        assert d.x == pytest.approx(cx, abs=0.05)


def test_detect_returns_icd_message_with_correct_azimuth():
    cam = make_camera()
    az_true, el_true = 0.01, np.deg2rad(3.0)
    pos = aer_to_enu(500.0, az_true, el_true)
    fr = cam.frame(0.0, pos, az_true + 7 * CAM.ifov, el_true, np.random.default_rng(1))
    m = detect(fr)
    assert isinstance(m, EOMeasurement)
    assert m.az == pytest.approx(az_true, abs=0.3 * CAM.ifov)


def test_empty_sky_gives_no_detection():
    cam = make_camera()
    behind = aer_to_enu(1000.0, np.pi, 0.0)
    fr = cam.frame(0.0, behind, 0.0, 0.0, np.random.default_rng(2))
    assert detect(fr) is None


def test_very_faint_target_is_not_detected():
    cam = ThermalCamera(CAM, SIG, rng=np.random.default_rng(0), gamma=1.0e-3)   # haze
    pos = aer_to_enu(3000.0, 0.0, 0.05)
    fr = cam.frame(0.0, pos, 0.0, 0.05, np.random.default_rng(3))
    assert detect(fr) is None


def test_gate_selects_blob_near_prediction():
    cam = make_camera()
    pos = aer_to_enu(400.0, 0.0, 0.0)
    fr = cam.frame(0.0, pos, 0.0, 0.0, np.random.default_rng(4))      # drone at (64, 64)
    img = fr.image + 2.0 * render_signature(SIG, 400.0, CAM, (25.0, 100.0))  # brighter decoy
    assert detect_px(img).x == pytest.approx(25.0, abs=1.0)              # strongest wins
    d = detect_px(img, predicted_px=(63.0, 65.0))
    assert d.x == pytest.approx(64.0, abs=0.5)                           # gate keeps the track


def test_gate_rejects_everything_far_from_prediction():
    cam = make_camera()
    fr = cam.frame(0.0, aer_to_enu(400.0, 0.0, 0.0), 0.0, 0.0, np.random.default_rng(5))
    assert detect_px(fr.image, predicted_px=(10.0, 10.0)) is None


def test_reported_azimuth_sigma_matches_actual_scatter():
    cam = make_camera()
    rng = np.random.default_rng(6)
    pos = aer_to_enu(1000.0, 0.0, 0.0)
    err, pred = [], []
    for _ in range(300):
        fr = cam.frame(0.0, pos, 0.4 * CAM.ifov, 0.0, rng)
        d = detect_px(fr.image, DetectorParams(floor_px=0.0, shape_frac=0.0))
        err.append(d.x - fr.truth_px[0])
        pred.append(d.sx)
    assert np.std(err) == pytest.approx(np.sqrt(np.mean(np.square(pred))), rel=0.3)



def test_shape_term_grows_with_target_size():
    cam = make_camera()
    rng = np.random.default_rng(7)
    near = detect_px(cam.frame(0.0, aer_to_enu(60.0, 0.0, 0.0), 0.0, 0.0, rng).image)
    far = detect_px(cam.frame(0.0, aer_to_enu(1500.0, 0.0, 0.0), 0.0, 0.0, rng).image)
    assert near.size_px > 10 * far.size_px
    assert near.sy > 10 * far.sy
    assert near.sy == pytest.approx(0.03 * near.size_px, rel=0.1)   # shape term dominates