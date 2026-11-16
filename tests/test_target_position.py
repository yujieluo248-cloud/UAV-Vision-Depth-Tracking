import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.camera.realsense_rgbd import RGBDFrame
from src.depth.target_depth import TargetDepthMeasurement
from src.detection.color_detector import ColorDetection, ColorCandidate
from tools.run_realsense_rgbd import target_position, save_rgbd_snapshot


def fixture():
    intr = SimpleNamespace(width=640, height=480, fx=600., fy=600., ppx=320., ppy=240., model='none', coeffs=[0.]*5)
    image = np.zeros((480, 640, 3), np.uint8)
    frame = RGBDFrame(image, np.ones((480, 640), np.float32), 123., intr)
    candidate = ColorCandidate(np.array([[[1, 1]]]), (300, 220, 40, 40), (320.2, 240.7), 1000., 1., .8)
    detection = ColorDetection(True, candidate.bbox, (320, 241), 1000., np.zeros((480, 640), np.uint8), candidates=(candidate,))
    measurement = TargetDepthMeasurement(True, 2., 1., 100, 100, .01)
    return frame, detection, measurement


def test_selected_float_center_and_metadata(tmp_path):
    frame, detection, measurement = fixture()
    def deproject(intr, pixel, depth):
        assert pixel == [320.2, 240.7]
        assert depth == 2.
        return [.001, .002, depth]
    position = target_position(frame, detection, measurement, deproject_fn=deproject)
    assert position.valid
    paths = save_rgbd_snapshot(frame, detection.mask, frame.color_bgr, frame.color_bgr,
                               measurement, output_root=tmp_path, position=position)
    data = json.loads(paths[-1].read_text())
    assert data['target_pixel'] == dict(u=320.2, v=240.7)
    assert data['camera_position_m'] == dict(x=.001, y=.002, z=2.)
    assert data['intrinsics']['fx'] == 600.
    assert data['intrinsics']['coeffs'] == [0.]*5


@pytest.mark.parametrize('no_target', [True, False])
def test_no_target_or_invalid_depth_returns_current_invalid(no_target):
    frame, detection, measurement = fixture()
    if no_target:
        detection.detected = False
    else:
        measurement = TargetDepthMeasurement(False, None, .1, 10, 100, .5)
    assert not target_position(frame, detection, measurement).valid
