import json
from datetime import datetime

import cv2
import numpy as np
import pytest
import yaml

from tools.calibrate_target_color import CalibrationSession, run_calibration


def bgr(values):
    return cv2.cvtColor(np.array([values], dtype=np.uint8), cv2.COLOR_HSV2BGR)


def test_roi_statistics_filtering_and_merge():
    session = CalibrationSession()
    stats = session.add_roi(bgr([[10, 255, 255], [20, 255, 255], [15, 0, 200], [15, 200, 20]]))
    assert stats['valid_pixel_count'] == 2
    assert stats['roi_pixel_count'] == 4
    assert stats['valid_ratio'] == .5
    assert stats['h_median'] == 15
    assert stats['h_p05'] == pytest.approx(10.5)
    assert stats['h_p95'] == pytest.approx(19.5)
    session.add_roi(bgr([[15, 255, 255]]))
    assert session.report()['sample_count'] == 2
    assert session.report()['valid_pixel_count'] == 3


def test_range_margin_clamp_and_yaml(tmp_path):
    session = CalibrationSession()
    session.add_roi(bgr([[15, 255, 255]]))
    path = tmp_path / 'tape.yaml'
    report_path = session.save(path, tmp_path, now=datetime(2026, 11, 16))
    assert yaml.safe_load(path.read_text()) == dict(lower_hsv=[13, 240, 240], upper_hsv=[17, 255, 255], min_area=100, morph_kernel_size=5)
    assert json.loads(report_path.read_text())['sample_count'] == 1
    with pytest.raises(FileExistsError):
        session.save(path, tmp_path)


def test_empty_and_invalid_roi_do_not_generate_config():
    session = CalibrationSession()
    assert session.add_roi(np.zeros((0, 0, 3), np.uint8)) is None
    assert session.add_roi(np.zeros((2, 2, 3), np.uint8))['valid_pixel_count'] == 0
    with pytest.raises(ValueError):
        session.report()


def test_hue_wraparound_refused():
    session = CalibrationSession()
    session.add_roi(bgr([[179, 255, 255], [0, 255, 255]]))
    with pytest.raises(ValueError, match='Hue wraparound'):
        session.report()


def test_sparse_high_hue_outliers_allow_robust_orange_range_and_save(tmp_path):
    session = CalibrationSession()
    hues = [6, 7, 8, 8, 9] * 100 + [178, 179]
    session.add_roi(bgr([[hue, 255, 255] for hue in hues]))

    report = session.report()

    assert report['h_median'] == 8
    assert report['h_p05'] == 6
    assert report['h_p95'] == 9
    assert report['generated_lower_hsv'] == [4, 240, 240]
    assert report['generated_upper_hsv'] == [11, 255, 255]
    path = tmp_path / 'tape.yaml'
    session.save(path, tmp_path / 'reports')
    assert yaml.safe_load(path.read_text())['upper_hsv'] == [11, 255, 255]


def test_substantial_samples_on_both_sides_of_seam_refuse_save(tmp_path):
    session = CalibrationSession()
    hues = [0, 1, 2, 3, 176, 177, 178, 179] * 50
    session.add_roi(bgr([[hue, 255, 255] for hue in hues]))
    path = tmp_path / 'tape.yaml'

    with pytest.raises(ValueError, match='Hue wraparound'):
        session.save(path, tmp_path / 'reports')

    assert not path.exists()


def test_fake_camera_stops_on_quit():
    class Camera:
        stopped = False
        def start(self): pass
        def read(self): return np.zeros((10, 10, 3), np.uint8)
        def stop(self): self.stopped = True
    camera = Camera()
    run_calibration(camera, setup_window=lambda callback: None, show=lambda image: None, read_key=lambda: ord('q'), cleanup=lambda: None)
    assert camera.stopped


def test_margin_crossing_hue_seam_is_refused():
    session = CalibrationSession()
    session.add_roi(bgr([[1, 255, 255]]))
    with pytest.raises(ValueError, match='Hue wraparound'):
        session.report()


def test_confirmation_does_not_duplicate_roi():
    class Camera:
        def start(self): pass
        def read(self): return bgr([[15, 255, 255]] * 4).repeat(4, axis=0)
        def stop(self): pass
    callback = []
    keys = iter([ord('c'), ord('c'), ord('q')])
    def show(image):
        if not callback[1:]:
            callback[0](cv2.EVENT_LBUTTONDOWN, 0, 0, 0, None)
            callback[0](cv2.EVENT_LBUTTONUP, 3, 3, 0, None)
            callback.append(True)
    # Count the actual session additions, while keeping statistics real.
    from unittest.mock import patch
    original = CalibrationSession.add_roi
    additions = []
    def add(self, image):
        additions.append(image.shape)
        return original(self, image)
    with patch.object(CalibrationSession, 'add_roi', add):
        run_calibration(Camera(), setup_window=lambda cb: callback.append(cb), show=show,
                        read_key=lambda: next(keys), cleanup=lambda: None)
    assert additions == [(3, 3, 3)]
