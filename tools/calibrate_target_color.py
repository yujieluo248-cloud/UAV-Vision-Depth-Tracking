"""Calibrate a specific target tape from user-confirmed D435i RGB ROIs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from uuid import uuid4

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.realsense_rgb import RealSenseRGBSource

DEFAULT_CONFIG = PROJECT_ROOT / 'configs/color_orange_tape_d435i.yaml'
DEFAULT_REPORT_ROOT = Path('outputs/color_calibration')
WINDOW = 'Target tape color calibration'


def roi_samples(image_bgr):
    """Discard obviously achromatic/dark pixels, independently of detection."""
    if image_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return hsv[(hsv[:, 1] >= 80) & (hsv[:, 2] >= 50)]


def statistics(samples, roi_pixel_count):
    result = dict(valid_pixel_count=len(samples), roi_pixel_count=roi_pixel_count,
                  valid_ratio=len(samples) / roi_pixel_count if roi_pixel_count else 0.0)
    for index, channel in enumerate('hsv'):
        for label, percentile in [('median', 50), ('p05', 5), ('p95', 95)]:
            result[f'{channel}_{label}'] = (
                float(np.percentile(samples[:, index], percentile)) if len(samples) else None
            )
    return result


class CalibrationSession:
    """Accumulate each confirmed ROI once; weight samples by pixel count."""

    def __init__(self):
        self.samples = []
        self.roi_pixel_count = 0

    def add_roi(self, image_bgr):
        pixels = roi_samples(image_bgr)
        if pixels is None:
            return None
        count = image_bgr.shape[0] * image_bgr.shape[1]
        result = statistics(pixels, count)
        if len(pixels):
            self.samples.append(pixels.copy())
            self.roi_pixel_count += count
        return result

    def report(self):
        if not self.samples:
            raise ValueError('No valid ROI samples; select tape and press c first.')
        pixels = np.concatenate(self.samples)
        report = statistics(pixels, self.roi_pixel_count)
        # Use the same robust interval as range generation: a few raw hue
        # outliers must not veto an otherwise compact orange distribution.
        # Substantial populations across the seam still span a wide p5–p95
        # interval and cannot safely fit the detector's single hue range.
        if report['h_p95'] - report['h_p05'] > 90:
            raise ValueError('Hue wraparound or ambiguous hue spread: cannot encode safely in one HSV interval.')
        lower, upper = [], []
        for channel, margin, limit in [('h', 2, 179), ('s', 15, 255), ('v', 15, 255)]:
            lo = int(np.floor(report[f'{channel}_p05'] - margin))
            hi = int(np.ceil(report[f'{channel}_p95'] + margin))
            if channel == 'h' and (lo < 0 or hi > 179):
                raise ValueError('Hue wraparound: margin crosses the hue seam; one HSV interval is insufficient.')
            lower.append(max(0, lo))
            upper.append(min(limit, hi))
        report.update(sample_count=len(self.samples), generated_lower_hsv=lower,
                      generated_upper_hsv=upper, hue_status='current sample does not cross hue wraparound',
                      prefilter=dict(min_s=80, min_v=50), margins=dict(h=2, s=15, v=15))
        return report

    def save(self, config_path, report_root=DEFAULT_REPORT_ROOT, *, now=None):
        report = self.report()
        config_path = Path(config_path)
        config = dict(lower_hsv=report['generated_lower_hsv'], upper_hsv=report['generated_upper_hsv'],
                      min_area=100, morph_kernel_size=5)
        timestamp = now or datetime.now()
        folder = Path(report_root) / timestamp.strftime('%Y-%m-%d')
        folder.mkdir(parents=True, exist_ok=True)
        report_path = folder / (timestamp.strftime('%H%M%S_%f') + '_' + uuid4().hex[:8] + '_color_calibration_report.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental replacement, including races.
        with config_path.open('x', encoding='utf-8') as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        return report_path


class ROISelection:
    def __init__(self):
        self.frame = None
        self.origin = None
        self.end = None
        self.roi = None
        self.confirmed = False

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.frame is not None:
            self.origin = (x, y)
            self.end = (x, y)
            self.frozen = self.frame.copy()
            self.roi = None
            self.confirmed = False
        elif self.origin is not None and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
            if event == cv2.EVENT_MOUSEMOVE and not flags & cv2.EVENT_FLAG_LBUTTON:
                return
            self.end = (x, y)
            if event == cv2.EVENT_LBUTTONUP:
                height, width = self.frozen.shape[:2]
                x0, x1 = sorted((max(0, min(width, self.origin[0])), max(0, min(width, x))))
                y0, y1 = sorted((max(0, min(height, self.origin[1])), max(0, min(height, y))))
                self.roi = self.frozen[y0:y1, x0:x1].copy()
                pixels = roi_samples(self.roi)
                print('ROI preview:', statistics(pixels, self.roi.shape[0] * self.roi.shape[1]) if pixels is not None else 'Empty ROI')


def run_calibration(source, *, save_config=DEFAULT_CONFIG, report_root=DEFAULT_REPORT_ROOT,
                    setup_window=None, show=None, read_key=None, cleanup=None):
    selection = ROISelection()
    session = CalibrationSession()
    if setup_window is None:
        def setup_window(callback):
            cv2.namedWindow(WINDOW)
            cv2.setMouseCallback(WINDOW, callback)
    show = show or (lambda image: cv2.imshow(WINDOW, image))
    read_key = read_key or (lambda: cv2.waitKey(1) & 255)
    cleanup = cleanup or cv2.destroyAllWindows
    try:
        source.start()
        setup_window(selection.mouse)
        while True:
            frame = source.read()
            if frame is not None:
                selection.frame = frame
                # Freeze selected image until confirmed, so ROI and pixels agree.
                display = selection.frozen.copy() if selection.origin is not None and not selection.confirmed else frame.copy()
                if selection.origin is not None:
                    cv2.rectangle(display, selection.origin, selection.end, (0, 255, 0), 2)
                cv2.putText(display, f'c: add | r: reset | s: save | q: quit | samples: {len(session.samples)}',
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                show(display)
            key = read_key()
            if key in (ord('q'), 27):
                break
            if key == ord('c') and selection.roi is not None and not selection.confirmed:
                print('Confirmed ROI:', session.add_roi(selection.roi))
                selection.confirmed = True
            elif key == ord('r'):
                session = CalibrationSession()
                selection = ROISelection()
                setup_window(selection.mouse)
                print('Calibration session cleared.')
            elif key == ord('s'):
                try:
                    report = session.report()
                    path = session.save(save_config, report_root)
                    print(json.dumps(report, indent=2))
                    print(f'Saved config: {save_config}; report: {path}')
                except (ValueError, OSError) as exc:
                    print(f'Cannot save: {exc}. Use --save-config with a new filename if it already exists.')
    finally:
        try:
            source.stop()
        finally:
            cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--save-config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--report-root', type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args(argv)
    try:
        run_calibration(RealSenseRGBSource(args.width, args.height, args.fps),
                        save_config=args.save_config, report_root=args.report_root)
    except (ValueError, RuntimeError, cv2.error, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
