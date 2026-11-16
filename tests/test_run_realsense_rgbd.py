import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.camera.realsense_rgbd import RGBDFrame
from src.depth.target_depth import TargetDepthExtractor, TargetDepthMeasurement
from src.detection.color_detector import ColorDetector
from tools import run_realsense_rgbd


class FakeSource:
    def __init__(self, frame: RGBDFrame, *, fail_read: bool = False) -> None:
        self.frame = frame
        self.fail_read = fail_read
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def read(self) -> RGBDFrame:
        if self.fail_read:
            raise RuntimeError("frame failure")
        return self.frame

    def stop(self) -> None:
        self.stopped = True


class MissingThenFailSource:
    def __init__(self) -> None:
        self.read_count = 0
        self.stopped = False

    def start(self) -> None:
        pass

    def read(self) -> None:
        self.read_count += 1
        if self.read_count == 1:
            return None
        raise AssertionError("keyboard was not polled after missing frame")

    def stop(self) -> None:
        self.stopped = True


def make_target_frame() -> RGBDFrame:
    color = np.zeros((100, 120, 3), dtype=np.uint8)
    orange = cv2.cvtColor(
        np.uint8([[[15, 220, 220]]]), cv2.COLOR_HSV2BGR
    )[0, 0].tolist()
    cv2.rectangle(color, (20, 20), (90, 80), orange, -1)
    depth = np.full((100, 120), 8.0, dtype=np.float32)
    depth[20:81, 20:91] = 2.0
    return RGBDFrame(color_bgr=color, depth_m=depth, timestamp_ms=456.0)


def permissive_extractor() -> TargetDepthExtractor:
    return TargetDepthExtractor(
        min_valid_pixels=1,
        min_valid_ratio=0.0,
        max_depth_std_m=100.0,
    )


def test_process_rgbd_frame_measures_selected_target_depth() -> None:
    frame = make_target_frame()

    detection, measurement, annotated, depth_visualization = (
        run_realsense_rgbd.process_rgbd_frame(
            frame,
            ColorDetector(min_area=100),
            permissive_extractor(),
        )
    )

    assert detection.detected
    assert measurement.valid
    assert measurement.depth_m == pytest.approx(2.0)
    assert measurement.valid_ratio == pytest.approx(1.0)
    assert annotated.shape == frame.color_bgr.shape
    assert depth_visualization.shape == frame.color_bgr.shape


def test_depth_visualization_does_not_modify_meter_depth() -> None:
    depth = np.array(
        [[0.0, 0.5, 2.0], [np.nan, np.inf, 10.0]], dtype=np.float32
    )
    original = depth.copy()

    visualization = run_realsense_rgbd.make_depth_visualization(
        depth, min_depth_m=0.2, max_depth_m=10.0
    )

    assert visualization.shape == (2, 3, 3)
    assert visualization.dtype == np.uint8
    assert np.array_equal(depth, original, equal_nan=True)
    assert np.array_equal(visualization[0, 0], np.zeros(3, dtype=np.uint8))
    assert np.array_equal(visualization[1, 0], np.zeros(3, dtype=np.uint8))


def test_save_snapshot_writes_four_images_and_json_metadata(tmp_path: Path) -> None:
    frame = make_target_frame()
    mask = np.full(frame.depth_m.shape, 255, dtype=np.uint8)
    annotated = frame.color_bgr.copy()
    depth_visualization = np.zeros_like(frame.color_bgr)
    measurement = TargetDepthMeasurement(
        valid=True,
        depth_m=2.0,
        valid_ratio=0.91,
        valid_count=523,
        total_count=575,
        depth_std_m=0.018,
    )

    paths = run_realsense_rgbd.save_rgbd_snapshot(
        frame,
        mask,
        annotated,
        depth_visualization,
        measurement,
        output_root=tmp_path,
        now=datetime(2026, 11, 16, 14, 5, 9),
    )

    assert [path.name for path in paths] == [
        "2026-11-16_140509_000000_rgb.png",
        "2026-11-16_140509_000000_mask.png",
        "2026-11-16_140509_000000_annotated.png",
        "2026-11-16_140509_000000_depth_visualization.png",
        "2026-11-16_140509_000000_metadata.json",
    ]
    assert all(path.is_file() for path in paths)
    metadata = json.loads(paths[-1].read_text(encoding="utf-8"))
    assert metadata == {
        "timestamp_ms": 456.0,
        "depth_m": 2.0,
        "valid": True,
        "valid_ratio": 0.91,
        "valid_count": 523,
        "total_count": 575,
        "depth_std_m": 0.018,
    }


def test_two_snapshots_in_one_second_use_distinct_names(tmp_path: Path) -> None:
    frame = make_target_frame()
    mask = np.full(frame.depth_m.shape, 255, dtype=np.uint8)
    measurement = TargetDepthMeasurement(
        valid=False,
        depth_m=None,
        valid_ratio=0.0,
        valid_count=0,
        total_count=0,
        depth_std_m=None,
    )

    first = run_realsense_rgbd.save_rgbd_snapshot(
        frame,
        mask,
        frame.color_bgr,
        frame.color_bgr,
        measurement,
        output_root=tmp_path,
        now=datetime(2026, 11, 16, 14, 5, 9, 100),
    )
    second = run_realsense_rgbd.save_rgbd_snapshot(
        frame,
        mask,
        frame.color_bgr,
        frame.color_bgr,
        measurement,
        output_root=tmp_path,
        now=datetime(2026, 11, 16, 14, 5, 9, 200),
    )

    assert first[0].name != second[0].name
    assert len(list((tmp_path / "2026-11-16").iterdir())) == 10


def test_preview_shows_four_windows_and_stops_normally() -> None:
    source = FakeSource(make_target_frame())
    shown = []
    destroyed = []

    run_realsense_rgbd.run_preview(
        source,
        ColorDetector(min_area=100),
        permissive_extractor(),
        show_frame=lambda name, image: shown.append((name, image.shape)),
        read_key=lambda _delay: ord("q"),
        destroy_windows=lambda: destroyed.append(True),
    )

    assert source.started
    assert source.stopped
    assert [name for name, _shape in shown] == [
        "RGB",
        "Mask",
        "Detection",
        "Aligned Depth",
    ]
    assert destroyed == [True]


def test_preview_stops_source_when_read_raises() -> None:
    source = FakeSource(make_target_frame(), fail_read=True)

    with pytest.raises(RuntimeError, match="frame failure"):
        run_realsense_rgbd.run_preview(
            source,
            ColorDetector(min_area=100),
            permissive_extractor(),
            show_frame=lambda _name, _image: None,
            read_key=lambda _delay: ord("q"),
            destroy_windows=lambda: None,
        )

    assert source.stopped


def test_preview_polls_quit_key_when_aligned_frame_is_missing() -> None:
    source = MissingThenFailSource()
    key_polls = []

    run_realsense_rgbd.run_preview(
        source,  # type: ignore[arg-type]
        ColorDetector(min_area=100),
        permissive_extractor(),
        show_frame=lambda _name, _image: None,
        read_key=lambda delay: key_polls.append(delay) or ord("q"),
        destroy_windows=lambda: None,
    )

    assert key_polls == [1]
    assert source.read_count == 1
    assert source.stopped
