from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.detection.color_detector import ColorDetector
from src.detection.candidate_selector import CandidateSelector
from tools import run_realsense_rgb


class FakeSource:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def read(self) -> np.ndarray:
        return self.frame.copy()

    def stop(self) -> None:
        self.stopped = True


def test_process_frame_returns_detection_mask_and_annotation() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    orange = cv2.cvtColor(
        np.uint8([[[15, 220, 220]]]), cv2.COLOR_HSV2BGR
    )[0, 0].tolist()
    cv2.rectangle(frame, (20, 20), (90, 80), orange, -1)

    detection, annotated = run_realsense_rgb.process_frame(
        frame, ColorDetector(min_area=100)
    )

    assert detection.detected
    assert detection.mask.shape == frame.shape[:2]
    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)


def test_process_frame_uses_history_to_keep_smaller_near_target() -> None:
    detector = ColorDetector(min_area=100)
    selector = CandidateSelector(max_center_jump_ratio=0.25)
    state = run_realsense_rgb.TemporalSelectionState(max_missed_frames=2)
    orange = cv2.cvtColor(
        np.uint8([[[15, 220, 220]]]), cv2.COLOR_HSV2BGR
    )[0, 0].tolist()
    first = np.zeros((200, 320, 3), dtype=np.uint8)
    cv2.rectangle(first, (30, 60), (70, 100), orange, -1)

    first_detection, _ = run_realsense_rgb.process_frame(
        first, detector, selector=selector, state=state
    )

    assert first_detection.selection_mode == "ACQUIRE"
    assert state.previous_center == (50.0, 80.0)

    second = np.zeros_like(first)
    cv2.rectangle(second, (35, 62), (75, 102), orange, -1)
    cv2.rectangle(second, (180, 30), (300, 170), orange, -1)

    second_detection, _ = run_realsense_rgb.process_frame(
        second, detector, selector=selector, state=state
    )

    assert second_detection.selection_mode == "TRACK_ASSOCIATION"
    assert second_detection.bbox == (35, 62, 41, 41)
    assert second_detection.color_mask[50, 200] == 255
    assert second_detection.target_mask[50, 200] == 0
    assert second_detection.target_mask[80, 50] == 255


def test_temporal_state_resets_after_miss_limit() -> None:
    state = run_realsense_rgb.TemporalSelectionState(max_missed_frames=2)
    state.previous_center = (50.0, 60.0)
    state.previous_area = 1000.0

    state.record_miss()
    state.record_miss()

    assert state.previous_center == (50.0, 60.0)
    assert state.missed_frames == 2

    state.record_miss()

    assert state.previous_center is None
    assert state.previous_area is None
    assert state.missed_frames == 0


def test_blank_frames_reset_history_then_next_target_reacquires() -> None:
    detector = ColorDetector(min_area=100)
    selector = CandidateSelector(max_center_jump_ratio=0.1)
    state = run_realsense_rgb.TemporalSelectionState(max_missed_frames=1)
    orange = cv2.cvtColor(
        np.uint8([[[15, 220, 220]]]), cv2.COLOR_HSV2BGR
    )[0, 0].tolist()
    first = np.zeros((200, 320, 3), dtype=np.uint8)
    cv2.rectangle(first, (20, 30), (60, 70), orange, -1)
    blank = np.zeros_like(first)
    far_reappearance = np.zeros_like(first)
    cv2.rectangle(far_reappearance, (240, 130), (280, 170), orange, -1)

    run_realsense_rgb.process_frame(
        first, detector, selector=selector, state=state
    )
    run_realsense_rgb.process_frame(
        blank, detector, selector=selector, state=state
    )
    run_realsense_rgb.process_frame(
        blank, detector, selector=selector, state=state
    )
    detection, _ = run_realsense_rgb.process_frame(
        far_reappearance, detector, selector=selector, state=state
    )

    assert detection.detected
    assert detection.selection_mode == "ACQUIRE"
    assert detection.bbox == (240, 130, 41, 41)


def test_save_snapshot_uses_one_timestamp_and_date_directory(tmp_path: Path) -> None:
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    mask = np.zeros((10, 12), dtype=np.uint8)
    annotated = rgb.copy()

    paths = run_realsense_rgb.save_snapshot(
        rgb,
        mask,
        annotated,
        output_root=tmp_path,
        now=datetime(2026, 11, 13, 14, 5, 9),
    )

    assert [path.name for path in paths] == [
        "2026-11-13_140509_rgb.png",
        "2026-11-13_140509_mask.png",
        "2026-11-13_140509_annotated.png",
    ]
    assert all(path.parent == tmp_path / "2026-11-13" for path in paths)
    assert all(path.is_file() for path in paths)


def test_preview_loop_always_stops_source_without_gui() -> None:
    source = FakeSource(np.zeros((20, 30, 3), dtype=np.uint8))
    shown = []
    destroyed = []

    run_realsense_rgb.run_preview(
        source,
        ColorDetector(min_area=1),
        show_frame=lambda name, image: shown.append((name, image.shape)),
        read_key=lambda _delay: ord("q"),
        destroy_windows=lambda: destroyed.append(True),
    )

    assert source.started
    assert source.stopped
    assert [name for name, _shape in shown] == ["RGB", "Mask", "Detection"]
    assert destroyed == [True]


def test_preview_loop_stops_source_when_frame_processing_raises() -> None:
    source = FakeSource(np.zeros((20, 30), dtype=np.uint8))

    try:
        run_realsense_rgb.run_preview(
            source,
            ColorDetector(),
            show_frame=lambda _name, _image: None,
            read_key=lambda _delay: ord("q"),
            destroy_windows=lambda: None,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid frame should fail detection")

    assert source.stopped


def test_preview_s_key_saves_rgb_mask_and_annotation(tmp_path: Path) -> None:
    source = FakeSource(np.zeros((20, 30, 3), dtype=np.uint8))
    keys = iter((ord("s"), ord("q")))

    run_realsense_rgb.run_preview(
        source,
        ColorDetector(min_area=1),
        output_root=tmp_path,
        show_frame=lambda _name, _image: None,
        read_key=lambda _delay: next(keys),
        destroy_windows=lambda: None,
    )

    saved = sorted(path.name for path in tmp_path.glob("*/*.png"))
    assert len(saved) == 3
    assert any(name.endswith("_rgb.png") for name in saved)
    assert any(name.endswith("_mask.png") for name in saved)
    assert any(name.endswith("_annotated.png") for name in saved)
