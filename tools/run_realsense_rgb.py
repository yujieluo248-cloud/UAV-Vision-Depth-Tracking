"""Preview RGB-only D435i color detection and save snapshots."""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.realsense_rgb import RealSenseRGBSource
from src.detection.candidate_selector import CandidateSelector
from src.detection.color_detector import ColorDetection, ColorDetector
from tools.run_color_image import DEFAULT_CONFIG, draw_detection, load_config


DEFAULT_OUTPUT_ROOT = Path("outputs/realsense_rgb")
DEFAULT_MAX_MISSED_FRAMES = 5


@dataclass
class TemporalSelectionState:
    """Minimal association history; intentionally not a full tracker."""

    previous_center: tuple[float, float] | None = None
    previous_area: float | None = None
    missed_frames: int = 0
    max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES

    def __post_init__(self) -> None:
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")

    def record_selection(
        self, center: tuple[float, float], area: float
    ) -> None:
        self.previous_center = center
        self.previous_area = area
        self.missed_frames = 0

    def record_miss(self) -> None:
        self.missed_frames += 1
        if self.missed_frames > self.max_missed_frames:
            self.previous_center = None
            self.previous_area = None
            self.missed_frames = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RGB-only HSV target detection from an Intel D435i."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Detector YAML config"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args(argv)


def process_frame(
    image_bgr: np.ndarray,
    detector: ColorDetector,
    *,
    selector: CandidateSelector | None = None,
    state: TemporalSelectionState | None = None,
) -> tuple[ColorDetection, np.ndarray]:
    active_selector = selector if selector is not None else CandidateSelector()
    active_state = state if state is not None else TemporalSelectionState()
    candidate_result = detector.detect_candidates(image_bgr)
    selection = active_selector.select(
        candidate_result.candidates,
        image_bgr.shape,
        previous_center=active_state.previous_center,
        previous_area=active_state.previous_area,
    )
    detection = detector.build_detection(
        candidate_result, selection.candidate, selection.mode
    )
    if selection.candidate is None:
        active_state.record_miss()
    else:
        active_state.record_selection(
            selection.candidate.center, selection.candidate.area
        )

    annotated = image_bgr.copy()
    for candidate_id, candidate in enumerate(candidate_result.candidates):
        x, y, width, height = candidate.bbox
        cv2.rectangle(
            annotated,
            (x, y),
            (x + width - 1, y + height - 1),
            (0, 200, 255),
            1,
        )
        cv2.putText(
            annotated,
            f"C{candidate_id} A={candidate.area:.0f}",
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

    annotated = draw_detection(annotated, detection)
    if detection.detected:
        assert detection.bbox is not None
        x, y, _width, height = detection.bbox
        cv2.putText(
            annotated,
            "SELECTED",
            (x, y + height + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    if not detection.detected:
        cv2.putText(
            annotated,
            "NO DETECTION",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        annotated,
        f"candidates: {len(candidate_result.candidates)}",
        (10, max(55, image_bgr.shape[0] - 40)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"mode: {detection.selection_mode}",
        (10, max(75, image_bgr.shape[0] - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return detection, annotated


def save_snapshot(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    annotated: np.ndarray,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    now: datetime | None = None,
) -> tuple[Path, Path, Path]:
    timestamp = now if now is not None else datetime.now()
    date_text = timestamp.strftime("%Y-%m-%d")
    stem = timestamp.strftime("%Y-%m-%d_%H%M%S")
    output_dir = output_root / date_text
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / f"{stem}_rgb.png",
        output_dir / f"{stem}_mask.png",
        output_dir / f"{stem}_annotated.png",
    )
    for path, image in zip(paths, (image_bgr, mask, annotated)):
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write output: {path}")
    return paths


def run_preview(
    source: RealSenseRGBSource,
    detector: ColorDetector,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    show_frame: Callable[[str, np.ndarray], None] | None = None,
    read_key: Callable[[int], int] | None = None,
    destroy_windows: Callable[[], None] | None = None,
) -> None:
    show = show_frame if show_frame is not None else cv2.imshow
    get_key = read_key if read_key is not None else cv2.waitKey
    destroy = destroy_windows if destroy_windows is not None else cv2.destroyAllWindows
    selector = CandidateSelector()
    state = TemporalSelectionState()
    try:
        source.start()
        while True:
            image_bgr = source.read()
            if image_bgr is None:
                continue
            detection, annotated = process_frame(
                image_bgr,
                detector,
                selector=selector,
                state=state,
            )
            show("RGB", image_bgr)
            show("Mask", detection.mask)
            show("Detection", annotated)

            key = get_key(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                paths = save_snapshot(
                    image_bgr,
                    detection.mask,
                    annotated,
                    output_root=output_root,
                )
                print(f"Saved RGB snapshot to: {paths[0].parent}")
    finally:
        try:
            source.stop()
        finally:
            destroy()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        detector = ColorDetector(
            lower_hsv=tuple(config["lower_hsv"]),
            upper_hsv=tuple(config["upper_hsv"]),
            min_area=config["min_area"],
            morph_kernel_size=config["morph_kernel_size"],
        )
        source = RealSenseRGBSource(args.width, args.height, args.fps)
        run_preview(source, detector)
    except (ValueError, RuntimeError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
