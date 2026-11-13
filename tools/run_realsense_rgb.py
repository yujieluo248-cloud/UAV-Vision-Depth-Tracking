"""Preview RGB-only D435i color detection and save snapshots."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.realsense_rgb import RealSenseRGBSource
from src.detection.color_detector import ColorDetection, ColorDetector
from tools.run_color_image import DEFAULT_CONFIG, draw_detection, load_config


DEFAULT_OUTPUT_ROOT = Path("outputs/realsense_rgb")


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
    image_bgr: np.ndarray, detector: ColorDetector
) -> tuple[ColorDetection, np.ndarray]:
    detection = detector.detect(image_bgr)
    annotated = draw_detection(image_bgr, detection)
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
    try:
        source.start()
        while True:
            image_bgr = source.read()
            if image_bgr is None:
                continue
            detection, annotated = process_frame(image_bgr, detector)
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
