"""Preview aligned D435i RGB-D target depth measurement."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.realsense_rgbd import RGBDFrame, RealSenseRGBDSource
from src.depth.target_depth import TargetDepthExtractor, TargetDepthMeasurement
from src.detection.candidate_selector import CandidateSelector
from src.detection.color_detector import ColorDetection, ColorDetector
from tools.run_color_image import load_config
from tools.run_realsense_rgb import TemporalSelectionState, process_frame


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "color_orange_d435i.yaml"
DEFAULT_OUTPUT_ROOT = Path("outputs/realsense_rgbd")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure selected-target depth from aligned D435i RGB-D."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Detector YAML config"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mask-erode-kernel", type=int, default=3)
    parser.add_argument("--mask-erode-iterations", type=int, default=1)
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--min-valid-pixels", type=int, default=20)
    parser.add_argument("--min-valid-ratio", type=float, default=0.5)
    parser.add_argument("--max-depth-std-m", type=float, default=0.3)
    return parser.parse_args(argv)


def make_depth_visualization(
    depth_m: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Create a display-only colormap without modifying meter-valued depth."""
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    if min_depth_m < 0 or max_depth_m <= min_depth_m:
        raise ValueError("visualization depth range is invalid")

    valid = np.isfinite(depth_m) & (depth_m > 0)
    scaled = np.zeros(depth_m.shape, dtype=np.uint8)
    clipped = np.clip(depth_m[valid], min_depth_m, max_depth_m)
    scaled[valid] = np.rint(
        (clipped - min_depth_m) * 255.0 / (max_depth_m - min_depth_m)
    ).astype(np.uint8)
    visualization = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    visualization[~valid] = 0
    return visualization


def _annotate_depth(
    annotated: np.ndarray,
    measurement: TargetDepthMeasurement,
) -> np.ndarray:
    output = annotated.copy()
    if measurement.valid:
        assert measurement.depth_m is not None
        headline = f"depth: {measurement.depth_m:.3f} m"
        color = (0, 255, 0)
    else:
        headline = "DEPTH INVALID"
        color = (0, 0, 255)
    std_text = (
        "N/A"
        if measurement.depth_std_m is None
        else f"{measurement.depth_std_m:.3f} m"
    )
    lines = (
        headline,
        f"valid_ratio: {measurement.valid_ratio:.1%}",
        f"depth_std: {std_text}",
    )
    for index, text in enumerate(lines):
        cv2.putText(
            output,
            text,
            (10, 30 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def process_rgbd_frame(
    frame: RGBDFrame,
    detector: ColorDetector,
    depth_extractor: TargetDepthExtractor,
    *,
    selector: CandidateSelector | None = None,
    state: TemporalSelectionState | None = None,
) -> tuple[
    ColorDetection,
    TargetDepthMeasurement,
    np.ndarray,
    np.ndarray,
]:
    """Detect/select in RGB, then measure the aligned selected-mask depth."""
    detection, color_annotated = process_frame(
        frame.color_bgr,
        detector,
        selector=selector,
        state=state,
    )
    measurement = depth_extractor.measure(frame.depth_m, detection.target_mask)
    annotated = _annotate_depth(color_annotated, measurement)
    depth_visualization = make_depth_visualization(
        frame.depth_m,
        min_depth_m=depth_extractor.min_depth_m,
        max_depth_m=depth_extractor.max_depth_m,
    )
    return detection, measurement, annotated, depth_visualization


def save_rgbd_snapshot(
    frame: RGBDFrame,
    target_mask: np.ndarray,
    annotated: np.ndarray,
    depth_visualization: np.ndarray,
    measurement: TargetDepthMeasurement,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    now: datetime | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    timestamp = now if now is not None else datetime.now()
    date_text = timestamp.strftime("%Y-%m-%d")
    stem = timestamp.strftime("%Y-%m-%d_%H%M%S_%f")
    output_dir = output_root / date_text
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / f"{stem}_rgb.png",
        output_dir / f"{stem}_mask.png",
        output_dir / f"{stem}_annotated.png",
        output_dir / f"{stem}_depth_visualization.png",
        output_dir / f"{stem}_metadata.json",
    )
    for path, image in zip(
        paths[:4],
        (frame.color_bgr, target_mask, annotated, depth_visualization),
    ):
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write output: {path}")

    metadata = {
        "timestamp_ms": frame.timestamp_ms,
        "depth_m": measurement.depth_m,
        "valid": measurement.valid,
        "valid_ratio": measurement.valid_ratio,
        "valid_count": measurement.valid_count,
        "total_count": measurement.total_count,
        "depth_std_m": measurement.depth_std_m,
    }
    try:
        paths[4].write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"Could not write output: {paths[4]}") from exc
    return paths


def run_preview(
    source: RealSenseRGBDSource,
    detector: ColorDetector,
    depth_extractor: TargetDepthExtractor,
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
            frame = source.read()
            if frame is not None:
                detection, measurement, annotated, depth_visualization = (
                    process_rgbd_frame(
                        frame,
                        detector,
                        depth_extractor,
                        selector=selector,
                        state=state,
                    )
                )
                show("RGB", frame.color_bgr)
                show("Mask", detection.target_mask)
                show("Detection", annotated)
                show("Aligned Depth", depth_visualization)

            key = get_key(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and frame is not None:
                paths = save_rgbd_snapshot(
                    frame,
                    detection.target_mask,
                    annotated,
                    depth_visualization,
                    measurement,
                    output_root=output_root,
                )
                print(f"Saved RGB-D snapshot to: {paths[0].parent}")
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
        depth_extractor = TargetDepthExtractor(
            mask_erode_kernel=args.mask_erode_kernel,
            mask_erode_iterations=args.mask_erode_iterations,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            min_valid_pixels=args.min_valid_pixels,
            min_valid_ratio=args.min_valid_ratio,
            max_depth_std_m=args.max_depth_std_m,
        )
        source = RealSenseRGBDSource(args.width, args.height, args.fps)
        run_preview(source, detector, depth_extractor)
    except (ValueError, RuntimeError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
