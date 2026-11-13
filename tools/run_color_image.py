"""Run the color detector on a static image and save visual outputs."""

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.color_detector import ColorDetection, ColorDetector


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "color_orange.yaml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect an orange target in a static RGB image."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Detector YAML configuration (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/color_test"),
        help="Directory for mask.png and annotated.png",
    )
    return parser.parse_args(argv)


def load_config(config_path: Path) -> dict[str, Any]:
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as exc:
        raise ValueError(f"Could not read config: {config_path}") from exc

    required_keys = {
        "lower_hsv",
        "upper_hsv",
        "min_area",
        "morph_kernel_size",
    }
    if not isinstance(config, dict) or not required_keys.issubset(config):
        raise ValueError(
            "Config must define lower_hsv, upper_hsv, min_area, "
            "and morph_kernel_size"
        )
    return config


def draw_detection(image_bgr: Any, detection: ColorDetection) -> Any:
    annotated = image_bgr.copy()
    if not detection.detected:
        return annotated

    assert detection.bbox is not None
    assert detection.center is not None
    x, y, width, height = detection.bbox
    cv2.rectangle(
        annotated,
        (x, y),
        (x + width - 1, y + height - 1),
        (0, 255, 0),
        2,
    )
    cv2.circle(annotated, detection.center, 4, (0, 0, 255), thickness=-1)
    cv2.putText(
        annotated,
        f"area: {detection.area:.1f}",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    image_bgr = cv2.imread(str(args.image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {args.image_path}")

    detector = ColorDetector(
        lower_hsv=tuple(config["lower_hsv"]),
        upper_hsv=tuple(config["upper_hsv"]),
        min_area=config["min_area"],
        morph_kernel_size=config["morph_kernel_size"],
    )
    detection = detector.detect(image_bgr)
    annotated = draw_detection(image_bgr, detection)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.output_dir / "mask.png"
    annotated_path = args.output_dir / "annotated.png"
    if not cv2.imwrite(str(mask_path), detection.mask):
        raise SystemExit(f"Could not write output: {mask_path}")
    if not cv2.imwrite(str(annotated_path), annotated):
        raise SystemExit(f"Could not write output: {annotated_path}")

    if detection.detected:
        print("Target detected")
        print(f"bbox: {detection.bbox}")
        print(f"center: {detection.center}")
        print(f"area: {detection.area:.1f}")
    else:
        print("Target not detected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
