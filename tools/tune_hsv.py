"""Interactively tune HSV thresholds for a static image."""

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.color_detector import ColorDetector, HSVRange
from tools.run_color_image import draw_detection, load_config


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "color_orange.yaml"
ORIGINAL_WINDOW = "Original"
MASK_WINDOW = "Mask"
ANNOTATED_WINDOW = "Annotated"
TRACKBARS = (
    ("H min", 179),
    ("H max", 179),
    ("S min", 255),
    ("S max", 255),
    ("V min", 255),
    ("V max", 255),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively tune HSV thresholds on a static image."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Initial detector configuration (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--save-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Save destination when pressing s (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args(argv)


def trackbar_values_to_hsv(
    *,
    h_min: int,
    h_max: int,
    s_min: int,
    s_max: int,
    v_min: int,
    v_max: int,
) -> tuple[HSVRange, HSVRange]:
    """Convert six OpenCV trackbar positions into HSV bound tuples."""
    return (h_min, s_min, v_min), (h_max, s_max, v_max)


def save_config(
    config_path: Path,
    *,
    lower_hsv: HSVRange,
    upper_hsv: HSVRange,
    base_config: dict[str, Any],
) -> None:
    """Save HSV bounds while retaining non-HSV detector parameters."""
    saved_config = {
        "lower_hsv": list(lower_hsv),
        "upper_hsv": list(upper_hsv),
        "min_area": base_config["min_area"],
        "morph_kernel_size": base_config["morph_kernel_size"],
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(saved_config, config_file, sort_keys=False)


def scale_for_display(
    image: np.ndarray,
    *,
    max_width: int = 1280,
    max_height: int = 720,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Return a display copy that fits the requested dimensions."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image.copy()

    display_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return cv2.resize(image, display_size, interpolation=interpolation)


def _noop_trackbar_callback(_: int) -> None:
    """OpenCV requires a callback; the event loop reads all positions."""


def _read_trackbar_hsv() -> tuple[HSVRange, HSVRange]:
    values = {
        name.replace(" ", "_").lower(): cv2.getTrackbarPos(
            name, ANNOTATED_WINDOW
        )
        for name, _ in TRACKBARS
    }
    return trackbar_values_to_hsv(**values)


def _create_windows(config: dict[str, Any]) -> None:
    for window_name in (ORIGINAL_WINDOW, MASK_WINDOW, ANNOTATED_WINDOW):
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    lower_hsv = config["lower_hsv"]
    upper_hsv = config["upper_hsv"]
    initial_values = (
        int(lower_hsv[0]),
        int(upper_hsv[0]),
        int(lower_hsv[1]),
        int(upper_hsv[1]),
        int(lower_hsv[2]),
        int(upper_hsv[2]),
    )
    for (name, maximum), initial_value in zip(TRACKBARS, initial_values):
        cv2.createTrackbar(
            name,
            ANNOTATED_WINDOW,
            max(0, min(initial_value, maximum)),
            maximum,
            _noop_trackbar_callback,
        )


def _annotate_hsv(
    image_bgr: np.ndarray,
    detection: Any,
    lower_hsv: HSVRange,
    upper_hsv: HSVRange,
) -> np.ndarray:
    annotated = draw_detection(image_bgr, detection)
    cv2.putText(
        annotated,
        f"HSV: {lower_hsv} - {upper_hsv}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
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

    _create_windows(config)
    cv2.imshow(ORIGINAL_WINDOW, scale_for_display(image_bgr))
    current_hsv: tuple[HSVRange, HSVRange] | None = None

    try:
        while True:
            selected_hsv = _read_trackbar_hsv()
            if selected_hsv != current_hsv:
                lower_hsv, upper_hsv = selected_hsv
                detector.set_hsv_range(lower_hsv, upper_hsv)
                detection = detector.detect(image_bgr)
                annotated = _annotate_hsv(
                    image_bgr, detection, lower_hsv, upper_hsv
                )
                cv2.imshow(
                    MASK_WINDOW,
                    scale_for_display(
                        detection.mask, interpolation=cv2.INTER_NEAREST
                    ),
                )
                cv2.imshow(
                    ANNOTATED_WINDOW, scale_for_display(annotated)
                )
                current_hsv = selected_hsv

            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                lower_hsv, upper_hsv = _read_trackbar_hsv()
                save_config(
                    args.save_config,
                    lower_hsv=lower_hsv,
                    upper_hsv=upper_hsv,
                    base_config=config,
                )
                print(f"Saved HSV config to: {args.save_config}")
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
