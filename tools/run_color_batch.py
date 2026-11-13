"""Batch-run the color detector over a directory of static images."""

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.color_detector import ColorDetection, ColorDetector
from tools.run_color_image import draw_detection, load_config


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "color_orange.yaml"
DEFAULT_OUTPUT_ROOT = Path("outputs/test")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CSV_FIELDS = (
    "filename",
    "expected",
    "detected",
    "classification",
    "center_x",
    "center_y",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "area",
)


@dataclass(frozen=True)
class BatchSummary:
    output_dir: Path
    processed: int
    detected: int
    no_detection: int
    errors: int
    labeled_positive: int
    labeled_negative: int
    unlabeled: int
    true_positive: int
    false_negative: int
    true_negative: int
    false_positive: int


def expected_from_filename(filename: str) -> str:
    """Infer the expected class from the dataset filename convention."""
    name = Path(filename).name.casefold()
    if name.startswith("orange_"):
        return "positive"
    if name.startswith("negative_"):
        return "negative"
    return "unknown"


def classify_detection(expected: str, detected: bool) -> str:
    """Return a confusion-matrix class for one detection result."""
    if expected == "positive":
        return "TP" if detected else "FN"
    if expected == "negative":
        return "FP" if detected else "TN"
    return "UNLABELED"


def format_rate(numerator: int, denominator: int) -> str:
    """Format an evaluation rate, including empty labeled subsets."""
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator:.4f}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the HSV color detector over a directory of images."
    )
    parser.add_argument("input_dir", type=Path, help="Directory of input images")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Detector YAML configuration (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args(argv)


def get_dated_output_dir(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    now: datetime | None = None,
) -> Path:
    """Create and return the local-date output directory."""
    current_time = now if now is not None else datetime.now()
    output_dir = output_root / current_time.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def list_image_files(input_dir: Path) -> list[Path]:
    """Return supported top-level image files in stable filename order."""
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    image_files = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )
    if not image_files:
        raise ValueError(f"No supported images found in: {input_dir}")
    return image_files


def _annotate_result(
    image_bgr: np.ndarray, detection: ColorDetection
) -> np.ndarray:
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
    return annotated


def _result_row(filename: str, detection: ColorDetection) -> dict[str, object]:
    expected = expected_from_filename(filename)
    row: dict[str, object] = {
        "filename": filename,
        "expected": expected,
        "detected": str(detection.detected).lower(),
        "classification": classify_detection(expected, detection.detected),
        "center_x": "",
        "center_y": "",
        "bbox_x": "",
        "bbox_y": "",
        "bbox_w": "",
        "bbox_h": "",
        "area": "",
    }
    if detection.detected:
        assert detection.center is not None
        assert detection.bbox is not None
        center_x, center_y = detection.center
        bbox_x, bbox_y, bbox_w, bbox_h = detection.bbox
        row.update(
            {
                "center_x": center_x,
                "center_y": center_y,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "area": detection.area,
            }
        )
    return row


def run_batch(
    input_dir: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    now: datetime | None = None,
) -> BatchSummary:
    """Process supported images, save outputs, and write a CSV summary."""
    image_files = list_image_files(input_dir)
    config = load_config(config_path)
    detector = ColorDetector(
        lower_hsv=tuple(config["lower_hsv"]),
        upper_hsv=tuple(config["upper_hsv"]),
        min_area=config["min_area"],
        morph_kernel_size=config["morph_kernel_size"],
    )
    output_dir = get_dated_output_dir(output_root, now=now)

    processed = 0
    detected_count = 0
    error_count = 0
    counts = {
        "positive": 0,
        "negative": 0,
        "unknown": 0,
        "TP": 0,
        "FN": 0,
        "TN": 0,
        "FP": 0,
    }
    total = len(image_files)

    with (output_dir / "results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for index, image_path in enumerate(image_files, start=1):
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                error_count += 1
                print(
                    f"[{index}/{total}] {image_path.name} "
                    "-> ERROR: could not read image"
                )
                continue

            detection = detector.detect(image_bgr)
            annotated = _annotate_result(image_bgr, detection)
            mask_path = output_dir / f"{image_path.stem}_mask.png"
            annotated_path = output_dir / f"{image_path.stem}_annotated.png"
            if not cv2.imwrite(str(mask_path), detection.mask):
                raise RuntimeError(f"Could not write output: {mask_path}")
            if not cv2.imwrite(str(annotated_path), annotated):
                raise RuntimeError(f"Could not write output: {annotated_path}")

            row = _result_row(image_path.name, detection)
            writer.writerow(row)
            counts[str(row["expected"])] += 1
            if row["classification"] != "UNLABELED":
                counts[str(row["classification"])] += 1
            processed += 1
            if detection.detected:
                detected_count += 1
                status = "DETECTED"
            else:
                status = "NO DETECTION"
            print(f"[{index}/{total}] {image_path.name} -> {status}")

    no_detection_count = processed - detected_count
    print(f"Processed: {processed}")
    print(f"Detected: {detected_count}")
    print(f"No detection: {no_detection_count}")
    print(f"Labeled positive: {counts['positive']}")
    print(f"Labeled negative: {counts['negative']}")
    print(f"Unlabeled: {counts['unknown']}")
    print(f"TP: {counts['TP']}")
    print(f"FN: {counts['FN']}")
    print(f"TN: {counts['TN']}")
    print(f"FP: {counts['FP']}")
    print(
        "Recall: "
        f"{format_rate(counts['TP'], counts['TP'] + counts['FN'])}"
    )
    print(
        "False positive rate: "
        f"{format_rate(counts['FP'], counts['FP'] + counts['TN'])}"
    )
    print(f"Output directory: {output_dir}")

    return BatchSummary(
        output_dir=output_dir,
        processed=processed,
        detected=detected_count,
        no_detection=no_detection_count,
        errors=error_count,
        labeled_positive=counts["positive"],
        labeled_negative=counts["negative"],
        unlabeled=counts["unknown"],
        true_positive=counts["TP"],
        false_negative=counts["FN"],
        true_negative=counts["TN"],
        false_positive=counts["FP"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_batch(args.input_dir, config_path=args.config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
