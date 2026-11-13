import csv
import importlib
from datetime import datetime
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest


def load_batch_module() -> ModuleType:
    return importlib.import_module("tools.run_color_batch")


def write_config(path: Path) -> None:
    path.write_text(
        "lower_hsv: [5, 100, 100]\n"
        "upper_hsv: [25, 255, 255]\n"
        "min_area: 500\n"
        "morph_kernel_size: 5\n",
        encoding="utf-8",
    )


def write_image(path: Path, *, orange_target: bool) -> None:
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    if orange_target:
        orange_bgr = cv2.cvtColor(
            np.uint8([[[15, 220, 220]]]), cv2.COLOR_HSV2BGR
        )[0, 0].tolist()
        cv2.rectangle(image, (40, 30), (140, 100), orange_bgr, thickness=-1)
    assert cv2.imwrite(str(path), image)


def test_get_dated_output_dir_uses_local_date_and_creates_directory(
    tmp_path: Path,
) -> None:
    batch = load_batch_module()

    output_dir = batch.get_dated_output_dir(
        tmp_path / "outputs", now=datetime(2026, 11, 13, 23, 59)
    )

    assert output_dir == tmp_path / "outputs" / "2026-11-13"
    assert output_dir.is_dir()


def test_list_image_files_sorts_supported_extensions_and_ignores_others(
    tmp_path: Path,
) -> None:
    batch = load_batch_module()
    (tmp_path / "z.jpeg").touch()
    (tmp_path / "B.PNG").touch()
    (tmp_path / "a.bmp").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "nested.jpg").mkdir()

    image_files = batch.list_image_files(tmp_path)

    assert [path.name for path in image_files] == ["a.bmp", "B.PNG", "z.jpeg"]


def test_batch_writes_two_output_pairs_no_detection_annotation_and_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = load_batch_module()
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    write_image(input_dir / "orange_001.jpg", orange_target=True)
    write_image(input_dir / "negative_001.png", orange_target=False)
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    summary = batch.run_batch(
        input_dir,
        config_path=config_path,
        output_root=tmp_path / "outputs" / "test",
        now=datetime(2026, 11, 13),
    )

    output_dir = tmp_path / "outputs" / "test" / "2026-11-13"
    assert summary.output_dir == output_dir
    assert summary.processed == 2
    assert summary.detected == 1
    assert summary.no_detection == 1
    assert summary.errors == 0
    assert summary.labeled_positive == 1
    assert summary.labeled_negative == 1
    assert summary.unlabeled == 0
    assert (summary.true_positive, summary.false_negative) == (1, 0)
    assert (summary.true_negative, summary.false_positive) == (1, 0)
    assert sorted(path.name for path in output_dir.glob("*_mask.png")) == [
        "negative_001_mask.png",
        "orange_001_mask.png",
    ]
    assert sorted(path.name for path in output_dir.glob("*_annotated.png")) == [
        "negative_001_annotated.png",
        "orange_001_annotated.png",
    ]

    black_annotated = cv2.imread(str(output_dir / "negative_001_annotated.png"))
    assert black_annotated is not None
    assert cv2.countNonZero(cv2.cvtColor(black_annotated, cv2.COLOR_BGR2GRAY)) > 0

    with (output_dir / "results.csv").open(
        "r", encoding="utf-8", newline=""
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert list(rows[0]) == [
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
    ]
    assert [row["filename"] for row in rows] == [
        "negative_001.png",
        "orange_001.jpg",
    ]
    assert rows[0] == {
        "filename": "negative_001.png",
        "expected": "negative",
        "detected": "false",
        "classification": "TN",
        "center_x": "",
        "center_y": "",
        "bbox_x": "",
        "bbox_y": "",
        "bbox_w": "",
        "bbox_h": "",
        "area": "",
    }
    assert rows[1]["expected"] == "positive"
    assert rows[1]["detected"] == "true"
    assert rows[1]["classification"] == "TP"
    assert rows[1]["center_x"] != ""
    assert rows[1]["bbox_w"] != ""
    assert float(rows[1]["area"]) > 0

    terminal_output = capsys.readouterr().out
    assert "[1/2] negative_001.png -> NO DETECTION" in terminal_output
    assert "[2/2] orange_001.jpg -> DETECTED" in terminal_output
    assert "Processed: 2" in terminal_output
    assert "Detected: 1" in terminal_output
    assert "No detection: 1" in terminal_output
    assert "Labeled positive: 1" in terminal_output
    assert "Labeled negative: 1" in terminal_output
    assert "Unlabeled: 0" in terminal_output
    assert "TP: 1" in terminal_output
    assert "FN: 0" in terminal_output
    assert "TN: 1" in terminal_output
    assert "FP: 0" in terminal_output
    assert "Recall: 1.0000" in terminal_output
    assert "False positive rate: 0.0000" in terminal_output
    assert f"Output directory: {output_dir}" in terminal_output


def test_empty_and_missing_input_directories_raise_clear_errors(
    tmp_path: Path,
) -> None:
    batch = load_batch_module()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No supported images"):
        batch.list_image_files(empty_dir)
    with pytest.raises(ValueError, match="Input directory does not exist"):
        batch.list_image_files(tmp_path / "missing")


def test_corrupt_image_does_not_stop_later_valid_image(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = load_batch_module()
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "broken.jpg").write_bytes(b"not an image")
    write_image(input_dir / "valid.png", orange_target=False)
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    summary = batch.run_batch(
        input_dir,
        config_path=config_path,
        output_root=tmp_path / "outputs",
        now=datetime(2026, 11, 13),
    )

    assert summary.processed == 1
    assert summary.no_detection == 1
    assert summary.errors == 1
    assert not (summary.output_dir / "broken_mask.png").exists()
    assert (summary.output_dir / "valid_mask.png").is_file()
    assert (summary.output_dir / "valid_annotated.png").is_file()
    with (summary.output_dir / "results.csv").open(
        "r", encoding="utf-8", newline=""
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [row["filename"] for row in rows] == ["valid.png"]
    assert "[1/2] broken.jpg -> ERROR: could not read image" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("filename", "detected", "expected", "classification"),
    [
        ("orange_001.jpg", True, "positive", "TP"),
        ("orange_002.png", False, "positive", "FN"),
        ("negative_001.jpeg", False, "negative", "TN"),
        ("negative_002.bmp", True, "negative", "FP"),
        ("other.jpg", True, "unknown", "UNLABELED"),
    ],
)
def test_filename_expectation_and_classification(
    filename: str,
    detected: bool,
    expected: str,
    classification: str,
) -> None:
    batch = load_batch_module()

    assert batch.expected_from_filename(filename) == expected
    assert batch.classify_detection(expected, detected) == classification


def test_rate_formatting_handles_zero_denominator() -> None:
    batch = load_batch_module()

    assert batch.format_rate(3, 4) == "0.7500"
    assert batch.format_rate(0, 0) == "N/A"


def test_batch_counts_all_classes_and_unlabeled(
    tmp_path: Path,
) -> None:
    batch = load_batch_module()
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    write_image(input_dir / "orange_tp.jpg", orange_target=True)
    write_image(input_dir / "orange_fn.jpg", orange_target=False)
    write_image(input_dir / "negative_tn.jpg", orange_target=False)
    write_image(input_dir / "negative_fp.jpg", orange_target=True)
    write_image(input_dir / "scene.jpg", orange_target=True)
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    summary = batch.run_batch(
        input_dir,
        config_path=config_path,
        output_root=tmp_path / "outputs",
        now=datetime(2026, 11, 13),
    )

    assert summary.processed == 5
    assert summary.labeled_positive == 2
    assert summary.labeled_negative == 2
    assert summary.unlabeled == 1
    assert summary.true_positive == 1
    assert summary.false_negative == 1
    assert summary.true_negative == 1
    assert summary.false_positive == 1


def test_unlabeled_only_batch_reports_na_rates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = load_batch_module()
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    write_image(input_dir / "scene.jpg", orange_target=False)
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    batch.run_batch(
        input_dir,
        config_path=config_path,
        output_root=tmp_path / "outputs",
        now=datetime(2026, 11, 13),
    )

    terminal_output = capsys.readouterr().out
    assert "Recall: N/A" in terminal_output
    assert "False positive rate: N/A" in terminal_output
