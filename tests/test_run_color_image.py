import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cli_saves_mask_and_annotated_image(tmp_path: Path) -> None:
    image_path = tmp_path / "orange.png"
    output_dir = tmp_path / "results"
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    orange_bgr = cv2.cvtColor(
        np.uint8([[[15, 220, 220]]]), cv2.COLOR_HSV2BGR
    )[0, 0].tolist()
    cv2.rectangle(image, (50, 40), (170, 130), orange_bgr, thickness=-1)
    assert cv2.imwrite(str(image_path), image)

    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_color_image.py",
            str(image_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Target detected" in completed.stdout
    assert "bbox:" in completed.stdout
    assert "center:" in completed.stdout
    assert "area:" in completed.stdout
    mask = cv2.imread(str(output_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    annotated = cv2.imread(str(output_dir / "annotated.png"))
    assert mask is not None and cv2.countNonZero(mask) > 0
    assert annotated is not None and annotated.shape == image.shape
