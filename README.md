# UAV Vision Depth Tracking

D435i-based UAV target visual detection and relative-state estimation.

## Current phase

Phase 1 - color-based RGB target segmentation baseline.

The current implementation processes BGR images with HSV color thresholding,
morphology, and contour filtering. It produces a target mask, bounding box,
center, and contour area. Static-image tools and an optional D435i RGB-only
preview are available. Depth, tracking, and ROS are not implemented.

## Usage

```bash
conda activate uav_vision
python tools/run_color_image.py data/orange_test.jpg
```

The default configuration is `configs/color_orange.yaml`. It can also be
provided explicitly, together with a custom output directory:

```bash
python tools/run_color_image.py data/orange_test.jpg \
  --config configs/color_orange.yaml \
  --output-dir outputs/color_test
```

The command saves:

- `outputs/color_test/mask.png` - final binary mask for the selected target
- `outputs/color_test/annotated.png` - input image with bounding box, center,
  and contour area

The HSV threshold is a baseline and must be calibrated for the actual
target/material/lighting.

## HSV tuning

```bash
python tools/tune_hsv.py data/orange_test.jpg
```

Controls:

- sliders: adjust HSV
- `s`: save config
- `q` / `ESC`: quit

Online test images can only verify the algorithm workflow. Final HSV values
must be calibrated with the real target color material, a real D435i, and the
actual deployment lighting.

## Batch image testing

```bash
python tools/run_color_batch.py data/test_images/
```

Each run writes masks, annotated images, and a CSV summary to the current local
date directory:

```text
outputs/test/YYYY-MM-DD/
```

For example:

```text
outputs/test/2026-11-13/
├── orange_01_mask.png
├── orange_01_annotated.png
├── orange_02_mask.png
├── orange_02_annotated.png
└── results.csv
```

This batch tool is only for offline evaluation of the current color-detection
baseline. Internet images cannot determine the final D435i HSV parameters;
calibrate them again using the real target material, a real D435i, and the
actual experiment environment.

Files named `orange_*` are evaluated as positive samples and files named
`negative_*` as negative samples. Other names are included as `UNLABELED`.
`results.csv` records TP, FN, TN, or FP for labeled images, and the terminal
summary reports recall and false-positive rate (`N/A` when the corresponding
labeled subset is empty).

This classification only evaluates whether the detector produced any candidate.
It does not measure bounding-box or mask quality. Review annotated outputs and
record PASS/PARTIAL/FAIL manually when assessing localization and segmentation.

## D435i RGB-only preview

Install the Intel RealSense SDK and `pyrealsense2` in the runtime environment,
then connect the D435i and run:

```bash
python tools/run_realsense_rgb.py
```

An explicit detector configuration and color-stream profile can be selected:

```bash
python tools/run_realsense_rgb.py \
  --config configs/color_orange.yaml \
  --width 640 --height 480 --fps 30
```

Controls:

- `s`: save the current RGB frame, mask, and annotated image under
  `outputs/realsense_rgb/YYYY-MM-DD/`
- `q` / `ESC`: quit

The default `640x480@30` profile is a starting point, not a guaranteed final
deployment profile. Final HSV values must be recalibrated using the real target
material, the real D435i color stream, and the actual deployment lighting.

This tool uses only the D435i color stream. Depth, RGB-depth alignment, 3D
position, tracking, ROS, and UAV control are not implemented.

## Image renaming

Preview a rename plan before changing files:

```bash
python tools/rename_images.py data/test_images \
  --prefix orange \
  --dry-run
```

Apply the reviewed plan:

```bash
python tools/rename_images.py data/test_images \
  --prefix orange
```

Rename a negative-image set:

```bash
python tools/rename_images.py data/test_images \
  --prefix negative
```
