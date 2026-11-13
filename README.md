# UAV Vision Depth Tracking

D435i-based UAV target visual detection and relative-state estimation.

## Current phase

Phase 1 - color-based RGB target segmentation baseline.

The current implementation processes a static BGR image with HSV color
thresholding, morphology, and contour filtering. It produces a target mask,
bounding box, center, and contour area. Depth, tracking, ROS, and camera access
are not implemented in this phase.

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
