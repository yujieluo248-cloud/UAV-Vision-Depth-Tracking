# UAV Vision Depth Tracking

D435i-based UAV target visual detection and relative-state estimation.

## Current phase

Phase 1 - color-based RGB target segmentation baseline.

The current implementation processes BGR images with HSV color thresholding,
morphology, and contour filtering. HSV is used for candidate generation, not
final target identity. Static-image tools and an optional D435i RGB-only
preview are available. Depth, full tracking, and ROS are not implemented.

## Multi-candidate selection V2

```text
ColorDetector
→ multiple color candidates
→ CandidateSelector
→ selected target mask
→ future Depth / Tracking
```

`ColorDetector.detect_candidates()` exposes every connected component that
passes the baseline minimum-area filter. Each candidate contains its contour,
bounding box, center, area, aspect ratio, and fill ratio.

`CandidateSelector` applies baseline bbox-size, fill-ratio, and aspect-ratio
gates. With no history, acquisition uses the largest remaining valid candidate
as a fallback. With a previous center and area, temporal association prefers a
nearby candidate with a reasonable area change instead of switching directly
to a larger orange region.

The masks have different meanings:

- `color_mask`: the full cleaned HSV segmentation, including all possible
  orange regions
- `target_mask` (legacy alias: `mask`): only the contour selected as the current
  target

The temporal association and its gate/score defaults are only a baseline, not
a complete tracker or final optimized parameters. First acquisition remains
ambiguous when multiple genuinely similar orange objects are visible: the
largest-valid fallback cannot establish target identity. Future work may add
tracking prediction, Depth, and 3D-size constraints to improve association.

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

The Detection window shows all candidates with thin boxes and highlights the
selected target. The preview retains only the previous selected center and
area; after a small number of consecutive misses it clears that history and
returns to acquisition. This is minimal temporal continuity, not a tracker.

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
