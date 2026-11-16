# UAV Vision Depth Tracking

## Camera-frame target position

The RGBD preview now deprojects the selected candidate's floating-point center
using its target-mask median depth. This produces a representative target point,
not a point-cloud centroid. The intrinsics come directly from each aligned
depth frame's `profile.as_video_stream_profile().get_intrinsics()`; color-frame
intrinsics are also read for diagnostics. The first valid frame prints the
aligned intrinsics once, with a warning if color and aligned-depth profiles
disagree. No stored or hard-coded camera calibration is used.

Production deprojection calls RealSense `rs2_deproject_pixel_to_point`, retaining
the SDK distortion model and coefficients. Coordinates are meters in the camera
optical frame: +X right, +Y down, +Z forward. Z must agree with input depth.
Invalid target, depth, pixel, intrinsics, or SDK output produces `3D INVALID`.
The Detection window shows current X/Y/Z alongside depth quality. Snapshots add
`position_valid`, `target_pixel`, `camera_position_m`, and a frame intrinsics
snapshot (including model and coefficients); invalid positions are JSON null.

```bash
python tools/run_realsense_rgbd.py \
  --config configs/color_orange_tape_d435i.yaml \
  --width 640 --height 480 --fps 30
```

Hardware acceptance is manual: center gives X/Y near zero; right/left changes
X positive/negative; down/up changes Y positive/negative; farther/nearer changes
Z larger/smaller. No body/world transform, extrinsics, velocity estimation,
ROS, PX4, flight control, YOLO, or complete tracking is implemented.

D435i-based UAV target visual detection and relative-state estimation.

## Current phase

Phase 2 - color-selected target depth measurement baseline.

The current implementation processes BGR images with HSV color thresholding,
morphology, and contour filtering. HSV is used for candidate generation, not
final target identity. Static-image tools, a D435i RGB-only preview, and an
SDK-aligned RGB-D camera optical position preview are available. Full tracking
and ROS are not implemented.

## Multi-candidate selection V2

```text
ColorDetector
→ multiple color candidates
→ CandidateSelector
→ selected target mask
→ aligned target depth
→ camera optical position
→ future Tracking
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

This RGB-only tool intentionally uses only the D435i color stream. Use the RGB-D
tool below when aligned target depth is required.

## D435i aligned RGB-D target depth

The RGB-D source enables color and depth streams, then uses the RealSense SDK:

```text
rs.align(rs.stream.color)
```

Depth is aligned to the color pixel grid before the selected target mask is
applied. Raw Z16 values are converted with the device depth scale, and all
calculation depth values use meters (`float32`). No handwritten RGB-depth
extrinsic projection is used.

Run with the current D435i color baseline:

```bash
python tools/run_realsense_rgbd.py \
  --config configs/color_orange_d435i.yaml \
  --width 640 --height 480 --fps 30
```

The defaults remain configurable. Relevant experimental depth defaults are:

```text
mask erosion:        3x3, 1 iteration
valid depth range:   0.2–10.0 m
minimum pixels:      20
minimum valid ratio: 0.5
maximum depth std:   0.3 m
```

The target mask is eroded before depth lookup to reduce mixed target/background
edge pixels. If erosion completely removes a non-empty small target, the
baseline falls back to the original target mask. Zero, NaN, Inf, and
out-of-range values are removed. The reported distance is the median of valid
target pixels; validity also requires the pixel-count, valid-ratio, and depth
standard-deviation thresholds to pass. Invalid measurements display `DEPTH
INVALID` and do not expose a numeric depth.

Windows:

- `RGB`
- `Mask` (selected target only)
- `Detection` (selection plus depth quality)
- `Aligned Depth` (display-only colormap; never used for measurement)

Controls:

- `s`: save RGB, target mask, annotation, depth visualization, and JSON metadata
  under `outputs/realsense_rgbd/YYYY-MM-DD/`
- `q` / `ESC`: quit

The current orange test can is a reflective curved metal target. Its highlights
and depth edges may be unstable, so it is suitable only for validating the
RGB-depth pipeline. Validate later with a matte orange object and the final UAV
orange marking. No can-specific behavior is hard-coded.

The depth extractor measures scalar depth; the camera-position stage then uses
SDK deprojection. Camera-to-body extrinsics, world coordinates, ROS, PX4, YOLO,
complete tracking, and UAV control are not implemented.

## Target tape color calibration

Calibrate the specific saturated orange tape used on the UAV with real D435i
RGB frames. The system can target this material rather than all orange objects.
Phone-photo colors cannot establish the final HSV thresholds. Collect multiple
ROIs across distances, viewing angles, lighting conditions, and tape locations;
do not calibrate from just one frame.

```bash
python tools/calibrate_target_color.py --width 640 --height 480 --fps 30
```

Drag with the left mouse button to select an ROI. The image freezes while the
ROI is reviewed; release prints its statistics. Press `c` to add it once, then
select another ROI. `r` clears the session, `s` generates YAML/report, and
`q` / ESC exits. ROI bounds use the original camera pixel dimensions.

Calibration excludes pixels with S < 80 or V < 50. This prefilter only removes
obviously achromatic/dark pixels; users should tightly select tape to avoid
colored background contamination. All accepted ROI pixels are pooled with pixel
weighting. H/S/V median and p5/p95 are reported with valid counts and ratios.
The generated range uses p5/p95 plus baseline margins H=2, S=15, V=15, rounded
outward and bounded to OpenCV HSV ranges. These margins are experimental, not
optimal final parameters. Hue seam-crossing or ambiguous wide samples are
explicitly rejected because the detector accepts only one non-wrapping range.

Default YAML: `configs/color_orange_tape_d435i.yaml`, with fixed `min_area: 100`
and `morph_kernel_size: 5`. Existing YAML files are never overwritten: choose a
new path using `--save-config`. Timestamped reports go under
`outputs/color_calibration/YYYY-MM-DD/`. The generated HSV range is a baseline
for the actual tape, camera, and sampled deployment conditions.

## Image renaming commands

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
