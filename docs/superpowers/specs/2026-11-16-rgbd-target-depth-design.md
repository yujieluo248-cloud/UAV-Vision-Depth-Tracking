# RGB–Depth Alignment and Target Depth Design

## Intent

Measure a robust scalar distance for the currently selected RGB target. Depth
must be SDK-aligned to color so the selected target mask can index the same
pixel coordinates without handwritten calibration projection.

## Scope

- Add a separate RGBD RealSense source; preserve the verified RGB-only source.
- Add a NumPy/OpenCV target-depth extractor independent of camera hardware.
- Add a separate RGBD preview CLI with aligned-depth visualization and snapshot
  metadata.
- Preserve current HSV, candidate selection, and temporal association behavior.
- Do not add 3D deprojection, world/body coordinates, ROS, PX4, YOLO, complete
  tracking, or flight control.

## Approaches

1. Replace the RGB-only source and CLI with RGBD: rejected because it risks the
   already verified RGB path and changes its API.
2. Add `RealSenseRGBDSource` and `run_realsense_rgbd.py`, reusing detection and
   temporal-selection functions: selected because it isolates hardware changes
   while avoiding duplicate color-selection logic.
3. Put depth statistics inside the camera source: rejected because it couples
   hardware acquisition to target semantics and makes unit testing harder.

## Data flow

`RealSenseRGBDSource` enables configurable color BGR8 and depth Z16 streams,
uses `rs.align(rs.stream.color)`, and obtains the hardware depth scale from the
active profile. It returns `RGBDFrame` containing BGR color, aligned `float32`
depth in meters, and the color-frame SDK timestamp in milliseconds.

`TargetDepthExtractor` erodes a binary target mask before indexing depth. If a
non-empty target disappears completely, the baseline policy falls back to the
original mask. It removes zero, non-finite, and out-of-range values; computes
the median, valid-pixel ratio, population standard deviation, counts; and marks
the measurement valid only when all quality thresholds pass. Invalid
measurements expose `depth_m=None` while retaining diagnostics.

`run_realsense_rgbd.py` reuses the existing detector, selector, and temporal
state. Its display-only colormap never feeds the measurement path. Snapshots
save RGB, target mask, annotation, depth visualization, and JSON under one wall
clock timestamp.

## Baseline parameters

- Mask erosion: 3x3 kernel, 1 iteration, fallback to original mask when empty.
- Depth range: 0.2–10.0 m.
- Quality: at least 20 valid pixels, valid ratio at least 0.5, depth standard
  deviation at most 0.3 m.

These are experimental defaults, not final calibrated thresholds.

## Target note

The orange can is acceptable for pipeline verification, but its reflective
curved metal surface and unstable depth edges are not representative of the
final target. Later validation should use matte orange material and the final
UAV marking.
