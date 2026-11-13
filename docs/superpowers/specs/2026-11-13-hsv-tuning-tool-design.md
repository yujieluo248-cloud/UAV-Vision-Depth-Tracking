# HSV Tuning Tool Design

## Intent

Add a static-image OpenCV GUI for interactively tuning the orange HSV range while reusing the existing detector's morphology, contour filtering, and target selection behavior.

## Scope and constraints

- Load one image and one YAML configuration.
- Show Original, Mask, and Annotated windows; attach six standard OpenCV HSV trackbars to the Annotated window.
- Re-run full-resolution detection after slider changes; only display copies may be scaled.
- Save the current HSV range with the loaded `min_area` and `morph_kernel_size` when `s` is pressed.
- Exit on `q` or Escape.
- Do not add camera, RealSense, ROS, depth, tracking, or YOLO functionality.

## Approaches

1. **Selected: small detector update method plus thin GUI.** Add `ColorDetector.set_hsv_range()` and keep all image processing in `detect()`. The GUI owns trackbars, drawing, display scaling, and YAML persistence.
2. Reconstruct a detector on every slider change. This avoids changing the detector API but needlessly recreates fixed morphology state.
3. Duplicate thresholding and contour logic in the GUI. This risks the tuning preview diverging from production detection and is rejected.

## Design decisions

- `set_hsv_range(lower_hsv, upper_hsv)` updates only thresholds and does not reject lower values greater than upper values; OpenCV then produces an empty mask without crashing.
- Reuse `tools.run_color_image.load_config()` for YAML loading and add a pure `save_config()` function to the tuner.
- Default input config and default save target are both `configs/color_orange.yaml`; `--save-config` overrides only the save destination.
- Pure tests cover trackbar conversion, detector dynamic updates, YAML load/save shape, and display-only scaling.
