# Color Detection Baseline Design

## Intent

Build the first static-image RGB baseline for detecting a conspicuous orange target in D435i imagery. The result is a binary pixel mask plus the largest valid target's bounding box, center, and contour area.

## Scope

- Use OpenCV BGR-to-HSV thresholding, morphology, and external contours.
- Keep `ColorDetector` independent of YAML and ROS.
- Let the command-line tool load YAML, annotate an input image, and save `mask.png` and `annotated.png` without requiring a display.
- Cover synthetic target, no-target, small-noise, largest-region, and invalid-input behavior with tests.
- Document only the current Phase 1 baseline.

## Non-goals

No depth processing, RGB-depth registration, 3D localization, tracking, camera access, ROS, PX4, VINS, flight control, YOLO, or real image artifacts.

## Constraints

- Python 3.10 with OpenCV, NumPy, PyYAML, and pytest for tests.
- Default orange range is a starting point only: lower HSV `(5, 100, 100)`, upper HSV `(25, 255, 255)`.
- Default minimum contour area is `500` pixels and the morphology kernel size is configurable.
- Preserve the existing ignore policy and `.gitkeep` files.

## Approaches

1. **Constructor-configured detector with YAML at the CLI boundary (selected).** Keeps detection reusable and easy to replace while the tool owns filesystem/config concerns.
2. Detector loads YAML internally. Shorter CLI code, but tightly couples the reusable image algorithm to a particular configuration format and path.
3. Module-level detection function. Minimal, but less clear for carrying calibrated parameters and future detector substitution.

The first approach best matches the requested separation and gives a small, testable API.

## Design

`ColorDetection` is a dataclass containing `detected`, optional `bbox`, optional `center`, contour `area`, and the final binary `mask`. `ColorDetector.detect()` validates a non-empty three-channel BGR NumPy image, thresholds HSV, applies open then close morphology, filters contours by `min_area`, and returns the largest remaining contour.

The CLI validates its input/config, constructs the detector, writes both output images, and prints a concise detection result. Configuration values remain ordinary YAML lists/scalars and are converted at the CLI boundary.
