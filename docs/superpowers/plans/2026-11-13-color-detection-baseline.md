# Color Detection Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify static RGB orange-target segmentation with mask, bounding box, center, and area output.

**Architecture:** `ColorDetector` is a YAML-independent OpenCV component configured through constructor arguments. A separate CLI loads YAML, reads/writes images, annotates detections, and reports results.

**Tech Stack:** Python 3.10, NumPy, OpenCV, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-11-13-color-detection-baseline-design.md`

## Global Constraints

- Scope is static RGB image detection only.
- Do not add depth, tracking, camera, ROS, flight-control, or neural-network dependencies.
- Treat HSV values as baseline starting points requiring real-world calibration.
- Do not alter `.gitignore`, `data/.gitkeep`, or `outputs/.gitkeep`.

## Review Focus

- Empty or malformed image input raises `ValueError` instead of failing inside OpenCV.
- Morphology preserves a clear target while removing isolated noise.
- Contours exactly at the configured minimum area are accepted consistently.
- Multiple valid regions select the one with the largest contour area.
- CLI paths work when invoked from the repository root in a headless environment.

---

### Task 1: Detector behavior

**Files:**
- Modify: `src/detection/color_detector.py`
- Create: `tests/test_color_detector.py`

**Interfaces:**
- Produces: `ColorDetection` and `ColorDetector.detect(image_bgr: np.ndarray) -> ColorDetection`

- [ ] Write synthetic-image tests for target detection, black image, small noise, largest target, and invalid input.
- [ ] Run the detector tests and confirm they fail because the API is missing.
- [ ] Implement input validation, HSV thresholding, open/close morphology, external contour filtering, and largest-contour result construction.
- [ ] Run the detector tests and confirm they pass.

### Task 2: Configuration and static-image CLI

**Files:**
- Create: `configs/color_orange.yaml`
- Modify: `tools/run_color_image.py`

**Interfaces:**
- Consumes: `ColorDetector(lower_hsv, upper_hsv, min_area, morph_kernel_size)`
- Produces: CLI accepting `image_path`, `--config`, and `--output-dir`

- [ ] Add a CLI integration test using temporary files and subprocess execution.
- [ ] Run it and confirm it fails because CLI behavior is missing.
- [ ] Add baseline YAML and implement config loading, image loading, annotations, saving, and terminal reporting.
- [ ] Run the CLI integration test and confirm it passes.

### Task 3: Dependencies, documentation, and full verification

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`

- [ ] Record only OpenCV, NumPy, PyYAML, and pytest dependencies.
- [ ] Document purpose, current phase, invocation, outputs, and HSV calibration warning.
- [ ] Run the requested `py_compile` command with the `uav_vision` Python.
- [ ] Run the full pytest suite with the `uav_vision` Python.
- [ ] Inspect Git diff/status and confirm protected `.gitkeep` and ignore files are unchanged.
