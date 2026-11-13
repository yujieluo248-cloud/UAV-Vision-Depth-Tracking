# HSV Tuning Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a static-image OpenCV HSV tuning GUI that previews the existing detector and saves calibrated YAML values.

**Architecture:** `ColorDetector` gains a threshold-only update method while retaining all processing in `detect()`. `tools/tune_hsv.py` provides testable state conversion, persistence, scaling, annotation, and a thin OpenCV event loop.

**Tech Stack:** Python 3.10, OpenCV, NumPy, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-11-13-hsv-tuning-tool-design.md`

## Global Constraints

- Static images only; do not access D435i or add ROS, depth, tracking, or YOLO.
- Detection must run on the original image dimensions.
- HSV limits are H `0..179` and S/V `0..255`.
- Existing image detector behavior must remain unchanged.
- Do not add images, outputs, or screenshots.

## Review Focus

- A lower HSV bound above its upper bound must return an empty mask rather than raise.
- YAML saving must retain the loaded `min_area` and `morph_kernel_size`.
- Display scaling must not mutate or replace the full-resolution detection input.
- Missing/unreadable images must exit with a clear error before creating GUI state.
- Default save location must remain `configs/color_orange.yaml` even when a different input config is supplied.

---

### Task 1: Dynamically update detector thresholds

**Files:**
- Modify: `src/detection/color_detector.py`
- Modify: `tests/test_color_detector.py`

**Interfaces:**
- Produces: `ColorDetector.set_hsv_range(lower_hsv: HSVRange, upper_hsv: HSVRange) -> None`

- [ ] Add tests proving an updated range changes detection and reversed bounds return an empty mask.
- [ ] Run the focused tests and confirm failure because the method is missing.
- [ ] Implement the threshold-only update method and route constructor initialization through it.
- [ ] Run detector tests and confirm they pass.

### Task 2: Pure tuner functions and GUI

**Files:**
- Create: `tools/tune_hsv.py`
- Create: `tests/test_tune_hsv.py`

**Interfaces:**
- Consumes: `ColorDetector.set_hsv_range()` and `tools.run_color_image.load_config()`
- Produces: `trackbar_values_to_hsv()`, `save_config()`, `scale_for_display()`, and CLI `main()`

- [ ] Add pure tests for trackbar mapping, YAML load/save preservation, and non-mutating display scaling.
- [ ] Run tuner tests and confirm failure because the module/functions are missing.
- [ ] Implement pure helpers, annotation, GUI windows/trackbars, keyboard handling, input errors, and full-resolution re-detection.
- [ ] Run tuner tests and confirm they pass.

### Task 3: Documentation and verification

**Files:**
- Modify: `README.md`

- [ ] Add the short HSV tuning command, controls, and real-hardware calibration warning.
- [ ] Compile detector, static runner, and tuner with the specified Conda Python.
- [ ] Run the complete pytest suite.
- [ ] Confirm protected data/output files and ignore rules are unchanged.
