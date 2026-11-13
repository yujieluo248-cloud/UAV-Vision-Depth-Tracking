# Color Batch Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested CLI that batch-processes static images with the current color detector into dated image outputs and a CSV summary.

**Architecture:** `tools/run_color_batch.py` exposes pure path/enumeration helpers and a `run_batch()` orchestration function. It reuses detector/config/annotation code already used by the single-image tool.

**Tech Stack:** Python 3.10, OpenCV, csv, pathlib, datetime, pytest

**Spec:** `docs/superpowers/specs/2026-11-13-color-batch-tool-design.md`

## Global Constraints

- Keep the current HSV YAML unchanged.
- Do not add depth, RealSense, ROS, tracking, YOLO, or auto-tuning.
- Do not add test images or generated output files to Git.
- Do not commit any code.

## Review Focus

- Uppercase supported extensions must be included while unrelated files are ignored.
- Duplicate stems with different extensions overwrite their shared output names as explicitly allowed for same-name results.
- Unreadable images must not abort later valid images or inflate no-detection counts.
- CSV rows must use lowercase `true`/`false` and blank unavailable geometry fields.
- Re-running on the same date must reuse the same directory and replace `results.csv` deterministically.

---

### Task 1: Batch processing and tests

**Files:**
- Create: `tools/run_color_batch.py`
- Create: `tests/test_run_color_batch.py`

**Interfaces:**
- Produces: `get_dated_output_dir()`, `list_image_files()`, `run_batch()`, and CLI `main()`

- [ ] Add tests for fixed-date output paths, sorted supported images, ignored files, two-image output pairs, no-detection output, CSV fields, empty directory, missing directory, and corrupt-image continuation.
- [ ] Run focused tests and confirm failure because the module is missing.
- [ ] Implement the batch tool by reusing `ColorDetector`, `load_config`, and `draw_detection`.
- [ ] Run focused and full tests and confirm they pass.

### Task 2: Documentation and verification

**Files:**
- Modify: `README.md`

- [ ] Document the batch command, dated output structure, offline-only purpose, and real D435i calibration warning.
- [ ] Compile all tools and detector with the `uav_vision` Python.
- [ ] Run the complete pytest suite.
- [ ] Verify `.gitignore` covers batch inputs/outputs, protected files are unchanged, HSV configuration is unchanged, and no commit was created.
