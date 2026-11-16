# RGB–Depth Target Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align D435i depth to RGB and calculate a quality-gated median depth from the selected target mask.

**Architecture:** A new optional-SDK RGBD source owns stream setup, alignment, and meter conversion. A camera-independent extractor owns erosion, filtering, statistics, and validity; a new CLI composes it with existing candidate selection.

**Tech Stack:** Python 3.10, NumPy, OpenCV, pyrealsense2 (optional runtime), pytest

**Spec:** `docs/superpowers/specs/2026-11-16-rgbd-target-depth-design.md`

## Global Constraints

- Preserve the existing RGB-only source and CLI.
- Use `rs.align(rs.stream.color)`; do not hand-code RGB-depth projection.
- Do not modify the current D435i HSV configuration.
- Do not connect real hardware in automated tests.
- Do not implement 3D, world/body coordinates, ROS, PX4, YOLO, full tracking, or flight control.
- Do not commit or push.

## Review Focus

- Missing aligned color or depth frame returns no sample without shape mismatch.
- Device start/read exceptions still stop the pipeline through the preview lifecycle.
- Empty/eroded-away masks do not produce a fake depth.
- NaN, Inf, zero, and range-outlier values cannot affect median or std.
- Display normalization cannot mutate or replace meter-valued calculation input.

---

### Task 1: Target depth extraction

**Files:**
- Create: `src/depth/__init__.py`
- Create: `src/depth/target_depth.py`
- Create: `tests/test_target_depth.py`

**Interfaces:**
- Produces: `TargetDepthMeasurement` and `TargetDepthExtractor.measure(aligned_depth_m, target_mask)`.

- [ ] Write failing tests for mask selection, invalid filtering, median, ratio, std, validity thresholds, empty masks, and erosion fallback.
- [ ] Run focused tests and verify missing-module failure.
- [ ] Implement the minimal extractor and baseline validation.
- [ ] Run focused tests until green.

### Task 2: RealSense aligned RGBD source

**Files:**
- Create: `src/camera/realsense_rgbd.py`
- Modify: `src/camera/__init__.py`
- Create: `tests/test_realsense_rgbd.py`

**Interfaces:**
- Produces: `RGBDFrame` and `RealSenseRGBDSource.start/read/stop`.

- [ ] Write fake-SDK tests for dual streams, `align(color)`, meter conversion, equal shapes, timestamps, missing frames, and cleanup.
- [ ] Run focused tests and verify missing-module failure.
- [ ] Implement optional SDK loading, aligned read, and idempotent stop.
- [ ] Run focused tests until green.

### Task 3: RGBD preview and snapshots

**Files:**
- Create: `tools/run_realsense_rgbd.py`
- Create: `tests/test_run_realsense_rgbd.py`

**Interfaces:**
- Consumes: `RGBDFrame`, `process_frame`, temporal state, and `TargetDepthExtractor`.
- Produces: four debug windows and same-timestamp PNG/JSON snapshot artifacts.

- [ ] Write failing tests for measurement composition, display-only depth colormap, JSON artifacts, normal stop, and exception stop.
- [ ] Run focused tests and verify missing-module failure.
- [ ] Implement the preview with dependency-injected GUI callbacks.
- [ ] Run focused tests until green.

### Task 4: Documentation and verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents alignment, units, quality thresholds, controls, target limitations, and explicit non-goals.

- [ ] Add the RGBD pipeline and can-target caveats without claiming 3D support.
- [ ] Run the complete pytest suite.
- [ ] Run py_compile, RGBD CLI `--help`, and `git diff --check`.
- [ ] Confirm user experiment configs remain untouched in `git status`.
