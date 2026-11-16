# Multi-Candidate Selection V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate all valid HSV candidates, select one with baseline temporal continuity, and expose a selected-only target mask while retaining existing APIs.

**Architecture:** `ColorDetector` remains responsible for segmentation and candidate geometry. A stateless `CandidateSelector` owns gates and scoring; the RealSense tool alone owns short-lived previous-center/area/miss state.

**Tech Stack:** Python 3.10, NumPy, OpenCV, pytest

**Spec:** `docs/superpowers/specs/2026-11-16-multi-candidate-selection-design.md`

## Global Constraints

- Do not change current HSV configuration values.
- Do not implement Depth, RGB-depth alignment, 3D, ROS, PX4, YOLO, Kalman, Alpha-Beta, or a full tracker.
- Preserve existing static tools and batch metrics.
- Do not connect real D435i in automated verification.
- Do not commit or push.

## Review Focus

- No candidates: acquisition and association return no target and an empty target mask.
- Two plausible candidates: acquisition has deterministic largest-valid fallback.
- Far candidate with history: center gate prevents an identity jump.
- Large area change near the previous center: area gate rejects implausible association.
- Repeated misses: history resets only after the configured miss allowance.

---

### Task 1: Candidate extraction and compatibility detection

**Files:**
- Modify: `src/detection/color_detector.py`
- Modify: `tests/test_color_detector.py`

**Interfaces:**
- Produces: `ColorCandidate`, `ColorCandidateResult`, and `ColorDetector.detect_candidates(image_bgr)`.
- Preserves: `ColorDetector.detect(image_bgr) -> ColorDetection`.

- [ ] Write failing tests for two candidates, geometry, sorting, and selected-only mask.
- [ ] Run focused tests and confirm missing API failures.
- [ ] Implement candidate extraction and compatibility acquisition.
- [ ] Run focused and legacy detector tests until green.

### Task 2: CandidateSelector gates and temporal score

**Files:**
- Create: `src/detection/candidate_selector.py`
- Create: `tests/test_candidate_selector.py`

**Interfaces:**
- Consumes: `ColorCandidate`, image shape, optional previous center and area.
- Produces: `SelectionResult(candidate, mode)` from `CandidateSelector.select(...)`.

- [ ] Write failing acquisition, near-small-vs-far-large, and jump-gate tests.
- [ ] Run focused tests and confirm import/API failures.
- [ ] Implement clear baseline gates and score.
- [ ] Run focused tests until green.

### Task 3: RealSense minimal temporal state and visualization

**Files:**
- Modify: `tools/run_realsense_rgb.py`
- Modify: `tests/test_run_realsense_rgb.py`

**Interfaces:**
- Consumes: candidate extraction and `CandidateSelector`.
- Produces: `TemporalSelectionState` and a per-frame selected target mask.

- [ ] Write failing tests for association continuity, misses, reset, mask, and display metadata.
- [ ] Run focused tests and confirm failures.
- [ ] Implement state update and candidate visualization without GUI automation.
- [ ] Run RealSense fake-source tests until green.

### Task 4: Documentation and regression verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents candidate generation, selection, target mask, and limitations.

- [ ] Update README without claiming identity recognition or a complete tracker.
- [ ] Run the complete pytest suite.
- [ ] Run `py_compile`, both CLI help commands, and `git diff --check`.
- [ ] Inspect `git status` and confirm user-owned config files remain untouched.
