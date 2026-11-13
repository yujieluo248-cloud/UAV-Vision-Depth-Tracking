# Safe Image Renaming Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested CLI for deterministic dry-run and collision-safe two-phase image renaming.

**Architecture:** `tools/rename_images.py` separates supported-image discovery, validated plan construction, preview output, and two-phase execution. Filesystem mutation happens only after the complete plan passes all validation.

**Tech Stack:** Python 3.10 standard library, pathlib, uuid, pytest

**Spec:** `docs/superpowers/specs/2026-11-13-rename-images-design.md`

## Global Constraints

- Do not recurse into subdirectories.
- Do not overwrite any non-participating filesystem entry.
- Do not modify ColorDetector, HSV parameters, or batch detector logic.
- Do not add data/output artifacts or create a Git commit.

## Review Focus

- A target occupied by another participating source must succeed without losing either file.
- A target occupied by a directory or non-participating entry must stop before any rename.
- Mixed-case extensions must remain byte-for-byte unchanged in the target suffix.
- Prefix path traversal or separators must not move files outside the selected directory.
- A failure during either rename phase must make a best-effort rollback and surface a clear error.

---

### Task 1: Plan construction, dry-run, and validation

**Files:**
- Create: `tools/rename_images.py`
- Create: `tests/test_rename_images.py`

**Interfaces:**
- Produces: `list_image_files()`, `build_rename_plan()`, `rename_images()`, and CLI `main()`

- [ ] Add tests for sorting/filtering, extension preservation, prefix/start/digits formatting, dry-run immutability, empty/missing directories, invalid arguments, digit overflow, and unrelated target conflicts.
- [ ] Run focused tests and confirm failure because the module is missing.
- [ ] Implement immutable plan construction and dry-run output without mutating files.
- [ ] Run focused tests and confirm planning/dry-run behavior passes.

### Task 2: Two-phase execution safety

**Files:**
- Modify: `tools/rename_images.py`
- Modify: `tests/test_rename_images.py`

**Interfaces:**
- Consumes: validated rename operations from Task 1
- Produces: safe applied renames with best-effort rollback

- [ ] Add execution tests for ordinary renames and an existing participating target-name permutation.
- [ ] Run them and confirm failure because apply behavior is absent.
- [ ] Implement UUID temporary paths, two rename phases, and rollback on exceptions.
- [ ] Run focused and full test suites.

### Task 3: Documentation and final verification

**Files:**
- Modify: `README.md`

- [ ] Add preview, apply, and negative-prefix examples.
- [ ] Compile the new tool with the `uav_vision` Python and run the full pytest suite.
- [ ] Verify Git ignores data/output artifacts, protected detector/config/batch files are unchanged, and HEAD is unchanged.
