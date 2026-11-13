# Color Batch Tool Design

## Intent

Add a deterministic offline batch runner for evaluating the existing HSV color detector over a directory of static images. Each readable image produces a mask, an annotated image, and one CSV result row inside a local-date output directory.

## Scope and constraints

- Accept `.jpg`, `.jpeg`, `.png`, and `.bmp`, case-insensitively, sorted by filename.
- Use `ColorDetector.detect()` and the existing annotation helper; do not duplicate detection logic.
- Write directly to `outputs/test/YYYY-MM-DD/` with `<stem>_mask.png`, `<stem>_annotated.png`, and `results.csv`.
- Always write both images for a readable input, including no-detection cases; add `NO DETECTION` to those annotations.
- Continue after unreadable image files and report them as errors rather than as detector misses.
- Do not change HSV configuration or add depth, RealSense, ROS, tracking, YOLO, or auto-tuning.

## Approaches

1. **Selected: testable batch function with filesystem helpers.** Pure helpers enumerate images and create the dated path; `run_batch()` owns reading, detection, output, CSV, and progress.
2. Put all behavior directly in CLI `main()`. Shorter initially, but difficult to test without subprocesses and date monkeypatching.
3. Run one subprocess per image through `run_color_image.py`. Reuses the CLI but adds overhead and makes aggregate CSV/error handling fragile.

## Decisions

- `Processed` counts successfully decoded images. Corrupt images print an `ERROR` line, produce no synthetic outputs, and do not count as no-detection.
- `results.csv` contains rows for successfully decoded images only because the required schema has no error field and a decode error is not a detection result.
- Tests inject a fixed `datetime` and temporary output root; production uses local `datetime.now()` and `outputs/test`.
