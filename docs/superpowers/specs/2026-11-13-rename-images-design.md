# Safe Image Renaming Tool Design

## Intent

Add a deterministic, previewable tool for renaming supported images in one directory while preserving extensions and never overwriting unrelated entries.

## Scope and constraints

- Process only first-level `.jpg`, `.jpeg`, `.png`, and `.bmp` files, sorted by current filename.
- Require a nonblank prefix; support nonnegative start and positive digit width.
- Preserve each original suffix exactly, including suffix letter case.
- Reject index ranges that exceed the selected digit capacity.
- `--dry-run` prints the full plan without filesystem changes.
- Apply with a two-phase source-to-UUID-temporary-to-final sequence.
- Reject target paths occupied by entries that are not participating sources.
- Do not modify detection, HSV, batch-processing logic, data images, or outputs; do not commit.

## Approaches

1. **Selected: plan first, then two-phase rename with rollback.** A pure plan builder validates every target before mutation. Execution moves all sources to unique temporary paths, then moves them to final paths; failures trigger best-effort restoration.
2. Sequential direct rename. Simpler, but a target that is another participating source can overwrite or block data.
3. Copy then delete. Avoids name cycles but duplicates large image data and introduces partial-copy/data-loss risks.

## Decisions

- Sorting uses Unicode case-folded filenames with the original name as a tie-breaker for deterministic mixed-case behavior.
- Prefixes that are blank, `.`/`..`, or contain path separators are rejected so outputs cannot escape the selected directory.
- Existing target names are allowed only when the occupant is itself in the rename plan; the temporary phase makes those permutations safe.
