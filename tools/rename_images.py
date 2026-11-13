"""Safely preview and apply deterministic image renames."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from uuid import uuid4


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class RenameOperation:
    source: Path
    target: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely rename images in one directory."
    )
    parser.add_argument("input_dir", type=Path, help="Directory of images")
    parser.add_argument("--prefix", required=True, help="New filename prefix")
    parser.add_argument("--start", type=int, default=1, help="Starting index")
    parser.add_argument(
        "--digits", type=int, default=3, help="Zero-padded index width"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rename plan without changing files",
    )
    return parser.parse_args(argv)


def list_image_files(input_dir: Path) -> list[Path]:
    """Return supported first-level image files in deterministic order."""
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    image_files = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not image_files:
        raise ValueError(f"No supported images found in: {input_dir}")
    return image_files


def _validate_parameters(prefix: str, start: int, digits: int) -> None:
    if not prefix.strip():
        raise ValueError("prefix must not be empty")
    if (
        prefix in {".", ".."}
        or "/" in prefix
        or "\\" in prefix
        or Path(prefix).name != prefix
    ):
        raise ValueError("prefix must be a filename prefix without separators")
    if start < 0:
        raise ValueError("start must be non-negative")
    if digits <= 0:
        raise ValueError("digits must be positive")


def build_rename_plan(
    input_dir: Path,
    *,
    prefix: str,
    start: int = 1,
    digits: int = 3,
) -> list[RenameOperation]:
    """Build and validate the complete rename plan without changing files."""
    _validate_parameters(prefix, start, digits)
    image_files = list_image_files(input_dir)
    final_index = start + len(image_files) - 1
    if final_index > (10**digits) - 1:
        raise ValueError(
            f"Index range ending at {final_index} exceeds {digits} digits"
        )

    plan = [
        RenameOperation(
            source=source,
            target=input_dir
            / f"{prefix}_{index:0{digits}d}{source.suffix}",
        )
        for index, source in enumerate(image_files, start=start)
    ]
    source_paths = {operation.source for operation in plan}
    for operation in plan:
        if (
            operation.target.exists() or operation.target.is_symlink()
        ) and operation.target not in source_paths:
            raise ValueError(f"Target already exists: {operation.target}")
    return plan


def _print_plan(plan: list[RenameOperation]) -> None:
    total = len(plan)
    for index, operation in enumerate(plan, start=1):
        print(
            f"[{index}/{total}] {operation.source.name} "
            f"-> {operation.target.name}"
        )


def _path_is_occupied(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _unique_temporary_path(source: Path) -> Path:
    while True:
        temporary = source.parent / (
            f".__rename_tmp_{uuid4().hex}{source.suffix}"
        )
        if not _path_is_occupied(temporary):
            return temporary


def _rollback_renames(
    plan: list[RenameOperation],
    temporary_paths: dict[RenameOperation, Path],
    finalized: list[RenameOperation],
) -> list[str]:
    errors: list[str] = []

    for operation in reversed(finalized):
        temporary = temporary_paths[operation]
        try:
            if _path_is_occupied(temporary):
                raise FileExistsError(f"rollback temporary occupied: {temporary}")
            operation.target.rename(temporary)
        except OSError as exc:
            errors.append(str(exc))

    for operation in reversed(plan):
        temporary = temporary_paths.get(operation)
        if temporary is None or not _path_is_occupied(temporary):
            continue
        try:
            if _path_is_occupied(operation.source):
                raise FileExistsError(
                    f"rollback source occupied: {operation.source}"
                )
            temporary.rename(operation.source)
        except OSError as exc:
            errors.append(str(exc))

    return errors


def _apply_rename_plan(plan: list[RenameOperation]) -> None:
    temporary_paths: dict[RenameOperation, Path] = {}
    finalized: list[RenameOperation] = []
    try:
        for operation in plan:
            temporary = _unique_temporary_path(operation.source)
            operation.source.rename(temporary)
            temporary_paths[operation] = temporary

        for operation in plan:
            if _path_is_occupied(operation.target):
                raise FileExistsError(
                    f"Target became occupied during rename: {operation.target}"
                )
            temporary_paths[operation].rename(operation.target)
            finalized.append(operation)
    except OSError as exc:
        rollback_errors = _rollback_renames(plan, temporary_paths, finalized)
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                f"Rename failed and rollback was incomplete: {details}"
            ) from exc
        raise RuntimeError("Rename failed; original files restored") from exc


def rename_images(
    input_dir: Path,
    *,
    prefix: str,
    start: int = 1,
    digits: int = 3,
    dry_run: bool = False,
) -> list[RenameOperation]:
    """Preview or safely apply a validated two-phase rename plan."""
    plan = build_rename_plan(
        input_dir, prefix=prefix, start=start, digits=digits
    )
    if dry_run:
        print("Rename preview:\n")
        _print_plan(plan)
        print("\nNo files were changed. Remove --dry-run to apply.")
        return plan

    _apply_rename_plan(plan)
    _print_plan(plan)
    print(f"\nRenamed: {len(plan)} images")
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rename_images(
            args.input_dir,
            prefix=args.prefix,
            start=args.start,
            digits=args.digits,
            dry_run=args.dry_run,
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
