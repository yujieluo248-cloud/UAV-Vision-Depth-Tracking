import importlib
from pathlib import Path
from types import ModuleType

import pytest


def load_rename_module() -> ModuleType:
    return importlib.import_module("tools.rename_images")


def test_list_image_files_sorts_names_and_ignores_non_images(
    tmp_path: Path,
) -> None:
    rename = load_rename_module()
    (tmp_path / "z.jpeg").write_bytes(b"z")
    (tmp_path / "B.PNG").write_bytes(b"b")
    (tmp_path / "a.jpg").write_bytes(b"a")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "nested.bmp").mkdir()

    image_files = rename.list_image_files(tmp_path)

    assert [path.name for path in image_files] == ["a.jpg", "B.PNG", "z.jpeg"]


def test_build_plan_preserves_extensions_prefix_start_and_digits(
    tmp_path: Path,
) -> None:
    rename = load_rename_module()
    (tmp_path / "first.JPG").write_bytes(b"one")
    (tmp_path / "second.png").write_bytes(b"two")

    plan = rename.build_rename_plan(
        tmp_path, prefix="negative", start=21, digits=4
    )

    assert [operation.source.name for operation in plan] == [
        "first.JPG",
        "second.png",
    ]
    assert [operation.target.name for operation in plan] == [
        "negative_0021.JPG",
        "negative_0022.png",
    ]


def test_dry_run_prints_preview_without_changing_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rename = load_rename_module()
    source = tmp_path / "IMG_1829.jpg"
    source.write_bytes(b"original")

    plan = rename.rename_images(
        tmp_path, prefix="orange", start=1, digits=3, dry_run=True
    )

    assert source.read_bytes() == b"original"
    assert not (tmp_path / "orange_001.jpg").exists()
    assert plan[0].target.name == "orange_001.jpg"
    assert capsys.readouterr().out == (
        "Rename preview:\n\n"
        "[1/1] IMG_1829.jpg -> orange_001.jpg\n\n"
        "No files were changed. Remove --dry-run to apply.\n"
    )


def test_empty_and_missing_directories_raise_clear_errors(tmp_path: Path) -> None:
    rename = load_rename_module()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No supported images"):
        rename.list_image_files(empty_dir)
    with pytest.raises(ValueError, match="Input directory does not exist"):
        rename.list_image_files(tmp_path / "missing")


@pytest.mark.parametrize(
    ("prefix", "start", "digits", "message"),
    [
        ("", 1, 3, "prefix must not be empty"),
        ("   ", 1, 3, "prefix must not be empty"),
        ("../orange", 1, 3, "prefix must be a filename prefix"),
        ("orange/name", 1, 3, "prefix must be a filename prefix"),
        ("orange", -1, 3, "start must be non-negative"),
        ("orange", 1, 0, "digits must be positive"),
    ],
)
def test_invalid_parameters_are_rejected(
    tmp_path: Path,
    prefix: str,
    start: int,
    digits: int,
    message: str,
) -> None:
    rename = load_rename_module()
    (tmp_path / "source.jpg").write_bytes(b"source")

    with pytest.raises(ValueError, match=message):
        rename.build_rename_plan(
            tmp_path, prefix=prefix, start=start, digits=digits
        )


def test_digit_capacity_overflow_is_rejected(tmp_path: Path) -> None:
    rename = load_rename_module()
    (tmp_path / "a.jpg").write_bytes(b"a")
    (tmp_path / "b.jpg").write_bytes(b"b")

    with pytest.raises(ValueError, match="exceeds 1 digits"):
        rename.build_rename_plan(tmp_path, prefix="orange", start=9, digits=1)


def test_non_participating_target_conflict_stops_before_changes(
    tmp_path: Path,
) -> None:
    rename = load_rename_module()
    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    (tmp_path / "orange_001.jpg").mkdir()

    with pytest.raises(ValueError, match="Target already exists"):
        rename.build_rename_plan(tmp_path, prefix="orange", start=1, digits=3)

    assert source.read_bytes() == b"source"


def test_apply_renames_files_and_preserves_contents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rename = load_rename_module()
    (tmp_path / "b.png").write_bytes(b"second")
    (tmp_path / "a.jpg").write_bytes(b"first")

    rename.rename_images(
        tmp_path, prefix="negative", start=21, digits=3, dry_run=False
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "negative_021.jpg",
        "negative_022.png",
    ]
    assert (tmp_path / "negative_021.jpg").read_bytes() == b"first"
    assert (tmp_path / "negative_022.png").read_bytes() == b"second"
    assert capsys.readouterr().out == (
        "[1/2] a.jpg -> negative_021.jpg\n"
        "[2/2] b.png -> negative_022.png\n\n"
        "Renamed: 2 images\n"
    )


def test_existing_participating_target_name_is_not_overwritten(
    tmp_path: Path,
) -> None:
    rename = load_rename_module()
    (tmp_path / "A.jpg").write_bytes(b"from A")
    (tmp_path / "orange_001.jpg").write_bytes(b"original numbered file")

    rename.rename_images(tmp_path, prefix="orange", dry_run=False)

    assert (tmp_path / "orange_001.jpg").read_bytes() == b"from A"
    assert (tmp_path / "orange_002.jpg").read_bytes() == b"original numbered file"


def test_apply_failure_rolls_back_original_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rename = load_rename_module()
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    real_rename = Path.rename
    call_count = 0

    def fail_once_on_third_rename(source: Path, target: Path) -> Path:
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise OSError("injected rename failure")
        return real_rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_once_on_third_rename)

    with pytest.raises(RuntimeError, match="Rename failed; original files restored"):
        rename.rename_images(tmp_path, prefix="orange", dry_run=False)

    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.jpg", "b.jpg"]
