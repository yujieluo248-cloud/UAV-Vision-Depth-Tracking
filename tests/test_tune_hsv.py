import importlib
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import yaml


def load_tuner() -> ModuleType:
    return importlib.import_module("tools.tune_hsv")


def test_trackbar_values_map_to_lower_and_upper_hsv() -> None:
    tuner = load_tuner()

    lower_hsv, upper_hsv = tuner.trackbar_values_to_hsv(
        h_min=5,
        h_max=22,
        s_min=150,
        s_max=255,
        v_min=140,
        v_max=250,
    )

    assert lower_hsv == (5, 150, 140)
    assert upper_hsv == (22, 255, 250)


def test_load_and_save_config_preserve_non_hsv_values(tmp_path: Path) -> None:
    tuner = load_tuner()
    source_path = tmp_path / "source.yaml"
    save_path = tmp_path / "nested" / "saved.yaml"
    source_path.write_text(
        "lower_hsv: [5, 150, 150]\n"
        "upper_hsv: [22, 255, 255]\n"
        "min_area: 725\n"
        "morph_kernel_size: 7\n",
        encoding="utf-8",
    )
    config = tuner.load_config(source_path)

    tuner.save_config(
        save_path,
        lower_hsv=(8, 120, 130),
        upper_hsv=(28, 240, 245),
        base_config=config,
    )

    saved = yaml.safe_load(save_path.read_text(encoding="utf-8"))
    assert saved == {
        "lower_hsv": [8, 120, 130],
        "upper_hsv": [28, 240, 245],
        "min_area": 725,
        "morph_kernel_size": 7,
    }


def test_scale_for_display_does_not_modify_original_shape() -> None:
    tuner = load_tuner()
    original = np.zeros((1000, 2000, 3), dtype=np.uint8)

    displayed = tuner.scale_for_display(
        original, max_width=500, max_height=500
    )

    assert original.shape == (1000, 2000, 3)
    assert displayed.shape == (250, 500, 3)
    assert not np.shares_memory(original, displayed)


def test_default_save_path_is_independent_of_input_config() -> None:
    tuner = load_tuner()

    args = tuner.parse_args(
        ["image.jpg", "--config", "configs/temporary.yaml"]
    )

    assert args.save_config == tuner.DEFAULT_CONFIG


def test_missing_image_exits_before_opening_gui(tmp_path: Path) -> None:
    tuner = load_tuner()
    missing_image = tmp_path / "missing.jpg"

    with pytest.raises(SystemExit, match="Could not read image"):
        tuner.main([str(missing_image)])
