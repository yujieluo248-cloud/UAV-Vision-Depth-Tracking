import numpy as np
import pytest

from src.depth.target_depth import TargetDepthExtractor


def permissive_extractor(**overrides: object) -> TargetDepthExtractor:
    params: dict[str, object] = {
        "mask_erode_kernel": 1,
        "mask_erode_iterations": 0,
        "min_valid_pixels": 1,
        "min_valid_ratio": 0.0,
        "max_depth_std_m": 100.0,
    }
    params.update(overrides)
    return TargetDepthExtractor(**params)


def test_mask_excludes_background_depth_from_statistics() -> None:
    depth = np.full((5, 5), 9.0, dtype=np.float32)
    depth[1:4, 1:4] = 2.0
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 255

    measurement = permissive_extractor().measure(depth, mask)

    assert measurement.valid
    assert measurement.depth_m == pytest.approx(2.0)
    assert measurement.valid_count == 9
    assert measurement.total_count == 9
    assert measurement.valid_ratio == pytest.approx(1.0)
    assert measurement.depth_std_m == pytest.approx(0.0)


@pytest.mark.parametrize(
    "invalid_depth",
    [0.0, np.nan, np.inf, -np.inf, 0.1, 10.1],
)
def test_invalid_depth_values_are_removed(invalid_depth: float) -> None:
    depth = np.array([[1.0, invalid_depth]], dtype=np.float32)
    mask = np.full((1, 2), 255, dtype=np.uint8)

    measurement = permissive_extractor().measure(depth, mask)

    assert measurement.valid
    assert measurement.depth_m == pytest.approx(1.0)
    assert measurement.valid_count == 1
    assert measurement.total_count == 2
    assert measurement.valid_ratio == pytest.approx(0.5)


def test_median_is_robust_to_an_outlier() -> None:
    depth = np.array([[1.0, 1.0, 1.0, 9.0]], dtype=np.float32)
    mask = np.full((1, 4), 255, dtype=np.uint8)

    measurement = permissive_extractor().measure(depth, mask)

    assert measurement.valid
    assert measurement.depth_m == pytest.approx(1.0)


def test_valid_ratio_and_population_std_are_reported() -> None:
    depth = np.array([[1.0, 2.0, 0.0, np.nan]], dtype=np.float32)
    mask = np.full((1, 4), 255, dtype=np.uint8)

    measurement = permissive_extractor().measure(depth, mask)

    assert measurement.valid_count == 2
    assert measurement.total_count == 4
    assert measurement.valid_ratio == pytest.approx(0.5)
    assert measurement.depth_std_m == pytest.approx(0.5)
    assert measurement.depth_m == pytest.approx(1.5)


def test_too_few_valid_pixels_marks_measurement_invalid() -> None:
    depth = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    mask = np.full((1, 3), 255, dtype=np.uint8)

    measurement = permissive_extractor(min_valid_pixels=3).measure(depth, mask)

    assert not measurement.valid
    assert measurement.depth_m is None
    assert measurement.valid_count == 2


def test_low_valid_ratio_marks_measurement_invalid() -> None:
    depth = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    mask = np.full((1, 4), 255, dtype=np.uint8)

    measurement = permissive_extractor(min_valid_ratio=0.5).measure(depth, mask)

    assert not measurement.valid
    assert measurement.depth_m is None
    assert measurement.valid_ratio == pytest.approx(0.25)


def test_high_depth_std_marks_measurement_invalid() -> None:
    depth = np.array([[1.0, 2.0]], dtype=np.float32)
    mask = np.full((1, 2), 255, dtype=np.uint8)

    measurement = permissive_extractor(max_depth_std_m=0.4).measure(depth, mask)

    assert not measurement.valid
    assert measurement.depth_m is None
    assert measurement.depth_std_m == pytest.approx(0.5)


def test_empty_target_mask_is_invalid() -> None:
    depth = np.ones((5, 5), dtype=np.float32)
    mask = np.zeros((5, 5), dtype=np.uint8)

    measurement = permissive_extractor().measure(depth, mask)

    assert not measurement.valid
    assert measurement.depth_m is None
    assert measurement.valid_count == 0
    assert measurement.total_count == 0
    assert measurement.valid_ratio == 0.0
    assert measurement.depth_std_m is None


def test_erosion_removes_target_edge_depth() -> None:
    depth = np.full((5, 5), 8.0, dtype=np.float32)
    depth[2, 2] = 2.0
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 255
    extractor = permissive_extractor(
        mask_erode_kernel=3,
        mask_erode_iterations=1,
    )

    measurement = extractor.measure(depth, mask)

    assert measurement.valid
    assert measurement.depth_m == pytest.approx(2.0)
    assert measurement.total_count == 1


def test_eroded_away_small_target_falls_back_to_original_mask() -> None:
    depth = np.full((5, 5), 9.0, dtype=np.float32)
    depth[2, 2] = 3.0
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 255
    extractor = permissive_extractor(
        mask_erode_kernel=3,
        mask_erode_iterations=1,
    )

    measurement = extractor.measure(depth, mask)

    assert measurement.valid
    assert measurement.depth_m == pytest.approx(3.0)
    assert measurement.total_count == 1


def test_depth_and_mask_shape_must_match() -> None:
    with pytest.raises(ValueError, match="same shape"):
        permissive_extractor().measure(
            np.ones((3, 4), dtype=np.float32),
            np.ones((3, 5), dtype=np.uint8),
        )
