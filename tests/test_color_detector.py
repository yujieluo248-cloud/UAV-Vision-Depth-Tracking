import cv2
import numpy as np
import pytest

from src.detection import color_detector


ORANGE_HSV = (15, 220, 220)


def make_bgr_image(width: int = 200, height: int = 150) -> np.ndarray:
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_orange_rectangle(
    image_bgr: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
) -> None:
    orange_bgr = cv2.cvtColor(
        np.uint8([[ORANGE_HSV]]), cv2.COLOR_HSV2BGR
    )[0, 0].tolist()
    cv2.rectangle(image_bgr, top_left, bottom_right, orange_bgr, thickness=-1)


def test_detects_orange_rectangle_with_geometry_and_mask() -> None:
    image = make_bgr_image()
    draw_orange_rectangle(image, (40, 30), (140, 110))

    result = color_detector.ColorDetector().detect(image)

    assert result.detected is True
    assert result.bbox == (40, 30, 101, 81)
    assert result.center == (90, 70)
    assert result.area == pytest.approx(8000.0)
    assert result.mask.dtype == np.uint8
    assert result.mask.shape == image.shape[:2]
    assert cv2.countNonZero(result.mask) > 0


def test_black_image_has_no_detection() -> None:
    image = make_bgr_image()

    result = color_detector.ColorDetector().detect(image)

    assert result.detected is False
    assert result.bbox is None
    assert result.center is None
    assert result.area == 0.0
    assert cv2.countNonZero(result.mask) == 0


def test_small_orange_region_below_min_area_is_rejected() -> None:
    image = make_bgr_image()
    draw_orange_rectangle(image, (20, 20), (30, 30))

    result = color_detector.ColorDetector(min_area=500).detect(image)

    assert result.detected is False
    assert result.bbox is None
    assert result.center is None


def test_largest_valid_orange_region_is_selected() -> None:
    image = make_bgr_image(width=300, height=200)
    draw_orange_rectangle(image, (20, 20), (50, 50))
    draw_orange_rectangle(image, (140, 60), (240, 160))

    result = color_detector.ColorDetector(min_area=500).detect(image)

    assert result.detected is True
    assert result.bbox == (140, 60, 101, 101)
    assert result.center == (190, 110)
    assert result.area == pytest.approx(10000.0)


def test_detect_candidates_returns_every_valid_region_sorted_by_area() -> None:
    image = make_bgr_image(width=320, height=220)
    draw_orange_rectangle(image, (20, 30), (60, 70))
    draw_orange_rectangle(image, (150, 40), (260, 170))

    result = color_detector.ColorDetector(min_area=500).detect_candidates(image)

    assert len(result.candidates) == 2
    assert [candidate.bbox for candidate in result.candidates] == [
        (150, 40, 111, 131),
        (20, 30, 41, 41),
    ]
    large, small = result.candidates
    assert large.center == pytest.approx((205.0, 105.0))
    assert large.area == pytest.approx(14300.0)
    assert large.aspect_ratio == pytest.approx(111 / 131)
    assert large.fill_ratio == pytest.approx(14300 / (111 * 131))
    assert small.area == pytest.approx(1600.0)
    assert cv2.countNonZero(result.color_mask) > 0


def test_color_mask_contains_all_candidates_but_target_mask_only_selected() -> None:
    image = make_bgr_image(width=320, height=220)
    draw_orange_rectangle(image, (20, 30), (70, 80))
    draw_orange_rectangle(image, (150, 40), (270, 180))

    result = color_detector.ColorDetector(min_area=500).detect(image)

    assert result.detected
    assert result.bbox == (150, 40, 121, 141)
    assert result.color_mask[50, 30] == 255
    assert result.color_mask[50, 170] == 255
    assert result.target_mask[50, 30] == 0
    assert result.target_mask[50, 170] == 255
    assert np.array_equal(result.mask, result.target_mask)


def test_color_detection_keeps_legacy_five_argument_constructor() -> None:
    mask = np.zeros((10, 12), dtype=np.uint8)

    result = color_detector.ColorDetection(False, None, None, 0.0, mask)

    assert result.mask is mask
    assert result.target_mask is mask
    assert np.array_equal(result.color_mask, mask)


def test_set_hsv_range_changes_subsequent_detection() -> None:
    image = make_bgr_image()
    green_bgr = cv2.cvtColor(
        np.uint8([[[60, 220, 220]]]), cv2.COLOR_HSV2BGR
    )[0, 0].tolist()
    cv2.rectangle(image, (40, 30), (140, 110), green_bgr, thickness=-1)
    detector = color_detector.ColorDetector()

    assert detector.detect(image).detected is False

    detector.set_hsv_range((50, 100, 100), (70, 255, 255))

    assert detector.detect(image).detected is True


def test_reversed_hsv_bounds_return_empty_mask() -> None:
    image = make_bgr_image()
    draw_orange_rectangle(image, (40, 30), (140, 110))
    detector = color_detector.ColorDetector()

    detector.set_hsv_range((25, 255, 255), (5, 100, 100))
    result = detector.detect(image)

    assert result.detected is False
    assert cv2.countNonZero(result.mask) == 0


@pytest.mark.parametrize(
    "invalid_image",
    [None, np.array([], dtype=np.uint8), np.zeros((20, 20), dtype=np.uint8)],
)
def test_invalid_or_empty_input_raises_value_error(invalid_image: object) -> None:
    with pytest.raises(ValueError, match="non-empty BGR image"):
        color_detector.ColorDetector().detect(invalid_image)  # type: ignore[arg-type]
