import numpy as np

from src.detection.candidate_selector import CandidateSelector
from src.detection.color_detector import ColorCandidate


def candidate(
    *,
    center: tuple[float, float],
    area: float,
    bbox: tuple[int, int, int, int],
    fill_ratio: float = 0.8,
) -> ColorCandidate:
    contour = np.array(
        [[[bbox[0], bbox[1]]], [[bbox[0] + bbox[2] - 1, bbox[1]]]],
        dtype=np.int32,
    )
    return ColorCandidate(
        contour=contour,
        bbox=bbox,
        center=center,
        area=area,
        aspect_ratio=bbox[2] / bbox[3],
        fill_ratio=fill_ratio,
    )


def test_acquisition_selects_largest_valid_candidate() -> None:
    small = candidate(center=(50.0, 50.0), area=900.0, bbox=(35, 35, 31, 31))
    large = candidate(
        center=(180.0, 120.0), area=3500.0, bbox=(145, 85, 71, 71)
    )

    result = CandidateSelector().select([small, large], (240, 320))

    assert result.candidate is large
    assert result.mode == "ACQUIRE"


def test_history_keeps_near_target_instead_of_larger_far_distractor() -> None:
    target = candidate(
        center=(58.0, 54.0), area=950.0, bbox=(43, 39, 31, 31)
    )
    distractor = candidate(
        center=(250.0, 170.0), area=7000.0, bbox=(200, 120, 101, 101)
    )

    result = CandidateSelector().select(
        [distractor, target],
        (240, 320),
        previous_center=(50.0, 50.0),
        previous_area=900.0,
    )

    assert result.candidate is target
    assert result.mode == "TRACK_ASSOCIATION"


def test_score_prefers_near_candidate_when_both_candidates_pass_gates() -> None:
    near_changed_area = candidate(
        center=(51.0, 51.0), area=400.0, bbox=(41, 41, 20, 20), fill_ratio=1.0
    )
    farther_same_area = candidate(
        center=(100.0, 50.0),
        area=1024.0,
        bbox=(84, 34, 32, 32),
        fill_ratio=1.0,
    )

    result = CandidateSelector().select(
        [farther_same_area, near_changed_area],
        (240, 320),
        previous_center=(50.0, 50.0),
        previous_area=1024.0,
    )

    assert result.candidate is near_changed_area
    assert result.mode == "TRACK_ASSOCIATION"


def test_far_jump_is_rejected_when_history_exists() -> None:
    far_candidate = candidate(
        center=(285.0, 205.0), area=1100.0, bbox=(265, 185, 41, 41)
    )

    result = CandidateSelector(max_center_jump_ratio=0.2).select(
        [far_candidate],
        (240, 320),
        previous_center=(30.0, 30.0),
        previous_area=1000.0,
    )

    assert result.candidate is None
    assert result.mode == "NO_TARGET"


def test_implausible_area_change_is_rejected_when_position_is_near() -> None:
    changed_area = candidate(
        center=(52.0, 51.0), area=5000.0, bbox=(17, 16, 71, 71)
    )

    result = CandidateSelector(max_area_change_ratio=4.0).select(
        [changed_area],
        (240, 320),
        previous_center=(50.0, 50.0),
        previous_area=1000.0,
    )

    assert result.candidate is None
    assert result.mode == "NO_TARGET"


def test_shape_gates_reject_screen_filling_and_sparse_regions() -> None:
    screen_filling = candidate(
        center=(160.0, 120.0), area=50000.0, bbox=(0, 0, 300, 220)
    )
    sparse = candidate(
        center=(80.0, 80.0),
        area=200.0,
        bbox=(50, 50, 61, 61),
        fill_ratio=0.05,
    )

    result = CandidateSelector().select(
        [screen_filling, sparse], (240, 320)
    )

    assert result.candidate is None
    assert result.mode == "NO_TARGET"
