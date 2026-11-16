"""Baseline acquisition and temporal association for color candidates."""

from dataclasses import dataclass
import math
from typing import Optional, Sequence

from src.detection.color_detector import ColorCandidate


@dataclass(frozen=True)
class SelectionResult:
    """One selected candidate and the selection mode used."""

    candidate: Optional[ColorCandidate]
    mode: str


class CandidateSelector:
    """Select a plausible target using simple gates and one-frame history.

    All defaults are baseline starting values, not optimized final parameters.
    """

    def __init__(
        self,
        *,
        max_center_jump_ratio: float = 0.25,
        max_area_change_ratio: float = 4.0,
        max_bbox_area_ratio: float = 0.5,
        min_fill_ratio: float = 0.2,
        min_aspect_ratio: float = 0.2,
        max_aspect_ratio: float = 5.0,
        position_weight: float = 0.7,
        area_weight: float = 0.25,
        shape_weight: float = 0.05,
    ) -> None:
        if not 0 < max_center_jump_ratio <= 1:
            raise ValueError("max_center_jump_ratio must be in (0, 1]")
        if max_area_change_ratio < 1:
            raise ValueError("max_area_change_ratio must be at least 1")
        if not 0 < max_bbox_area_ratio <= 1:
            raise ValueError("max_bbox_area_ratio must be in (0, 1]")
        if not 0 <= min_fill_ratio <= 1:
            raise ValueError("min_fill_ratio must be in [0, 1]")
        if min_aspect_ratio <= 0 or max_aspect_ratio < min_aspect_ratio:
            raise ValueError("aspect ratio limits are invalid")

        self.max_center_jump_ratio = max_center_jump_ratio
        self.max_area_change_ratio = max_area_change_ratio
        self.max_bbox_area_ratio = max_bbox_area_ratio
        self.min_fill_ratio = min_fill_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.position_weight = position_weight
        self.area_weight = area_weight
        self.shape_weight = shape_weight

    def _passes_shape_gates(
        self,
        candidate: ColorCandidate,
        image_area: float,
    ) -> bool:
        _x, _y, width, height = candidate.bbox
        bbox_area_ratio = (width * height) / image_area
        return (
            bbox_area_ratio <= self.max_bbox_area_ratio
            and candidate.fill_ratio >= self.min_fill_ratio
            and self.min_aspect_ratio
            <= candidate.aspect_ratio
            <= self.max_aspect_ratio
        )

    def select(
        self,
        candidates: Sequence[ColorCandidate],
        image_shape: tuple[int, ...],
        *,
        previous_center: tuple[float, float] | None = None,
        previous_area: float | None = None,
    ) -> SelectionResult:
        """Select by largest-valid acquisition or gated temporal association."""
        if len(image_shape) < 2 or image_shape[0] <= 0 or image_shape[1] <= 0:
            raise ValueError("image_shape must contain positive height and width")
        height, width = image_shape[:2]
        image_area = float(height * width)
        valid = [
            candidate
            for candidate in candidates
            if self._passes_shape_gates(candidate, image_area)
        ]

        has_history = (
            previous_center is not None
            and previous_area is not None
            and previous_area > 0
        )
        if not has_history:
            if not valid:
                return SelectionResult(None, "NO_TARGET")
            return SelectionResult(
                max(valid, key=lambda candidate: candidate.area), "ACQUIRE"
            )

        assert previous_center is not None
        assert previous_area is not None
        image_diagonal = math.hypot(width, height)
        scored: list[tuple[float, ColorCandidate]] = []
        for candidate in valid:
            distance_ratio = math.dist(candidate.center, previous_center) / image_diagonal
            if distance_ratio > self.max_center_jump_ratio:
                continue
            area_change_ratio = max(
                candidate.area / previous_area,
                previous_area / candidate.area,
            )
            if area_change_ratio > self.max_area_change_ratio:
                continue
            position_penalty = distance_ratio / self.max_center_jump_ratio
            if self.max_area_change_ratio == 1:
                area_penalty = 0.0
            else:
                area_penalty = abs(
                    math.log(candidate.area / previous_area)
                ) / math.log(self.max_area_change_ratio)
            max_shape_penalty = (1.0 - self.min_fill_ratio) + max(
                abs(math.log(self.min_aspect_ratio)),
                abs(math.log(self.max_aspect_ratio)),
            )
            if max_shape_penalty == 0:
                shape_penalty = 0.0
            else:
                shape_penalty = (
                    (1.0 - candidate.fill_ratio)
                    + abs(math.log(candidate.aspect_ratio))
                ) / max_shape_penalty
            score = (
                self.position_weight * position_penalty
                + self.area_weight * area_penalty
                + self.shape_weight * shape_penalty
            )
            scored.append((score, candidate))

        if not scored:
            return SelectionResult(None, "NO_TARGET")
        return SelectionResult(
            min(scored, key=lambda item: item[0])[1],
            "TRACK_ASSOCIATION",
        )
