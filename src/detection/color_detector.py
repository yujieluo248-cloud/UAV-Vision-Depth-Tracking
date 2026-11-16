"""OpenCV-based HSV color candidate generation."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]
Point = Tuple[int, int]
HSVRange = Tuple[int, int, int]


@dataclass
class ColorCandidate:
    """Geometry and contour for one valid HSV connected component."""

    contour: np.ndarray
    bbox: BBox
    center: tuple[float, float]
    area: float
    aspect_ratio: float
    fill_ratio: float


@dataclass
class ColorCandidateResult:
    """Full cleaned color segmentation and its valid candidates."""

    color_mask: np.ndarray
    candidates: list[ColorCandidate]


@dataclass
class ColorDetection:
    """Selected target result returned by :class:`ColorDetector`."""

    detected: bool
    bbox: Optional[BBox]
    center: Optional[Point]
    area: float
    mask: np.ndarray
    color_mask: np.ndarray | None = None
    candidates: tuple[ColorCandidate, ...] = field(default_factory=tuple)
    selection_mode: str = "NO_TARGET"

    def __post_init__(self) -> None:
        if self.color_mask is None:
            self.color_mask = self.mask

    @property
    def target_mask(self) -> np.ndarray:
        """Explicit name for the selected-only legacy mask field."""
        return self.mask


class ColorDetector:
    """Generate HSV candidates and select a baseline acquisition target.

    The default orange thresholds are baseline starting values. They must be
    calibrated for the actual target material and lighting conditions.
    """

    def __init__(
        self,
        lower_hsv: HSVRange = (5, 100, 100),
        upper_hsv: HSVRange = (25, 255, 255),
        min_area: float = 500.0,
        morph_kernel_size: int = 5,
    ) -> None:
        if morph_kernel_size <= 0:
            raise ValueError("morph_kernel_size must be positive")
        if min_area < 0:
            raise ValueError("min_area must be non-negative")

        self.set_hsv_range(lower_hsv, upper_hsv)
        self.min_area = float(min_area)
        self.kernel = np.ones(
            (morph_kernel_size, morph_kernel_size), dtype=np.uint8
        )

    def set_hsv_range(
        self, lower_hsv: HSVRange, upper_hsv: HSVRange
    ) -> None:
        """Update the HSV thresholds used by subsequent detections."""
        self.lower_hsv = np.asarray(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.asarray(upper_hsv, dtype=np.uint8)

    @staticmethod
    def _validate_image(image_bgr: np.ndarray) -> None:
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.size == 0
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
        ):
            raise ValueError("image_bgr must be a non-empty BGR image")

    def detect_candidates(self, image_bgr: np.ndarray) -> ColorCandidateResult:
        """Return every sufficiently large connected component and color mask."""
        self._validate_image(image_bgr)

        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        threshold_mask = cv2.inRange(
            image_hsv, self.lower_hsv, self.upper_hsv
        )
        opened_mask = cv2.morphologyEx(
            threshold_mask, cv2.MORPH_OPEN, self.kernel
        )
        cleaned_mask = cv2.morphologyEx(
            opened_mask, cv2.MORPH_CLOSE, self.kernel
        )

        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if moments["m00"]:
                center = (
                    float(moments["m10"] / moments["m00"]),
                    float(moments["m01"] / moments["m00"]),
                )
            else:
                center = (x + (width - 1) / 2, y + (height - 1) / 2)
            candidates.append(
                ColorCandidate(
                    contour=contour,
                    bbox=(x, y, width, height),
                    center=center,
                    area=area,
                    aspect_ratio=width / height,
                    fill_ratio=area / (width * height),
                )
            )

        candidates.sort(key=lambda candidate: candidate.area, reverse=True)
        return ColorCandidateResult(color_mask=cleaned_mask, candidates=candidates)

    def detect(self, image_bgr: np.ndarray) -> ColorDetection:
        """Select one acquisition target while preserving the V1 API."""
        from src.detection.candidate_selector import CandidateSelector

        candidate_result = self.detect_candidates(image_bgr)
        selection = CandidateSelector().select(
            candidate_result.candidates, image_bgr.shape
        )
        return self.build_detection(
            candidate_result, selection.candidate, selection.mode
        )

    def build_detection(
        self,
        candidate_result: ColorCandidateResult,
        selected_candidate: ColorCandidate | None,
        selection_mode: str,
    ) -> ColorDetection:
        """Build a selected-only target result from generated candidates."""
        target_mask = np.zeros_like(candidate_result.color_mask)
        if selected_candidate is None:
            return ColorDetection(
                detected=False,
                bbox=None,
                center=None,
                area=0.0,
                mask=target_mask,
                color_mask=candidate_result.color_mask,
                candidates=tuple(candidate_result.candidates),
                selection_mode=selection_mode,
            )

        target = selected_candidate
        cv2.drawContours(
            target_mask, [target.contour], contourIdx=-1, color=255, thickness=-1
        )

        return ColorDetection(
            detected=True,
            bbox=target.bbox,
            center=(int(round(target.center[0])), int(round(target.center[1]))),
            area=target.area,
            mask=target_mask,
            color_mask=candidate_result.color_mask,
            candidates=tuple(candidate_result.candidates),
            selection_mode=selection_mode,
        )
