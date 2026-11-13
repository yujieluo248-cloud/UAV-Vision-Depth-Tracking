"""OpenCV-based HSV color target detection baseline."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]
Point = Tuple[int, int]
HSVRange = Tuple[int, int, int]


@dataclass
class ColorDetection:
    """Result returned by :class:`ColorDetector`."""

    detected: bool
    bbox: Optional[BBox]
    center: Optional[Point]
    area: float
    mask: np.ndarray


class ColorDetector:
    """Detect the largest sufficiently large region in an HSV color range.

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

    def detect(self, image_bgr: np.ndarray) -> ColorDetection:
        """Detect an HSV-colored target in a non-empty three-channel image."""
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.size == 0
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
        ):
            raise ValueError("image_bgr must be a non-empty BGR image")

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
        valid_contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= self.min_area
        ]

        target_mask = np.zeros_like(cleaned_mask)
        if not valid_contours:
            return ColorDetection(False, None, None, 0.0, target_mask)

        target_contour = max(valid_contours, key=cv2.contourArea)
        area = float(cv2.contourArea(target_contour))
        x, y, width, height = cv2.boundingRect(target_contour)
        moments = cv2.moments(target_contour)
        center = (
            int(round(moments["m10"] / moments["m00"])),
            int(round(moments["m01"] / moments["m00"])),
        )
        cv2.drawContours(
            target_mask, [target_contour], contourIdx=-1, color=255, thickness=-1
        )

        return ColorDetection(
            detected=True,
            bbox=(x, y, width, height),
            center=center,
            area=area,
            mask=target_mask,
        )
