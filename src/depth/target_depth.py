"""Robust scalar depth extraction from an aligned target mask."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TargetDepthMeasurement:
    """Depth estimate and the quality evidence used to accept it."""

    valid: bool
    depth_m: float | None
    valid_ratio: float
    valid_count: int
    total_count: int
    depth_std_m: float | None


class TargetDepthExtractor:
    """Measure median target depth after mask erosion and quality filtering.

    Defaults are experimental baseline parameters and require validation for
    the actual target, range, camera settings, and environment.
    """

    def __init__(
        self,
        *,
        mask_erode_kernel: int = 3,
        mask_erode_iterations: int = 1,
        fallback_to_original_mask: bool = True,
        min_depth_m: float = 0.2,
        max_depth_m: float = 10.0,
        min_valid_pixels: int = 20,
        min_valid_ratio: float = 0.5,
        max_depth_std_m: float = 0.3,
    ) -> None:
        if mask_erode_kernel <= 0:
            raise ValueError("mask_erode_kernel must be positive")
        if mask_erode_iterations < 0:
            raise ValueError("mask_erode_iterations must be non-negative")
        if min_depth_m < 0 or max_depth_m <= min_depth_m:
            raise ValueError("depth range is invalid")
        if min_valid_pixels < 1:
            raise ValueError("min_valid_pixels must be positive")
        if not 0 <= min_valid_ratio <= 1:
            raise ValueError("min_valid_ratio must be in [0, 1]")
        if max_depth_std_m < 0:
            raise ValueError("max_depth_std_m must be non-negative")

        self.mask_erode_iterations = mask_erode_iterations
        self.fallback_to_original_mask = fallback_to_original_mask
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.min_valid_pixels = min_valid_pixels
        self.min_valid_ratio = float(min_valid_ratio)
        self.max_depth_std_m = float(max_depth_std_m)
        self._erode_kernel = np.ones(
            (mask_erode_kernel, mask_erode_kernel), dtype=np.uint8
        )

    @staticmethod
    def _invalid_empty(total_count: int = 0) -> TargetDepthMeasurement:
        return TargetDepthMeasurement(
            valid=False,
            depth_m=None,
            valid_ratio=0.0,
            valid_count=0,
            total_count=total_count,
            depth_std_m=None,
        )

    def _prepare_mask(self, target_mask: np.ndarray) -> np.ndarray:
        binary_mask = np.where(target_mask > 0, 255, 0).astype(np.uint8)
        original_count = int(cv2.countNonZero(binary_mask))
        if original_count == 0 or self.mask_erode_iterations == 0:
            return binary_mask

        eroded = cv2.erode(
            binary_mask,
            self._erode_kernel,
            iterations=self.mask_erode_iterations,
        )
        if cv2.countNonZero(eroded) > 0:
            return eroded
        if self.fallback_to_original_mask:
            return binary_mask
        return eroded

    def measure(
        self,
        aligned_depth_m: np.ndarray,
        target_mask: np.ndarray,
    ) -> TargetDepthMeasurement:
        """Return a quality-gated median depth from aligned target pixels."""
        if (
            not isinstance(aligned_depth_m, np.ndarray)
            or not isinstance(target_mask, np.ndarray)
            or aligned_depth_m.ndim != 2
            or target_mask.ndim != 2
        ):
            raise ValueError("aligned depth and target mask must be 2D arrays")
        if aligned_depth_m.shape != target_mask.shape:
            raise ValueError("aligned depth and target mask must have the same shape")

        measurement_mask = self._prepare_mask(target_mask)
        total_count = int(cv2.countNonZero(measurement_mask))
        if total_count == 0:
            return self._invalid_empty()

        masked_depths = aligned_depth_m[measurement_mask > 0]
        valid_values = masked_depths[
            np.isfinite(masked_depths)
            & (masked_depths != 0)
            & (masked_depths >= self.min_depth_m)
            & (masked_depths <= self.max_depth_m)
        ].astype(np.float64, copy=False)
        valid_count = int(valid_values.size)
        valid_ratio = valid_count / total_count
        if valid_count == 0:
            return TargetDepthMeasurement(
                valid=False,
                depth_m=None,
                valid_ratio=valid_ratio,
                valid_count=0,
                total_count=total_count,
                depth_std_m=None,
            )

        median_depth_m = float(np.median(valid_values))
        depth_std_m = float(np.std(valid_values))
        valid = (
            valid_count >= self.min_valid_pixels
            and valid_ratio >= self.min_valid_ratio
            and depth_std_m <= self.max_depth_std_m
        )
        return TargetDepthMeasurement(
            valid=valid,
            depth_m=median_depth_m if valid else None,
            valid_ratio=valid_ratio,
            valid_count=valid_count,
            total_count=total_count,
            depth_std_m=depth_std_m,
        )
