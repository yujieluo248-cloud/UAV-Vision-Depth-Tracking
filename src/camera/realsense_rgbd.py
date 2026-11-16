"""Optional Intel RealSense RGB and aligned-depth frame source."""

from dataclasses import dataclass
import importlib
import warnings
from typing import Any

import numpy as np


DEPENDENCY_ERROR = (
    "pyrealsense2 is required for D435i RGB-D capture. "
    "Install Intel RealSense SDK / pyrealsense2 first."
)


@dataclass(frozen=True)
class RGBDFrame:
    """One BGR color frame and SDK-aligned depth image in meters."""

    color_bgr: np.ndarray
    depth_m: np.ndarray
    timestamp_ms: float
    aligned_intrinsics: Any | None = None
    color_intrinsics: Any | None = None


class RealSenseRGBDSource:
    """Acquire D435i color and depth aligned into the color pixel grid."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        *,
        rs_module: Any | None = None,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("width, height, and fps must be positive")
        self.width = width
        self.height = height
        self.fps = fps
        self._rs_module = rs_module
        self._pipeline: Any | None = None
        self._aligner: Any | None = None
        self._depth_scale: float | None = None
        self._printed_intrinsics = False

    def _get_rs_module(self) -> Any:
        if self._rs_module is not None:
            return self._rs_module
        try:
            self._rs_module = importlib.import_module("pyrealsense2")
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(DEPENDENCY_ERROR) from exc
        return self._rs_module

    def start(self) -> None:
        """Start configurable color/depth streams and color alignment."""
        if self._pipeline is not None:
            return
        rs = self._get_rs_module()
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        config.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps,
        )
        try:
            profile = pipeline.start(config)
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = float(depth_sensor.get_depth_scale())
            if depth_scale <= 0:
                raise RuntimeError("RealSense depth scale must be positive")
            aligner = rs.align(rs.stream.color)
        except Exception as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise RuntimeError(f"Could not start D435i RGB-D stream: {exc}") from exc

        self._pipeline = pipeline
        self._aligner = aligner
        self._depth_scale = depth_scale
        self._printed_intrinsics = False

    def read(self) -> RGBDFrame | None:
        """Return aligned BGR/depth data, with depth converted to meters."""
        if (
            self._pipeline is None
            or self._aligner is None
            or self._depth_scale is None
        ):
            raise RuntimeError("RealSense RGB-D source is not started")

        frames = self._pipeline.wait_for_frames()
        aligned_frames = self._aligner.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        color_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
            raise RuntimeError("Aligned color frame is not a three-channel image")
        if depth_raw.ndim != 2:
            raise RuntimeError("Aligned depth frame is not a single-channel image")
        if color_bgr.shape[:2] != depth_raw.shape:
            raise RuntimeError(
                "Aligned color and depth frames must have the same dimensions"
            )

        depth_m = depth_raw.astype(np.float32) * self._depth_scale
        aligned_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        color_intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
        if (aligned_intrinsics.width, aligned_intrinsics.height) != (color_bgr.shape[1], color_bgr.shape[0]):
            raise RuntimeError('Aligned intrinsics dimensions do not match current pixel grid')
        if not self._printed_intrinsics:
            from src.geometry.deprojection import intrinsics_snapshot
            print('Aligned camera intrinsics:')
            for name, value in intrinsics_snapshot(aligned_intrinsics).items():
                print(f'{name} = {value}')
            mismatch = any(
                not np.isclose(getattr(aligned_intrinsics, key), getattr(color_intrinsics, key), rtol=1e-4, atol=1e-3)
                for key in ('width', 'height', 'fx', 'fy', 'ppx', 'ppy')
            )
            if mismatch or aligned_intrinsics.model != color_intrinsics.model or not np.allclose(aligned_intrinsics.coeffs, color_intrinsics.coeffs):
                warnings.warn('Aligned depth and color intrinsics differ; check current profiles and pixel-grid consistency.', RuntimeWarning)
            self._printed_intrinsics = True
        return RGBDFrame(
            color_bgr=color_bgr,
            depth_m=depth_m,
            timestamp_ms=float(color_frame.get_timestamp()),
            aligned_intrinsics=aligned_intrinsics,
            color_intrinsics=color_intrinsics,
        )

    def stop(self) -> None:
        """Stop the stream; safe before start and on repeated calls."""
        pipeline = self._pipeline
        self._pipeline = None
        self._aligner = None
        self._depth_scale = None
        if pipeline is not None:
            pipeline.stop()
