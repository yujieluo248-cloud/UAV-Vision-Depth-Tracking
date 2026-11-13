"""Optional Intel RealSense RGB-only frame source."""

import importlib
from typing import Any

import numpy as np


DEPENDENCY_ERROR = (
    "pyrealsense2 is required for D435i RGB capture. "
    "Install Intel RealSense SDK / pyrealsense2 first."
)


class RealSenseRGBSource:
    """Acquire OpenCV-ready BGR frames from the RealSense color stream only."""

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

    def _get_rs_module(self) -> Any:
        if self._rs_module is not None:
            return self._rs_module
        try:
            self._rs_module = importlib.import_module("pyrealsense2")
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(DEPENDENCY_ERROR) from exc
        return self._rs_module

    def start(self) -> None:
        """Start an RGB-only stream in SDK-native BGR8 format."""
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
        try:
            pipeline.start(config)
        except Exception as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise RuntimeError(f"Could not start D435i RGB stream: {exc}") from exc
        self._pipeline = pipeline

    def read(self) -> np.ndarray | None:
        """Wait for and return one BGR frame, or None if color is unavailable."""
        if self._pipeline is None:
            raise RuntimeError("RealSense RGB source is not started")
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        image_bgr = np.asanyarray(color_frame.get_data())
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise RuntimeError("RealSense color frame is not a three-channel image")
        return image_bgr

    def stop(self) -> None:
        """Stop the stream; safe to call repeatedly or before start."""
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            pipeline.stop()
