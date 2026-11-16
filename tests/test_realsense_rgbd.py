import importlib
from types import SimpleNamespace

import numpy as np
import pytest


class FakeFrame:
    def __init__(self, data: np.ndarray, timestamp_ms: float = 0.0) -> None:
        self._data = data
        self._timestamp_ms = timestamp_ms
        height, width = data.shape[:2]
        intr = SimpleNamespace(width=width, height=height, fx=600., fy=600., ppx=width/2, ppy=height/2, model='none', coeffs=[0.]*5)
        self.profile = SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intr))

    def get_data(self) -> np.ndarray:
        return self._data

    def get_timestamp(self) -> float:
        return self._timestamp_ms


class FakeAlignedFrames:
    def __init__(
        self,
        color_frame: FakeFrame | None,
        depth_frame: FakeFrame | None,
    ) -> None:
        self._color_frame = color_frame
        self._depth_frame = depth_frame

    def get_color_frame(self) -> FakeFrame | None:
        return self._color_frame

    def get_depth_frame(self) -> FakeFrame | None:
        return self._depth_frame


class FakeAlign:
    def __init__(self, aligned_frames: FakeAlignedFrames) -> None:
        self.aligned_frames = aligned_frames
        self.processed = []

    def process(self, frames: object) -> FakeAlignedFrames:
        self.processed.append(frames)
        return self.aligned_frames


class FakeConfig:
    def __init__(self) -> None:
        self.enabled = []

    def enable_stream(self, *args: object) -> None:
        self.enabled.append(args)


class FakeDepthSensor:
    def __init__(self, depth_scale: float) -> None:
        self.depth_scale = depth_scale

    def get_depth_scale(self) -> float:
        return self.depth_scale


class FakeProfile:
    def __init__(self, depth_scale: float) -> None:
        sensor = FakeDepthSensor(depth_scale)
        self._device = SimpleNamespace(first_depth_sensor=lambda: sensor)

    def get_device(self) -> object:
        return self._device


class FakePipeline:
    def __init__(self, raw_frames: object, depth_scale: float) -> None:
        self.raw_frames = raw_frames
        self.profile = FakeProfile(depth_scale)
        self.started_with = None
        self.stopped = False

    def start(self, config: object) -> FakeProfile:
        self.started_with = config
        return self.profile

    def wait_for_frames(self) -> object:
        return self.raw_frames

    def stop(self) -> None:
        self.stopped = True


def make_fake_rs(
    color: np.ndarray,
    depth_raw: np.ndarray,
    *,
    depth_scale: float = 0.001,
    timestamp_ms: float = 1234.5,
) -> tuple[object, FakePipeline, FakeConfig, FakeAlign, list[object]]:
    raw_frames = object()
    aligned_frames = FakeAlignedFrames(
        FakeFrame(color, timestamp_ms),
        FakeFrame(depth_raw),
    )
    pipeline = FakePipeline(raw_frames, depth_scale)
    config = FakeConfig()
    aligner = FakeAlign(aligned_frames)
    align_targets = []

    def create_aligner(target: object) -> FakeAlign:
        align_targets.append(target)
        return aligner

    fake_rs = SimpleNamespace(
        pipeline=lambda: pipeline,
        config=lambda: config,
        align=create_aligner,
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(bgr8="bgr8", z16="z16"),
    )
    return fake_rs, pipeline, config, aligner, align_targets


def test_module_import_does_not_require_pyrealsense2() -> None:
    module = importlib.import_module("src.camera.realsense_rgbd")

    assert hasattr(module, "RealSenseRGBDSource")
    assert hasattr(module, "RGBDFrame")


def test_intrinsics_are_frame_profiles_printed_once_and_mismatch_warns(capsys):
    module = importlib.import_module('src.camera.realsense_rgbd')
    fake, pipeline, config, aligner, targets = make_fake_rs(
        np.zeros((2, 3, 3), np.uint8), np.ones((2, 3), np.uint16))
    aligned = aligner.aligned_frames.get_depth_frame().profile.as_video_stream_profile().get_intrinsics()
    color = aligner.aligned_frames.get_color_frame().profile.as_video_stream_profile().get_intrinsics()
    aligned.fx = 601.
    source = module.RealSenseRGBDSource(rs_module=fake)
    source.start()
    with pytest.warns(RuntimeWarning, match='intrinsics differ'):
        frame = source.read()
    source.read()
    source.stop()
    assert frame.aligned_intrinsics is aligned
    assert frame.color_intrinsics is color
    assert capsys.readouterr().out.count('Aligned camera intrinsics:') == 1


def test_start_configures_color_depth_and_alignment_then_reads_meters() -> None:
    module = importlib.import_module("src.camera.realsense_rgbd")
    color = np.zeros((2, 3, 3), dtype=np.uint8)
    color[..., 1] = 77
    depth_raw = np.array([[1000, 1500, 0], [2000, 2500, 3000]], dtype=np.uint16)
    fake_rs, pipeline, config, aligner, align_targets = make_fake_rs(
        color, depth_raw
    )
    source = module.RealSenseRGBDSource(
        width=3, height=2, fps=30, rs_module=fake_rs
    )

    source.start()
    frame = source.read()

    assert frame is not None
    assert config.enabled == [
        ("color", 3, 2, "bgr8", 30),
        ("depth", 3, 2, "z16", 30),
    ]
    assert pipeline.started_with is config
    assert align_targets == ["color"]
    assert aligner.processed == [pipeline.raw_frames]
    assert np.array_equal(frame.color_bgr, color)
    assert frame.color_bgr.shape[:2] == frame.depth_m.shape
    assert frame.depth_m.dtype == np.float32
    assert frame.depth_m == pytest.approx(depth_raw.astype(np.float32) * 0.001)
    assert frame.timestamp_ms == pytest.approx(1234.5)


def test_missing_aligned_frame_returns_none() -> None:
    module = importlib.import_module("src.camera.realsense_rgbd")
    color = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.zeros((2, 3), dtype=np.uint16)
    fake_rs, _pipeline, _config, aligner, _targets = make_fake_rs(color, depth)
    aligner.aligned_frames = FakeAlignedFrames(FakeFrame(color), None)
    source = module.RealSenseRGBDSource(
        width=3, height=2, rs_module=fake_rs
    )
    source.start()

    assert source.read() is None


def test_mismatched_aligned_shapes_raise_clear_error() -> None:
    module = importlib.import_module("src.camera.realsense_rgbd")
    color = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.zeros((4, 5), dtype=np.uint16)
    fake_rs, _pipeline, _config, _aligner, _targets = make_fake_rs(color, depth)
    source = module.RealSenseRGBDSource(rs_module=fake_rs)
    source.start()

    with pytest.raises(RuntimeError, match="same dimensions"):
        source.read()


def test_start_failure_stops_partial_pipeline() -> None:
    module = importlib.import_module("src.camera.realsense_rgbd")
    color = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.zeros((2, 3), dtype=np.uint16)
    fake_rs, pipeline, _config, _aligner, _targets = make_fake_rs(color, depth)

    def fail_start(_config: object) -> object:
        raise RuntimeError("device unavailable")

    pipeline.start = fail_start  # type: ignore[method-assign]
    source = module.RealSenseRGBDSource(rs_module=fake_rs)

    with pytest.raises(RuntimeError, match="Could not start D435i RGB-D stream"):
        source.start()

    assert pipeline.stopped


def test_stop_is_safe_and_idempotent() -> None:
    module = importlib.import_module("src.camera.realsense_rgbd")
    color = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.zeros((2, 3), dtype=np.uint16)
    fake_rs, pipeline, _config, _aligner, _targets = make_fake_rs(color, depth)
    source = module.RealSenseRGBDSource(rs_module=fake_rs)

    source.stop()
    source.start()
    source.stop()
    source.stop()

    assert pipeline.stopped
