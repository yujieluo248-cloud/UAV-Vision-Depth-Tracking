import importlib
from types import SimpleNamespace

import numpy as np
import pytest


class FakeColorFrame:
    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def get_data(self) -> np.ndarray:
        return self._data


class FakeFrames:
    def __init__(self, color_frame: FakeColorFrame | None) -> None:
        self._color_frame = color_frame

    def get_color_frame(self) -> FakeColorFrame | None:
        return self._color_frame


class FakePipeline:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.started_with = None
        self.stopped = False

    def start(self, config: object) -> None:
        self.started_with = config

    def wait_for_frames(self) -> FakeFrames:
        return FakeFrames(FakeColorFrame(self.frame))

    def stop(self) -> None:
        self.stopped = True


class FakeConfig:
    def __init__(self) -> None:
        self.enabled = []

    def enable_stream(self, *args: object) -> None:
        self.enabled.append(args)


def make_fake_rs(frame: np.ndarray) -> tuple[object, FakePipeline, FakeConfig]:
    pipeline = FakePipeline(frame)
    config = FakeConfig()
    fake_rs = SimpleNamespace(
        pipeline=lambda: pipeline,
        config=lambda: config,
        stream=SimpleNamespace(color="color"),
        format=SimpleNamespace(bgr8="bgr8"),
    )
    return fake_rs, pipeline, config


def test_module_import_does_not_require_pyrealsense2() -> None:
    module = importlib.import_module("src.camera.realsense_rgb")

    assert hasattr(module, "RealSenseRGBSource")


def test_start_reports_clear_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("src.camera.realsense_rgb")

    def missing_module(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(module.importlib, "import_module", missing_module)
    source = module.RealSenseRGBSource()

    with pytest.raises(
        RuntimeError,
        match="pyrealsense2 is required for D435i RGB capture",
    ):
        source.start()


def test_fake_sdk_reads_bgr_frame_and_configures_only_color() -> None:
    module = importlib.import_module("src.camera.realsense_rgb")
    expected = np.zeros((480, 640, 3), dtype=np.uint8)
    expected[..., 0] = 123
    fake_rs, pipeline, config = make_fake_rs(expected)
    source = module.RealSenseRGBSource(rs_module=fake_rs)

    source.start()
    actual = source.read()

    assert actual is not None
    assert actual.shape == (480, 640, 3)
    assert np.array_equal(actual, expected)
    assert config.enabled == [("color", 640, 480, "bgr8", 30)]
    assert pipeline.started_with is config


def test_stop_is_safe_and_idempotent() -> None:
    module = importlib.import_module("src.camera.realsense_rgb")
    fake_rs, pipeline, _ = make_fake_rs(np.zeros((2, 3, 3), dtype=np.uint8))
    source = module.RealSenseRGBSource(width=3, height=2, rs_module=fake_rs)

    source.stop()
    source.start()
    source.stop()
    source.stop()

    assert pipeline.stopped


def test_failed_start_stops_partial_pipeline() -> None:
    module = importlib.import_module("src.camera.realsense_rgb")
    fake_rs, pipeline, _ = make_fake_rs(np.zeros((2, 3, 3), dtype=np.uint8))

    def fail_start(_config: object) -> None:
        raise RuntimeError("camera unavailable")

    pipeline.start = fail_start  # type: ignore[method-assign]
    source = module.RealSenseRGBSource(rs_module=fake_rs)

    with pytest.raises(RuntimeError, match="Could not start D435i RGB stream"):
        source.start()

    assert pipeline.stopped
