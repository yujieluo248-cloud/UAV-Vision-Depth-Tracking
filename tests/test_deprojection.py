from types import SimpleNamespace

import numpy as np
import pytest

from src.geometry.deprojection import deproject_target


def intrinsics(**overrides):
    values = dict(width=640, height=480, fx=600., fy=600., ppx=320., ppy=240., model='none', coeffs=[0.] * 5)
    values.update(overrides)
    return SimpleNamespace(**values)


def pinhole(intr, pixel, depth):
    return [(pixel[0] - intr.ppx) * depth / intr.fx,
            (pixel[1] - intr.ppy) * depth / intr.fy, depth]


@pytest.mark.parametrize('pixel, xyz', [
    ((320, 240), (0, 0, 2)), ((380, 240), (.2, 0, 2)),
    ((260, 240), (-.2, 0, 2)), ((320, 300), (0, .2, 2)),
    ((320, 180), (0, -.2, 2)),
])
def test_optical_coordinates(pixel, xyz):
    result = deproject_target(*pixel, 2., intrinsics(), deproject_fn=pinhole)
    assert result.valid
    assert (result.x_m, result.y_m, result.z_m) == pytest.approx(xyz)
    assert result.z_m == result.depth_m


@pytest.mark.parametrize('u,v,depth', [
    (320, 240, 0), (320, 240, -1), (320, 240, np.nan),
    (320, 240, np.inf), (np.nan, 240, 2), (320, np.inf, 2),
    (-1, 240, 2), (640, 240, 2), (320, 480, 2),
    (None, None, 2), (320, 240, None),
])
def test_invalid_inputs_never_produce_xyz(u, v, depth):
    result = deproject_target(u, v, depth, intrinsics(), deproject_fn=pinhole)
    assert not result.valid
    assert result.x_m is result.y_m is result.z_m is None


@pytest.mark.parametrize('intr', [None, intrinsics(fx=0), intrinsics(fy=np.nan), intrinsics(coeffs=[np.inf]*5)])
def test_invalid_intrinsics(intr):
    assert not deproject_target(320, 240, 2, intr, deproject_fn=pinhole).valid


def test_sdk_output_must_be_finite_and_z_consistent():
    assert not deproject_target(320, 240, 2, intrinsics(), deproject_fn=lambda *args: [0, 0, 4]).valid
    assert not deproject_target(320, 240, 2, intrinsics(), deproject_fn=lambda *args: [np.nan, 0, 2]).valid


def test_production_path_calls_sdk_helper_with_original_intrinsics(monkeypatch):
    from src.geometry import deprojection
    intr = intrinsics(model='sdk-distortion', coeffs=[.1, .2, 0., 0., 0.])
    calls = []
    def sdk_helper(actual, pixel, depth):
        calls.append((actual, pixel, depth))
        return [.3, -.2, depth]
    monkeypatch.setattr(deprojection.importlib, 'import_module',
                        lambda name: SimpleNamespace(rs2_deproject_pixel_to_point=sdk_helper))
    result = deproject_target(380, 240, 2, intr)
    assert result.valid
    assert calls == [(intr, [380., 240.], 2.)]
    assert result.x_m == .3
