"""SDK deprojection of a selected center plus mask median depth."""

from dataclasses import dataclass
import importlib
import math


@dataclass(frozen=True)
class TargetPosition3D:
    valid: bool = False
    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    pixel_u: float | None = None
    pixel_v: float | None = None
    depth_m: float | None = None


def intrinsics_snapshot(intrinsics):
    """Serializable snapshot of the actual frame's SDK intrinsics."""
    if intrinsics is None:
        return None
    return dict(width=int(intrinsics.width), height=int(intrinsics.height),
                fx=float(intrinsics.fx), fy=float(intrinsics.fy),
                ppx=float(intrinsics.ppx), ppy=float(intrinsics.ppy),
                model=str(intrinsics.model), coeffs=[float(c) for c in intrinsics.coeffs])


def deproject_target(pixel_u, pixel_v, depth_m, intrinsics, *, deproject_fn=None):
    """Return meters in RealSense optical axes: right, down, forward.

    The production implementation delegates distortion handling to RealSense.
    deproject_fn permits hardware-independent tests using ideal intrinsics.
    """
    invalid = TargetPosition3D()
    try:
        u, v, depth = float(pixel_u), float(pixel_v), float(depth_m)
        info = intrinsics_snapshot(intrinsics)
        if info is None:
            return invalid
        numbers = [u, v, depth, info['fx'], info['fy'], info['ppx'], info['ppy'], *info['coeffs']]
        if not all(math.isfinite(n) for n in numbers):
            return invalid
        if (depth <= 0 or info['fx'] <= 0 or info['fy'] <= 0
                or not 0 <= u < info['width'] or not 0 <= v < info['height']):
            return invalid
        if deproject_fn is None:
            deproject_fn = importlib.import_module('pyrealsense2').rs2_deproject_pixel_to_point
        point = deproject_fn(intrinsics, [u, v], depth)
        x, y, z = map(float, point)
        if not all(math.isfinite(n) for n in (x, y, z)) or not math.isclose(z, depth, rel_tol=1e-5, abs_tol=1e-6):
            return invalid
        return TargetPosition3D(True, x, y, z, u, v, depth)
    except (TypeError, ValueError, AttributeError, ImportError, RuntimeError, OverflowError):
        return invalid
