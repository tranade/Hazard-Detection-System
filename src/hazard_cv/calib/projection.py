"""Pinhole ray construction (camera frame: x right, y down, z forward)."""

from __future__ import annotations

import numpy as np

from .colmap_model import CameraPinhole


def pixel_ray_direction(cam: CameraPinhole, u: float, v: float) -> np.ndarray:
    """Unit direction in camera frame for pixel (u, v), z > 0."""
    x = (u - cam.cx) / cam.fx
    y = (v - cam.cy) / cam.fy
    d = np.array([x, y, 1.0], dtype=np.float64)
    return d / np.linalg.norm(d)


def backproject_pinhole(cam: CameraPinhole, u: float, v: float, z: float) -> np.ndarray:
    """3D point in camera coordinates given pixel and depth Z (along optical axis)."""
    x = (u - cam.cx) * z / cam.fx
    y = (v - cam.cy) * z / cam.fy
    return np.array([x, y, z], dtype=np.float64)
