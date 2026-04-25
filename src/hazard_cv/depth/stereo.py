"""Simple SGBM disparity for two-view or rectified stereo pairs."""

from __future__ import annotations

import cv2
import numpy as np


def suggest_num_disparities(image_width: int, cap: int = 1024) -> int:
    """Heuristic numDisparities (multiple of 16) for wide ETH3D DSLR frames."""
    v = min(cap, max(192, image_width // 5))
    return max(16, (int(v) // 16) * 16)


def _sgbm_create(
    min_disparity: int,
    num_disparities: int,
    block_size: int = 5,
) -> cv2.StereoSGBM:
    num_disparities = int(num_disparities)
    num_disparities = max(16, (num_disparities // 16) * 16)
    window_size = block_size if block_size % 2 == 1 else block_size + 1
    return cv2.StereoSGBM_create(
        minDisparity=int(min_disparity),
        numDisparities=num_disparities,
        blockSize=window_size,
        P1=8 * 1 * window_size**2,
        P2=32 * 1 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def sgbm_disparity_rectified(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    num_disparities: int = 128,
    min_disparity: int = 0,
    block_size: int = 5,
) -> np.ndarray:
    """SGBM on an already rectified / row-aligned stereo pair."""
    left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    stereo = _sgbm_create(min_disparity, num_disparities, block_size)
    disp = stereo.compute(left, right).astype(np.float32) / 16.0
    return disp


def sgbm_disparity(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    """Default ETH3D two-view settings (legacy wrapper)."""
    return sgbm_disparity_rectified(left_bgr, right_bgr, num_disparities=128, min_disparity=0)
