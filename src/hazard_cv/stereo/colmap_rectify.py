"""Rectified stereo from two COLMAP / ETH3D PINHOLE views (e.g. terrace multi-view)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from hazard_cv.calib.colmap_model import CameraPinhole, ImagePose, load_colmap_model
from hazard_cv.depth.stereo import sgbm_disparity_rectified, suggest_num_disparities


def resolve_eth3d_image_path(colmap_dir: Path, image_rel: str) -> Path:
    """ETH3D packs often store COLMAP under dslr_calibration_* and RGB under images/."""
    colmap_dir = Path(colmap_dir).resolve()
    rel = image_rel.replace("\\", "/").strip()
    candidates = [
        colmap_dir / rel,
        colmap_dir.parent / rel,
        colmap_dir.parent / "images" / rel,
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Could not locate image {rel!r} under {colmap_dir} (tried images/ subfolder).")


def _K(cam: CameraPinhole) -> np.ndarray:
    return np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], dtype=np.float64)


def relative_rt(R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform from camera-1 frame to camera-2: X2 = R @ X1 + T."""
    R = R2 @ R1.T
    T = (t2.reshape(3) - R @ t1.reshape(3)).reshape(3, 1)
    return R.astype(np.float64), T.astype(np.float64)


@dataclass
class StereoRectifiedResult:
    disparity: np.ndarray  # HxW float, rectified left
    points_3d: np.ndarray  # HxWx3 float, camera-1 (left) frame after rectify... actually Q gives in rectified coords
    Q: np.ndarray
    left_rect: np.ndarray
    right_rect: np.ndarray
    baseline_m: float


def stereo_disparity_from_colmap_pair(
    scene_dir: Path,
    left_rel: str,
    right_rel: str,
    num_disparities: int = 0,
    min_disparity: int = -64,
) -> StereoRectifiedResult:
    """Load two images by COLMAP-relative paths, rectify, run SGBM, reproject to 3D."""
    scene_dir = Path(scene_dir)
    cams, imgs = load_colmap_model(scene_dir, pinhole_only=True)
    if not cams:
        raise ValueError(
            f"No PINHOLE cameras in {scene_dir / 'cameras.txt'}. "
            "Use undistorted ETH3D packs for multi-view stereo."
        )

    def _find(name: str) -> ImagePose:
        key = name.replace("\\", "/").strip()
        for im in imgs:
            if im.name == key or im.name.endswith(key):
                return im
        preview = "\n".join(im.name for im in imgs[:12])
        raise FileNotFoundError(f"No image matching {key!r} in images.txt. Examples:\n{preview}")

    pL = _find(left_rel)
    pR = _find(right_rel)
    camL = cams[pL.camera_id]
    camR = cams[pR.camera_id]
    if camL.width != camR.width or camL.height != camR.height:
        raise ValueError("Left/right cameras must share resolution for this pipeline.")

    path_l = resolve_eth3d_image_path(scene_dir, pL.name)
    path_r = resolve_eth3d_image_path(scene_dir, pR.name)
    left = cv2.imread(str(path_l))
    right = cv2.imread(str(path_r))
    if left is None or right is None:
        raise FileNotFoundError(f"Missing images: {path_l} or {path_r}")
    h, w = left.shape[:2]
    if (w, h) != (camL.width, camL.height):
        raise ValueError(
            f"Image size {(w, h)} != cameras.txt {(camL.width, camL.height)}; "
            "resize images or use matching exports."
        )

    if num_disparities <= 0:
        num_disparities = suggest_num_disparities(w)

    K1 = _K(camL)
    K2 = _K(camR)
    d1 = np.zeros(5, dtype=np.float64)
    d2 = np.zeros(5, dtype=np.float64)
    R, T = relative_rt(pL.R_wc, pL.t_wc, pR.R_wc, pR.t_wc)
    baseline_m = float(np.linalg.norm(T))

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1,
        d1,
        K2,
        d2,
        (w, h),
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=-1,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K1, d1, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, d2, R2, P2, (w, h), cv2.CV_32FC1)
    left_r = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    right_r = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

    disp = sgbm_disparity_rectified(left_r, right_r, num_disparities=num_disparities, min_disparity=min_disparity)
    points_3d = cv2.reprojectImageTo3D(disp, Q).astype(np.float32)
    return StereoRectifiedResult(
        disparity=disp,
        points_3d=points_3d,
        Q=Q,
        left_rect=left_r,
        right_rect=right_r,
        baseline_m=baseline_m,
    )


def pick_best_baseline_pair(
    scene_dir: Path, max_images: int = 24
) -> Tuple[str, str, float]:
    """Choose two registered images (by order in images.txt) with largest stereo baseline."""
    scene_dir = Path(scene_dir)
    _, imgs = load_colmap_model(scene_dir, pinhole_only=True)
    if len(imgs) < 2:
        raise ValueError("Need at least two images in images.txt")
    use = imgs[:max_images]
    best = ("", "", 0.0)
    for i in range(len(use)):
        for j in range(i + 1, len(use)):
            R, T = relative_rt(use[i].R_wc, use[i].t_wc, use[j].R_wc, use[j].t_wc)
            b = float(np.linalg.norm(T))
            if b > best[2]:
                best = (use[i].name, use[j].name, b)
    if best[2] <= 0:
        raise ValueError("Could not find a pair with positive baseline.")
    return best


def pick_reasonable_baseline_pair(
    scene_dir: Path,
    max_images: int = 30,
    b_lo: float = 0.02,
    b_hi: float = 1.5,
) -> Tuple[str, str, float]:
    """Pick a pair whose camera translation is in a band that usually preserves stereo overlap."""
    scene_dir = Path(scene_dir)
    _, imgs = load_colmap_model(scene_dir, pinhole_only=True)
    if len(imgs) < 2:
        raise ValueError("Need at least two images in images.txt")
    use = imgs[:max_images]
    target = 0.5 * (b_lo + b_hi)
    best: Optional[Tuple[str, str, float, float]] = None  # left, right, b, score
    for i in range(len(use)):
        for j in range(i + 1, len(use)):
            _, T = relative_rt(use[i].R_wc, use[i].t_wc, use[j].R_wc, use[j].t_wc)
            b = float(np.linalg.norm(T))
            if b_lo <= b <= b_hi:
                score = abs(b - target)
                if best is None or score < best[3]:
                    best = (use[i].name, use[j].name, b, score)
    if best is not None:
        return best[0], best[1], best[2]
    return pick_best_baseline_pair(scene_dir, max_images=max_images)


def depth_m_at_uv(result: StereoRectifiedResult, u: float, v: float) -> Tuple[float, bool]:
    """Z depth in meters from reprojectImageTo3D at integer pixel (clamped)."""
    h, w = result.disparity.shape[:2]
    ui = int(round(np.clip(u, 0, w - 1)))
    vi = int(round(np.clip(v, 0, h - 1)))
    p = result.points_3d[vi, ui]
    if not np.all(np.isfinite(p)):
        return float("nan"), False
    z = float(p[2])
    if z <= 0 or z > 200.0:
        return float("nan"), False
    return z, True
