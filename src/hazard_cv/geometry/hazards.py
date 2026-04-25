"""Harris corners, local edge orientation, angle-based hazard levels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class CornerHazard:
    u: float
    v: float
    angle_deg: float
    score: float
    level: str  # "low" | "medium" | "high"


def non_max_suppression_grid(
    pts: np.ndarray, scores: np.ndarray, min_dist: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy NMS on 2D points by score."""
    if pts.size == 0:
        return pts, scores
    order = np.argsort(-scores)
    pts = pts[order]
    scores = scores[order]
    keep_pts: List[np.ndarray] = []
    keep_sc: List[float] = []
    for p, s in zip(pts, scores):
        if not keep_pts:
            keep_pts.append(p)
            keep_sc.append(float(s))
            continue
        d = np.hypot(p[0] - np.array([q[0] for q in keep_pts]), p[1] - np.array([q[1] for q in keep_pts]))
        if np.all(d >= min_dist):
            keep_pts.append(p)
            keep_sc.append(float(s))
    return np.array(keep_pts, dtype=np.float32), np.array(keep_sc, dtype=np.float32)


def _estimate_corner_angle(gray: np.ndarray, u: int, v: int, win: int = 21) -> float:
    """Approximate opening angle at corner using gradient directions in a window."""
    h, w = gray.shape
    r = win // 2
    x0, x1 = max(0, u - r), min(w, u + r + 1)
    y0, y1 = max(0, v - r), min(h, v + r + 1)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    if patch.size < 100:
        return 90.0
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mask = mag > np.percentile(mag, 70)
    ang = np.arctan2(gy[mask], gx[mask])
    if ang.size < 8:
        return 90.0
    ang = np.unwrap(ang)
    # Two dominant directions via histogram peaks (coarse)
    bins = np.linspace(-np.pi, np.pi, 37)
    hist, _ = np.histogram(ang, bins=bins)
    peaks = np.argsort(hist)[-4:]
    peak_angles = [(bins[i] + bins[i + 1]) / 2 for i in peaks if hist[i] > 0]
    if len(peak_angles) < 2:
        return 90.0
    peak_angles = sorted(peak_angles)
    diffs = []
    for i in range(len(peak_angles)):
        for j in range(i + 1, len(peak_angles)):
            da = abs(peak_angles[i] - peak_angles[j])
            da = min(da, 2 * np.pi - da)
            diffs.append(da)
    if not diffs:
        return 90.0
    sep = max(diffs)
    angle_deg = float(np.degrees(np.pi - sep))
    angle_deg = max(0.0, min(180.0, angle_deg))
    return angle_deg


def _angle_to_level(angle_deg: float, sharp_deg: float = 55.0, blunt_deg: float = 100.0) -> str:
    if angle_deg < sharp_deg:
        return "high"
    if angle_deg < blunt_deg:
        return "medium"
    return "low"


def detect_corner_hazards(
    bgr: np.ndarray,
    harris_block_size: int = 2,
    harris_ksize: int = 3,
    harris_k: float = 0.04,
    harris_thresh: float = 1e-3,
    nms_dist: float = 12.0,
    max_corners: int = 80,
) -> List[CornerHazard]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    h = cv2.cornerHarris(gray, harris_block_size, harris_ksize, harris_k)
    h = cv2.dilate(h, None)
    pts = np.argwhere(h > harris_thresh)
    if pts.size == 0:
        return []
    scores = h[pts[:, 0], pts[:, 1]].astype(np.float32)
    xy = np.stack([pts[:, 1].astype(np.float32), pts[:, 0].astype(np.float32)], axis=1)
    xy, scores = non_max_suppression_grid(xy, scores, min_dist=nms_dist)
    order = np.argsort(-scores)[:max_corners]
    out: List[CornerHazard] = []
    for (u, v), sc in zip(xy[order], scores[order]):
        ui, vi = int(round(u)), int(round(v))
        ang = _estimate_corner_angle(gray, ui, vi)
        level = _angle_to_level(ang)
        out.append(CornerHazard(u=float(u), v=float(v), angle_deg=ang, score=float(sc), level=level))
    return out


def match_label_level(pred_level: str, true_level: str) -> bool:
    order = {"low": 0, "medium": 1, "high": 2}
    return abs(order[pred_level] - order[true_level]) <= 1
