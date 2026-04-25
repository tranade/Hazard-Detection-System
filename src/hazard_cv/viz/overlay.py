"""Debug visualizations for disparity and corner hazards."""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from hazard_cv.geometry.hazards import CornerHazard


def colorize_disparity(disp: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Map disparity to BGR for saving (invalid -> black)."""
    d = disp.astype(np.float32)
    if valid_mask is None:
        valid = np.isfinite(d) & (d > 0)
    else:
        valid = valid_mask & np.isfinite(d) & (d > 0)
    out = np.zeros((*d.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return out
    lo, hi = np.percentile(d[valid], 5), np.percentile(d[valid], 95)
    if hi <= lo:
        hi = lo + 1.0
    g = np.clip((d - lo) / (hi - lo), 0, 1)
    g = (g * 255).astype(np.uint8)
    cm = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
    out[valid] = cm[valid]
    return out


def draw_corner_hazards(bgr: np.ndarray, corners: List[CornerHazard]) -> np.ndarray:
    vis = bgr.copy()
    colors = {"high": (0, 0, 255), "medium": (0, 165, 255), "low": (0, 200, 0)}
    for c in corners:
        col = colors.get(c.level, (200, 200, 200))
        u, v = int(round(c.u)), int(round(c.v))
        cv2.circle(vis, (u, v), 6, col, 2)
        cv2.putText(
            vis,
            f"{c.level[0].upper()}{int(c.angle_deg)}",
            (u + 8, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )
    return vis


def save_image(path: Path, bgr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)
