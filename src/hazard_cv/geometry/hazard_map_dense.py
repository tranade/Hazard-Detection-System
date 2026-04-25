"""Dense hazard heatmap from sparse corner detections (Gaussian splat)."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from hazard_cv.geometry.hazards import CornerHazard

_LEVEL_WEIGHT = {"low": 0.35, "medium": 0.65, "high": 1.0}


def build_hazard_heatmap(
    height: int,
    width: int,
    corners: List[CornerHazard],
    sigma_px: float = 45.0,
    corner_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return float32 HxW in [0,1]: splatted hazard with Gaussian falloff.

    Weighting is proximity-first:
    - disparity proximity drives most of the hazard weight
    - corner severity provides an additional boost
    This enforces: near objects > far objects, and near+sharp corners are strongest.
    """
    h = np.zeros((height, width), dtype=np.float32)
    if not corners or sigma_px <= 0:
        return h
    sig2 = 2.0 * float(sigma_px) ** 2
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    if corner_weights is None:
        weights = np.ones((len(corners),), dtype=np.float32)
    else:
        weights = np.asarray(corner_weights, dtype=np.float32)
        if weights.shape[0] != len(corners):
            raise ValueError("corner_weights must have same length as corners")
        weights = np.clip(weights, 0.0, 1.0)
    for c, prox_w in zip(corners, weights):
        level_w = _LEVEL_WEIGHT.get(c.level, 0.3)
        prox = float(np.clip(prox_w, 0.0, 1.0))
        # Proximity dominates; corner level amplifies the close hazards.
        base = 0.85 * prox + 0.15 * level_w
        synergy = 1.0 + 0.45 * prox * level_w
        w0 = float(np.clip(base * synergy, 0.0, 1.5))
        dy = yy - c.v
        dx = xx - c.u
        g = np.exp(-(dx * dx + dy * dy) / sig2) * w0
        h = np.maximum(h, g)  # max-pool so overlapping high hazards stay salient
    m = float(h.max()) if h.size else 0.0
    if m > 1e-6:
        h = h / m
    return h


def heatmap_to_bgr(heatmap: np.ndarray) -> np.ndarray:
    """Map [0,1] heat to BGR (COLORMAP_TURBO)."""
    g = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_TURBO)


def blend_hazard_map(
    left_bgr: np.ndarray, heat_bgr: np.ndarray, alpha: float = 0.55
) -> np.ndarray:
    """Alpha-blend left RGB image with heat overlay."""
    if left_bgr.shape[:2] != heat_bgr.shape[:2]:
        raise ValueError("Shape mismatch in blend_hazard_map")
    return cv2.addWeighted(left_bgr, 1.0 - alpha, heat_bgr, alpha, 0.0)
