"""Sample depth at 2D locations; MAE vs ground truth."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


def sample_depth_bilinear(z: np.ndarray, us: np.ndarray, vs: np.ndarray, invalid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinear sample depth map z (H,W). Returns (values, valid)."""
    h, w = z.shape
    u = np.asarray(us, dtype=np.float64)
    v = np.asarray(vs, dtype=np.float64)
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1
    valid = (u0 >= 0) & (v0 >= 0) & (u1 < w) & (v1 < h)
    out = np.full(u.shape, np.nan, dtype=np.float64)
    for i in range(u.size):
        if not valid[i]:
            continue
        x0, y0, x1, y1 = u0[i], v0[i], u1[i], v1[i]
        if invalid[y0, x0] or invalid[y0, x1] or invalid[y1, x0] or invalid[y1, x1]:
            continue
        wa = (x1 - u[i]) * (y1 - v[i])
        wb = (u[i] - x0) * (y1 - v[i])
        wc = (x1 - u[i]) * (v[i] - y0)
        wd = (u[i] - x0) * (v[i] - y0)
        out[i] = wa * z[y0, x0] + wb * z[y0, x1] + wc * z[y1, x0] + wd * z[y1, x1]
    fin = np.isfinite(out)
    return out.astype(np.float32), fin


def depth_mae_at_points(
    z_pred: np.ndarray,
    invalid_pred: np.ndarray,
    z_gt: np.ndarray,
    invalid_gt: np.ndarray,
    us: Iterable[float],
    vs: Iterable[float],
) -> Tuple[float, int]:
    """Mean absolute error in meters where both GT and pred are valid."""
    us = np.array(list(us), dtype=np.float64)
    vs = np.array(list(vs), dtype=np.float64)
    gv, vg = sample_depth_bilinear(z_gt, us, vs, invalid_gt)
    pv, vp = sample_depth_bilinear(z_pred, us, vs, invalid_pred)
    m = vg & vp & np.isfinite(gv) & np.isfinite(pv)
    if not np.any(m):
        return float("nan"), 0
    err = np.abs(pv[m] - gv[m])
    return float(np.mean(err)), int(np.sum(m))
