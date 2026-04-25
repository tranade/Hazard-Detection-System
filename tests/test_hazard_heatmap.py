import numpy as np

from hazard_cv.geometry.hazard_map_dense import build_hazard_heatmap, heatmap_to_bgr
from hazard_cv.geometry.hazards import CornerHazard


def test_heatmap_splat() -> None:
    corners = [
        CornerHazard(50.0, 50.0, 30.0, 1.0, "high"),
        CornerHazard(10.0, 10.0, 100.0, 1.0, "low"),
    ]
    h = build_hazard_heatmap(100, 100, corners, sigma_px=10.0)
    assert h.shape == (100, 100)
    assert h[50, 50] > h[0, 0]
    bgr = heatmap_to_bgr(h)
    assert bgr.shape == (100, 100, 3)
