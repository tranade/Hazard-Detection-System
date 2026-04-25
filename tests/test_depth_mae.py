import numpy as np

from hazard_cv.depth.eval import depth_mae_at_points


def test_mae_self_zero() -> None:
    z = np.ones((32, 32), dtype=np.float32) * 2.0
    inv = np.zeros_like(z, dtype=bool)
    mae, n = depth_mae_at_points(z, inv, z, inv, [10.5], [10.5])
    assert n == 1
    assert mae < 1e-5
