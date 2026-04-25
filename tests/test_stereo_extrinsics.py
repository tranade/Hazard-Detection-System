import numpy as np

from hazard_cv.stereo.colmap_rectify import relative_rt


def test_relative_rt_identity_translation() -> None:
    R1 = np.eye(3)
    t1 = np.zeros(3)
    R2 = np.eye(3)
    t2 = np.array([0.1, 0, 0])
    R, T = relative_rt(R1, t1, R2, t2)
    assert np.allclose(R, np.eye(3))
    assert np.allclose(T.ravel(), t2)
