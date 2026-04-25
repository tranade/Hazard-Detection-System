import struct
from pathlib import Path

import numpy as np

from hazard_cv.eth3d.two_view import (
    disparity_to_depth_z,
    load_disparity_pfm,
    read_middlebury_calib,
)
from hazard_cv.geometry.hazards import detect_corner_hazards


def _write_pfm(path: Path, arr: np.ndarray) -> None:
    """Write little-endian grayscale PFM, bottom-up storage per PFM spec."""
    h, w = arr.shape
    flipped = np.flipud(arr).astype(np.float32)
    header = f"Pf\n{w} {h}\n-1.0\n".encode("ascii")
    path.write_bytes(header + flipped.tobytes())


def test_disparity_roundtrip_pfm(tmp_path: Path) -> None:
    d = tmp_path / "scene"
    d.mkdir()
    disp = np.zeros((10, 12), dtype=np.float32)
    disp[:, :] = np.linspace(2, 5, 12)
    _write_pfm(d / "disp0GT.pfm", disp)
    out, valid = load_disparity_pfm(d / "disp0GT.pfm")
    assert out.shape == (10, 12)
    assert np.allclose(out[5, 6], disp[5, 6])


def test_middlebury_calib_parse(tmp_path: Path) -> None:
    txt = """cam0=[500 0 100; 0 500 80; 0 0 1]
cam1=[500 0 100; 0 500 80; 0 0 1]
doffs=0
baseline=0.15
width=200
height=150
ndisp=64
"""
    p = tmp_path / "calib.txt"
    p.write_text(txt)
    c = read_middlebury_calib(p)
    assert c.width == 200 and c.height == 150
    z = disparity_to_depth_z(c, np.full((4, 4), 10.0, dtype=np.float32))
    assert z.shape == (4, 4)
    assert np.allclose(z[0, 0], (500 * 0.15) / 10.0)


def test_corners_run_on_checkerboard() -> None:
    im = np.zeros((240, 320, 3), dtype=np.uint8)
    for i in range(0, 320, 40):
        for j in range(0, 240, 40):
            c = 255 if ((i // 40) + (j // 40)) % 2 == 0 else 0
            im[j : j + 40, i : i + 40] = c
    corners = detect_corner_hazards(im, harris_thresh=1e-5)
    assert len(corners) > 0
