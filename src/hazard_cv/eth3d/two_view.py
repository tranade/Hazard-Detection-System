"""ETH3D low-res two-view: Middlebury calib + PFM disparity (bottom-up rows)."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class MiddleburyCalib:
    cam0: np.ndarray  # 3x3
    cam1: np.ndarray  # 3x3
    baseline: float
    doffs: float
    width: int
    height: int


def read_middlebury_calib(path: Path) -> MiddleburyCalib:
    text = Path(path).read_text()

    def _mat(name: str) -> np.ndarray:
        m = re.search(rf"{name}=\[(.*?)\]", text, re.S)
        if not m:
            raise ValueError(f"Missing {name} in {path}")
        body = m.group(1).replace(";", " ")
        vals = [float(x) for x in body.split() if x.strip()]
        if len(vals) != 9:
            raise ValueError(f"{name} expected 9 numbers in {path}")
        return np.array(vals, dtype=np.float64).reshape(3, 3)

    def _scalar(name: str, default: float | None = None) -> float:
        m = re.search(rf"{name}=([-\d.]+)", text)
        if not m:
            if default is not None:
                return default
            raise ValueError(f"Missing {name} in {path}")
        return float(m.group(1))

    cam0 = _mat("cam0")
    cam1 = _mat("cam1")
    baseline = abs(_scalar("baseline"))
    doffs = _scalar("doffs", default=0.0)
    width = int(_scalar("width"))
    height = int(_scalar("height"))
    return MiddleburyCalib(cam0=cam0, cam1=cam1, baseline=baseline, doffs=doffs, width=width, height=height)


def load_disparity_pfm(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load PFM; return (disp, valid) with disp in pixels; rows top-down in output array."""
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").strip()
        if header not in ("Pf", "PF"):
            raise ValueError(f"Not a grayscale PFM: {header}")
        dims = f.readline().decode("ascii").split()
        width, height = int(dims[0]), int(dims[1])
        scale_line = f.readline().decode("ascii").strip()
        scale = float(scale_line)
        little_endian = scale < 0
        endian = "<" if little_endian else ">"
        count = width * height
        data = struct.unpack(endian + f"{count}f", f.read(4 * count))
    arr = np.array(data, dtype=np.float32).reshape((height, width))
    # PFM stores rows bottom-to-top; flip to top-down for OpenCV consistency
    arr = np.flipud(arr)
    valid = np.isfinite(arr) & (arr < width) & (arr > -width)
    return arr, valid


def disparity_to_depth_z(calib: MiddleburyCalib, disp: np.ndarray) -> np.ndarray:
    """Z depth (m) for rectified left camera: Z = f * B / (disp + doffs)."""
    fx = float(calib.cam0[0, 0])
    d = disp + calib.doffs
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (fx * calib.baseline) / np.maximum(d, 1e-6)
    return z.astype(np.float32)


def load_two_view_mask(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """mask0nocc.png: valid_both = 255 on all channels (per ETH3D doc)."""
    import cv2

    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        both = np.all(m >= 250, axis=2)
    else:
        both = m >= 250
    invalid = ~both
    return both, invalid
