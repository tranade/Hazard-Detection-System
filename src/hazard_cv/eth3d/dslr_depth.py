"""ETH3D rendered depth: float32 row-major; invalid pixels are +inf (per ETH3D docs)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class DepthMap:
    """Depth in meters along camera Z (rendered MVS depth); invalid mask excludes inf/nan."""

    z: np.ndarray  # HxW float32
    invalid: np.ndarray  # HxW bool, True where no valid depth

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.z.shape[0]), int(self.z.shape[1])


def load_dslr_depth_binary(path: Path, width: int, height: int) -> DepthMap:
    raw = np.fromfile(path, dtype=np.float32)
    expected = width * height
    if raw.size != expected:
        raise ValueError(f"{path}: expected {expected} floats ({width}x{height}), got {raw.size}")
    z = raw.reshape((height, width))
    invalid = ~np.isfinite(z) | (z <= 0)
    return DepthMap(z=z.astype(np.float32), invalid=invalid)


def load_occlusion_mask_png(path: Path) -> Optional[np.ndarray]:
    """ETH3D occlusion PNG: discard pixels with value 2 in mask (grayscale)."""
    import cv2

    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    discard = m == 2
    return discard


def find_depth_file_for_image(image_path: Path, depth_root: Path) -> Path:
    """Match depth binary to an image basename (ETH3D uses same name, .jpg depth is binary)."""
    depth_root = Path(depth_root)
    name = image_path.name
    cand = depth_root / name
    if cand.is_file():
        return cand
    # Some archives nest in dslr_depth_jpg/
    for p in depth_root.rglob(name):
        if p.is_file():
            return p
    raise FileNotFoundError(f"No depth file for {image_path.name} under {depth_root}")
