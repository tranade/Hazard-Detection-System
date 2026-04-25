"""Minimal COLMAP text loader for ETH3D (PINHOLE undistorted and pose lines)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class CameraPinhole:
    camera_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class ImagePose:
    image_id: int
    name: str
    camera_id: int
    R_wc: np.ndarray  # 3x3 world->camera rotation (COLMAP: X_cam = R * X_world + t)
    t_wc: np.ndarray  # 3 translation


def _parse_cameras(cameras_path: Path, pinhole_only: bool = True) -> Dict[int, CameraPinhole]:
    cams: Dict[int, CameraPinhole] = {}
    for line in cameras_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cid = int(parts[0])
        model = parts[1]
        w, h = int(parts[2]), int(parts[3])
        if model != "PINHOLE":
            if pinhole_only:
                continue
            raise ValueError(f"Camera {cid} uses {model}; expected PINHOLE.")
        fx, fy, cx, cy = map(float, parts[4:8])
        cams[cid] = CameraPinhole(cid, w, h, fx, fy, cx, cy)
    return cams


def read_any_camera_resolution(cameras_path: Path) -> Tuple[int, int]:
    """First camera entry width/height (works for THIN_PRISM_FISHEYE DSLR packs)."""
    for line in Path(cameras_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        return int(parts[2]), int(parts[3])
    raise ValueError(f"No cameras in {cameras_path}")


def _quat_wxyz_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Hamilton quaternion w,x,y,z to rotation matrix (world to camera)."""
    r = Rotation.from_quat([qw, qx, qy, qz], scalar_first=True)
    return r.as_matrix()


def _parse_images(images_path: Path) -> List[ImagePose]:
    lines = [ln for ln in images_path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    images: List[ImagePose] = []
    i = 0
    while i < len(lines):
        toks = lines[i].split()
        if len(toks) < 10:
            i += 1
            continue
        image_id = int(toks[0])
        qw, qx, qy, qz = map(float, toks[1:5])
        tx, ty, tz = map(float, toks[5:8])
        cam_id = int(toks[8])
        name = toks[9]
        R = _quat_wxyz_to_R(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)
        images.append(ImagePose(image_id, name, cam_id, R, t))
        i += 2  # skip POINTS2D line
    return images


def load_colmap_model(
    scene_dir: Path, pinhole_only: bool = True
) -> Tuple[Dict[int, CameraPinhole], List[ImagePose]]:
    """Load cameras.txt and images.txt from a COLMAP / ETH3D scene directory."""
    scene_dir = Path(scene_dir)
    cams = _parse_cameras(scene_dir / "cameras.txt", pinhole_only=pinhole_only)
    imgs = _parse_images(scene_dir / "images.txt")
    return cams, imgs


def find_colmap_scene_roots(eth3d_root: Path, max_results: int = 200) -> List[Path]:
    """Return directories containing both cameras.txt and images.txt."""
    root = Path(eth3d_root)
    found: List[Path] = []
    for p in root.rglob("cameras.txt"):
        parent = p.parent
        if (parent / "images.txt").is_file():
            found.append(parent)
            if len(found) >= max_results:
                break
    return found
