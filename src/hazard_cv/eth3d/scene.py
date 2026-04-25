"""Locate ETH3D scene assets (images, depth directory) under an extracted root."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple


def resolve_colmap_scene_dir(user_path: Path) -> Path:
    """Resolve a path that may be the scene root, a parent of it, or an extract folder."""
    p = Path(user_path).expanduser().resolve()
    if (p / "cameras.txt").is_file() and (p / "images.txt").is_file():
        return p
    if p.is_dir():
        for sub in sorted(p.iterdir()):
            if sub.is_dir() and (sub / "cameras.txt").is_file() and (sub / "images.txt").is_file():
                return sub
    hits = [x.parent for x in p.rglob("cameras.txt") if (x.parent / "images.txt").is_file()]
    if not hits:
        raise FileNotFoundError(
            f"No COLMAP scene (cameras.txt + images.txt) under {user_path}. "
            "Extract a multi-view .7z and point --scene-dir at the folder that contains cameras.txt."
        )
    hits.sort(key=lambda x: (len(x.parts), str(x)))
    return hits[0]

DEPTH_DIR_NAMES = (
    "dslr_depth_jpg",
    "dslr_depth",
    "rig_depth",
    "depth",
)


def find_depth_directory(scene_dir: Path) -> Optional[Path]:
    scene_dir = Path(scene_dir)
    for name in DEPTH_DIR_NAMES:
        p = scene_dir / name
        if p.is_dir():
            return p
    for p in scene_dir.rglob("dslr_depth_jpg"):
        if p.is_dir():
            return p
    for p in scene_dir.rglob("rig_depth"):
        if p.is_dir():
            return p
    return None


def find_image_directories(scene_dir: Path) -> List[Path]:
    scene_dir = Path(scene_dir)
    candidates = []
    for name in ("dslr_images_jpg", "dslr_images", "images", "images_rig_cam4"):
        p = scene_dir / name
        if p.is_dir():
            candidates.append(p)
    if not candidates:
        for p in scene_dir.rglob("dslr_images_jpg"):
            if p.is_dir():
                candidates.append(p)
    return list(dict.fromkeys(candidates))


def list_scene_images(scene_dir: Path, limit: int = 500) -> List[Path]:
    exts = {".jpg", ".jpeg", ".JPG", ".png", ".PNG"}
    imgs: List[Path] = []
    for folder in find_image_directories(scene_dir):
        for p in sorted(folder.iterdir()):
            if p.suffix in exts and p.is_file():
                imgs.append(p)
                if len(imgs) >= limit:
                    return imgs
    return imgs


def image_size_from_colmap(scene_dir: Path) -> Optional[Tuple[int, int]]:
    from hazard_cv.calib.colmap_model import load_colmap_model, read_any_camera_resolution

    scene_dir = Path(scene_dir)
    try:
        cams, _ = load_colmap_model(scene_dir, pinhole_only=True)
        if cams:
            c0 = next(iter(cams.values()))
            return c0.width, c0.height
        return read_any_camera_resolution(scene_dir / "cameras.txt")
    except (FileNotFoundError, ValueError, StopIteration):
        return None
