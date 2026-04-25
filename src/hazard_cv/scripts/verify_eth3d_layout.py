"""Scan an extracted ETH3D root for expected layout (COLMAP, two-view, depth dirs)."""

from __future__ import annotations

import argparse
from pathlib import Path

from hazard_cv.calib.colmap_model import find_colmap_scene_roots
from hazard_cv.eth3d.scene import find_depth_directory, find_image_directories


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify ETH3D extract layout under --root")
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    print(f"Scanning {root} ...")
    scenes = find_colmap_scene_roots(root)
    print(f"Found {len(scenes)} COLMAP scene roots (cameras.txt + images.txt).")
    for s in scenes[:8]:
        imgs = find_image_directories(s)
        dep = find_depth_directory(s)
        print(f"  {s}")
        print(f"    images: {[str(p) for p in imgs][:3]}{'...' if len(imgs) > 3 else ''}")
        print(f"    depth:  {dep}")
    tw = list(root.rglob("im0.png"))
    print(f"Found {len(tw)} two-view left images (im0.png).")
    for p in tw[:6]:
        print(f"  {p.parent}")


if __name__ == "__main__":
    main()
