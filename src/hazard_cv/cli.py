"""Command-line entry points for static eval, dynamic bench, and voice demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np

from hazard_cv.depth.eval import depth_mae_at_points
from hazard_cv.depth.stereo import sgbm_disparity
from hazard_cv.eth3d.dslr_depth import find_depth_file_for_image, load_dslr_depth_binary
from hazard_cv.eth3d.scene import find_depth_directory, list_scene_images, resolve_colmap_scene_dir
from hazard_cv.eth3d.two_view import (
    disparity_to_depth_z,
    load_disparity_pfm,
    load_two_view_mask,
    read_middlebury_calib,
)
from hazard_cv.geometry.hazards import CornerHazard, detect_corner_hazards
from hazard_cv.geometry.labels import evaluate_label_file
from hazard_cv.temporal.smoothing import TemporalCornerFilter, bench_fps
from hazard_cv.stereo.colmap_rectify import (
    depth_m_at_uv,
    pick_best_baseline_pair,
    pick_reasonable_baseline_pair,
    stereo_disparity_from_colmap_pair,
)
from hazard_cv.viz.overlay import colorize_disparity, draw_corner_hazards, save_image
from hazard_cv.voice.narrator import HazardNarrator


def _load_bgr(path: Path) -> np.ndarray:
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(path)
    return im


def cmd_verify(args: argparse.Namespace) -> int:
    from hazard_cv.calib.colmap_model import find_colmap_scene_roots
    from hazard_cv.eth3d.scene import find_depth_directory, find_image_directories

    root = Path(args.root).expanduser().resolve()
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
    return 0


def cmd_static_two_view(args: argparse.Namespace) -> int:
    d = Path(args.scene_dir).expanduser().resolve()
    left = _load_bgr(d / "im0.png")
    right = _load_bgr(d / "im1.png")
    calib = read_middlebury_calib(d / "calib.txt")
    disp_gt, v_gt = load_disparity_pfm(d / "disp0GT.pfm")
    mask_path = d / "mask0nocc.png"
    if mask_path.is_file():
        _, invalid_stereo = load_two_view_mask(mask_path)
    else:
        invalid_stereo = ~v_gt
    z_gt = disparity_to_depth_z(calib, disp_gt)
    invalid_gt = invalid_stereo | ~v_gt | ~np.isfinite(z_gt) | (z_gt <= 0)

    disp_pr = sgbm_disparity(left, right)
    z_pr = disparity_to_depth_z(calib, disp_pr)
    invalid_pr = ~np.isfinite(z_pr) | (z_pr <= 0) | (disp_pr <= 0)

    corners = detect_corner_hazards(left)
    us = [c.u for c in corners]
    vs = [c.v for c in corners]
    mae, n = depth_mae_at_points(z_pr, invalid_pr, z_gt, invalid_gt, us, vs)
    print(json.dumps({"corners": len(corners), "depth_mae_m": mae, "depth_samples": n}, indent=2))

    if args.labels:
        preds = {"im0": corners}
        rep = evaluate_label_file(preds, Path(args.labels))
        print(json.dumps({"label_eval": rep}, indent=2))
    return 0


def cmd_static_dslr(args: argparse.Namespace) -> int:
    scene = Path(args.scene_dir).expanduser().resolve()
    imgs = list_scene_images(scene)
    if not imgs:
        raise SystemExit("No images found under scene_dir")
    img_path = imgs[0]
    bgr = _load_bgr(img_path)
    h, w = bgr.shape[:2]
    depth_dir = find_depth_directory(scene)
    if depth_dir is None:
        raise SystemExit("No depth directory found (dslr_depth_jpg / rig_depth / ...)")
    depth_path = find_depth_file_for_image(img_path, depth_dir)
    dm = load_dslr_depth_binary(depth_path, w, h)
    corners = detect_corner_hazards(bgr)
    us = [c.u for c in corners]
    vs = [c.v for c in corners]
    if args.self_check:
        mae, n = depth_mae_at_points(dm.z, dm.invalid, dm.z, dm.invalid, us, vs)
    else:
        mae, n = float("nan"), 0
    print(
        json.dumps(
            {
                "image": str(img_path),
                "depth_file": str(depth_path),
                "corners": len(corners),
                "depth_mae_m_selfcheck": mae,
                "samples": n,
            },
            indent=2,
        )
    )
    return 0


def cmd_dynamic(args: argparse.Namespace) -> int:
    from hazard_cv.eth3d.scene import find_image_directories

    scene = Path(args.scene_dir).expanduser().resolve()
    paths: List[Path] = []
    for folder in find_image_directories(scene):
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.is_file():
                paths.append(p)
    if not paths:
        paths = sorted(
            p
            for p in scene.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.is_file()
        )
    if args.limit:
        paths = paths[: int(args.limit)]
    filt = TemporalCornerFilter(alpha=args.alpha)

    def step(bgr: np.ndarray) -> None:
        corners = detect_corner_hazards(bgr)
        filt.update(corners)

    frames = [_load_bgr(p) for p in paths]
    fps, ms = bench_fps(frames, step)
    print(json.dumps({"frames": len(paths), "mean_fps": fps, "mean_ms_per_frame": ms}, indent=2))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from hazard_cv.data.extract_7z import extract_7z_archive

    extract_7z_archive(Path(args.archive), Path(args.out))
    print(f"Extracted to {Path(args.out).resolve()}")
    return 0


def cmd_stereo_hazard(args: argparse.Namespace) -> int:
    """COLMAP-calibrated pair: rectify, SGBM, Harris hazards, depth at corners (rectified left)."""
    scene = resolve_colmap_scene_dir(Path(args.scene_dir))
    if args.auto_pair:
        if args.pair_strategy == "max":
            left, right, b = pick_best_baseline_pair(scene, max_images=int(args.max_images))
        else:
            left, right, b = pick_reasonable_baseline_pair(scene, max_images=int(args.max_images))
        pair_info = {"left": left, "right": right, "baseline_m_approx": b, "pair_strategy": args.pair_strategy}
    else:
        if not args.left or not args.right:
            raise SystemExit("Use --auto-pair or set both --left and --right (paths exactly as in images.txt).")
        left, right = str(args.left), str(args.right)
        pair_info = {"left": left, "right": right}

    res = stereo_disparity_from_colmap_pair(
        scene,
        left,
        right,
        num_disparities=int(args.num_disp),
        min_disparity=int(args.min_disp),
    )
    corners = detect_corner_hazards(res.left_rect, harris_thresh=float(args.harris_thresh))
    records = []
    for c in corners:
        z, ok = depth_m_at_uv(res, c.u, c.v)
        records.append(
            {
                "u": round(c.u, 2),
                "v": round(c.v, 2),
                "hazard_level": c.level,
                "angle_deg": round(c.angle_deg, 2),
                "depth_m": (round(z, 3) if ok else None),
            }
        )
    out = {
        "scene_dir": str(scene),
        "pair": pair_info,
        "stereo_baseline_m": round(res.baseline_m, 4),
        "corners": len(corners),
        "hazards": records[: int(args.top)] if args.top else records,
    }
    print(json.dumps(out, indent=2))

    if args.out_dir:
        od = Path(args.out_dir)
        valid = np.isfinite(res.disparity) & (res.disparity > 0)
        dvis = colorize_disparity(res.disparity, valid)
        save_image(od / "disparity_color.png", dvis)
        save_image(od / "left_rectified.png", res.left_rect)
        hvis = draw_corner_hazards(res.left_rect, corners)
        save_image(od / "corners_hazard.png", hvis)
        print(f"Wrote visuals to {od.resolve()}")
    return 0


def cmd_voice(args: argparse.Namespace) -> int:
    left = _load_bgr(Path(args.image))
    corners = detect_corner_hazards(left)
    h, w = left.shape[:2]
    z_fake = [1.2 for _ in corners]
    items = list(zip(corners, z_fake)) if corners else []
    narr = HazardNarrator(min_interval_s=0.0)
    lat, phrase = narr.announce_hazards(items, image_width=float(w), speak=not args.no_speak)
    print(json.dumps({"latency_s": lat, "phrase": phrase}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hazard-cv")
    sub = p.add_subparsers(dest="cmd", required=True)

    s0 = sub.add_parser("verify", help="List COLMAP scenes and two-view folders")
    s0.add_argument("--root", type=Path, required=True)
    s0.set_defaults(func=cmd_verify)

    s1 = sub.add_parser("static-two-view", help="Corners + SGBM depth MAE on ETH3D two-view scene")
    s1.add_argument("--scene-dir", type=Path, required=True, help="Folder with im0.png, im1.png, calib.txt, disp0GT.pfm")
    s1.add_argument("--labels", type=Path, default=None, help="Optional corner_labels.json for PR/F1")
    s1.set_defaults(func=cmd_static_two_view)

    s2 = sub.add_parser("static-dslr", help="Corners + DSLR float depth (self-check MAE optional)")
    s2.add_argument("--scene-dir", type=Path, required=True)
    s2.add_argument("--self-check", action="store_true", help="MAE(pred=GT)=0 sanity check")
    s2.set_defaults(func=cmd_static_dslr)

    s3 = sub.add_parser("dynamic", help="FPS / temporal smoothing on an image sequence folder")
    s3.add_argument("--scene-dir", type=Path, required=True)
    s3.add_argument("--limit", type=int, default=0)
    s3.add_argument("--alpha", type=float, default=0.35)
    s3.set_defaults(func=cmd_dynamic)

    s4 = sub.add_parser("voice", help="Speak top hazard for one image")
    s4.add_argument("--image", type=Path, required=True)
    s4.add_argument("--no-speak", action="store_true", help="Compute phrase only (CI / headless)")
    s4.set_defaults(func=cmd_voice)

    s5 = sub.add_parser("extract", help="Extract a .7z ETH3D archive (needs 7z CLI or pip install py7zr)")
    s5.add_argument("--archive", type=Path, required=True)
    s5.add_argument("--out", type=Path, required=True, help="Output directory (contents of archive appear here)")
    s5.set_defaults(func=cmd_extract)

    s6 = sub.add_parser(
        "stereo-hazard",
        help="Multi-view ETH3D: stereo via two COLMAP poses + corner hazards + depth at corners",
    )
    s6.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
        help="Extracted scene or parent folder (we locate cameras.txt + images.txt)",
    )
    s6.add_argument("--left", type=str, default=None, help="Left image path as in images.txt")
    s6.add_argument("--right", type=str, default=None, help="Right image path as in images.txt")
    s6.add_argument("--auto-pair", action="store_true", help="Pick two views automatically among first N images")
    s6.add_argument(
        "--pair-strategy",
        choices=("reasonable", "max"),
        default="reasonable",
        help="reasonable: baseline in ~0.02–0.45 m (better overlap); max: largest baseline",
    )
    s6.add_argument("--max-images", type=int, default=24, help="Search window for --auto-pair")
    s6.add_argument(
        "--num-disp",
        type=int,
        default=0,
        help="SGBM numDisparities (0 = auto from image width; else multiple of 16)",
    )
    s6.add_argument(
        "--min-disp",
        type=int,
        default=-64,
        help="SGBM minDisparity (often negative after stereoRectify on ETH3D)",
    )
    s6.add_argument("--top", type=int, default=30, help="Max hazards to print in JSON")
    s6.add_argument("--out-dir", type=Path, default=None, help="Save disparity + corner overlay PNGs")
    s6.add_argument(
        "--harris-thresh",
        type=float,
        default=1e-5,
        help="Harris peak threshold (lower = more corners on low-texture scenes)",
    )
    s6.set_defaults(func=cmd_stereo_hazard)

    s7 = sub.add_parser(
        "pair",
        help="Two images (stereo) → hazard map PNG (SGBM + corner heat; use with ETH3D two_view im0/im1)",
    )
    s7.add_argument("left", type=Path, help="Left / reference image (e.g. im0.png)")
    s7.add_argument("right", type=Path, help="Right image (e.g. im1.png)")
    s7.add_argument("-o", "--output", type=Path, default=Path("hazard_map.png"))
    s7.add_argument("--no-extra", action="store_true", help="Only write the main blended image")
    s7.add_argument("--min-disp", type=int, default=-64)
    s7.add_argument("--num-disp", type=int, default=0)
    s7.add_argument("--harris-thresh", type=float, default=1e-5)
    s7.add_argument("--sigma", type=float, default=45.0)
    s7.add_argument("--blend", type=float, default=0.55)
    s7.set_defaults(func=cmd_pair)

    return p


def cmd_pair(args: argparse.Namespace) -> int:
    from hazard_cv.pair_stereo import main_pair_cli

    a = [str(args.left), str(args.right), "-o", str(args.output)]
    if args.no_extra:
        a.append("--no-extra")
    a += [
        "--min-disp",
        str(args.min_disp),
        "--num-disp",
        str(args.num_disp),
        "--harris-thresh",
        str(args.harris_thresh),
        "--sigma",
        str(args.sigma),
        "--blend",
        str(args.blend),
    ]
    return main_pair_cli(a)


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()

