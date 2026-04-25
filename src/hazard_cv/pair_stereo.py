"""Rectified-stereo pair from two image paths (ETH3D two_view style) → disparity + hazard map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from hazard_cv.depth.stereo import sgbm_disparity_rectified, suggest_num_disparities
from hazard_cv.geometry.hazard_map_dense import blend_hazard_map, build_hazard_heatmap, heatmap_to_bgr
from hazard_cv.geometry.hazards import CornerHazard, detect_corner_hazards
from hazard_cv.viz.overlay import colorize_disparity, draw_corner_hazards, save_image


@dataclass
class PairHazardResult:
    left_bgr: np.ndarray
    right_bgr: np.ndarray
    disparity: np.ndarray
    corners: List[CornerHazard]
    heatmap: np.ndarray
    heat_bgr: np.ndarray
    overlay_bgr: np.ndarray
    heat_only_bgr: np.ndarray


def load_stereo_pair(left_path: Path, right_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    l = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    r = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if l is None:
        raise FileNotFoundError(left_path)
    if r is None:
        raise FileNotFoundError(right_path)
    if l.shape != r.shape:
        h, w = l.shape[:2]
        r = cv2.resize(r, (w, h), interpolation=cv2.INTER_AREA)
    return l, r


def run_stereo_hazard_map(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    num_disparities: int = 0,
    min_disparity: int = -64,
    harris_thresh: float = 1e-5,
    sigma_px: float = 45.0,
    blend_alpha: float = 0.55,
) -> PairHazardResult:
    h, w = left_bgr.shape[:2]
    if num_disparities <= 0:
        num_disparities = suggest_num_disparities(w)
    disp = sgbm_disparity_rectified(
        left_bgr, right_bgr, num_disparities=num_disparities, min_disparity=min_disparity
    )
    corners = detect_corner_hazards(left_bgr, harris_thresh=harris_thresh)
    heat = build_hazard_heatmap(h, w, corners, sigma_px=sigma_px)
    heat_bgr = heatmap_to_bgr(heat)
    overlay = blend_hazard_map(left_bgr, heat_bgr, alpha=blend_alpha)
    return PairHazardResult(
        left_bgr=left_bgr,
        right_bgr=right_bgr,
        disparity=disp,
        corners=corners,
        heatmap=heat,
        heat_bgr=heat_bgr,
        overlay_bgr=overlay,
        heat_only_bgr=heat_bgr,
    )


def save_hazard_map_bundle(
    result: PairHazardResult,
    out_path: Path,
    save_sidecars: bool = True,
) -> None:
    """Write main hazard map (blended) to out_path; optional disparity + corner overlay."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(out_path, result.overlay_bgr)
    if not save_sidecars:
        return
    stem = out_path.with_suffix("")
    dvis = colorize_disparity(
        result.disparity, valid_mask=np.isfinite(result.disparity) & (result.disparity > 0)
    )
    save_image(stem.parent / f"{stem.name}_disparity.png", dvis)
    save_image(stem.parent / f"{stem.name}_hazard_heat.png", result.heat_only_bgr)
    corn = draw_corner_hazards(result.left_bgr, result.corners)
    save_image(stem.parent / f"{stem.name}_corners.png", corn)


def main_pair_cli(argv: list[str] | None = None) -> int:
    """Entry point: ``python -m hazard_cv.pair_stereo left.png right.png -o out.png``."""
    import argparse
    import sys

    a = (argv if argv is not None else sys.argv[1:]) or []
    p = argparse.ArgumentParser(
        description="Build a hazard map from a rectified stereo pair (e.g. ETH3D im0 / im1).",
    )
    p.add_argument("left", type=Path, help="Left (reference) image, e.g. im0.png")
    p.add_argument("right", type=Path, help="Right image, e.g. im1.png")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("hazard_map.png"),
        help="Output blended hazard map (default: ./hazard_map.png)",
    )
    p.add_argument(
        "--no-extra",
        action="store_true",
        help="Do not write *_disparity.png, *_hazard_heat.png, *_corners.png next to the output",
    )
    p.add_argument("--min-disp", type=int, default=-64, help="SGBM minDisparity")
    p.add_argument("--num-disp", type=int, default=0, help="SGBM numDisparities (0=auto from width)")
    p.add_argument("--harris-thresh", type=float, default=1e-5)
    p.add_argument("--sigma", type=float, default=45.0, help="Gaussian falloff (pixels) for heat splat")
    p.add_argument("--blend", type=float, default=0.55, help="Alpha for heat over left image")
    args = p.parse_args(a)

    left, right = load_stereo_pair(args.left, args.right)
    res = run_stereo_hazard_map(
        left,
        right,
        num_disparities=int(args.num_disp),
        min_disparity=int(args.min_disp),
        harris_thresh=float(args.harris_thresh),
        sigma_px=float(args.sigma),
        blend_alpha=float(args.blend),
    )
    save_hazard_map_bundle(res, args.output, save_sidecars=not args.no_extra)
    print(str(Path(args.output).resolve()))
    return 0


def main() -> int:
    """Console entry for setuptools / ``python -m hazard_cv.pair_stereo``."""
    import sys

    return main_pair_cli(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
