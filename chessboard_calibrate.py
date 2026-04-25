import argparse
import glob
import os
from typing import List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate mono/stereo cameras from chessboard image pairs."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="chessboard-calibration",
        help="Directory containing chessboard images (e.g. *_left.png, *_right.png).",
    )
    parser.add_argument(
        "--left-suffix",
        type=str,
        default="_left.png",
        help="Suffix used for left camera images.",
    )
    parser.add_argument(
        "--right-suffix",
        type=str,
        default="_right.png",
        help="Suffix used for right camera images.",
    )
    parser.add_argument(
        "--pattern-cols",
        type=int,
        default=9,
        help="Number of inner corners per chessboard row (columns).",
    )
    parser.add_argument(
        "--pattern-rows",
        type=int,
        default=6,
        help="Number of inner corners per chessboard column (rows).",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=1.0,
        help="Chessboard square size in your chosen unit (e.g. meters).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="camera_calibration.npz",
        help="Output .npz file for calibration results.",
    )
    parser.add_argument(
        "--show-detections",
        action="store_true",
        help="Display corner detections while processing images.",
    )
    return parser.parse_args()


def build_pairs(images_dir: str, left_suffix: str, right_suffix: str) -> List[Tuple[str, str]]:
    left_pattern = os.path.join(images_dir, f"*{left_suffix}")
    left_paths = sorted(glob.glob(left_pattern))
    pairs: List[Tuple[str, str]] = []

    for left_path in left_paths:
        base = os.path.basename(left_path)
        if not base.endswith(left_suffix):
            continue
        stem = base[: -len(left_suffix)]
        right_path = os.path.join(images_dir, f"{stem}{right_suffix}")
        if os.path.exists(right_path):
            pairs.append((left_path, right_path))

    return pairs


def detect_corners(
    image_path: str,
    pattern_size: Tuple[int, int],
    criteria: Tuple[int, int, float],
):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    if not found:
        return image, gray, None

    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return image, gray, refined


def main() -> None:
    args = parse_args()
    pattern_size = (args.pattern_cols, args.pattern_rows)
    term_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    pairs = build_pairs(args.images_dir, args.left_suffix, args.right_suffix)
    if not pairs:
        raise RuntimeError(
            f"No matching image pairs found in '{args.images_dir}' "
            f"for suffixes '{args.left_suffix}' and '{args.right_suffix}'."
        )

    objp = np.zeros((args.pattern_rows * args.pattern_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.pattern_cols, 0 : args.pattern_rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    image_size = None
    used_pairs = 0

    for left_path, right_path in pairs:
        left_img, left_gray, left_corners = detect_corners(left_path, pattern_size, term_criteria)
        right_img, right_gray, right_corners = detect_corners(right_path, pattern_size, term_criteria)

        if left_gray is None or right_gray is None:
            continue

        if image_size is None:
            image_size = (left_gray.shape[1], left_gray.shape[0])

        if left_corners is None or right_corners is None:
            continue

        objpoints.append(objp.copy())
        imgpoints_left.append(left_corners)
        imgpoints_right.append(right_corners)
        used_pairs += 1

        if args.show_detections:
            cv2.drawChessboardCorners(left_img, pattern_size, left_corners, True)
            cv2.drawChessboardCorners(right_img, pattern_size, right_corners, True)
            preview = np.hstack([left_img, right_img])
            cv2.imshow("Left/Right detections", preview)
            cv2.waitKey(200)

    if args.show_detections:
        cv2.destroyAllWindows()

    if used_pairs < 3:
        raise RuntimeError(
            f"Only {used_pairs} valid pair(s) detected. "
            "Need at least 3 for stable calibration."
        )

    ret_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_left, image_size, None, None
    )
    ret_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_right, image_size, None, None
    )

    stereo_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5,
    )
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    ret_s, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        K_l,
        D_l,
        K_r,
        D_r,
        image_size,
        criteria=stereo_criteria,
        flags=stereo_flags,
    )

    np.savez(
        args.output,
        image_size=np.array(image_size, dtype=np.int32),
        pattern_size=np.array(pattern_size, dtype=np.int32),
        square_size=np.array(args.square_size, dtype=np.float64),
        used_pairs=np.array(used_pairs, dtype=np.int32),
        K_left=K_l,
        D_left=D_l,
        K_right=K_r,
        D_right=D_r,
        R=R,
        T=T,
        E=E,
        F=F,
        mono_rms_left=np.array(ret_l, dtype=np.float64),
        mono_rms_right=np.array(ret_r, dtype=np.float64),
        stereo_rms=np.array(ret_s, dtype=np.float64),
    )

    np.set_printoptions(precision=6, suppress=True)
    print(f"Used valid pairs: {used_pairs}/{len(pairs)}")
    print(f"Left mono RMS reprojection error:  {ret_l:.6f}")
    print(f"Right mono RMS reprojection error: {ret_r:.6f}")
    print(f"Stereo RMS reprojection error:     {ret_s:.6f}")
    print(
        f"\nLeft camera intrinsics: fx={K_l[0,0]:.3f}px, fy={K_l[1,1]:.3f}px, cx={K_l[0,2]:.3f}px, cy={K_l[1,2]:.3f}px"
    )
    print(
        f"Right camera intrinsics: fx={K_r[0,0]:.3f}px, fy={K_r[1,1]:.3f}px, cx={K_r[0,2]:.3f}px, cy={K_r[1,2]:.3f}px"
    )
    print("\nK_left:")
    print(K_l)
    print("\nD_left:")
    print(D_l.ravel())
    print("\nK_right:")
    print(K_r)
    print("\nD_right:")
    print(D_r.ravel())
    print("\nR (left->right):")
    print(R)
    print("\nT (left->right):")
    print(T.ravel())
    print(f"\nSaved calibration to: {args.output}")


if __name__ == "__main__":
    main()
