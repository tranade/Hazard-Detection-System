import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Hardcoded stereo calibration defaults from your dataset
DEFAULT_FOCAL_PX = 411 #711.499
DEFAULT_BASELINE_M =  0.018 #0.072000 #0.0599432


def resolve_existing_path(raw_path: str) -> Path:
    path = Path(raw_path.strip().strip('"').strip("'")).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Path does not exist: {path}\n"
            "Tip: pass a full path and wrap it in quotes if it has spaces."
        )
    return path


def load_bgr_image(path_str: str):
    file_path = resolve_existing_path(path_str)
    image = cv2.imread(str(file_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {file_path}")
    return image


def compute_disparity(left_bgr: np.ndarray, right_bgr: np.ndarray):
    if left_bgr.shape[:2] != right_bgr.shape[:2]:
        right_bgr = cv2.resize(
            right_bgr,
            (left_bgr.shape[1], left_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    gray_l = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    block = 7
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 8,
        blockSize=block,
        P1=8 * block**2,
        P2=32 * block**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=120,
        speckleRange=16,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
    disparity[disparity <= 0] = np.nan
    return disparity


def disparity_to_colormap(disparity: np.ndarray):
    valid = np.isfinite(disparity)
    if not np.any(valid):
        return np.zeros((disparity.shape[0], disparity.shape[1], 3), dtype=np.uint8)

    lo, hi = np.percentile(disparity[valid], [2, 98])
    clipped = np.clip(disparity, lo, hi)
    vis = np.zeros_like(clipped, dtype=np.float32)
    vis[valid] = (clipped[valid] - lo) / max(hi - lo, 1e-6)
    vis_u8 = (vis * 255).astype(np.uint8)
    return cv2.applyColorMap(vis_u8, cv2.COLORMAP_TURBO)


def detect_objects_and_disparity(
    left_image_path: str,
    right_image_path: str | None,
    model_path: str = "yolo11n-seg.pt",
    conf_threshold: float = 0.1,
    focal_px: float | None = None,
    baseline_m: float | None = None,
):
    left_img = load_bgr_image(left_image_path)
    right_img = load_bgr_image(right_image_path) if right_image_path else None

    model = YOLO(model_path)
    result = model.predict(left_img, conf=conf_threshold, verbose=False)[0]

    annotated = left_img.copy()
    detections = []
    combined_mask = np.zeros(left_img.shape[:2], dtype=bool)
    object_masks = []

    boxes = result.boxes
    masks = result.masks
    mask_data = masks.data.cpu().numpy() if masks is not None else None

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            score = float(confs[i])
            class_id = int(classes[i])
            label = model.names.get(class_id, str(class_id))
            color = (
                int((37 * (class_id + 3)) % 255),
                int((17 * (class_id + 5)) % 255),
                int((29 * (class_id + 7)) % 255),
            )

            if mask_data is not None and i < len(mask_data):
                raw_mask = mask_data[i]
                if raw_mask.shape[:2] != annotated.shape[:2]:
                    raw_mask = cv2.resize(
                        raw_mask,
                        (annotated.shape[1], annotated.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                obj_mask = raw_mask > 0.5
                combined_mask |= obj_mask
                object_masks.append(obj_mask)
                annotated[obj_mask] = (
                    0.6 * annotated[obj_mask] + 0.4 * np.array(color)
                ).astype(np.uint8)
            else:
                object_masks.append(None)

            detections.append(
                {
                    "class": label,
                    "confidence": score,
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{label} {score:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    disparity_vis = None
    masked_disparity_vis = None
    depth_map_m = None
    if right_img is not None:
        disparity = compute_disparity(left_img, right_img)
        masked_disparity = np.where(combined_mask, disparity, np.nan).astype(np.float32)
        disparity_vis = disparity_to_colormap(disparity)
        masked_disparity_vis = disparity_to_colormap(masked_disparity)
        if focal_px is not None and baseline_m is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                depth_map_m = (focal_px * baseline_m) / disparity
            depth_map_m[~np.isfinite(depth_map_m)] = np.nan
            depth_map_m[depth_map_m <= 0] = np.nan

            for i, det in enumerate(detections):
                obj_mask = object_masks[i] if i < len(object_masks) else None
                if obj_mask is None:
                    det["closest_depth_m"] = None
                    continue

                valid = obj_mask & np.isfinite(depth_map_m) & (depth_map_m > 0)
                if np.any(valid):
                    closest_m = float(np.nanmin(depth_map_m[valid]))
                    det["closest_depth_m"] = closest_m
                    x1, y1, _, _ = det["bbox_xyxy"]
                    cv2.putText(
                        annotated,
                        f"{closest_m:.2f} m",
                        (x1, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    det["closest_depth_m"] = None

    return annotated, detections, disparity_vis, masked_disparity_vis, depth_map_m


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO segmentation on left image (masks + boxes + labels). "
            "Optional: provide right stereo image to generate full and YOLO-masked disparity maps."
        )
    )
    parser.add_argument("--image", required=True, help="Path to left input image.")
    parser.add_argument(
        "--right-image",
        default=None,
        help="Path to right stereo image (optional for disparity).",
    )
    parser.add_argument(
        "--model",
        default="yolo11n-seg.pt",
        help="YOLO segmentation model path or name.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--output",
        default="detections_output.jpg",
        help="Path to save annotated output image.",
    )
    parser.add_argument(
        "--disp-output",
        default="disparity_output.png",
        help="Path to save full disparity colormap.",
    )
    parser.add_argument(
        "--masked-disp-output",
        default="masked_disparity_output.png",
        help="Path to save disparity colormap only in segmented masks.",
    )
    parser.add_argument(
        "--focal-px",
        type=float,
        default=DEFAULT_FOCAL_PX,
        help="Stereo focal length in pixels. Needed for metric depth (meters).",
    )
    parser.add_argument(
        "--baseline-m",
        type=float,
        default=DEFAULT_BASELINE_M,
        help="Stereo baseline in meters. Needed for metric depth (meters).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display result window.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    annotated, detections, disparity_vis, masked_disparity_vis, depth_map_m = detect_objects_and_disparity(
        left_image_path=args.image,
        right_image_path=args.right_image,
        model_path=args.model,
        conf_threshold=args.conf,
        focal_px=args.focal_px,
        baseline_m=args.baseline_m,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    print(f"Saved annotated image to: {output_path}")
    if detections:
        print("Detections:")
        for d in detections:
            closest_depth = d.get("closest_depth_m", None)
            if closest_depth is None:
                print(
                    f"  - {d['class']} | conf={d['confidence']:.2f} | bbox={d['bbox_xyxy']}"
                )
            else:
                print(
                    f"  - {d['class']} | conf={d['confidence']:.2f} | closest_depth={closest_depth:.2f} m | bbox={d['bbox_xyxy']}"
                )
    else:
        print("No objects detected.")

    if disparity_vis is not None and masked_disparity_vis is not None:
        disp_output = Path(args.disp_output)
        disp_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(disp_output), disparity_vis)
        print(f"Saved full disparity map to: {disp_output}")

        masked_disp_output = Path(args.masked_disp_output)
        masked_disp_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(masked_disp_output), masked_disparity_vis)
        print(f"Saved masked disparity map to: {masked_disp_output}")
        if depth_map_m is None:
            print("Metric depth not computed. Provide --focal-px and --baseline-m.")
    else:
        print("Right image not provided; skipped disparity generation.")

    if args.show:
        cv2.imshow("YOLO Segmentation", annotated)
        if disparity_vis is not None and masked_disparity_vis is not None:
            cv2.imshow("Disparity (Full)", disparity_vis)
            cv2.imshow("Disparity (YOLO Masks Only)", masked_disparity_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
