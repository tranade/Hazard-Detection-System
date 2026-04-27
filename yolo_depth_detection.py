import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import pyttsx3
except Exception:  # pragma: no cover - optional runtime dependency
    pyttsx3 = None

try:
    import pyrealsense2 as rs
except Exception:  # pragma: no cover - optional runtime dependency
    rs = None

# Fallback values if no calibration file is provided/found.
DEFAULT_FOCAL_PX = 411.0
DEFAULT_BASELINE_M = 0.018
DEFAULT_PRINCIPAL_X = 320.818
METER_TO_INCH = 39.37007874
PAIR_CAPTURE_PATTERN = re.compile(r"^(?P<index>\d+)_(?P<side>left|right)\.png$")


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


def capture_stereo_pair_from_camera(save_dir: str = "camera_captures") -> tuple[Path, Path]:
    if rs is None:
        raise RuntimeError(
            "Camera capture requested, but pyrealsense2 is not installed. "
            "Install with: pip install pyrealsense2"
        )

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_idx = len(list(output_dir.glob("*_left.png"))) + 1

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    pipeline.start(config)

    print("Camera mode enabled.")
    print("Press SPACE to capture synchronized left/right images, or Q to quit.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            if not left_frame or not right_frame:
                continue

            left_img = np.asanyarray(left_frame.get_data())
            right_img = np.asanyarray(right_frame.get_data())
            preview = np.hstack((left_img, right_img))
            cv2.imshow("Left IR | Right IR", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                left_path = output_dir / f"{capture_idx}_left.png"
                right_path = output_dir / f"{capture_idx}_right.png"
                cv2.imwrite(str(left_path), left_img)
                cv2.imwrite(str(right_path), right_img)
                print(f"Captured left image: {left_path}")
                print(f"Captured right image: {right_path}")
                return left_path, right_path
            if key == ord("q"):
                raise RuntimeError("Camera capture cancelled by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def list_stereo_pairs(dataset_dir: str) -> list[tuple[Path, Path]]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    grouped: dict[int, dict[str, Path]] = {}
    for image_path in root.glob("*.png"):
        m = PAIR_CAPTURE_PATTERN.match(image_path.name)
        if not m:
            continue
        idx = int(m.group("index"))
        side = m.group("side")
        grouped.setdefault(idx, {})[side] = image_path

    pairs: list[tuple[Path, Path]] = []
    for idx in sorted(grouped.keys()):
        sides = grouped[idx]
        if "left" in sides and "right" in sides:
            pairs.append((sides["left"], sides["right"]))
    return pairs


def load_calibration_npz(calibration_path: str):
    cal_path = resolve_existing_path(calibration_path)
    with np.load(str(cal_path)) as data:
        required = ("K_left", "T")
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(
                f"Calibration file is missing keys: {missing}. "
                "Expected at least K_left and T."
            )

        k_left = data["K_left"].astype(np.float64)
        t = data["T"].astype(np.float64).reshape(-1)

        focal_px = float(k_left[0, 0])
        principal_x = float(k_left[0, 2])
        baseline_m = float(np.linalg.norm(t))

        return {
            "path": cal_path,
            "focal_px": focal_px,
            "principal_x": principal_x,
            "baseline_m": baseline_m,
        }


def load_depth_params_json(params_path: str):
    file_path = resolve_existing_path(params_path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    required = ("focal_px", "principal_x", "baseline_m")
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(
            f"Depth params file is missing keys: {missing}. "
            "Expected focal_px, principal_x, baseline_m."
        )
    return {
        "path": file_path,
        "focal_px": float(payload["focal_px"]),
        "principal_x": float(payload["principal_x"]),
        "baseline_m": float(payload["baseline_m"]),
    }


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


def bearing_to_direction(bearing_deg: float, deadzone_deg: float = 7.5) -> str:
    if abs(bearing_deg) <= deadzone_deg:
        return "ahead"
    if bearing_deg < 0:
        return "to your left"
    return "to your right"


def choose_path_instruction(detections: list[dict]) -> str:
    with_depth = [
        d for d in detections if d.get("closest_depth_m") is not None and d.get("bearing_deg") is not None
    ]
    if not with_depth:
        return "straight"

    front_min = float("inf")
    left_min = float("inf")
    right_min = float("inf")
    left_risk = 0.0
    right_risk = 0.0
    for det in with_depth:
        depth = float(det["closest_depth_m"])
        bearing = float(det["bearing_deg"])
        # Closer objects and headings near the centerline carry more path risk.
        dist_weight = 1.0 / max(depth, 0.10)
        center_weight = max(0.0, 1.0 - (abs(bearing) / 45.0))
        risk = dist_weight * (1.0 + center_weight)
        if abs(bearing) <= 7.5:
            front_min = min(front_min, depth)
            left_risk += risk
            right_risk += risk
        elif bearing < 0:
            left_min = min(left_min, depth)
            left_risk += risk
        else:
            right_min = min(right_min, depth)
            right_risk += risk

    front_blocked = front_min < 1.5
    left_blocked = left_min < 1.2
    right_blocked = right_min < 1.2

    if front_blocked:
        if not left_blocked and not right_blocked:
            return "left" if left_risk <= right_risk else "right"
        if (left_min > right_min) and not left_blocked:
            return "left"
        if (right_min >= left_min) and not right_blocked:
            return "right"
        return "turn around"

    if not left_blocked and not right_blocked and abs(left_risk - right_risk) > 0.05:
        return "left" if left_risk < right_risk else "right"
    if left_blocked and not right_blocked:
        return "right"
    if right_blocked and not left_blocked:
        return "left"
    return "straight"


def pick_closest_detections(detections: list[dict], limit: int = 2) -> list[dict]:
    with_depth = [d for d in detections if d.get("closest_depth_m") is not None]
    with_depth.sort(key=lambda d: float(d["closest_depth_m"]))
    return with_depth[: max(0, int(limit))]


def format_heading_phrase(bearing_deg: float) -> str:
    if abs(bearing_deg) <= 7.5:
        return "straight ahead"
    side = "left" if bearing_deg < 0 else "right"
    return f"{abs(bearing_deg):.0f} degrees {side}"


def meters_to_inches(distance_m: float) -> float:
    return float(distance_m) * METER_TO_INCH


def speak_detection_summary(
    detections: list[dict],
    enabled: bool,
    rate_wpm: int,
    pause_s: float,
) -> None:
    if not enabled:
        return
    if pyttsx3 is None:
        print("Speech requested, but pyttsx3 is not installed. Run: pip install pyttsx3")
        return

    engine = pyttsx3.init()
    engine.setProperty("rate", int(rate_wpm))
    if pause_s > 0:
        engine.setProperty("pause", float(pause_s))

    if not detections:
        engine.say("No objects detected.")
        engine.runAndWait()
        return

    path_instruction = choose_path_instruction(detections)
    engine.say(f"Path suggestion: {path_instruction}.")

    closest = pick_closest_detections(detections, limit=2)
    if not closest:
        engine.say("No reliable distance readings for nearby objects.")
    else:
        for det in closest:
            label = det.get("class", "object")
            bearing = float(det.get("bearing_deg", 0.0))
            heading = format_heading_phrase(bearing)
            depth = float(det["closest_depth_m"])
            depth_in = meters_to_inches(depth)
            speech = f"{label}. Distance {depth_in:.1f} inches. Heading {heading}."
            engine.say(speech)

    engine.runAndWait()


def detect_objects_and_disparity(
    left_image_path: str,
    right_image_path: str | None,
    model_path: str = "yolo11n-seg.pt",
    conf_threshold: float = 0.1,
    focal_px: float | None = None,
    baseline_m: float | None = None,
    principal_x: float = DEFAULT_PRINCIPAL_X,
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
            use_fx = focal_px if focal_px is not None else DEFAULT_FOCAL_PX
            obj_center_x = 0.5 * (x1 + x2)
            bearing_deg = float(np.degrees(np.arctan2((obj_center_x - principal_x), use_fx)))
            detections[-1]["bearing_deg"] = bearing_deg

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
            cv2.putText(
                annotated,
                f"angle {bearing_deg:+.1f} deg",
                (x1, y1 + 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
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
                        f"{meters_to_inches(closest_m):.1f} in",
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
    parser.add_argument(
        "--image",
        required=False,
        default=None,
        help="Path to left input image. Optional when using --camera.",
    )
    parser.add_argument(
        "--right-image",
        default=None,
        help="Path to right stereo image (optional for disparity).",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Use RealSense camera to capture left/right images on keypress.",
    )
    parser.add_argument(
        "--camera-save-dir",
        default="camera_captures",
        help="Folder to save camera-captured stereo image pairs.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Folder containing '<index>_left.png' and '<index>_right.png' pairs to process.",
    )
    parser.add_argument(
        "--dataset-interval-s",
        type=float,
        default=10.0,
        help="Seconds between each dataset pair when --dataset-dir is used.",
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
        default=None,
        help="Stereo focal length in pixels. Overrides calibration file if provided.",
    )
    parser.add_argument(
        "--baseline-m",
        type=float,
        default=None,
        help="Stereo baseline in meters. Overrides calibration file if provided.",
    )
    parser.add_argument(
        "--principal-x",
        type=float,
        default=None,
        help="Principal point x (cx) in pixels. Overrides calibration file if provided.",
    )
    parser.add_argument(
        "--params-file",
        default="depth_params.json",
        help=(
            "Intermediate JSON file with focal_px/principal_x/baseline_m "
            "(produced by export_depth_params.py). Set empty string to skip."
        ),
    )
    parser.add_argument(
        "--calibration",
        default="camera_calibration.npz",
        help=(
            "Optional fallback calibration .npz if params-file is missing/disabled. "
            "Set to an empty string to skip loading."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display result window.",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Speak detected object, direction, and depth using text-to-speech.",
    )
    parser.add_argument(
        "--speech-rate",
        type=int,
        default=175,
        help="Speech rate in words per minute (used when --speak is enabled).",
    )
    parser.add_argument(
        "--speech-pause",
        type=float,
        default=0.0,
        help="Optional pause between utterances in seconds (when supported).",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.camera and args.dataset_dir:
        raise ValueError("Use only one input mode: either --camera or --dataset-dir.")

    if args.camera:
        captured_left, captured_right = capture_stereo_pair_from_camera(args.camera_save_dir)
        args.image = str(captured_left)
        args.right_image = str(captured_right)
        args.speak = True
    elif args.dataset_dir:
        args.speak = True
    elif args.image is None:
        raise ValueError("Provide --image, or use --camera / --dataset-dir.")

    calib = None
    if args.params_file is not None and str(args.params_file).strip() != "":
        try:
            calib = load_depth_params_json(args.params_file)
            print(
                f"Loaded depth params from: {calib['path']} "
                f"(fx={calib['focal_px']:.3f}px, cx={calib['principal_x']:.3f}px, baseline={calib['baseline_m']:.6f}m)"
            )
        except FileNotFoundError:
            calib = None
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load params file '{args.params_file}': {exc}"
            ) from exc

    if calib is None and args.calibration is not None and str(args.calibration).strip() != "":
        try:
            calib = load_calibration_npz(args.calibration)
            print(
                f"Loaded fallback calibration from: {calib['path']} "
                f"(fx={calib['focal_px']:.3f}px, cx={calib['principal_x']:.3f}px, baseline={calib['baseline_m']:.6f}m)"
            )
        except FileNotFoundError:
            calib = None
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load calibration file '{args.calibration}': {exc}"
            ) from exc

    focal_px = args.focal_px
    baseline_m = args.baseline_m
    principal_x = args.principal_x
    if calib is not None:
        if focal_px is None:
            focal_px = calib["focal_px"]
        if baseline_m is None:
            baseline_m = calib["baseline_m"]
        if principal_x is None:
            principal_x = calib["principal_x"]

    if focal_px is None:
        focal_px = DEFAULT_FOCAL_PX
    if baseline_m is None:
        baseline_m = DEFAULT_BASELINE_M
    if principal_x is None:
        principal_x = DEFAULT_PRINCIPAL_X

    print(
        f"Using intrinsics/extrinsics: focal_px={focal_px:.3f}, principal_x={principal_x:.3f}, baseline_m={baseline_m:.6f}"
    )

    def run_once(left_path: str, right_path: str | None) -> None:
        annotated, detections, disparity_vis, masked_disparity_vis, depth_map_m = detect_objects_and_disparity(
            left_image_path=left_path,
            right_image_path=right_path,
            model_path=args.model,
            conf_threshold=args.conf,
            focal_px=focal_px,
            baseline_m=baseline_m,
            principal_x=principal_x,
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved annotated image to: {output_path}")

        if detections:
            print("Detections:")
            for d in detections:
                closest_depth = d.get("closest_depth_m", None)
                bearing_deg = d.get("bearing_deg", None)
                bearing_txt = f"angle={bearing_deg:+.2f} deg | " if bearing_deg is not None else ""
                if closest_depth is None:
                    print(
                        f"  - {d['class']} | conf={d['confidence']:.2f} | {bearing_txt}bbox={d['bbox_xyxy']}"
                    )
                else:
                    print(
                        f"  - {d['class']} | conf={d['confidence']:.2f} | {bearing_txt}closest_depth={meters_to_inches(closest_depth):.1f} in | bbox={d['bbox_xyxy']}"
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

        speak_detection_summary(
            detections=detections,
            enabled=args.speak,
            rate_wpm=args.speech_rate,
            pause_s=args.speech_pause,
        )

        if args.show:
            cv2.imshow("YOLO Segmentation", annotated)
            if disparity_vis is not None and masked_disparity_vis is not None:
                cv2.imshow("Disparity (Full)", disparity_vis)
                cv2.imshow("Disparity (YOLO Masks Only)", masked_disparity_vis)
            cv2.waitKey(1)

    if args.dataset_dir:
        pairs = list_stereo_pairs(args.dataset_dir)
        if not pairs:
            raise RuntimeError(
                f"No stereo pairs found in {args.dataset_dir}. Expected files like '1_left.png' and '1_right.png'."
            )
        print(
            f"Dataset mode: found {len(pairs)} stereo pair(s) in {args.dataset_dir}. "
            f"Processing one pair every {args.dataset_interval_s:.1f} seconds."
        )
        for i, (left_path, right_path) in enumerate(pairs, start=1):
            print(f"[{i}/{len(pairs)}] Processing {left_path.name} + {right_path.name}")
            run_once(str(left_path), str(right_path))
            if i < len(pairs):
                time.sleep(max(0.0, args.dataset_interval_s))
        if args.show:
            cv2.destroyAllWindows()
    else:
        run_once(args.image, args.right_image)
        if args.show:
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
