# Hazard Detection (Stereo Vision + YOLO Depth Estimation)
## Computer Vision Final Project Spring 2026 (Prof. Katyal)
## Tanvi Ranade, Ishita Unde, Ryanne Ma, Ria Dani

This project combines stereo vision and YOLO segmentation to estimate object distance from a left/right camera pair across general scenes (indoor or outdoor).

Current recommended pipeline:
1. Calibrate cameras from chessboard images.
2. Export runtime depth parameters to an intermediate JSON file.
3. Run YOLO + disparity + metric depth using those parameters.

## Project Layout

- `chessboard_calibrate.py`: Stereo calibration from chessboard pairs (`*_left.png`, `*_right.png`).
- `export_depth_params.py`: Exports depth runtime params to `depth_params.json`.
- `yolo_depth_detection.py`: Runs YOLO segmentation and disparity/depth estimation.
- `preprocess_bw.py`: Utility to convert images/folders to grayscale.
- `stereoCalibration.py`: Older COLMAP-based stereo/depth experiment script.
- `chessboard-calibration/`: Input chessboard calibration images.
- `images/`: Regular stereo image inputs for detection/depth.
- `camera_calibration.npz`: Calibration output (intrinsics + stereo extrinsics).
- `depth_params.json`: Intermediate runtime params used by `yolo_depth_detection.py`.

## Environment Setup

Python 3 + OpenCV + Ultralytics are required.

### 1) Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install numpy opencv-python ultralytics
```

If you want optional advanced OpenCV modules (`cv2.ximgproc`) in other scripts, use:

```bash
pip install opencv-contrib-python
```

## Calibration Workflow

### Step 1: Run chessboard calibration

Use chessboard image pairs in `chessboard-calibration/`:

```bash
venv/bin/python chessboard_calibrate.py \
  --images-dir chessboard-calibration \
  --pattern-cols 9 \
  --pattern-rows 6 \
  --square-size 0.024 \
  --output camera_calibration.npz
```

Important:
- `pattern-cols` and `pattern-rows` are **inner corner counts**.
- `square-size` is real square edge length in meters (or your chosen unit).
- If this value is off, stereo translation scale from calibration will also be off.

What it prints:
- Mono RMS reprojection error for left/right.
- Stereo RMS reprojection error.
- Intrinsics including focal lengths (`fx`, `fy`) and principal points (`cx`, `cy`).

What it saves:
- `K_left`, `D_left`, `K_right`, `D_right`, `R`, `T`, `E`, `F`
- Metadata (`image_size`, `pattern_size`, `square_size`, `used_pairs`)

## Intermediate Depth Params (Recommended Runtime Input)

### Step 2: Export runtime params to JSON

```bash
venv/bin/python export_depth_params.py \
  --calibration camera_calibration.npz \
  --output depth_params.json
```

This writes:
- `focal_px`
- `principal_x`
- `baseline_m`

Note on baseline:
- Calibration can estimate baseline from `T`, but measured physical baseline is often more reliable for metric depth.
- You can manually edit `depth_params.json` to set your measured `baseline_m`.

## YOLO + Depth Inference

### Step 3: Run detection and depth

```bash
venv/bin/python yolo_depth_detection.py \
  --image "images/6_left.png" \
  --right-image "images/6_right.png" \
  --params-file depth_params.json \
  --model yolo11n-seg.pt \
  --output detections_output.jpg \
  --disp-output disparity_output.png \
  --masked-disp-output masked_disparity_output.png
```

Outputs:
- Annotated detections image with masks, boxes, labels, bearing angle, and nearest depth.
- Full disparity colormap.
- Masked disparity colormap (only YOLO object regions).
- Console logs for detections and calibration/depth params used.

### Parameter precedence in `yolo_depth_detection.py`

From highest priority to lowest:
1. CLI overrides: `--focal-px`, `--principal-x`, `--baseline-m`
2. `--params-file` JSON (default `depth_params.json`)
3. Fallback `--calibration` NPZ (default `camera_calibration.npz`)
4. Hardcoded defaults in script

Skip file loading by passing empty strings:
- `--params-file ""`
- `--calibration ""`

## Script Reference

### `chessboard_calibrate.py`

Purpose:
- Performs mono + stereo calibration from chessboard left/right image pairs.

Key arguments:
- `--images-dir`
- `--left-suffix`, `--right-suffix`
- `--pattern-cols`, `--pattern-rows`
- `--square-size`
- `--output`
- `--show-detections`

### `export_depth_params.py`

Purpose:
- Extracts only runtime depth parameters from calibration output to a compact JSON.

Key arguments:
- `--calibration`
- `--output`

### `yolo_depth_detection.py`

Purpose:
- Runs YOLO segmentation on the left image.
- If right image is provided, computes disparity and metric depth.
- Reports nearest valid depth per segmented object.

Key arguments:
- `--image`, `--right-image`
- `--model`, `--conf`
- `--params-file`, `--calibration`
- `--focal-px`, `--principal-x`, `--baseline-m`
- `--output`, `--disp-output`, `--masked-disp-output`
- `--show`

### `preprocess_bw.py`

Purpose:
- Converts one image or a full folder tree to grayscale.

Example:

```bash
venv/bin/python preprocess_bw.py \
  --input images \
  --output preprocessed
```

## Typical End-to-End Run

```bash
# 1) Calibrate
venv/bin/python chessboard_calibrate.py --images-dir chessboard-calibration --pattern-cols 9 --pattern-rows 6 --square-size 0.024

# 2) Export params
venv/bin/python export_depth_params.py --calibration camera_calibration.npz --output depth_params.json

# 3) (Optional) edit baseline_m in depth_params.json to measured value

# 4) Run YOLO depth
venv/bin/python yolo_depth_detection.py --image "images/6_left.png" --right-image "images/6_right.png" --params-file depth_params.json
```

## Troubleshooting

- `ModuleNotFoundError: cv2`:
  - Use your project venv: `venv/bin/python ...`
  - Or install OpenCV in current environment.
- No depth appears:
  - Ensure `--right-image` is provided.
  - Check that left/right images correspond to the same scene/time.
  - Verify `baseline_m` and `focal_px` are reasonable.
- Bad metric scale:
  - Confirm measured baseline in `depth_params.json`.
  - Re-check chessboard `square-size` and corner pattern settings.
