# Hazard-Detection-System

Computer vision course project: **geometric hazard cues** (corners / sharpness), **ETH3D ground-truth depth** for distance evaluation, **temporal smoothing** on frame sequences, and **optional voice** feedback. ETH3D provides calibration and metric depth—not semantic “hazard class” labels; see [data/eth3d/DATASET.md](data/eth3d/DATASET.md) for what to download.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

For **`.7z` extraction** without a system `7z` binary: `pip install -e ".[data]"` (adds `py7zr`).

## Your ETH3D terrace archives (`data/*.7z`)

Multi-view DSLR packs (e.g. `terrace_dslr_undistorted.7z`) are **not** pre-rectified stereo pairs. This repo runs **COLMAP-calibrated two-view stereo**: pick two registered frames, **rectify**, run **SGBM**, then **Harris corner hazards** on the **rectified left** image and sample **depth** from the disparity-derived 3D cloud.

**Recommended:** use the **undistorted** archive for `stereo-hazard` (PINHOLE intrinsics). The distorted `*_dslr_jpg.7z` uses fisheye models and is better paired with ETH3D’s float depth maps, not this rectified-stereo shortcut.

```bash
# 1) Extract (7z CLI or py7zr)
pip install -e ".[data]"
hazard-cv extract --archive data/terrace_dslr_undistorted.7z --out data/eth3d_extracted/terrace_u

# 2) Stereo + corner/hazard (auto-pick a reasonable-baseline pair; saves PNGs)
hazard-cv stereo-hazard \
  --scene-dir data/eth3d_extracted/terrace_u/terrace/dslr_calibration_undistorted \
  --auto-pair \
  --out-dir data/out_terrace_vis

# Manual pair (paths must match `images.txt` lines, usually `dslr_images_undistorted/...`)
hazard-cv stereo-hazard \
  --scene-dir .../dslr_calibration_undistorted \
  --left dslr_images_undistorted/DSC_0273.JPG \
  --right dslr_images_undistorted/DSC_0274.JPG \
  --out-dir data/out_manual
```

Tuning: `--num-disp 0` (default) picks a width-based SGBM search range; `--min-disp` defaults to `-64` (often needed after `stereoRectify`). For a **fixed stereo rig** later, set `--left` / `--right` to your two cameras and pass a stable `--num-disp` / `--min-disp` from calibration.

### Two-view training / test: hazard map from `im0` + `im1`

Assumes **already rectified** stereo (ETH3D low-res two-view), same size. Produces a **blended hazard map** (turbo heat from corner hazards over the left image) plus optional sidecars.

```bash
# Installed entry point
hazard-cv pair path/to/im0.png path/to/im1.png -o hazard_map.png

# Or from repo root
python make_hazard_map.py path/to/im0.png path/to/im1.png -o hazard_map.png

# Same (after pip install -e .)
hazard-pair im0.png im1.png -o out.png
```

- **`--no-extra`**: only writes the main PNG (default also writes `*_disparity.png`, `*_hazard_heat.png`, `*_corners.png` next to `-o`).
- Pairs you upload later: use the same command; tune `--min-disp` / `--num-disp` if your camera differs from ETH3D.

## CLI (`hazard-cv`)

After install, the `hazard-cv` console script is available (or run `python -m hazard_cv`).

| Command | Purpose |
|--------|---------|
| `hazard-cv verify --root /path/to/extracted/eth3d` | List COLMAP scenes, image folders, depth dirs, and `im0.png` two-view scenes |
| `hazard-cv static-two-view --scene-dir .../delivery_area_1` | Harris corners + OpenCV **SGBM** disparity → depth vs **disp0GT.pfm** MAE |
| `hazard-cv static-dslr --scene-dir .../door --self-check` | DSLR float depth load + corner detection; `--self-check` verifies MAE(pred=GT)≈0 |
| `hazard-cv dynamic --scene-dir .../rig_images --limit 200` | Mean FPS + temporal EMA on a folder of frames |
| `hazard-cv voice --image path/to/im0.png --no-speak` | Build phrase + latency; omit `--no-speak` for real TTS (requires audio stack) |
| `hazard-cv extract --archive data/foo.7z --out data/extracted/foo` | Unpack ETH3D `.7z` (needs `7z` on PATH or `pip install -e ".[data]"`) |
| `hazard-cv stereo-hazard --scene-dir .../dslr_calibration_undistorted --auto-pair` | Rectified stereo + Harris hazards + depth samples (+ optional `--out-dir` PNGs) |
| `hazard-cv pair im0.png im1.png -o hazard_map.png` | Two rectified images → main hazard map PNG (+ extras unless `--no-extra`) |

Optional label evaluation (precision / recall / F1) against a small JSON file: `--labels data/eth3d/labels/corner_labels.example.json` on `static-two-view` (edit stems to match your frames).

## Optional semantics

`pip install -e ".[semantics]"` then use `hazard_cv.semantics.OutdoorSemanticDetector(weights="yolov8n.pt")` to fuse YOLO boxes with geometric corners.

## Layout

- `src/hazard_cv/` — library code (`calib`, `eth3d`, `stereo`, `geometry`, `depth`, `viz`, `temporal`, `voice`, `semantics`)
- `data/eth3d/` — dataset notes and example corner labels
- `tests/` — unit tests (no large ETH3D archives required)
