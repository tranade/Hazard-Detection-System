# ETH3D subset for this project

Use [ETH3D datasets](https://www.eth3d.net/datasets) archives. Extract with `7z x <file>.7z`.

## Recommended downloads (course-friendly)

### 1. Low-res two-view (quick stereo + disparity GT)

- [two_view_training.7z](https://www.eth3d.net/data/two_view_training.7z) (~13.6 MB)
- [two_view_training_gt.7z](https://www.eth3d.net/data/two_view_training_gt.7z) (~14.2 MB)

Each scene folder contains rectified `im0.png`, `im1.png`, `calib.txt`, and (in the GT archive) `disp0GT.pfm`, `mask0nocc.png`.

### 2. High-res multi-view (small indoor scenes)

Pick **distorted** JPEG packs so **rendered depth** aligns with RGB (depth maps match **original distorted** images per [ETH3D documentation](https://www.eth3d.net/documentation)):

- [door_dslr_jpg.7z](https://www.eth3d.net/data/door_dslr_jpg.7z) (~0.1 GB)
- [door_dslr_depth.7z](https://www.eth3d.net/data/door_dslr_depth.7z)
- [door_dslr_occlusion.7z](https://www.eth3d.net/data/door_dslr_occlusion.7z) (optional masks)

Repeat for **pipes** if you want a second scene:

- [pipes_dslr_jpg.7z](https://www.eth3d.net/data/pipes_dslr_jpg.7z)
- [pipes_dslr_depth.7z](https://www.eth3d.net/data/pipes_dslr_depth.7z)

Also download **COLMAP** `cameras.txt`, `images.txt`, `points3D.txt` for the scene (included inside the same multi-view archives alongside `dslr_images`).

### 3. Low-res many-view rig (“moving camera”)

- [delivery_area_rig_undistorted.7z](https://www.eth3d.net/data/delivery_area_rig_undistorted.7z) (or `storage_room_rig_undistorted.7z`)
- Matching [delivery_area_rig_depth.7z](https://www.eth3d.net/data/delivery_area_rig_depth.7z) and eval/occlusion if you need GT depth on rig frames

Rig depth uses the same float binary convention; filenames align with rig PNGs.

## Layout after extract

Point `--eth3d-root` at the parent folder that contains scene directories (names vary slightly by archive). The CLI searches recursively for `cameras.txt` / `calib.txt` and depth folders.

Example:

```text
eth3d_data/
  door/
    cameras.txt
    images.txt
    dslr_images_jpg/   # or dslr_images/ depending on archive
    dslr_depth_jpg/    # float32 raw files, same basename as images
```

Two-view:

```text
eth3d_data/two_view_training/delivery_area_1/im0.png
```

Use `python -m hazard_cv.scripts.verify_eth3d_layout --root /path/to/eth3d_data` after unpacking.
