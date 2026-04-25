import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation


# ── COLMAP loaders

def load_cameras(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model  = parts[1]
            w, h   = int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))
            if model == "PINHOLE":
                fx, fy, cx, cy = params
            else:
                fx = fy = params[0]
                cx, cy  = params[1], params[2]
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float64)
            cameras[cam_id] = {"K": K, "w": w, "h": h}
    return cameras


def load_images(path):
    images = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith("#"):
            i += 1
            continue
        parts  = lines[i].split()
        img_id = int(parts[0])
        q      = np.array(list(map(float, parts[1:5])))   # qw qx qy qz
        t      = np.array(list(map(float, parts[5:8])))
        cam_id = int(parts[8])
        name   = parts[9]
        images[img_id] = {"q": q, "t": t, "cam_id": cam_id, "name": name}
        i += 2   # skip keypoints line
    return images


def qvec2rotmat(q):
    # COLMAP order: qw qx qy qz  →  scipy order: qx qy qz qw
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


# camera helper 

def get_projection(cam_data):
    """Return R, t for a COLMAP image entry (world-to-camera)."""
    R = qvec2rotmat(cam_data["q"])
    t = cam_data["t"]
    return R, t


def camera_center(R, t):
    """World-space position of the camera optical centre."""
    return -R.T @ t


def baseline_and_relative_pose(R1, t1, R2, t2):
    """
    Relative rotation and translation that takes a point from
    camera-1 coordinates to camera-2 coordinates.

        X2 = R_rel @ X1 + t_rel
    """
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    return R_rel, t_rel


# pick best stero pair 

def pick_stereo_pair(images, cameras, image_dir, max_pairs=10):
    """
    Among the first `max_pairs` images choose the pair whose baseline
    is closest to 10 % of the average scene depth (good stereo overlap).
    Falls back to the pair with the largest baseline if none load.
    """
    ids = list(images.keys())[:max_pairs]
    loaded = {}
    for img_id in ids:
        path = os.path.join(image_dir, os.path.basename(images[img_id]["name"]))
        frame = cv2.imread(path)
        if frame is not None:
            loaded[img_id] = frame

    if len(loaded) < 2:
        raise RuntimeError("Need at least 2 loadable images.")

    # compute camera centres
    centres = {}
    for img_id in loaded:
        R, t = get_projection(images[img_id])
        centres[img_id] = camera_center(R, t)

    # choose pair with baseline closest to a target ratio
    best = (None, None)
    best_score = np.inf
    target_ratio = 0.10   # 10 % of scene scale is a good heuristic

    ids_loaded = list(loaded.keys())
    for i in range(len(ids_loaded)):
        for j in range(i + 1, len(ids_loaded)):
            a, b = ids_loaded[i], ids_loaded[j]
            baseline = np.linalg.norm(centres[a] - centres[b])
            score = abs(baseline - target_ratio)
            if score < best_score:
                best_score = score
                best = (a, b)

    print(f"Selected stereo pair: image IDs {best[0]} and {best[1]}")
    return best[0], best[1], loaded[best[0]], loaded[best[1]]


# stereo rectification, disparity, depth

def rectify_pair(img1, img2, K1, K2, R_rel, t_rel):
    """
    Stereo-rectify an image pair.
    Returns rectified images, the common focal length, and the baseline.
    """
    h, w = img1.shape[:2]

    # stereoRectify needs dist-coeffs; images are already undistorted
    D = np.zeros(5)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D, K2, D,
        (w, h),
        R_rel, t_rel,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0       # crop to valid pixels only
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D, R2, P2, (w, h), cv2.CV_32FC1)

    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

    # focal length and baseline from P matrices
    f        = P1[0, 0]           # shared focal length after rectification
    baseline = abs(P2[0, 3]) / f  # B = -Tx / f  (Tx is P2[0,3])

    return rect1, rect2, f, baseline, Q


# disparity / depth map 

def compute_disparity(rect1, rect2):
    """
    Dense disparity with StereoSGBM.
    Tune numDisparities / blockSize for your scene scale.
    """
    gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY)

    num_disp  = 16 * 8    # must be divisible by 16; increase for far scenes
    block     = 9         # odd, 3-11 typical

    sgbm = cv2.StereoSGBM_create(
        minDisparity    = 0,
        numDisparities  = num_disp,
        blockSize       = block,
        P1              = 8  * 3 * block ** 2,
        P2              = 32 * 3 * block ** 2,
        disp12MaxDiff   = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange    = 32,
        preFilterCap    = 63,
        mode            = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disp_raw = sgbm.compute(gray1, gray2).astype(np.float32) / 16.0

    # WLS filter to smooth / fill holes (left-right consistency)
    right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
    disp_right    = right_matcher.compute(gray2, gray1).astype(np.float32) / 16.0

    wls = cv2.ximgproc.createDisparityWLSFilter(sgbm)
    wls.setLambda(8000)
    wls.setSigmaColor(1.5)
    disp_filtered = wls.filter(disp_raw, gray1, disparity_map_right=disp_right)

    return disp_filtered.astype(np.float32)


def disparity_to_depth(disparity, f, baseline):
    """
    depth = f * baseline / disparity   (classic stereo formula)
    Invalid / zero disparities become NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(disparity > 0, f * baseline / disparity, np.nan)
    return depth.astype(np.float32)


# visualize 

def visualise(img_rgb, depth_map, rect1=None, rect2=None, disp=None):
    valid  = np.isfinite(depth_map) & (depth_map > 0)
    d_vis  = np.full_like(depth_map, np.nan)
    if valid.any():
        lo, hi = np.percentile(depth_map[valid], [2, 98])
        d_vis  = np.clip(depth_map, lo, hi)

    n_cols  = 4 if (rect1 is not None) else 2
    fig, ax = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    ax[0].imshow(img_rgb);           ax[0].set_title("Reference Image");   ax[0].axis("off")
    im = ax[1].imshow(d_vis, cmap="plasma")
    ax[1].set_title("Metric Depth (m)"); ax[1].axis("off")
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    if rect1 is not None:
        ax[2].imshow(cv2.cvtColor(rect1, cv2.COLOR_BGR2RGB))
        ax[2].set_title("Rectified Left");  ax[2].axis("off")
        ax[3].imshow(disp, cmap="jet")
        ax[3].set_title("Disparity");       ax[3].axis("off")

    plt.tight_layout()
    plt.show()


# main 

def main():
    # paths 
    base_dir  = r"C:\Users\ishit\CV\project\terrace\dslr_calibration_undistorted"
    image_dir = r"C:\Users\ishit\CV\project\terrace\images\dslr_images_undistorted"

    cam_path = os.path.join(base_dir, "cameras.txt")
    img_path = os.path.join(base_dir, "images.txt")

    # COLMAP data 
    print("Loading COLMAP data …")
    cameras = load_cameras(cam_path)
    images  = load_images(img_path)
    print(f"  {len(cameras)} camera(s), {len(images)} image(s)")

    # stereo pair 
    id1, id2, img1, img2 = pick_stereo_pair(images, cameras, image_dir)

    cam1 = cameras[images[id1]["cam_id"]]
    cam2 = cameras[images[id2]["cam_id"]]
    K1, K2 = cam1["K"], cam2["K"]

    R1, t1 = get_projection(images[id1])
    R2, t2 = get_projection(images[id2])
    R_rel, t_rel = baseline_and_relative_pose(R1, t1, R2, t2)

    # stereo rectification
    print("Rectifying …")
    rect1, rect2, f, baseline, Q = rectify_pair(img1, img2, K1, K2, R_rel, t_rel)
    print(f"  focal length = {f:.1f} px,  baseline = {baseline:.4f} m")

    # dense disparity 
    print("Computing disparity (StereoSGBM + WLS) …")
    try:
        disp = compute_disparity(rect1, rect2)
    except AttributeError:
        # opencv-contrib not installed → fall back to plain SGBM without WLS
        print("  [warning] opencv-contrib not found, skipping WLS filter")
        gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY)
        num_disp, block = 16 * 8, 9
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disp, blockSize=block,
            P1=8*3*block**2, P2=32*3*block**2,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32, preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        disp = sgbm.compute(gray1, gray2).astype(np.float32) / 16.0

    # disparity / metric map 
    depth = disparity_to_depth(disp, f, baseline)

    valid = np.isfinite(depth) & (depth > 0)
    if valid.any():
        print(f"  depth range: {np.nanmin(depth[valid]):.2f} m … {np.nanmax(depth[valid]):.2f} m")

    # visualize 
    ref_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    visualise(ref_rgb, depth, rect1=rect1, rect2=rect2, disp=disp)


if __name__ == "__main__":
    main()