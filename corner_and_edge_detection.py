import cv2
import numpy as np


def _validate_odd_kernel(kernel_size):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")


def _prepare_image_and_gray(image):
    if image is None:
        raise ValueError("image cannot be None")
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim < 2:
        raise ValueError("image must be at least 2-dimensional")

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Copy to avoid mutating a shared source image in downstream drawing steps.
    image = image.copy()
    return image, gray


def _normalize_to_u8(arr):
    norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def _gradient_magnitude(channel):
    grad_x = cv2.Scharr(channel, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(channel, cv2.CV_32F, 0, 1)
    return cv2.magnitude(grad_x, grad_y)


def _remove_small_components(edge_map, min_area=40):
    if min_area <= 0:
        return edge_map

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_map, connectivity=8)
    cleaned = np.zeros_like(edge_map)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def _resize_to_max_dim(img, max_dim):
    if max_dim is None or max_dim <= 0:
        return img
    h, w = img.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return img
    scale = max_dim / float(largest)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def detect_edges(
    image,
    blur_kernel_size=5,
    blur_sigma=1.0,
    apply_clahe=True,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
    luma_weight=0.65,
    color_weight=0.35,
    strong_percentile=88,
    weak_percentile=72,
    canny_threshold1=25,
    canny_threshold2=85,
    aperture_size=3,
    l2_gradient=True,
    open_kernel_size=3,
    open_iterations=1,
    close_kernel_size=5,
    close_iterations=1,
    min_component_area=40,
    edge_alpha=0.4,
    edge_color=(0, 0, 255),
    processing_max_dim=1800,
    return_tuning_mask=False,
):
    _validate_odd_kernel(blur_kernel_size)
    _validate_odd_kernel(open_kernel_size)
    _validate_odd_kernel(close_kernel_size)
    if aperture_size not in (3, 5, 7):
        raise ValueError("aperture_size must be 3, 5, or 7")
    if not 0.0 <= luma_weight <= 1.0 or not 0.0 <= color_weight <= 1.0:
        raise ValueError("luma_weight and color_weight must be in [0, 1]")
    if luma_weight + color_weight == 0:
        raise ValueError("luma_weight + color_weight must be > 0")
    if not 0 <= weak_percentile <= 100 or not 0 <= strong_percentile <= 100:
        raise ValueError("weak_percentile and strong_percentile must be in [0, 100]")
    if weak_percentile >= strong_percentile:
        raise ValueError("weak_percentile must be less than strong_percentile")

    image, gray = _prepare_image_and_gray(image)
    original_image = image
    original_h, original_w = image.shape[:2]

    proc_image = image
    proc_gray = gray
    if processing_max_dim is not None and processing_max_dim > 0:
        largest = max(original_h, original_w)
        if largest > processing_max_dim:
            scale = processing_max_dim / float(largest)
            proc_size = (int(original_w * scale), int(original_h * scale))
            proc_image = cv2.resize(image, proc_size, interpolation=cv2.INTER_AREA)
            proc_gray = cv2.resize(gray, proc_size, interpolation=cv2.INTER_AREA)

    gray_blur = cv2.GaussianBlur(proc_gray, (blur_kernel_size, blur_kernel_size), blur_sigma)

    if apply_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size,
        )
        gray_for_edges = clahe.apply(gray_blur)
    else:
        gray_for_edges = gray_blur

    luma_mag = _gradient_magnitude(gray_for_edges)
    luma_norm = _normalize_to_u8(luma_mag)

    if proc_image.ndim == 3:
        lab = cv2.cvtColor(proc_image, cv2.COLOR_BGR2LAB)
        chan_a = cv2.GaussianBlur(lab[:, :, 1], (blur_kernel_size, blur_kernel_size), blur_sigma)
        chan_b = cv2.GaussianBlur(lab[:, :, 2], (blur_kernel_size, blur_kernel_size), blur_sigma)
        color_mag = cv2.addWeighted(
            _gradient_magnitude(chan_a),
            0.5,
            _gradient_magnitude(chan_b),
            0.5,
            0,
        )
        color_norm = _normalize_to_u8(color_mag)
    else:
        color_norm = np.zeros_like(luma_norm)

    weight_sum = luma_weight + color_weight
    luma_w = luma_weight / weight_sum
    color_w = color_weight / weight_sum
    confidence = cv2.addWeighted(luma_norm, luma_w, color_norm, color_w, 0)
    confidence = _normalize_to_u8(confidence)

    strong_threshold = int(np.percentile(confidence, strong_percentile))
    weak_threshold = int(np.percentile(confidence, weak_percentile))

    strong_mask = np.zeros_like(confidence)
    strong_mask[confidence >= strong_threshold] = 255

    weak_mask = np.zeros_like(confidence)
    weak_mask[confidence >= weak_threshold] = 255

    canny_edges = cv2.Canny(
        gray_for_edges,
        threshold1=canny_threshold1,
        threshold2=canny_threshold2,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )

    # Keep Canny responses in plausible regions, then recover strong non-Canny gradients.
    edges = cv2.bitwise_and(canny_edges, weak_mask)
    edges = cv2.bitwise_or(edges, strong_mask)

    if open_iterations > 0:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_kernel_size, open_kernel_size),
        )
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, open_kernel, iterations=open_iterations)

    if close_iterations > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel_size, close_kernel_size),
        )
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)

    edges = _remove_small_components(edges, min_area=min_component_area)

    if proc_image.ndim == 2:
        image_for_overlay = cv2.cvtColor(proc_image, cv2.COLOR_GRAY2BGR)
    else:
        image_for_overlay = proc_image

    edge_overlay = np.zeros_like(image_for_overlay)
    edge_overlay[edges > 0] = edge_color
    result = cv2.addWeighted(image_for_overlay, 1.0, edge_overlay, edge_alpha, 0)

    if result.shape[:2] != original_image.shape[:2]:
        edges = cv2.resize(edges, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        result = cv2.resize(result, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    if return_tuning_mask:
        tuning_mask = build_tuning_mask(confidence, weak_mask, strong_mask, edges)
        return original_image, edges, result, tuning_mask

    return original_image, edges, result


def detect_corners(
    image,
    max_corners=300,
    quality_level=0.01,
    min_distance=10,
    blur_kernel_size=3,
    blur_sigma=0,
    apply_clahe=True,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
    block_size=3,
    use_harris_detector=False,
    harris_k=0.04,
    mask=None,
    corner_color=(255, 0, 0),
    corner_radius=6,
    corner_thickness=-1,
    base_image=None,
):
    _validate_odd_kernel(blur_kernel_size)
    image, gray = _prepare_image_and_gray(image)
    gray_blur = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), blur_sigma)
    corner_input = gray_blur

    if apply_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size,
        )
        corner_input = clahe.apply(gray_blur)

    corners = cv2.goodFeaturesToTrack(
        corner_input,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=mask,
        blockSize=block_size,
        useHarrisDetector=use_harris_detector,
        k=harris_k,
    )

    result = image.copy() if base_image is None else base_image.copy()

    if corners is not None:
        corners = np.int32(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result, (x, y), corner_radius, corner_color, corner_thickness)

    return image, corners, result


def get_display_size(img, max_width, max_height):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1:
        return int(w * scale), int(h * scale)
    return w, h


def get_screen_size(default_width=1920, default_height=1080):
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        return default_width, default_height


def _add_panel_label(img, label):
    labeled = img.copy()
    cv2.rectangle(labeled, (0, 0), (300, 44), (0, 0, 0), -1)
    cv2.putText(
        labeled,
        label,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def build_tuning_mask(confidence_map, weak_mask, strong_mask, final_edges, max_panel_dim=640):
    confidence_small = _resize_to_max_dim(confidence_map, max_panel_dim)
    weak_small = _resize_to_max_dim(weak_mask, max_panel_dim)
    strong_small = _resize_to_max_dim(strong_mask, max_panel_dim)
    edges_small = _resize_to_max_dim(final_edges, max_panel_dim)

    # Normalize all resized images to the same size to avoid hstack dimension mismatch
    target_h = confidence_small.shape[0]
    target_w = confidence_small.shape[1]
    
    if weak_small.shape[:2] != (target_h, target_w):
        weak_small = cv2.resize(weak_small, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if strong_small.shape[:2] != (target_h, target_w):
        strong_small = cv2.resize(strong_small, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if edges_small.shape[:2] != (target_h, target_w):
        edges_small = cv2.resize(edges_small, (target_w, target_h), interpolation=cv2.INTER_AREA)

    confidence_color = cv2.applyColorMap(confidence_small, cv2.COLORMAP_TURBO)
    weak_color = cv2.cvtColor(weak_small, cv2.COLOR_GRAY2BGR)
    strong_color = cv2.cvtColor(strong_small, cv2.COLOR_GRAY2BGR)
    edges_color = cv2.cvtColor(edges_small, cv2.COLOR_GRAY2BGR)

    top_row = np.hstack([
        _add_panel_label(confidence_color, "Edge Confidence"),
        _add_panel_label(weak_color, "Weak Mask"),
    ])
    bottom_row = np.hstack([
        _add_panel_label(strong_color, "Strong Mask"),
        _add_panel_label(edges_color, "Final Edge Mask"),
    ])
    return np.vstack([top_row, bottom_row])


def build_overview_grid(raw_img, corners_img, edges_img, both_img):
    top_row = np.hstack([
        _add_panel_label(raw_img, "Raw Image"),
        _add_panel_label(corners_img, "Corners Only"),
    ])
    bottom_row = np.hstack([
        _add_panel_label(edges_img, "Edges Only"),
        _add_panel_label(both_img, "Edges + Corners"),
    ])
    return np.vstack([top_row, bottom_row])


def build_edge_only_overview(edge_overlay_img):
    return _add_panel_label(edge_overlay_img, "Edges Only")


if __name__ == "__main__":
    image_path = r"C:\Users\ryann\Downloads\kicker_dslr_undistorted\kicker\images\dslr_images_undistorted\DSC_6514.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    _, _, edges_only, tuning_mask = detect_edges(image, return_tuning_mask=True)
    overview = build_edge_only_overview(edges_only)

    screen_w, screen_h = get_screen_size()
    max_w = int(screen_w * 0.9)
    max_h = int(screen_h * 0.9)

    cv2.namedWindow("Detection Overview", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Edge Tuning Mask", cv2.WINDOW_NORMAL)

    view_w, view_h = get_display_size(overview, max_w, max_h)
    tune_w, tune_h = get_display_size(tuning_mask, max_w, max_h)
    cv2.resizeWindow("Detection Overview", view_w, view_h)
    cv2.resizeWindow("Edge Tuning Mask", tune_w, tune_h)

    # Show full-resolution composite and let window size control on-screen fit.
    cv2.imshow("Detection Overview", overview)
    cv2.imshow("Edge Tuning Mask", tuning_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()