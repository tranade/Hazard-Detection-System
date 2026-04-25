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


def detect_edges(
    image,
    canny_threshold1=24,
    canny_threshold2=77,
    blur_kernel_size=5,
    blur_sigma=1.0,
    aperture_size=3,
    l2_gradient=False,
    close_kernel_size=5,
    close_iterations=3,
    dilate_iterations=2,
    min_contour_length=80,
    final_close_kernel_size=13,
    final_close_iterations=2,
    contour_draw_thickness=3,
    edge_alpha=0.4,
    edge_color=(0, 0, 255),
):
    _validate_odd_kernel(blur_kernel_size)
    _validate_odd_kernel(close_kernel_size)
    _validate_odd_kernel(final_close_kernel_size)
    if aperture_size not in (3, 5, 7):
        raise ValueError("aperture_size must be 3, 5, or 7")

    image, gray = _prepare_image_and_gray(image)

    gray_blur = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), blur_sigma)
    edges = cv2.Canny(
        gray_blur,
        threshold1=canny_threshold1,
        threshold2=canny_threshold2,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )

    if close_iterations > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel_size, close_kernel_size),
        )
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)

    if dilate_iterations > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, dilate_kernel, iterations=dilate_iterations)

    if final_close_iterations > 0:
        bridge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (final_close_kernel_size, final_close_kernel_size),
        )
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            bridge_kernel,
            iterations=final_close_iterations,
        )

    if min_contour_length > 0:
        edges_before_filter = edges.copy()
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filtered_edges = np.zeros_like(edges)
        kept_count = 0
        for contour in contours:
            if cv2.arcLength(contour, closed=False) >= min_contour_length:
                cv2.drawContours(filtered_edges, [contour], -1, 255, contour_draw_thickness)
                kept_count += 1

        # If filtering is too strict for this frame, keep the unfiltered connected edges.
        edges = filtered_edges if kept_count > 0 else edges_before_filter

    edge_overlay = np.zeros_like(image)
    edge_overlay[edges > 0] = edge_color
    result = cv2.addWeighted(image, 1.0, edge_overlay, edge_alpha, 0)

    return image, edges, result


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
    h, w = edge_overlay_img.shape[:2]
    expanded = cv2.resize(edge_overlay_img, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    return _add_panel_label(expanded, "Edges Only")


if __name__ == "__main__":
    image_path = r"C:\Users\ryann\Downloads\kicker_dslr_undistorted\kicker\images\dslr_images_undistorted\DSC_6514.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    _, _, edges_only = detect_edges(image)
    overview = build_edge_only_overview(edges_only)

    screen_w, screen_h = get_screen_size()
    max_w = int(screen_w * 0.9)
    max_h = int(screen_h * 0.9)

    cv2.namedWindow("Detection Overview", cv2.WINDOW_NORMAL)

    view_w, view_h = get_display_size(overview, max_w, max_h)
    cv2.resizeWindow("Detection Overview", view_w, view_h)

    # Show full-resolution composite and let window size control on-screen fit.
    cv2.imshow("Detection Overview", overview)

    cv2.waitKey(0)
    cv2.destroyAllWindows()