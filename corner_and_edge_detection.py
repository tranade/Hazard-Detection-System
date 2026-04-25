import cv2
import numpy as np


def detect_edges_and_corners_overlay(
    image_path,
    max_corners=300,
    quality_level=0.01,
    min_distance=10,
    canny_threshold1=30,
    canny_threshold2=100,
    edge_alpha=0.4,
):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # optional slight blur
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # edges
    edges = cv2.Canny(gray_blur, canny_threshold1, canny_threshold2)

    edge_overlay = np.zeros_like(image)
    edge_overlay[edges > 0] = (0, 0, 255)   # red in BGR

    # blend edges onto original
    result = cv2.addWeighted(image, 1.0, edge_overlay, edge_alpha, 0)

    # corners
    corners = cv2.goodFeaturesToTrack(
        gray_blur,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )

    # draw corners directly so they stay visible
    if corners is not None:
        corners = np.int32(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result, (x, y), 6, (255, 0, 0), -1)  # blue in BGR

    return image, edges, result


def resize_for_display(img, max_width=1200, max_height=800):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


if __name__ == "__main__":
    image_path = r"C:\Users\ryann\Downloads\kicker_dslr_undistorted\kicker\images\dslr_images_undistorted\DSC_6514.jpg"

    original, edges, result = detect_edges_and_corners_overlay(image_path)

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Overlay Result", cv2.WINDOW_NORMAL)

    cv2.imshow("Original", resize_for_display(original))
    cv2.imshow("Edges", resize_for_display(edges))
    cv2.imshow("Overlay Result", resize_for_display(result))

    cv2.waitKey(0)
    cv2.destroyAllWindows()