import pyrealsense2 as rs
import numpy as np
import cv2
import re
from pathlib import Path

# Folder where images will be saved
SAVE_DIR = Path("depth_dataset")
SAVE_DIR.mkdir(exist_ok=True)

NEW_STYLE_CAPTURE_PATTERN = re.compile(r"^(?P<index>\d+)_(?P<side>left|right)$")
OLD_STYLE_CAPTURE_PATTERN = re.compile(r"^(?:left|right)_(?P<timestamp>\d{8}_\d{6}_\d{6})$")


def get_next_capture_index(save_dir: Path) -> int:
    capture_keys = set()

    for image_path in save_dir.glob("*.png"):
        stem = image_path.stem

        new_style_match = NEW_STYLE_CAPTURE_PATTERN.match(stem)
        if new_style_match:
            capture_keys.add(f"new:{new_style_match.group('index')}")
            continue

        old_style_match = OLD_STYLE_CAPTURE_PATTERN.match(stem)
        if old_style_match:
            capture_keys.add(f"old:{old_style_match.group('timestamp')}")

    return len(capture_keys) + 1

# Create RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable left and right infrared streams
# IR stream 1 = left imager
# IR stream 2 = right imager
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

# Start camera
pipeline.start(config)

capture_index = get_next_capture_index(SAVE_DIR)

print("RealSense stereo capture started.")
print("Press SPACE to save left and right images at the same time.")
print("Press Q to quit.")

try:
    while True:
        # wait_for_frames gives a synchronized frameset
        frames = pipeline.wait_for_frames()

        left_frame = frames.get_infrared_frame(1)
        right_frame = frames.get_infrared_frame(2)

        if not left_frame or not right_frame:
            print("Could not get both IR frames.")
            continue

        # Convert RealSense frames to NumPy arrays
        left_img = np.asanyarray(left_frame.get_data())
        right_img = np.asanyarray(right_frame.get_data())

        # Show side-by-side preview
        combined = np.hstack((left_img, right_img))
        cv2.imshow("Left IR | Right IR", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            left_path = SAVE_DIR / f"{capture_index}_left.png"
            right_path = SAVE_DIR / f"{capture_index}_right.png"

            cv2.imwrite(str(left_path), left_img)
            cv2.imwrite(str(right_path), right_img)

            print("Saved synchronized stereo pair:")
            print(f"  {left_path}")
            print(f"  {right_path}")

            capture_index += 1

        elif key == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()