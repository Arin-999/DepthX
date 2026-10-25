
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "data" / "sample" / "person.jpg"
OUTPUT_DIR = ROOT / "outputs"
DEPTH_PATH = OUTPUT_DIR / "synthetic_depth.npy"
CAMERA_PATH = OUTPUT_DIR / "synthetic_camera.json"


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Cannot read RGB image: {IMAGE_PATH}"
        )

    height, width = image.shape[:2]

    # TEST DATA ONLY: every pixel is assigned 2 metres.
    # This is not a real measurement of the person's body.
    depth_m = np.full(
        (height, width),
        2.0,
        dtype=np.float32
    )

    # Approximate test intrinsics, NOT real camera calibration.
    camera = {
        "image_width": width,
        "image_height": height,
        "fx": float(width),
        "fy": float(width),
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
        "depth_unit": "metres",
        "source": "synthetic_test_only"
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DEPTH_PATH, depth_m)
    CAMERA_PATH.write_text(
        json.dumps(camera, indent=2),
        encoding="utf-8"
    )

    print("Synthetic RGB-D test data created.")
    print(f"RGB dimensions: {width} x {height}")
    print(f"Depth map: {DEPTH_PATH}")
    print(f"Camera parameters: {CAMERA_PATH}")
    print("WARNING: synthetic depth is not real body depth.")


if __name__ == "__main__":
    main()
