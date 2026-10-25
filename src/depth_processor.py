
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def sample_depth_median(depth_m, u, v, radius=3):
    """
    Return median valid depth in metres around a pixel.
    Invalid depths are rejected.
    """
    depth_m = np.asarray(depth_m)
    if depth_m.ndim != 2 or not np.issubdtype(depth_m.dtype, np.number):
        raise ValueError("Depth map must be a 2D numeric array.")

    if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)):
        raise ValueError("Sampling radius must be a non-negative integer.")
    if radius < 0:
        raise ValueError("Sampling radius must be a non-negative integer.")

    if not math.isfinite(float(u)) or not math.isfinite(float(v)):
        raise ValueError("Pixel coordinates must be finite.")

    height, width = depth_m.shape

    u = int(round(u))
    v = int(round(v))

    if not (0 <= u < width and 0 <= v < height):
        return None, 0

    x0 = max(0, u - radius)
    x1 = min(width, u + radius + 1)
    y0 = max(0, v - radius)
    y1 = min(height, v + radius + 1)

    patch = depth_m[y0:y1, x0:x1]

    valid = patch[
        np.isfinite(patch) & (patch > 0)
    ]

    if valid.size == 0:
        return None, 0

    return float(np.median(valid)), int(valid.size)


def pixel_to_3d(u, v, depth_m, camera):
    """
    Deproject a pixel using pinhole-camera intrinsics.
    All coordinates are in metres.
    """
    try:
        fx = float(camera["fx"])
        fy = float(camera["fy"])
        cx = float(camera["cx"])
        cy = float(camera["cy"])
        depth_m = float(depth_m)
        u = float(u)
        v = float(v)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Pixel, depth, and camera intrinsics must be numeric.") from exc

    if not all(math.isfinite(value) for value in (u, v, depth_m, fx, fy, cx, cy)):
        raise ValueError("Pixel, depth, and camera intrinsics must be finite.")

    if fx <= 0 or fy <= 0:
        raise ValueError("Camera focal lengths must be positive.")
    if depth_m <= 0:
        raise ValueError("Depth must be positive.")

    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m

    return {
        "x_m": float(X),
        "y_m": float(Y),
        "z_m": float(Z)
    }


def reconstruct_landmarks():
    rgb_path = ROOT / "data" / "sample" / "person.jpg"
    depth_path = OUTPUT_DIR / "synthetic_depth.npy"
    camera_path = OUTPUT_DIR / "synthetic_camera.json"
    landmarks_path = OUTPUT_DIR / "landmarks.json"

    required = [
        rgb_path,
        depth_path,
        camera_path,
        landmarks_path
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

    image = cv2.imread(str(rgb_path))
    if image is None:
        raise ValueError(f"OpenCV could not read the RGB image: {rgb_path}")

    depth_m = np.load(depth_path, allow_pickle=False)
    camera = load_json(camera_path)
    landmarks = load_json(landmarks_path)

    height, width = image.shape[:2]

    if depth_m.ndim != 2 or not np.issubdtype(depth_m.dtype, np.number):
        raise ValueError("Depth map must be a 2D numeric array.")

    if depth_m.shape != (height, width):
        raise ValueError(
            "RGB and depth dimensions do not match."
        )

    if (
        camera["image_width"] != width
        or camera["image_height"] != height
    ):
        raise ValueError(
            "Camera parameters do not match RGB dimensions."
        )

    output = {
        "schema_version": "1.0",
        "coordinate_frame": "camera",
        "units": "metres",
        "depth_source": camera["source"],
        "landmarks": {}
    }

    for name, data in landmarks.items():
        u = data["image"]["x_px"]
        v = data["image"]["y_px"]

        z, valid_count = sample_depth_median(
            depth_m, u, v, radius=3
        )

        if z is None:
            output["landmarks"][name] = {
                "pixel": {"u": u, "v": v},
                "xyz_m": None,
                "valid_depth_samples": 0,
                "status": "invalid_depth"
            }
            continue

        xyz = pixel_to_3d(u, v, z, camera)

        output["landmarks"][name] = {
            "pixel": {"u": u, "v": v},
            "depth_m": z,
            "xyz_m": xyz,
            "valid_depth_samples": valid_count,
            "status": "ok"
        }

    output_path = OUTPUT_DIR / "landmarks_3d.json"
    output_path.write_text(
        json.dumps(output, indent=2, allow_nan=False),
        encoding="utf-8"
    )

    print("3D reconstruction test completed.")
    print(f"Output: {output_path}")
    print(f"Depth source: {camera['source']}")

    for name, item in output["landmarks"].items():
        print(name, item["xyz_m"], item["status"])


if __name__ == "__main__":
    reconstruct_landmarks()
