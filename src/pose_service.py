"""MediaPipe pose inference and report construction for the local API."""

import csv
import threading
from datetime import datetime, timezone
from functools import lru_cache
from io import StringIO
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]
_INFERENCE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def load_pose_landmarker():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Pose model not found at {MODEL_PATH}")
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def analyze_uploaded_image(image: Image.Image):
    """Return an annotated RGB image and detected pose landmarks."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    rgb = np.asarray(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with _INFERENCE_LOCK:
        result = load_pose_landmarker().detect(mp_image)

    annotated = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    height, width = rgb.shape[:2]
    pose_landmarks = result.pose_landmarks[0] if result.pose_landmarks else []

    for start_idx, end_idx in POSE_CONNECTIONS:
        if (
            len(pose_landmarks) > max(start_idx, end_idx)
            and pose_landmarks[start_idx].visibility > 0.35
            and pose_landmarks[end_idx].visibility > 0.35
        ):
            start, end = pose_landmarks[start_idx], pose_landmarks[end_idx]
            p1 = (int(start.x * width), int(start.y * height))
            p2 = (int(end.x * width), int(end.y * height))
            cv2.line(annotated, p1, p2, (37, 99, 235), 3, cv2.LINE_AA)

    for landmark in pose_landmarks:
        if landmark.visibility > 0.35:
            point = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(annotated, point, 5, (0, 180, 90), -1, cv2.LINE_AA)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), pose_landmarks


def build_pose_report(
    source_name,
    image,
    landmarks,
    garment,
    elapsed_ms,
    mean_visibility,
):
    """Build a report that makes unavailable body measurements explicit."""
    pose_detected = bool(landmarks)
    return {
        "project": "DepthX | SahaAI Fit",
        "schema_version": "1.1",
        "status": "experimental",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_image": {
            "name": source_name,
            "image_dimensions_px": {"width": image.width, "height": image.height},
            "retained_on_disk": False,
        },
        "pose_analysis": {
            "status": "detected" if pose_detected else "not_detected",
            "landmark_count": len(landmarks),
            "dimension": "2D image coordinates",
            "mean_model_visibility": mean_visibility,
        },
        "performance": {
            "pose_analysis_elapsed_ms": round(elapsed_ms, 2),
            "note": "Single-run timing; not a benchmark or accuracy measure.",
        },
        "depth": {
            "source": "none",
            "real_depth_camera_connected": False,
            "note": (
                "Synthetic depth outputs elsewhere in the repository are "
                "test data only."
            ),
        },
        "body_measurements": {
            "chest_circumference": {
                "value_cm": None,
                "status": "not_measured",
                "reason": "No validated body-measurement method is connected.",
            },
            "waist_circumference": {
                "value_cm": None,
                "status": "not_measured",
                "reason": "No validated body-measurement method is connected.",
            },
        },
        "garment": garment,
        "fit_analysis": None,
        "fit_analysis_status": "unavailable",
        "fit_analysis_reason": (
            "Validated body chest and waist measurements are not available."
        ),
        "privacy": {
            "camera_accessed": False,
            "image_saved_to_application_storage": False,
        },
        "limitations": [
            "Pose landmarks come from a single RGB image.",
            "No real depth sensor data was used.",
            "Chest and waist circumferences were not measured.",
            "Garment fit has not been classified.",
        ],
    }


def report_as_csv(report):
    """Return a compact CSV export of the pose report."""
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["field", "value"])
    writer.writerow(["project", report["project"]])
    writer.writerow(["pose_status", report["pose_analysis"]["status"]])
    writer.writerow(["landmark_count", report["pose_analysis"]["landmark_count"]])
    writer.writerow(
        [
            "pose_analysis_elapsed_ms",
            report["performance"]["pose_analysis_elapsed_ms"],
        ]
    )
    writer.writerow(["depth_source", report["depth"]["source"]])
    writer.writerow(["real_depth_camera_connected", False])
    writer.writerow(["chest_circumference_cm", "not measured"])
    writer.writerow(["waist_circumference_cm", "not measured"])
    writer.writerow(["fit_analysis_status", report["fit_analysis_status"]])
    writer.writerow(["garment_name", report["garment"]["name"]])
    return buffer.getvalue()
