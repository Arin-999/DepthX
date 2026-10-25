
import json
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"
IMAGE_PATH = ROOT / "data" / "sample" / "person.jpg"
OUTPUT_DIR = ROOT / "outputs"

# MediaPipe landmark indices
LANDMARKS = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    23: "left_hip",
    24: "right_hip",
}


def detect_pose():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file missing: {MODEL_PATH}"
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Sample image missing: {IMAGE_PATH}"
        )

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise ValueError("OpenCV could not read the image.")

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )

    with vision.PoseLandmarker.create_from_options(
        options
    ) as detector:
        result = detector.detect(mp_image)

    if not result.pose_landmarks:
        print("No person detected. Try another image.")
        return

    landmarks = result.pose_landmarks[0]
    world = result.pose_world_landmarks[0]

    output = {}

    for index, name in LANDMARKS.items():
        lm = landmarks[index]
        wm = world[index]

        output[name] = {
            "image": {
                "x_px": round(lm.x * width, 2),
                "y_px": round(lm.y * height, 2),
                "visibility": round(lm.visibility, 4),
            },
            "model_world": {
                "x": round(wm.x, 5),
                "y": round(wm.y, 5),
                "z": round(wm.z, 5),
                "unit": "metres_estimated",
            },
        }

        x = int(lm.x * width)
        y = int(lm.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(
                image,
                name,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "landmarks.json"
    image_path = OUTPUT_DIR / "pose_result.jpg"

    json_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    cv2.imwrite(str(image_path), image)

    print("Pose detection successful.")
    print(f"Landmarks JSON: {json_path}")
    print(f"Visualized image: {image_path}")


if __name__ == "__main__":
    detect_pose()
