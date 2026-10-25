
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "landmarks_3d.json"
OUTPUT_PATH = ROOT / "outputs" / "body_measurements.json"


def distance_cm(point_a, point_b):
    """Calculate Euclidean distance between two 3D points."""
    coordinates_a = (
        point_a["x_m"],
        point_a["y_m"],
        point_a["z_m"],
    )
    coordinates_b = (
        point_b["x_m"],
        point_b["y_m"],
        point_b["z_m"],
    )

    if not all(
        math.isfinite(value)
        for value in (*coordinates_a, *coordinates_b)
    ):
        raise ValueError("Coordinates must be finite.")

    distance_m = math.sqrt(
        sum(
            (a - b) ** 2
            for a, b in zip(coordinates_a, coordinates_b)
        )
    )

    return distance_m * 100.0


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def make_unavailable(reason):
    return {
        "value_cm": None,
        "confidence": None,
        "status": "not_measured",
        "reason": reason,
    }


def estimate_measurements():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Run Sprint 2 first. Missing: {INPUT_PATH}"
        )

    data = load_json(INPUT_PATH)
    landmarks = data["landmarks"]

    if data.get("depth_source") != "synthetic_test_only":
        print("Input depth source:", data.get("depth_source"))

    left = landmarks.get("left_shoulder")
    right = landmarks.get("right_shoulder")

    shoulder = make_unavailable(
        "Shoulder landmarks unavailable or invalid."
    )

    if (
        left
        and right
        and left.get("status") == "ok"
        and right.get("status") == "ok"
        and left.get("xyz_m") is not None
        and right.get("xyz_m") is not None
    ):
        breadth = distance_cm(
            left["xyz_m"],
            right["xyz_m"],
        )

        shoulder = {
            "value_cm": round(breadth, 2),
            "confidence": None,
            "status": "simulated_geometry",
            "reason": (
                "Distance between detected shoulder landmarks; "
                "not yet calibrated to anatomical shoulder breadth."
            ),
        }

    result = {
        "schema_version": "1.0",
        "module": "SahaAI-Fit-BodyMeasurement",
        "input_depth_source": data.get("depth_source"),
        "units": "cm",
        "measurements": {
            "shoulder_breadth": shoulder,
            "chest_circumference": make_unavailable(
                "Requires torso surface reconstruction."
            ),
            "waist_circumference": make_unavailable(
                "Requires torso surface reconstruction."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("Measurement engine completed.")
    print(f"Results saved to: {OUTPUT_PATH}")
    print(json.dumps(result["measurements"], indent=2))

    return result


if __name__ == "__main__":
    estimate_measurements()
