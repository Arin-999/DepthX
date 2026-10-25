
import json
from pathlib import Path

from src.pose_detector import detect_pose
from src.simulate_rgbd import main as generate_synthetic_rgbd
from src.depth_processor import reconstruct_landmarks
from src.measurement_engine import estimate_measurements
from src.fit_analyzer import analyze_fit


ROOT = Path(__file__).resolve().parents[1]
GARMENT_PATH = ROOT / "data" / "garment_measurements.json"
REPORT_PATH = ROOT / "outputs" / "fit_report.json"


def load_garment_measurements():
    if not GARMENT_PATH.exists():
        print(f"Garment specifications not found: {GARMENT_PATH}")
        return None

    with GARMENT_PATH.open("r", encoding="utf-8") as file:
        garment = json.load(file)

    if garment.get("units") != "cm":
        raise ValueError("Garment measurements must use centimetres.")

    measurements = garment.get("measurements", {})
    if not isinstance(measurements, dict):
        raise ValueError("'measurements' must be a JSON object.")

    return measurements


def run_pipeline():
    print("\n=== SahaAI Fit Pipeline ===")

    print("\n[1/4] Detecting pose...")
    detect_pose()

    print("\n[2/4] Generating synthetic depth...")
    generate_synthetic_rgbd()

    print("\n[3/4] Reconstructing 3D landmarks...")
    reconstruct_landmarks()

    print("\n[4/4] Estimating body measurements...")
    body_result = estimate_measurements()

    body_measurements = {}
    measurement_map = {
        "chest_cm": "chest_circumference",
        "waist_cm": "waist_circumference",
    }

    for fit_key, body_key in measurement_map.items():
        measurement = body_result["measurements"][body_key]
        if (
            measurement["status"] == "measured"
            and measurement["value_cm"] is not None
        ):
            body_measurements[fit_key] = measurement["value_cm"]

    garment_measurements = load_garment_measurements()
    fit_result = None
    fit_error = None

    if garment_measurements is None:
        fit_error = "Garment specifications are missing."
    elif not body_measurements:
        fit_error = (
            "No validated chest or waist body measurements are available. "
            "Fit analysis cannot be performed."
        )
    else:
        try:
            fit_result = analyze_fit(
                body_measurements,
                garment_measurements,
            )
        except (TypeError, ValueError) as exc:
            fit_error = str(exc)

    report = {
        "project": "SahaAI Fit",
        "schema_version": "1.0",
        "status": "prototype",
        "depth_source": body_result.get("input_depth_source"),
        "body_measurements": body_result["measurements"],
        "fit_analysis": fit_result,
        "fit_analysis_status": (
            "estimated" if fit_result is not None else "unavailable"
        ),
        "fit_analysis_reason": fit_error,
        "limitations": [
            "The current pipeline uses synthetic depth.",
            "Shoulder landmark distance is not calibrated anatomical breadth.",
            "Chest and waist circumference remain unavailable until validated.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== Integrated Pipeline Report ===")
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {REPORT_PATH}")

    return report


if __name__ == "__main__":
    run_pipeline()
