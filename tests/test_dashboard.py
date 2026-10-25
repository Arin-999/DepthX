import csv
from io import StringIO

from PIL import Image

from src.pose_service import build_pose_report, report_as_csv


def test_pose_report_does_not_infer_body_measurements_or_real_depth():
    report = build_pose_report(
        source_name="sample.jpg",
        image=Image.new("RGB", (64, 96)),
        landmarks=[],
        garment={"name": "Sample tee", "units": "cm"},
        elapsed_ms=12.345,
        mean_visibility=None,
    )

    assert report["pose_analysis"]["status"] == "not_detected"
    assert report["depth"]["source"] == "none"
    assert report["depth"]["real_depth_camera_connected"] is False
    assert report["body_measurements"]["chest_circumference"]["value_cm"] is None
    assert report["body_measurements"]["waist_circumference"]["value_cm"] is None
    assert report["fit_analysis_status"] == "unavailable"
    assert report["performance"]["pose_analysis_elapsed_ms"] == 12.35
    assert report["privacy"]["camera_accessed"] is False
    assert report["privacy"]["image_saved_to_application_storage"] is False


def test_csv_report_contains_measurement_and_depth_status():
    report = build_pose_report(
        source_name="sample.jpg",
        image=Image.new("RGB", (32, 48)),
        landmarks=[],
        garment={"name": "Sample tee"},
        elapsed_ms=4.0,
        mean_visibility=None,
    )

    rows = dict(csv.reader(StringIO(report_as_csv(report))))

    assert rows["real_depth_camera_connected"] == "False"
    assert rows["chest_circumference_cm"] == "not measured"
    assert rows["fit_analysis_status"] == "unavailable"
