
import json

from src.pipeline import run_pipeline


def test_pipeline_creates_report():
    report = run_pipeline()

    assert report["project"] == "SahaAI Fit"
    assert report["status"] == "prototype"
    assert report["depth_source"] == "synthetic_test_only"
    assert report["fit_analysis_status"] == "unavailable"

    assert report["fit_analysis"] is None
    assert report["body_measurements"]["chest_circumference"]["value_cm"] is None
    assert report["body_measurements"]["waist_circumference"]["value_cm"] is None


def test_saved_report_matches_pipeline():
    report = run_pipeline()

    with open("outputs/fit_report.json", "r", encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report == report
