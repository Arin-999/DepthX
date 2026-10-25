import pytest

from src.fit_analyzer import analyze_fit, classify_fit


def test_tight_fit():
    result = classify_fit(100, 102, 4, 12)
    assert result["ease_cm"] == 2.0
    assert result["fit"] == "tight"


def test_regular_fit():
    result = classify_fit(100, 108, 4, 12)
    assert result["ease_cm"] == 8.0
    assert result["fit"] == "regular"


def test_loose_fit():
    result = classify_fit(100, 115, 4, 12)
    assert result["ease_cm"] == 15.0
    assert result["fit"] == "loose"


def test_negative_ease():
    result = classify_fit(100, 95, 4, 12)
    assert result["ease_cm"] == -5.0
    assert result["fit"] == "tight"


def test_analyze_chest_and_waist():
    body = {"chest_cm": 96, "waist_cm": 80}
    garment = {"chest_cm": 104, "waist_cm": 85}

    result = analyze_fit(body, garment)

    assert result["status"] == "estimated"
    assert result["dimensions"]["chest_cm"]["ease_cm"] == 8.0
    assert result["dimensions"]["chest_cm"]["fit"] == "regular"
    assert result["dimensions"]["waist_cm"]["ease_cm"] == 5.0
    assert result["dimensions"]["waist_cm"]["fit"] == "regular"


def test_rejects_zero_body_measurement():
    with pytest.raises(ValueError):
        classify_fit(0, 100, 4, 12)


def test_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        classify_fit(100, 108, 12, 4)


def test_rejects_missing_matching_measurements():
    with pytest.raises(ValueError):
        analyze_fit(
            {"shoulder_cm": 45},
            {"shoulder_cm": 48},
        )