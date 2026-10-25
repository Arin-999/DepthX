import pytest

from src.measurement_engine import distance_cm
from src.torso_geometry import cross_section_perimeter
from src.smoothing import ExponentialSmoother
from src.confidence import calculate_quality


def test_distance_cm():
    a = {"x_m": 0, "y_m": 0, "z_m": 1}
    b = {"x_m": 0.4, "y_m": 0, "z_m": 1}

    assert distance_cm(a, b) == pytest.approx(40.0)


def test_smoother():
    smoother = ExponentialSmoother(alpha=0.3)

    assert smoother.update(40) == pytest.approx(40)
    assert smoother.update(42) == pytest.approx(40.6)


def test_quality_score():
    result = calculate_quality(0.9, 45, 49)

    assert result["score"] == pytest.approx(0.909, abs=0.001)


def test_contour_perimeter():
    points = [
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [0, -1, 0],
    ]

    assert cross_section_perimeter(points) == pytest.approx(5.65685)


def test_invalid_coordinates():
    a = {"x_m": 0, "y_m": 0, "z_m": 1}
    b = {"x_m": float("nan"), "y_m": 0, "z_m": 1}

    with pytest.raises(ValueError):
        distance_cm(a, b)