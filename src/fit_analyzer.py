"""Compare body and garment circumferences to estimate garment fit."""

from typing import Any


DEFAULT_THRESHOLDS = {
    "chest_cm": {"regular_min": 4.0, "regular_max": 12.0},
    "waist_cm": {"regular_min": 3.0, "regular_max": 10.0},
}


def classify_fit(
    body_cm: float,
    garment_cm: float,
    regular_min: float,
    regular_max: float,
) -> dict[str, Any]:
    """Classify fit using circumference ease in centimetres."""
    values = (body_cm, garment_cm, regular_min, regular_max)

    if not all(isinstance(value, (int, float)) for value in values):
        raise TypeError("Measurements and thresholds must be numeric.")

    if body_cm <= 0 or garment_cm <= 0:
        raise ValueError("Body and garment measurements must be positive.")

    if regular_min < 0 or regular_max < regular_min:
        raise ValueError("Fit thresholds are invalid.")

    ease_cm = float(garment_cm - body_cm)

    if ease_cm < regular_min:
        fit = "tight"
    elif ease_cm <= regular_max:
        fit = "regular"
    else:
        fit = "loose"

    return {
        "body_cm": round(float(body_cm), 2),
        "garment_cm": round(float(garment_cm), 2),
        "ease_cm": round(ease_cm, 2),
        "fit": fit,
    }


def analyze_fit(
    body_measurements: dict[str, float],
    garment_measurements: dict[str, float],
    thresholds: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Analyze matching chest and waist circumference measurements."""
    if not isinstance(body_measurements, dict):
        raise TypeError("body_measurements must be a dictionary.")

    if not isinstance(garment_measurements, dict):
        raise TypeError("garment_measurements must be a dictionary.")

    selected_thresholds = (
        DEFAULT_THRESHOLDS if thresholds is None else thresholds
    )

    results = {}

    for dimension in ("chest_cm", "waist_cm"):
        if dimension not in body_measurements:
            continue

        if dimension not in garment_measurements:
            continue

        if dimension not in selected_thresholds:
            raise ValueError(f"Missing thresholds for {dimension}.")

        limits = selected_thresholds[dimension]

        if "regular_min" not in limits or "regular_max" not in limits:
            raise ValueError(f"Incomplete thresholds for {dimension}.")

        results[dimension] = classify_fit(
            body_cm=body_measurements[dimension],
            garment_cm=garment_measurements[dimension],
            regular_min=limits["regular_min"],
            regular_max=limits["regular_max"],
        )

    if not results:
        raise ValueError(
            "No matching chest_cm or waist_cm measurements were provided."
        )

    return {
        "status": "estimated",
        "unit": "cm",
        "dimensions": results,
        "note": (
            "Fit labels use configurable ease thresholds. "
            "They are estimates, not validated clothing-size advice."
        ),
    }