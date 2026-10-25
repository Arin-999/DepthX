
def calculate_quality(
    landmark_visibility,
    valid_depth_samples,
    total_depth_samples,
):
    """
    Return an input-quality score from 0 to 1.

    This is a heuristic quality indicator, not a calibrated
    probability that a body measurement is correct.
    """
    if total_depth_samples <= 0:
        return {
            "score": None,
            "status": "invalid_depth",
        }

    if valid_depth_samples < 0 or valid_depth_samples > total_depth_samples:
        raise ValueError("Invalid depth sample counts.")

    visibility = max(0.0, min(1.0, landmark_visibility))
    depth_ratio = valid_depth_samples / total_depth_samples

    score = 0.5 * visibility + 0.5 * depth_ratio

    if score >= 0.85:
        status = "high_input_quality"
    elif score >= 0.60:
        status = "medium_input_quality"
    else:
        status = "low_input_quality"

    return {
        "score": round(score, 3),
        "status": status,
    }
