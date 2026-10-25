
import numpy as np


def cross_section_perimeter(points_xyz):
    """
    Calculate the perimeter of a closed 3D contour.

    points_xyz:
        Ordered points around a closed contour, shape (N, 3).
        The first point is automatically connected to the last.

    Returns:
        Perimeter in the same units as the input coordinates.

    Important:
        The contour must represent the actual body surface.
        This function does not reconstruct a torso from landmarks.
    """
    points = np.asarray(points_xyz, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an array with shape (N, 3).")

    if len(points) < 3:
        raise ValueError("At least three contour points are required.")

    if not np.isfinite(points).all():
        raise ValueError("Contour points must be finite.")

    closed = np.vstack([points, points[0]])
    segment_lengths = np.linalg.norm(
        np.diff(closed, axis=0),
        axis=1,
    )

    return float(segment_lengths.sum())


def circumference_status(contour_available, depth_valid):
    """Report whether a torso circumference can be estimated."""
    if not contour_available:
        return {
            "value_cm": None,
            "status": "not_measured",
            "reason": "No reconstructed torso contour is available.",
        }

    if not depth_valid:
        return {
            "value_cm": None,
            "status": "low_confidence",
            "reason": "Insufficient valid depth data.",
        }

    return {
        "value_cm": None,
        "status": "ready_for_contour",
        "reason": (
            "Contour supplied; calculate its perimeter and "
            "validate against tape measurements."
        ),
    }
