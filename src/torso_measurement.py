
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
MESH_PATH = ROOT / "outputs" / "test_torso.stl"


def calculate_section_perimeter(mesh, z_level):
    """
    Intersect the mesh with a horizontal plane at z_level.
    Return the total closed-contour perimeter in mesh units.
    """
    section = mesh.section(
        plane_origin=[0.0, 0.0, z_level],
        plane_normal=[0.0, 0.0, 1.0],
    )

    if section is None:
        raise ValueError(
            f"No mesh intersection at z={z_level}"
        )

    total_perimeter = 0.0
    contour_count = 0

    for contour in section.discrete:
        points = np.asarray(contour, dtype=float)

        if len(points) < 3:
            continue

        # Close the contour by connecting the final point
        # back to the first point.
        closed = np.vstack([points, points[0]])

        segment_lengths = np.linalg.norm(
            np.diff(closed, axis=0),
            axis=1,
        )

        total_perimeter += float(segment_lengths.sum())
        contour_count += 1

    if contour_count == 0:
        raise ValueError("No valid closed contours found.")

    return total_perimeter


def main():
    if not MESH_PATH.exists():
        raise FileNotFoundError(
            "Run python src/body_mesh.py first."
        )

    mesh = trimesh.load_mesh(MESH_PATH)

    if not mesh.is_watertight:
        print("Warning: mesh is not watertight.")

    # These are test plane positions, not anatomical locations.
    measurement_levels = {
        "test_upper_torso": 0.20,
        "test_lower_torso": -0.20,
    }

    print("Mesh units: assumed metres for this test")
    print("Mesh extents:", mesh.extents)

    for name, z_level in measurement_levels.items():
        perimeter_m = calculate_section_perimeter(
            mesh, z_level
        )

        print(
            f"{name}: "
            f"{perimeter_m:.4f} m "
            f"({perimeter_m * 100:.2f} cm)"
        )


if __name__ == "__main__":
    main()
