
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_test_torso():
    """
    Create a synthetic torso with varying elliptical cross-sections.
    Dimensions are illustrative, not anatomical measurements.
    """
    # Height in metres, half-width in metres, half-depth in metres
    profile = [
        (-0.45, 0.14, 0.105),
        (-0.30, 0.16, 0.115),
        (-0.15, 0.18, 0.125),
        ( 0.00, 0.17, 0.120),
        ( 0.15, 0.20, 0.135),
        ( 0.30, 0.22, 0.145),
        ( 0.45, 0.18, 0.120),
    ]

    segments = 64
    vertices = []

    for z, rx, ry in profile:
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            x = rx * np.cos(theta)
            y = ry * np.sin(theta)
            vertices.append([x, y, z])

    faces = []

    # Join adjacent elliptical rings with triangles.
    for ring in range(len(profile) - 1):
        for i in range(segments):
            j = (i + 1) % segments

            a = ring * segments + i
            b = ring * segments + j
            c = (ring + 1) * segments + i
            d = (ring + 1) * segments + j

            faces.append([a, b, c])
            faces.append([b, d, c])

    # Close the bottom and top with centre vertices.
    bottom_center = len(vertices)
    vertices.append([0, 0, profile[0][0]])

    top_center = len(vertices)
    vertices.append([0, 0, profile[-1][0]])

    for i in range(segments):
        j = (i + 1) % segments

        faces.append([bottom_center, j, i])

        last_ring = (len(profile) - 1) * segments
        faces.append([
            top_center,
            last_ring + i,
            last_ring + j,
        ])

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices),
        faces=np.asarray(faces),
        process=True,
    )

    # Ensure consistent face winding and outward normals.
    trimesh.repair.fix_normals(mesh)

    return mesh


def main():
    mesh = create_test_torso()
    output_path = OUTPUT_DIR / "test_torso.stl"
    mesh.export(output_path)

    print("Variable-profile test torso created.")
    print("Saved to:", output_path)
    print("Vertices:", len(mesh.vertices))
    print("Faces:", len(mesh.faces))
    print("Watertight:", mesh.is_watertight)
    print("Extents:", mesh.extents)


if __name__ == "__main__":
    main()
