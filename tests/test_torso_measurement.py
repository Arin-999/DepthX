
import pytest
import trimesh

from src.torso_measurement import calculate_section_perimeter


def test_cylinder_cross_section():
    mesh = trimesh.creation.cylinder(
        radius=0.18,
        height=0.9,
        sections=256,
    )

    perimeter = calculate_section_perimeter(mesh, 0.0)

    # A polygon approximates the circular cross-section.
    expected = 2 * 256 * 0.18 * __import__("math").sin(
        __import__("math").pi / 256
    )

    assert perimeter == pytest.approx(expected, abs=0.01)


def test_no_intersection_raises_error():
    mesh = trimesh.creation.cylinder(
        radius=0.18,
        height=0.9,
        sections=64,
    )

    with pytest.raises(ValueError):
        calculate_section_perimeter(mesh, 1.0)
