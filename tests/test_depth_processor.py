import numpy as np
import pytest

from src.depth_processor import pixel_to_3d, sample_depth_median


CAMERA = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}


def test_sample_depth_median_ignores_invalid_samples():
    depth = np.array(
        [
            [0.0, np.nan, 1.8],
            [np.inf, 2.0, 2.2],
            [-1.0, 2.4, 0.0],
        ],
        dtype=np.float32,
    )

    value, valid_count = sample_depth_median(depth, 1, 1, radius=1)

    assert value == pytest.approx(2.1)
    assert valid_count == 4


@pytest.mark.parametrize(
    "depth, camera, expected",
    [
        (0.0, CAMERA, "positive"),
        (float("nan"), CAMERA, "finite"),
        (2.0, {**CAMERA, "fx": float("inf")}, "finite"),
        (2.0, {**CAMERA, "fy": 0.0}, "positive"),
    ],
)
def test_pixel_to_3d_rejects_invalid_depth_or_intrinsics(depth, camera, expected):
    with pytest.raises(ValueError, match=expected):
        pixel_to_3d(60.0, 40.0, depth, camera)


def test_pixel_to_3d_rejects_non_finite_pixel_coordinates():
    with pytest.raises(ValueError, match="finite"):
        pixel_to_3d(float("nan"), 40.0, 2.0, CAMERA)


def test_sample_depth_median_rejects_invalid_shape_and_radius():
    with pytest.raises(ValueError, match="2D numeric"):
        sample_depth_median(np.ones((4, 4, 2)), 1, 1)
    with pytest.raises(ValueError, match="non-negative integer"):
        sample_depth_median(np.ones((4, 4)), 1, 1, radius=-1)
