import numpy as np
import pytest

from visionguard.processing import ImageValidationError, canny_edges


def test_canny_detects_edges_and_returns_binary_uint8_map():
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 255

    edges = canny_edges(image)

    assert edges.shape == image.shape
    assert edges.dtype == np.uint8
    assert set(np.unique(edges)).issubset({0, 255})
    assert np.count_nonzero(edges) > 0


def test_canny_rgb_and_bgr_inputs_produce_the_same_edge_map():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[8:24, 8:24] = (240, 100, 30)
    bgr = rgb[..., ::-1].copy()

    rgb_edges = canny_edges(rgb, low_threshold=20, high_threshold=80, color_order="RGB")
    bgr_edges = canny_edges(bgr, low_threshold=20, high_threshold=80, color_order="BGR")

    np.testing.assert_array_equal(rgb_edges, bgr_edges)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"low_threshold": -1},
        {"high_threshold": 256},
        {"low_threshold": 100, "high_threshold": 100},
        {"aperture_size": 4},
        {"l2gradient": 1},
    ],
)
def test_canny_rejects_invalid_parameters(kwargs):
    with pytest.raises(ImageValidationError):
        canny_edges(np.zeros((8, 8), dtype=np.uint8), **kwargs)
