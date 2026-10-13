import numpy as np
import pytest

from visionguard.processing import (
    ImageValidationError,
    adaptive_threshold,
    global_threshold,
    otsu_threshold,
)


def test_global_threshold_returns_expected_binary_mask():
    image = np.array([[0, 100, 128, 255]], dtype=np.uint8)

    mask = global_threshold(image, threshold=127)
    inverted = global_threshold(image, threshold=127, invert=True)

    np.testing.assert_array_equal(mask, [[0, 0, 255, 255]])
    np.testing.assert_array_equal(inverted, [[255, 255, 0, 0]])


def test_otsu_selects_threshold_for_two_intensity_groups():
    image = np.full((16, 16), 20, dtype=np.uint8)
    image[:, 8:] = 220

    threshold, mask = otsu_threshold(image)

    assert 20 <= threshold < 220
    assert mask.shape == image.shape
    np.testing.assert_array_equal(mask[:, :8], 0)
    np.testing.assert_array_equal(mask[:, 8:], 255)


@pytest.mark.parametrize("method", ["mean", "gaussian"])
def test_adaptive_threshold_supports_local_methods(method):
    image = np.full((32, 32), 30, dtype=np.uint8)
    image[8:24, 8:24] = 220

    mask = adaptive_threshold(image, block_size=7, c=2, method=method)

    assert mask.shape == image.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert np.count_nonzero(mask) > 0


def test_color_inputs_respect_rgb_and_bgr_order():
    rgb = np.zeros((12, 12, 3), dtype=np.uint8)
    rgb[3:9, 3:9] = (255, 20, 5)
    bgr = rgb[..., ::-1].copy()

    rgb_mask = global_threshold(rgb, threshold=50, color_order="RGB")
    bgr_mask = global_threshold(bgr, threshold=50, color_order="BGR")

    np.testing.assert_array_equal(rgb_mask, bgr_mask)


@pytest.mark.parametrize(
    "call",
    [
        lambda image: global_threshold(image, threshold=-1),
        lambda image: global_threshold(image, invert=1),
        lambda image: adaptive_threshold(image, block_size=4),
        lambda image: adaptive_threshold(image, c=float("inf")),
        lambda image: adaptive_threshold(image, method="median"),
    ],
)
def test_thresholding_rejects_invalid_parameters(call):
    with pytest.raises(ImageValidationError):
        call(np.zeros((8, 8), dtype=np.uint8))
