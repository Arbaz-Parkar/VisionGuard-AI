import numpy as np
import pytest

from visionguard.processing import (
    ImageValidationError,
    apply_clahe,
    gaussian_blur,
    median_blur,
)


def test_gaussian_blur_smooths_and_preserves_image_shape():
    image = np.zeros((9, 9, 1), dtype=np.uint8)
    image[4, 4, 0] = 255

    blurred = gaussian_blur(image, kernel_size=3)

    assert blurred.shape == image.shape
    assert blurred.dtype == image.dtype
    assert 0 < blurred[4, 4, 0] < 255


def test_median_blur_removes_an_isolated_impulse():
    image = np.zeros((7, 7), dtype=np.uint8)
    image[3, 3] = 255

    filtered = median_blur(image, kernel_size=3)

    assert filtered[3, 3] == 0
    assert filtered.shape == image.shape


def test_clahe_accepts_grayscale_and_preserves_rgba_alpha():
    gray = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[..., 0] = np.arange(16, dtype=np.uint8)[None, :]
    rgba[..., 1] = 80
    rgba[..., 2] = 140
    rgba[..., 3] = np.arange(16, dtype=np.uint8)[:, None] * 17

    enhanced_gray = apply_clahe(gray)
    enhanced_rgba = apply_clahe(rgba)

    assert enhanced_gray.shape == gray.shape
    assert enhanced_gray.dtype == gray.dtype
    assert enhanced_rgba.shape == rgba.shape
    assert enhanced_rgba.dtype == rgba.dtype
    np.testing.assert_array_equal(enhanced_rgba[..., 3], rgba[..., 3])


def test_clahe_color_order_produces_equivalent_rgb_and_bgr_results():
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(24, dtype=np.uint8)[None, :] * 10
    rgb[..., 1] = np.arange(24, dtype=np.uint8)[:, None] * 8
    rgb[..., 2] = 120

    enhanced_rgb = apply_clahe(rgb, color_order="RGB")
    enhanced_bgr = apply_clahe(rgb[..., ::-1].copy(), color_order="BGR")

    np.testing.assert_array_equal(enhanced_rgb, enhanced_bgr[..., ::-1])


@pytest.mark.parametrize(
    "call",
    [
        lambda image: gaussian_blur(image, kernel_size=2),
        lambda image: gaussian_blur(image, sigma=-1),
        lambda image: median_blur(image, kernel_size=1),
        lambda image: apply_clahe(image, clip_limit=0),
        lambda image: apply_clahe(image, tile_grid_size=(0, 8)),
        lambda image: apply_clahe(image, color_order="XYZ"),
    ],
)
def test_filters_reject_invalid_parameters(call):
    with pytest.raises(ImageValidationError):
        call(np.zeros((8, 8), dtype=np.uint8))
