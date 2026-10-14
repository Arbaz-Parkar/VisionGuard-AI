import cv2
import numpy as np
import pytest

from visionguard.processing import ImageValidationError, convolve2d


def test_numpy_convolution_matches_opencv_reflect_border_on_synthetic_input():
    rng = np.random.default_rng(31)
    image = rng.normal(size=(11, 13)).astype(np.float64)
    kernel = np.array(
        [[0.25, -0.5, 0.75], [1.0, 0.125, -0.25], [0.0, 0.5, 0.25]],
        dtype=np.float64,
    )

    actual = convolve2d(image, kernel, border="reflect")
    opencv_kernel = kernel[::-1, ::-1].copy()
    expected = cv2.filter2D(
        image,
        ddepth=cv2.CV_64F,
        kernel=opencv_kernel,
        borderType=cv2.BORDER_REFLECT_101,
    )

    assert actual.shape == image.shape
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_numpy_convolution_matches_opencv_zero_constant_border():
    image = np.arange(1, 21, dtype=np.uint8).reshape(4, 5)
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    actual = convolve2d(image, kernel, border="constant")
    expected = cv2.filter2D(
        image,
        ddepth=cv2.CV_32F,
        kernel=kernel[::-1, ::-1].copy(),
        borderType=cv2.BORDER_CONSTANT,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_one_by_one_kernel_scales_without_changing_shape():
    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    result = convolve2d(image, np.array([[2]], dtype=np.float32))

    assert result.shape == image.shape
    np.testing.assert_array_equal(result, [[2, 4], [6, 8]])


@pytest.mark.parametrize(
    "image, kernel, kwargs",
    [
        (np.zeros((3, 3), dtype=np.uint8), np.ones((2, 3)), {}),
        (np.zeros((3, 3), dtype=np.uint8), np.array([[np.nan]]), {}),
        (np.array([1, 2, 3]), np.ones((1, 1)), {}),
        (np.zeros((3, 3)), np.ones((1, 1)), {"border": "wrap"}),
        (np.zeros((3, 3)), np.ones((1, 1)), {"constant_value": float("inf")}),
    ],
)
def test_convolution_rejects_invalid_images_kernels_or_options(image, kernel, kwargs):
    with pytest.raises(ImageValidationError):
        convolve2d(image, kernel, **kwargs)
