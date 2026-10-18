import cv2
import numpy as np
import pytest

from visionguard.processing import (
    ImageValidationError,
    contrast_stretch,
    equalize_histogram,
    gamma_transform,
    harris_corners,
    hog_features,
    hough_circles,
    hough_lines,
    laplacian_edges,
    laplacian_of_gaussian,
    laplacian_sharpen,
    log_transform,
    sobel_edges,
    to_hsv,
    to_lab,
    watershed_segmentation,
)


def test_hsv_and_lab_conversions_respect_rgb_and_bgr_order():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:] = (255, 0, 0)

    hsv_rgb = to_hsv(rgb)
    lab_rgb = to_lab(rgb)
    hsv_bgr = to_hsv(rgb[..., ::-1].copy(), color_order="BGR")
    lab_bgr = to_lab(rgb[..., ::-1].copy(), color_order="BGR")

    assert hsv_rgb.shape == lab_rgb.shape == rgb.shape
    assert hsv_rgb[0, 0, 0] == 0
    assert lab_rgb[0, 0, 0] > 100
    np.testing.assert_array_equal(hsv_rgb, hsv_bgr)
    np.testing.assert_array_equal(lab_rgb, lab_bgr)


def test_point_transforms_preserve_alpha_and_have_expected_monotonic_effect():
    pixels = np.array([[0, 64, 128, 255]], dtype=np.uint8)
    rgba = np.dstack((np.repeat(pixels[..., None], 3, axis=2), np.full((1, 4), 127, dtype=np.uint8)))

    gamma = gamma_transform(rgba, gamma=2.0)
    stretched = contrast_stretch(pixels, input_min=64, input_max=128)
    logged = log_transform(pixels)

    assert gamma[0, 1, 0] < rgba[0, 1, 0]
    np.testing.assert_array_equal(gamma[..., 3], rgba[..., 3])
    assert stretched.tolist() == [[0, 0, 255, 255]]
    assert logged[0, 2] > pixels[0, 2]


def test_laplacian_sharpening_preserves_shape_dtype_and_alpha():
    gray = np.zeros((15, 15), dtype=np.uint8)
    gray[5:10, 5:10] = 128
    rgba = np.dstack((np.repeat(gray[..., None], 3, axis=2), np.full(gray.shape, 91, dtype=np.uint8)))

    sharpened = laplacian_sharpen(rgba, amount=0.5)

    assert sharpened.shape == rgba.shape
    assert sharpened.dtype == np.uint8
    np.testing.assert_array_equal(sharpened[..., 3], rgba[..., 3])
    assert not np.array_equal(sharpened[..., :3], rgba[..., :3])


def test_histogram_equalization_changes_luminance_and_preserves_alpha():
    gray = np.tile(np.arange(64, dtype=np.uint8), (32, 1))
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgba = np.dstack((rgb, np.full(gray.shape, 137, dtype=np.uint8)))

    equalized = equalize_histogram(rgba)

    assert equalized.shape == rgba.shape
    np.testing.assert_array_equal(equalized[..., 3], rgba[..., 3])
    assert int(equalized[..., :3].max()) > int(rgba[..., :3].max())


@pytest.mark.parametrize("operation", [sobel_edges, laplacian_edges, laplacian_of_gaussian])
def test_extended_edges_return_displayable_maps(operation):
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 255

    edges = operation(image)

    assert edges.shape == image.shape
    assert edges.dtype == np.uint8
    assert np.count_nonzero(edges) > 0


def test_feature_detectors_and_watershed_return_visualizations():
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.rectangle(image, (18, 18), (42, 42), (255, 255, 255), -1)
    cv2.circle(image, (68, 68), 14, (200, 200, 200), -1)

    corners = harris_corners(image)
    hog = hog_features(image)
    lines = hough_lines(image, threshold=10, min_line_length=10)
    circles = hough_circles(image, param2=20)
    watershed = watershed_segmentation(image)

    assert corners.visualization.shape == image.shape
    assert len(corners.coordinates) > 0
    assert hog.visualization.shape == image.shape
    assert hog.descriptor.ndim == 1 and hog.descriptor.size > 0
    assert lines.visualization.shape == circles.visualization.shape == image.shape
    assert watershed.visualization.shape == image.shape
    assert watershed.labels.shape == image.shape[:2]


@pytest.mark.parametrize(
    "function,kwargs",
    [
        (gamma_transform, {"gamma": 0}),
        (contrast_stretch, {"input_min": 30, "input_max": 30}),
        (sobel_edges, {"aperture_size": 4}),
        (laplacian_edges, {"aperture_size": 4}),
        (laplacian_of_gaussian, {"sigma": 0}),
        (harris_corners, {"threshold": 0}),
        (hog_features, {"orientations": 1}),
        (watershed_segmentation, {"distance_ratio": 1}),
    ],
)
def test_new_lab_operations_reject_invalid_parameters(function, kwargs):
    with pytest.raises(ImageValidationError):
        function(np.zeros((16, 16), dtype=np.uint8), **kwargs)
