import cv2
import numpy as np
import pytest

from visionguard.lab_ui import _run


@pytest.fixture
def sample_image():
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.rectangle(image, (18, 18), (50, 50), (220, 220, 220), -1)
    cv2.circle(image, (68, 68), 14, (180, 180, 180), -1)
    return image


@pytest.mark.parametrize(
    "operation,params",
    [
        ("Grayscale", {}),
        ("RGB conversion", {}),
        ("HSV channels", {"channel": "H"}),
        ("Lab channels", {"channel": "L"}),
        ("Log transform", {}),
        ("Gamma transform", {"gamma": 1.4}),
        ("Contrast stretch", {"input_min": 20, "input_max": 235}),
        ("Histogram equalization", {}),
        ("CLAHE", {"clip_limit": 2.0, "tile": 8}),
        ("Histogram", {}),
        ("Gaussian smoothing", {"kernel_size": 5, "sigma": 0.0}),
        ("Median filtering", {"kernel_size": 5}),
        ("Gaussian denoising", {"kernel_size": 5, "sigma": 0.0}),
        ("Median denoising", {"kernel_size": 5}),
        ("Laplacian sharpening", {"amount": 0.5, "aperture_size": 3}),
        ("Unsharp sharpening", {"amount": 1.0, "sigma": 1.0}),
        ("Convolution", {"kernel": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)}),
        ("Gaussian noise", {"sigma": 20.0, "seed": 7}),
        ("Salt-and-pepper noise", {"amount": 0.05, "seed": 7}),
        ("FFT spectrum", {}),
        ("FFT filter", {"pass_type": "lowpass", "filter_type": "gaussian", "cutoff_frequency": 0.1}),
        ("Template matching", {"x": 0, "y": 0, "width": 16, "height": 16, "method": "ccoeff_normed"}),
        ("Sobel edges", {"aperture_size": 3}),
        ("Laplacian edges", {"aperture_size": 3}),
        ("LoG edges", {"sigma": 1.0}),
        ("Canny edges", {"low_threshold": 50, "high_threshold": 150}),
        ("Erode", {"kernel_size": 3, "kernel_shape": "rect", "iterations": 1}),
        ("Dilate", {"kernel_size": 3, "kernel_shape": "rect", "iterations": 1}),
        ("Open", {"kernel_size": 3, "kernel_shape": "rect", "iterations": 1}),
        ("Close", {"kernel_size": 3, "kernel_shape": "rect", "iterations": 1}),
        ("Top-hat", {"kernel_size": 3, "kernel_shape": "rect"}),
        ("Global threshold", {"threshold": 127, "invert": False}),
        ("Otsu threshold", {"invert": False}),
        ("Adaptive threshold", {"block_size": 11, "c": 2.0, "method": "gaussian", "invert": False}),
        ("Connected regions", {"threshold": 127, "min_area": 0.0, "invert": False}),
        ("Watershed", {"foreground": "bright", "distance_ratio": 0.4}),
        ("Harris corners", {"threshold": 0.01}),
        ("HOG features", {"cell_size": 8, "orientations": 9}),
        ("Hough lines", {"threshold": 10, "min_line_length": 10}),
        ("Hough circles", {"param2": 20.0, "min_distance": 20.0}),
        ("ORB features", {}),
    ],
)
def test_lab_dispatch_runs_each_operation(operation, params, sample_image):
    result = _run(operation, sample_image, params, "RGB")

    assert isinstance(result, tuple)
    assert len(result) in (2, 3)
