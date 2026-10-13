import numpy as np
import pytest
from matplotlib.figure import Figure

from visionguard.processing import (
    ImageValidationError,
    compute_histograms,
    plot_histograms,
)


def test_grayscale_histogram_counts_each_intensity():
    image = np.array([[0, 0, 7], [128, 255, 255]], dtype=np.uint8)

    histogram = compute_histograms(image)

    assert list(histogram) == ["gray"]
    assert histogram["gray"].shape == (256,)
    assert histogram["gray"].sum() == image.size
    assert histogram["gray"][0] == 2
    assert histogram["gray"][255] == 2


def test_rgb_and_bgr_histograms_use_semantic_channel_names():
    rgb = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    bgr = rgb[..., ::-1].copy()

    rgb_histograms = compute_histograms(rgb, color_order="RGB")
    bgr_histograms = compute_histograms(bgr, color_order="BGR")

    assert list(rgb_histograms) == ["red", "green", "blue"]
    for channel in rgb_histograms:
        np.testing.assert_array_equal(rgb_histograms[channel], bgr_histograms[channel])


def test_rgba_histogram_includes_alpha_channel():
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    image[..., 3] = np.array([[0, 64], [128, 255]], dtype=np.uint8)

    histograms = compute_histograms(image)

    assert list(histograms) == ["red", "green", "blue", "alpha"]
    assert histograms["alpha"][0] == 1
    assert histograms["alpha"][64] == 1
    assert histograms["alpha"][128] == 1
    assert histograms["alpha"][255] == 1


def test_plot_histograms_adds_labeled_channels_to_supplied_axes():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    ax = Figure().subplots()

    returned_axes = plot_histograms(rgb, ax=ax, title="Fixture histogram")

    assert returned_axes is ax
    assert ax.get_title() == "Fixture histogram"
    assert [line.get_label() for line in ax.lines] == ["Red", "Green", "Blue"]
    assert ax.get_xlabel() == "Intensity (0–255)"


def test_histogram_rejects_unsupported_color_order():
    with pytest.raises(ImageValidationError, match="color_order"):
        compute_histograms(np.zeros((2, 2, 3), dtype=np.uint8), color_order="XYZ")
