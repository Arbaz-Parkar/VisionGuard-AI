"""Intensity histogram calculation and plotting helpers."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from .preprocessing import ImageValidationError, validate_image

ColorOrder = Literal["RGB", "BGR"]
HISTOGRAM_BINS = 256
_PLOT_COLORS = {
    "gray": "dimgray",
    "red": "red",
    "green": "green",
    "blue": "blue",
    "alpha": "purple",
}


def compute_histograms(
    image: np.ndarray, *, color_order: ColorOrder = "RGB"
) -> dict[str, np.ndarray]:
    """Count uint8 intensities for a grayscale or color image.

    The returned arrays always have 256 entries, indexed by intensity 0–255.
    Color results use semantic channel names (red, green, blue) regardless of
    whether the input is RGB or BGR. Four-channel inputs also include alpha.
    """
    pixels = validate_image(image)
    if color_order not in ("RGB", "BGR"):
        raise ImageValidationError("color_order must be either 'RGB' or 'BGR'.")

    if pixels.ndim == 2 or pixels.shape[2] == 1:
        gray = pixels if pixels.ndim == 2 else pixels[..., 0]
        return {"gray": np.bincount(gray.ravel(), minlength=HISTOGRAM_BINS)}

    red_index, blue_index = ((0, 2) if color_order == "RGB" else (2, 0))
    histograms = {
        "red": np.bincount(pixels[..., red_index].ravel(), minlength=HISTOGRAM_BINS),
        "green": np.bincount(pixels[..., 1].ravel(), minlength=HISTOGRAM_BINS),
        "blue": np.bincount(pixels[..., blue_index].ravel(), minlength=HISTOGRAM_BINS),
    }
    if pixels.shape[2] == 4:
        histograms["alpha"] = np.bincount(
            pixels[..., 3].ravel(), minlength=HISTOGRAM_BINS
        )
    return histograms


def plot_histograms(
    image: np.ndarray,
    *,
    color_order: ColorOrder = "RGB",
    ax: Axes | None = None,
    title: str = "Image intensity histogram",
) -> Axes:
    """Plot per-channel uint8 histograms and return the Matplotlib axes."""
    histograms = compute_histograms(image, color_order=color_order)
    if ax is None:
        _, ax = plt.subplots()

    intensity_values = np.arange(HISTOGRAM_BINS)
    for channel, counts in histograms.items():
        ax.plot(
            intensity_values,
            counts,
            color=_PLOT_COLORS[channel],
            label=channel.title(),
        )
    ax.set_title(title)
    ax.set_xlabel("Intensity (0–255)")
    ax.set_ylabel("Pixel count")
    ax.set_xlim(0, HISTOGRAM_BINS - 1)
    ax.legend()
    return ax
