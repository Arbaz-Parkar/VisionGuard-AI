"""NumPy FFT spectrum and radial frequency-domain filters."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .preprocessing import ImageValidationError

PassType = Literal["lowpass", "highpass"]
FilterType = Literal["ideal", "gaussian", "butterworth"]


def _real_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ImageValidationError("image must be a non-empty 2D NumPy array.")
    if (
        not np.issubdtype(image.dtype, np.number)
        or np.issubdtype(image.dtype, np.complexfloating)
        or not np.isfinite(image).all()
    ):
        raise ImageValidationError("image must contain finite real-valued pixels.")
    return image.astype(np.float64, copy=False)


def fft_magnitude_spectrum(image: np.ndarray, *, log_scale: bool = True) -> np.ndarray:
    """Return the centered magnitude of the 2D FFT, log-scaled by default."""
    if not isinstance(log_scale, bool):
        raise ImageValidationError("log_scale must be a boolean.")
    pixels = _real_image(image)
    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(pixels)))
    return np.log1p(magnitude) if log_scale else magnitude


def _radial_response(
    shape: tuple[int, int],
    cutoff_frequency: float,
    filter_type: FilterType,
    order: int,
) -> np.ndarray:
    frequency_y = np.fft.fftfreq(shape[0])[:, np.newaxis]
    frequency_x = np.fft.fftfreq(shape[1])[np.newaxis, :]
    radius = np.sqrt(frequency_x**2 + frequency_y**2)

    if filter_type == "ideal":
        return (radius <= cutoff_frequency).astype(np.float64)
    if filter_type == "gaussian":
        return np.exp(-0.5 * (radius / cutoff_frequency) ** 2)
    return 1.0 / (1.0 + (radius / cutoff_frequency) ** (2 * order))


def frequency_filter(
    image: np.ndarray,
    *,
    pass_type: PassType = "lowpass",
    filter_type: FilterType = "gaussian",
    cutoff_frequency: float = 0.1,
    order: int = 2,
) -> np.ndarray:
    """Apply a radial ideal, Gaussian, or Butterworth FFT filter.

    ``cutoff_frequency`` is measured in cycles per pixel and must be in
    ``(0, 0.5]``. High-pass response is the complement of the selected low-pass
    response. The output is a float64 image with the same height and width.
    """
    pixels = _real_image(image)
    if pass_type not in ("lowpass", "highpass"):
        raise ImageValidationError("pass_type must be either 'lowpass' or 'highpass'.")
    if filter_type not in ("ideal", "gaussian", "butterworth"):
        raise ImageValidationError(
            "filter_type must be 'ideal', 'gaussian', or 'butterworth'."
        )
    if (
        isinstance(cutoff_frequency, bool)
        or not isinstance(cutoff_frequency, (int, float))
        or not math.isfinite(cutoff_frequency)
        or not 0 < cutoff_frequency <= 0.5
    ):
        raise ImageValidationError("cutoff_frequency must be greater than 0 and at most 0.5.")
    if isinstance(order, bool) or not isinstance(order, int) or order <= 0:
        raise ImageValidationError("order must be a positive integer.")

    response = _radial_response(
        pixels.shape, float(cutoff_frequency), filter_type, order
    )
    if pass_type == "highpass":
        response = 1.0 - response
    filtered = np.fft.ifft2(np.fft.fft2(pixels) * response)
    return filtered.real
