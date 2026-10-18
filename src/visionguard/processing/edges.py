"""Edge detection helpers."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from .preprocessing import ImageValidationError, to_grayscale

ColorOrder = Literal["RGB", "BGR"]
_SUPPORTED_APERTURES = frozenset({3, 5, 7})


def _normalized_magnitude(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values.astype(np.float32))
    maximum = float(magnitude.max(initial=0.0))
    if maximum <= 0:
        return np.zeros(magnitude.shape, dtype=np.uint8)
    return np.clip(np.rint(magnitude * (255.0 / maximum)), 0, 255).astype(np.uint8)


def sobel_edges(
    image: np.ndarray, *, aperture_size: int = 3, color_order: ColorOrder = "RGB"
) -> np.ndarray:
    """Return a display-normalized Sobel gradient-magnitude image."""
    if isinstance(aperture_size, bool) or not isinstance(aperture_size, int) or aperture_size not in _SUPPORTED_APERTURES:
        raise ImageValidationError("aperture_size must be one of 3, 5, or 7.")
    gray = to_grayscale(image, color_order=color_order)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=aperture_size)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=aperture_size)
    return _normalized_magnitude(cv2.magnitude(dx, dy))


def laplacian_edges(
    image: np.ndarray, *, aperture_size: int = 3, color_order: ColorOrder = "RGB"
) -> np.ndarray:
    """Return a display-normalized absolute Laplacian response."""
    if isinstance(aperture_size, bool) or not isinstance(aperture_size, int) or aperture_size not in _SUPPORTED_APERTURES:
        raise ImageValidationError("aperture_size must be one of 3, 5, or 7.")
    gray = to_grayscale(image, color_order=color_order)
    response = cv2.Laplacian(gray, cv2.CV_32F, ksize=aperture_size)
    return _normalized_magnitude(response)


def laplacian_of_gaussian(
    image: np.ndarray, *, sigma: float = 1.0, color_order: ColorOrder = "RGB"
) -> np.ndarray:
    """Smooth with a Gaussian, then return the normalized Laplacian response."""
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not np.isfinite(sigma) or sigma <= 0:
        raise ImageValidationError("sigma must be a finite number greater than 0.")
    gray = to_grayscale(image, color_order=color_order)
    radius = max(1, int(np.ceil(3 * float(sigma))))
    blurred = cv2.GaussianBlur(gray, (2 * radius + 1, 2 * radius + 1), sigmaX=float(sigma))
    return _normalized_magnitude(cv2.Laplacian(blurred, cv2.CV_32F, ksize=3))


def canny_edges(
    image: np.ndarray,
    *,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3,
    l2gradient: bool = False,
    color_order: ColorOrder = "RGB",
) -> np.ndarray:
    """Return a uint8 Canny edge map from a grayscale or color image.

    Color inputs are converted to grayscale using the declared channel order.
    The returned map has the source image's height and width, with edge pixels
    set to 255 and non-edge pixels set to 0.
    """
    for name, value in (("low_threshold", low_threshold), ("high_threshold", high_threshold)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
            raise ImageValidationError(f"{name} must be an integer from 0 through 255.")
    if low_threshold >= high_threshold:
        raise ImageValidationError("low_threshold must be less than high_threshold.")
    if (
        isinstance(aperture_size, bool)
        or not isinstance(aperture_size, int)
        or aperture_size not in _SUPPORTED_APERTURES
    ):
        raise ImageValidationError("aperture_size must be one of 3, 5, or 7.")
    if not isinstance(l2gradient, bool):
        raise ImageValidationError("l2gradient must be a boolean.")

    gray = to_grayscale(image, color_order=color_order)
    return cv2.Canny(
        gray,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=aperture_size,
        L2gradient=l2gradient,
    )
