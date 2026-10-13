"""Edge detection helpers."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from .preprocessing import ImageValidationError, to_grayscale

ColorOrder = Literal["RGB", "BGR"]
_SUPPORTED_APERTURES = frozenset({3, 5, 7})


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
