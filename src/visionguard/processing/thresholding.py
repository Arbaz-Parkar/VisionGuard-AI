"""Binary thresholding methods for grayscale and color images."""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

from .preprocessing import ImageValidationError, to_grayscale

ColorOrder = Literal["RGB", "BGR"]
AdaptiveMethod = Literal["mean", "gaussian"]


def _check_invert(invert: bool) -> None:
    if not isinstance(invert, bool):
        raise ImageValidationError("invert must be a boolean.")


def global_threshold(
    image: np.ndarray,
    threshold: int = 127,
    *,
    invert: bool = False,
    color_order: ColorOrder = "RGB",
) -> np.ndarray:
    """Apply a fixed intensity threshold and return a 0/255 uint8 mask."""
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 255:
        raise ImageValidationError("threshold must be an integer from 0 through 255.")
    _check_invert(invert)
    gray = to_grayscale(image, color_order=color_order)
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, threshold, 255, threshold_type)
    return mask


def otsu_threshold(
    image: np.ndarray,
    *,
    invert: bool = False,
    color_order: ColorOrder = "RGB",
) -> tuple[int, np.ndarray]:
    """Choose a global threshold with Otsu's method and return (value, mask)."""
    _check_invert(invert)
    gray = to_grayscale(image, color_order=color_order)
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    threshold_value, mask = cv2.threshold(
        gray, 0, 255, threshold_type | cv2.THRESH_OTSU
    )
    return int(round(threshold_value)), mask


def adaptive_threshold(
    image: np.ndarray,
    *,
    block_size: int = 11,
    c: float = 2.0,
    method: AdaptiveMethod = "gaussian",
    invert: bool = False,
    color_order: ColorOrder = "RGB",
) -> np.ndarray:
    """Apply local mean or Gaussian thresholding and return a binary mask.

    ``block_size`` is the odd neighborhood width. ``c`` is subtracted from the
    local neighborhood threshold, following OpenCV's adaptive-threshold rule.
    """
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 3
        or block_size % 2 == 0
    ):
        raise ImageValidationError("block_size must be an odd integer greater than or equal to 3.")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c):
        raise ImageValidationError("c must be a finite number.")
    if method not in ("mean", "gaussian"):
        raise ImageValidationError("method must be either 'mean' or 'gaussian'.")
    _check_invert(invert)

    gray = to_grayscale(image, color_order=color_order)
    adaptive_method = (
        cv2.ADAPTIVE_THRESH_MEAN_C if method == "mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    )
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=adaptive_method,
        thresholdType=threshold_type,
        blockSize=block_size,
        C=float(c),
    )
