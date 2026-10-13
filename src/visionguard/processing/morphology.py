"""Morphological operations for binary threshold and defect masks."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from .preprocessing import ImageValidationError, validate_image

MorphologyOperation = Literal["erode", "dilate", "open", "close", "gradient"]
KernelShape = Literal["rect", "ellipse", "cross"]
_OPERATIONS = frozenset({"erode", "dilate", "open", "close", "gradient"})
_KERNEL_SHAPES = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


def morphology(
    mask: np.ndarray,
    operation: MorphologyOperation,
    *,
    kernel_size: int = 3,
    kernel_shape: KernelShape = "rect",
    iterations: int = 1,
) -> np.ndarray:
    """Apply a morphological operation to a 2D uint8 binary mask.

    Masks must contain only 0 and 255 values. Opening removes small foreground
    specks, closing fills small holes, and gradient highlights mask boundaries.
    """
    pixels = validate_image(mask)
    if pixels.ndim != 2:
        raise ImageValidationError("Morphology expects a 2D binary mask.")
    if not np.isin(pixels, (0, 255)).all():
        raise ImageValidationError("Binary masks must contain only 0 and 255 values.")
    if operation not in _OPERATIONS:
        raise ImageValidationError(
            "operation must be one of 'erode', 'dilate', 'open', 'close', or 'gradient'."
        )
    if (
        isinstance(kernel_size, bool)
        or not isinstance(kernel_size, int)
        or kernel_size <= 0
        or kernel_size % 2 == 0
    ):
        raise ImageValidationError("kernel_size must be a positive odd integer.")
    if kernel_shape not in _KERNEL_SHAPES:
        raise ImageValidationError("kernel_shape must be 'rect', 'ellipse', or 'cross'.")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ImageValidationError("iterations must be a positive integer.")

    kernel = cv2.getStructuringElement(
        _KERNEL_SHAPES[kernel_shape], (kernel_size, kernel_size)
    )
    if operation == "erode":
        result = cv2.erode(pixels, kernel, iterations=iterations)
    elif operation == "dilate":
        result = cv2.dilate(pixels, kernel, iterations=iterations)
    else:
        operation_code = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
        }[operation]
        result = cv2.morphologyEx(
            pixels, operation_code, kernel, iterations=iterations
        )
    return result
