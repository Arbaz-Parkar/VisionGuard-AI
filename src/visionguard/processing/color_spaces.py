"""Color-space conversions used by the Image Processing Lab."""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import ColorOrder, validate_image, to_rgb


def to_hsv(image: np.ndarray, *, color_order: ColorOrder = "RGB") -> np.ndarray:
    """Convert to OpenCV uint8 HSV (H 0–179, S/V 0–255)."""
    pixels = validate_image(image)
    rgb = to_rgb(pixels, color_order=color_order)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def to_lab(image: np.ndarray, *, color_order: ColorOrder = "RGB") -> np.ndarray:
    """Convert to OpenCV uint8 CIELAB (L 0–255, a/b offset by 128)."""
    pixels = validate_image(image)
    rgb = to_rgb(pixels, color_order=color_order)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
