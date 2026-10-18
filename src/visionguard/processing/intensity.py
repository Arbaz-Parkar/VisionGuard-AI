"""Point transforms and histogram equalization for uint8 images."""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

from .preprocessing import ImageValidationError, validate_image

ColorOrder = Literal["RGB", "BGR"]


def _check_order(color_order: ColorOrder) -> None:
    if color_order not in ("RGB", "BGR"):
        raise ImageValidationError("color_order must be either 'RGB' or 'BGR'.")


def _map_color_channels(image: np.ndarray, table: np.ndarray) -> np.ndarray:
    result = image.copy()
    if result.ndim == 2:
        result[...] = table[result]
    elif result.shape[2] == 1:
        result[..., 0] = table[result[..., 0]]
    else:
        channels = 3 if result.shape[2] == 4 else result.shape[2]
        result[..., :channels] = table[result[..., :channels]]
    return result


def log_transform(image: np.ndarray) -> np.ndarray:
    """Apply a normalized logarithmic transform to intensity channels."""
    pixels = validate_image(image)
    table = np.rint(255.0 * np.log1p(np.arange(256) / 255.0) / math.log(2.0))
    return _map_color_channels(pixels, np.clip(table, 0, 255).astype(np.uint8))


def gamma_transform(image: np.ndarray, *, gamma: float = 1.0) -> np.ndarray:
    """Apply ``output = 255 * (input / 255) ** gamma`` to color/intensity channels."""
    pixels = validate_image(image)
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or gamma <= 0:
        raise ImageValidationError("gamma must be a finite number greater than 0.")
    table = np.rint(255.0 * (np.arange(256) / 255.0) ** float(gamma)).astype(np.uint8)
    return _map_color_channels(pixels, table)


def contrast_stretch(
    image: np.ndarray, *, input_min: int = 0, input_max: int = 255
) -> np.ndarray:
    """Linearly stretch a selected input intensity interval to 0–255."""
    pixels = validate_image(image)
    for name, value in (("input_min", input_min), ("input_max", input_max)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
            raise ImageValidationError(f"{name} must be an integer from 0 through 255.")
    if input_min >= input_max:
        raise ImageValidationError("input_min must be less than input_max.")
    table = np.rint(
        np.clip((np.arange(256) - input_min) * 255.0 / (input_max - input_min), 0, 255)
    ).astype(np.uint8)
    return _map_color_channels(pixels, table)


def equalize_histogram(
    image: np.ndarray, *, color_order: ColorOrder = "RGB"
) -> np.ndarray:
    """Equalize grayscale or the CIELAB lightness of a color image."""
    pixels = validate_image(image)
    _check_order(color_order)
    if pixels.ndim == 2:
        return cv2.equalizeHist(pixels)
    if pixels.shape[2] == 1:
        return cv2.equalizeHist(pixels[..., 0])[..., np.newaxis]
    conversion = cv2.COLOR_RGB2LAB if color_order == "RGB" else cv2.COLOR_BGR2LAB
    inverse = cv2.COLOR_LAB2RGB if color_order == "RGB" else cv2.COLOR_LAB2BGR
    lab = cv2.cvtColor(pixels[..., :3], conversion)
    lab[..., 0] = cv2.equalizeHist(lab[..., 0])
    enhanced = cv2.cvtColor(lab, inverse)
    if pixels.shape[2] == 4:
        return np.dstack((enhanced, pixels[..., 3]))
    return enhanced
