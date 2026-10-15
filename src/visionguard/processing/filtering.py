"""Smoothing, sharpening, and local contrast enhancement for uint8 images."""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

from .preprocessing import ImageValidationError, validate_image

ColorOrder = Literal["RGB", "BGR"]


def _positive_odd_kernel(kernel_size: int, *, minimum: int = 1) -> int:
    if (
        isinstance(kernel_size, bool)
        or not isinstance(kernel_size, int)
        or kernel_size < minimum
        or kernel_size % 2 == 0
    ):
        raise ImageValidationError(
            f"kernel_size must be an odd integer greater than or equal to {minimum}."
        )
    return kernel_size


def gaussian_blur(
    image: np.ndarray,
    *,
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> np.ndarray:
    """Apply Gaussian smoothing with a square odd-sized kernel."""
    pixels = validate_image(image)
    kernel_size = _positive_odd_kernel(kernel_size)
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise ImageValidationError("sigma must be a finite non-negative number.")
    if not math.isfinite(sigma) or sigma < 0:
        raise ImageValidationError("sigma must be a finite non-negative number.")

    result = cv2.GaussianBlur(pixels, (kernel_size, kernel_size), sigmaX=float(sigma))
    if pixels.ndim == 3 and pixels.shape[2] == 1 and result.ndim == 2:
        result = result[..., np.newaxis]
    return result


def median_blur(image: np.ndarray, *, kernel_size: int = 5) -> np.ndarray:
    """Apply median filtering, useful for reducing isolated impulse noise."""
    pixels = validate_image(image)
    kernel_size = _positive_odd_kernel(kernel_size, minimum=3)
    result = cv2.medianBlur(pixels, kernel_size)
    if pixels.ndim == 3 and pixels.shape[2] == 1 and result.ndim == 2:
        result = result[..., np.newaxis]
    return result


def sharpen_image(
    image: np.ndarray,
    *,
    amount: float = 1.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """Sharpen with an unsharp mask while preserving an RGBA alpha channel.

    ``sigma`` controls how much Gaussian blur is subtracted from the original;
    ``amount`` controls the strength of that high-frequency detail. Amount zero
    returns an unchanged copy. Values are clipped back to the uint8 range.
    """
    pixels = validate_image(image)
    if (
        isinstance(amount, bool)
        or not isinstance(amount, (int, float))
        or not math.isfinite(amount)
        or not 0 <= amount <= 5
    ):
        raise ImageValidationError("amount must be a finite number from 0 through 5.")
    if (
        isinstance(sigma, bool)
        or not isinstance(sigma, (int, float))
        or not math.isfinite(sigma)
        or not 0 < sigma <= 100
    ):
        raise ImageValidationError("sigma must be a finite number greater than 0 and at most 100.")

    if amount == 0:
        return pixels.copy()

    result = pixels.copy()
    target = pixels[..., :3] if pixels.ndim == 3 and pixels.shape[2] == 4 else pixels
    blurred = cv2.GaussianBlur(target, (0, 0), sigmaX=float(sigma))
    original_float = target.astype(np.float32)
    detail = original_float - blurred.astype(np.float32)
    sharpened = np.clip(
        np.rint(original_float + float(amount) * detail), 0, 255
    ).astype(np.uint8)
    if result.ndim == 3 and result.shape[2] == 4:
        result[..., :3] = sharpened
    else:
        result[...] = sharpened
    return result


def _clahe_parameters(
    clip_limit: float, tile_grid_size: tuple[int, int]
) -> tuple[float, tuple[int, int]]:
    if isinstance(clip_limit, bool) or not isinstance(clip_limit, (int, float)):
        raise ImageValidationError("clip_limit must be a finite positive number.")
    if not math.isfinite(clip_limit) or clip_limit <= 0:
        raise ImageValidationError("clip_limit must be a finite positive number.")
    if (
        not isinstance(tile_grid_size, tuple)
        or len(tile_grid_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in tile_grid_size
        )
    ):
        raise ImageValidationError("tile_grid_size must contain two positive integers.")
    return float(clip_limit), tile_grid_size


def apply_clahe(
    image: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    color_order: ColorOrder = "RGB",
) -> np.ndarray:
    """Apply CLAHE to grayscale or to the lightness of an RGB/BGR image.

    Color images are converted to CIELAB so contrast is adjusted on lightness
    while chroma is retained. Four-channel images keep their alpha channel
    unchanged. The default color order for three- and four-channel arrays is RGB.
    """
    pixels = validate_image(image)
    clip_limit, tile_grid_size = _clahe_parameters(clip_limit, tile_grid_size)
    if color_order not in ("RGB", "BGR"):
        raise ImageValidationError("color_order must be either 'RGB' or 'BGR'.")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if pixels.ndim == 2:
        return clahe.apply(pixels)
    if pixels.shape[2] == 1:
        enhanced = clahe.apply(pixels[..., 0])
        return enhanced[..., np.newaxis]

    color = pixels[..., :3]
    to_lab = cv2.COLOR_RGB2LAB if color_order == "RGB" else cv2.COLOR_BGR2LAB
    from_lab = cv2.COLOR_LAB2RGB if color_order == "RGB" else cv2.COLOR_LAB2BGR
    lab = cv2.cvtColor(color, to_lab)
    lab[..., 0] = clahe.apply(lab[..., 0])
    enhanced_color = cv2.cvtColor(lab, from_lab)
    if pixels.shape[2] == 4:
        return np.dstack((enhanced_color, pixels[..., 3]))
    return enhanced_color
