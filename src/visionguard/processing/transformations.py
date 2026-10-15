"""Basic geometric image transformations for Lab experiments."""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

from .preprocessing import ImageValidationError, MAX_IMAGE_PIXELS, validate_image

FlipDirection = Literal["horizontal", "vertical", "both"]
_INTERPOLATIONS = {
    cv2.INTER_NEAREST,
    cv2.INTER_LINEAR,
    cv2.INTER_CUBIC,
    cv2.INTER_LANCZOS4,
}


def _warp_options(interpolation: int, border_value: int) -> None:
    if (
        isinstance(interpolation, bool)
        or not isinstance(interpolation, int)
        or interpolation not in _INTERPOLATIONS
    ):
        raise ImageValidationError(
            "interpolation must be INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, or INTER_LANCZOS4."
        )
    if (
        isinstance(border_value, bool)
        or not isinstance(border_value, int)
        or not 0 <= border_value <= 255
    ):
        raise ImageValidationError("border_value must be an integer from 0 through 255.")


def _warp_affine(
    pixels: np.ndarray,
    matrix: np.ndarray,
    output_size: tuple[int, int],
    *,
    interpolation: int,
    border_value: int,
) -> np.ndarray:
    output = cv2.warpAffine(
        pixels,
        matrix,
        output_size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    if pixels.ndim == 3 and pixels.shape[2] == 1 and output.ndim == 2:
        output = output[..., np.newaxis]
    return output


def flip_image(image: np.ndarray, *, direction: FlipDirection = "horizontal") -> np.ndarray:
    """Flip an image horizontally, vertically, or across both axes."""
    pixels = validate_image(image)
    flip_codes = {"horizontal": 1, "vertical": 0, "both": -1}
    if not isinstance(direction, str) or direction not in flip_codes:
        raise ImageValidationError("direction must be 'horizontal', 'vertical', or 'both'.")
    return cv2.flip(pixels, flip_codes[direction])


def rotate_image(
    image: np.ndarray,
    angle_degrees: float,
    *,
    expand: bool = False,
    interpolation: int = cv2.INTER_LINEAR,
    border_value: int = 0,
) -> np.ndarray:
    """Rotate around the image center; optionally expand to fit the full result.

    Positive angles rotate counterclockwise. When ``expand`` is false, pixels
    outside the original canvas are clipped and empty areas use ``border_value``.
    """
    pixels = validate_image(image)
    if (
        isinstance(angle_degrees, bool)
        or not isinstance(angle_degrees, (int, float))
        or not math.isfinite(angle_degrees)
    ):
        raise ImageValidationError("angle_degrees must be a finite number.")
    if not isinstance(expand, bool):
        raise ImageValidationError("expand must be a boolean.")
    _warp_options(interpolation, border_value)

    height, width = pixels.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    angle = float(angle_degrees) % 360.0
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    output_width, output_height = width, height

    if expand:
        cosine = abs(math.cos(math.radians(angle)))
        sine = abs(math.sin(math.radians(angle)))
        if math.isclose(cosine, 0.0, abs_tol=1e-12):
            cosine = 0.0
        elif math.isclose(cosine, 1.0, abs_tol=1e-12):
            cosine = 1.0
        if math.isclose(sine, 0.0, abs_tol=1e-12):
            sine = 0.0
        elif math.isclose(sine, 1.0, abs_tol=1e-12):
            sine = 1.0
        output_width = int(math.ceil(width * cosine + height * sine))
        output_height = int(math.ceil(height * cosine + width * sine))
        if output_width * output_height > MAX_IMAGE_PIXELS:
            raise ImageValidationError(
                f"Rotated image would exceed the {MAX_IMAGE_PIXELS:,}-pixel limit."
            )
        matrix[0, 2] += (output_width - 1) / 2.0 - center[0]
        matrix[1, 2] += (output_height - 1) / 2.0 - center[1]

    return _warp_affine(
        pixels,
        matrix,
        (output_width, output_height),
        interpolation=interpolation,
        border_value=border_value,
    )


def translate_image(
    image: np.ndarray,
    offset_x: float,
    offset_y: float,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_value: int = 0,
) -> np.ndarray:
    """Shift an image by ``(offset_x, offset_y)`` while keeping its canvas size."""
    pixels = validate_image(image)
    for name, offset in (("offset_x", offset_x), ("offset_y", offset_y)):
        if (
            isinstance(offset, bool)
            or not isinstance(offset, (int, float))
            or not math.isfinite(offset)
        ):
            raise ImageValidationError(f"{name} must be a finite number.")
    _warp_options(interpolation, border_value)

    height, width = pixels.shape[:2]
    matrix = np.array(
        [[1.0, 0.0, float(offset_x)], [0.0, 1.0, float(offset_y)]],
        dtype=np.float64,
    )
    return _warp_affine(
        pixels,
        matrix,
        (width, height),
        interpolation=interpolation,
        border_value=border_value,
    )
