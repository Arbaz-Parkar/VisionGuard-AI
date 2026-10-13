"""Safe image loading and size/color utilities for classical processing."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import BinaryIO, Literal
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_PIXELS = 40_000_000
ColorOrder = Literal["RGB", "BGR"]
ImageSource = str | PathLike[str] | bytes | bytearray | BinaryIO


class ImageValidationError(ValueError):
    """Raised when an image cannot be safely loaded or processed."""


@dataclass(frozen=True)
class ResizeTransform:
    """Geometry needed to align masks or restore a map after image resizing."""

    original_height: int
    original_width: int
    target_height: int
    target_width: int
    resized_height: int
    resized_width: int
    offset_y: int
    offset_x: int


def _check_pixel_limit(height: int, width: int, max_pixels: int) -> None:
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer.")
    if height <= 0 or width <= 0:
        raise ImageValidationError("Image dimensions must be greater than zero.")
    if height * width > max_pixels:
        raise ImageValidationError(
            f"Image has {height * width:,} pixels; the configured limit is {max_pixels:,}."
        )


def validate_image(image: np.ndarray, *, max_pixels: int = MAX_IMAGE_PIXELS) -> np.ndarray:
    """Validate a uint8 image in HxW, HxWx1, HxWx3, or HxWx4 layout."""
    if not isinstance(image, np.ndarray):
        raise ImageValidationError("Image data must be a NumPy array.")
    if image.ndim == 2:
        height, width = image.shape
    elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
        height, width = image.shape[:2]
    else:
        raise ImageValidationError(
            "Expected an image shaped HxW, HxWx1, HxWx3, or HxWx4."
        )
    _check_pixel_limit(height, width, max_pixels)
    if image.dtype != np.uint8:
        raise ImageValidationError("Image data must use uint8 pixels in the range 0–255.")
    return np.ascontiguousarray(image)


def load_image(
    source: ImageSource, *, max_pixels: int = MAX_IMAGE_PIXELS
) -> np.ndarray:
    """Load a local image as uint8, returning grayscale, RGB, or RGBA pixels.

    EXIF orientation is applied. Byte sources are useful for uploaded files; paths
    and binary file objects are also accepted. Palette images with transparency
    are returned as RGBA, and other palette images as RGB.
    """
    if isinstance(source, (bytes, bytearray)):
        if not source:
            raise ImageValidationError("Image data is empty.")
        image_source: object = BytesIO(source)
    elif isinstance(source, (str, PathLike)) or hasattr(source, "read"):
        image_source = source
    else:
        raise ImageValidationError("Source must be a path, image bytes, or binary file object.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(image_source) as opened:
                oriented = ImageOps.exif_transpose(opened)
                width, height = oriented.size
                _check_pixel_limit(height, width, max_pixels)
                oriented.load()

                if oriented.mode in ("I;16", "I;16L", "I;16B"):
                    pixels = np.asarray(oriented, dtype=np.uint16)
                    pixels = ((pixels.astype(np.uint32) + 128) // 257).astype(np.uint8)
                elif oriented.mode == "I":
                    pixels = np.asarray(oriented)
                    if pixels.size and (pixels.min() < 0 or pixels.max() > 65535):
                        raise ImageValidationError(
                            "32-bit integer images must contain values from 0 through 65535."
                        )
                    pixels = ((pixels.astype(np.uint32) + 128) // 257).astype(np.uint8)
                elif oriented.mode in ("1", "L"):
                    pixels = np.asarray(oriented.convert("L"), dtype=np.uint8)
                elif oriented.mode == "RGB":
                    pixels = np.asarray(oriented, dtype=np.uint8)
                elif oriented.mode == "RGBA":
                    pixels = np.asarray(oriented, dtype=np.uint8)
                elif oriented.mode == "P" and "transparency" in oriented.info:
                    pixels = np.asarray(oriented.convert("RGBA"), dtype=np.uint8)
                else:
                    pixels = np.asarray(oriented.convert("RGB"), dtype=np.uint8)
    except ImageValidationError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ImageValidationError(f"Unable to load image: {exc}") from exc

    return validate_image(pixels, max_pixels=max_pixels)


def _color_order(color_order: str) -> ColorOrder:
    if color_order not in ("RGB", "BGR"):
        raise ImageValidationError("color_order must be either 'RGB' or 'BGR'.")
    return color_order  # type: ignore[return-value]


def _composite_white(image: np.ndarray) -> np.ndarray:
    """Composite the first three color channels over white using alpha."""
    colors = image[..., :3].astype(np.float32)
    alpha = image[..., 3:4].astype(np.float32) / 255.0
    return np.rint(colors * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)


def to_grayscale(image: np.ndarray, *, color_order: ColorOrder = "RGB") -> np.ndarray:
    """Convert an image to grayscale; color inputs default to RGB channel order.

    RGBA/BGRA input is composited over white before conversion so transparent
    pixels do not become unexpectedly black.
    """
    order = _color_order(color_order)
    pixels = validate_image(image)
    if pixels.ndim == 2:
        return pixels.copy()
    if pixels.shape[2] == 1:
        return pixels[..., 0].copy()
    if pixels.shape[2] == 4:
        pixels = _composite_white(pixels)
    conversion = cv2.COLOR_RGB2GRAY if order == "RGB" else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(pixels, conversion)


def to_rgb(image: np.ndarray, *, color_order: ColorOrder = "RGB") -> np.ndarray:
    """Return a three-channel RGB image, compositing alpha over white."""
    order = _color_order(color_order)
    pixels = validate_image(image)
    if pixels.ndim == 2:
        return cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
    if pixels.shape[2] == 1:
        return cv2.cvtColor(pixels[..., 0], cv2.COLOR_GRAY2RGB)
    if pixels.shape[2] == 4:
        pixels = _composite_white(pixels)
    if order == "BGR":
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    return pixels.copy()


def _target_size(target_size: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(target_size, tuple)
        or len(target_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in target_size)
        or any(value <= 0 for value in target_size)
    ):
        raise ImageValidationError("target_size must be a pair of positive (height, width) integers.")
    return target_size


def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int],
    *,
    preserve_aspect_ratio: bool = True,
    pad_value: int = 0,
    interpolation: int | None = None,
) -> tuple[np.ndarray, ResizeTransform]:
    """Resize to (height, width), optionally padding to preserve aspect ratio."""
    pixels = validate_image(image)
    target_height, target_width = _target_size(target_size)
    if not isinstance(preserve_aspect_ratio, bool):
        raise ImageValidationError("preserve_aspect_ratio must be a boolean.")
    if isinstance(pad_value, bool) or not isinstance(pad_value, int) or not 0 <= pad_value <= 255:
        raise ImageValidationError("pad_value must be an integer from 0 through 255.")

    original_height, original_width = pixels.shape[:2]
    if preserve_aspect_ratio:
        scale = min(target_width / original_width, target_height / original_height)
        resized_width = max(1, min(target_width, int(round(original_width * scale))))
        resized_height = max(1, min(target_height, int(round(original_height * scale))))
    else:
        resized_height, resized_width = target_height, target_width

    if interpolation is None:
        interpolation = (
            cv2.INTER_AREA
            if resized_height < original_height or resized_width < original_width
            else cv2.INTER_LINEAR
        )
    resized = cv2.resize(pixels, (resized_width, resized_height), interpolation=interpolation)
    if pixels.ndim == 3 and pixels.shape[2] == 1 and resized.ndim == 2:
        resized = resized[..., np.newaxis]
    offset_y = (target_height - resized_height) // 2
    offset_x = (target_width - resized_width) // 2

    if preserve_aspect_ratio:
        if pixels.ndim == 2:
            output = np.full((target_height, target_width), pad_value, dtype=np.uint8)
        else:
            output = np.full(
                (target_height, target_width, pixels.shape[2]), pad_value, dtype=np.uint8
            )
        output[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
            ...,
        ] = resized
    else:
        output = resized

    transform = ResizeTransform(
        original_height=original_height,
        original_width=original_width,
        target_height=target_height,
        target_width=target_width,
        resized_height=resized_height,
        resized_width=resized_width,
        offset_y=offset_y,
        offset_x=offset_x,
    )
    return output, transform


def resize_mask(mask: np.ndarray, transform: ResizeTransform) -> np.ndarray:
    """Resize an original-size mask with nearest-neighbor interpolation and zero padding."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ImageValidationError("Mask must be a 2D uint8 NumPy array.")
    if mask.shape != (transform.original_height, transform.original_width):
        raise ImageValidationError("Mask dimensions must match the transform's original image size.")
    scaled = cv2.resize(
        mask,
        (transform.resized_width, transform.resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    output = np.zeros((transform.target_height, transform.target_width), dtype=np.uint8)
    output[
        transform.offset_y : transform.offset_y + transform.resized_height,
        transform.offset_x : transform.offset_x + transform.resized_width,
    ] = scaled
    return output


def restore_map(anomaly_map: np.ndarray, transform: ResizeTransform) -> np.ndarray:
    """Crop padding and resize a target-size map back to original image dimensions."""
    if not isinstance(anomaly_map, np.ndarray) or anomaly_map.ndim != 2:
        raise ImageValidationError("Map must be a 2D NumPy array.")
    if not np.issubdtype(anomaly_map.dtype, np.number) or not np.isfinite(anomaly_map).all():
        raise ImageValidationError("Map values must be finite numbers.")
    if anomaly_map.shape != (transform.target_height, transform.target_width):
        raise ImageValidationError("Map dimensions must match the transform's target size.")
    y0, x0 = transform.offset_y, transform.offset_x
    cropped = anomaly_map[
        y0 : y0 + transform.resized_height,
        x0 : x0 + transform.resized_width,
    ].astype(np.float32, copy=False)
    return cv2.resize(
        cropped,
        (transform.original_width, transform.original_height),
        interpolation=cv2.INTER_LINEAR,
    )
