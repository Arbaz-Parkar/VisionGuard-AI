"""NumPy implementation of same-size 2D spatial convolution."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .preprocessing import ImageValidationError

BorderMode = Literal["reflect", "constant"]


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    border: BorderMode = "reflect",
    constant_value: float = 0.0,
) -> np.ndarray:
    """Convolve a 2D numeric image using NumPy and return a same-size result.

    The kernel is flipped on both axes, as required for convolution. ``reflect``
    uses edge-excluding reflection (matching OpenCV ``BORDER_REFLECT_101``);
    ``constant`` pads with ``constant_value``. Integer input is promoted so the
    accumulation does not overflow in the source pixel type.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ImageValidationError("image must be a non-empty 2D NumPy array.")
    if (
        not np.issubdtype(image.dtype, np.number)
        or np.issubdtype(image.dtype, np.complexfloating)
        or not np.isfinite(image).all()
    ):
        raise ImageValidationError("image must contain finite real-valued pixels.")
    if not isinstance(kernel, np.ndarray) or kernel.ndim != 2 or kernel.size == 0:
        raise ImageValidationError("kernel must be a non-empty 2D NumPy array.")
    if (
        not np.issubdtype(kernel.dtype, np.number)
        or np.issubdtype(kernel.dtype, np.complexfloating)
        or not np.isfinite(kernel).all()
    ):
        raise ImageValidationError("kernel must contain finite real-valued weights.")
    if any(size % 2 == 0 for size in kernel.shape):
        raise ImageValidationError("kernel dimensions must both be odd.")
    if border not in ("reflect", "constant"):
        raise ImageValidationError("border must be either 'reflect' or 'constant'.")
    if (
        isinstance(constant_value, bool)
        or not isinstance(constant_value, (int, float))
        or not math.isfinite(constant_value)
    ):
        raise ImageValidationError("constant_value must be a finite real number.")

    kernel_height, kernel_width = kernel.shape
    pad_y, pad_x = kernel_height // 2, kernel_width // 2
    source_dtype = np.result_type(image.dtype, kernel.dtype, np.float32)
    source = image.astype(source_dtype, copy=False)
    weights = kernel.astype(source_dtype, copy=False)
    padding = ((pad_y, pad_y), (pad_x, pad_x))
    if border == "reflect":
        padded = np.pad(source, padding, mode="reflect")
    else:
        padded = np.pad(
            source,
            padding,
            mode="constant",
            constant_values=constant_value,
        )

    windows = sliding_window_view(padded, (kernel_height, kernel_width))
    return np.einsum("ijmn,mn->ij", windows, weights[::-1, ::-1], optimize=True)
