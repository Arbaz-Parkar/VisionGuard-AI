"""Template matching helpers for locating a small image patch in a larger one."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .preprocessing import ColorOrder, ImageValidationError, to_grayscale

TemplateMethod = Literal[
    "sqdiff",
    "sqdiff_normed",
    "ccorr",
    "ccorr_normed",
    "ccoeff",
    "ccoeff_normed",
]

_METHODS: dict[TemplateMethod, int] = {
    "sqdiff": cv2.TM_SQDIFF,
    "sqdiff_normed": cv2.TM_SQDIFF_NORMED,
    "ccorr": cv2.TM_CCORR,
    "ccorr_normed": cv2.TM_CCORR_NORMED,
    "ccoeff": cv2.TM_CCOEFF,
    "ccoeff_normed": cv2.TM_CCOEFF_NORMED,
}
_LOWER_IS_BETTER = {"sqdiff", "sqdiff_normed"}


@dataclass(frozen=True)
class TemplateMatchResult:
    """Best match and the full response map returned by OpenCV.

    Coordinates are ``(x, y)``. ``bottom_right`` is the exclusive edge, so
    ``top_left`` and ``bottom_right`` can be used directly as an array slice.
    For squared-difference methods a lower score is better; for the other
    methods a higher score is better.
    """

    response: np.ndarray
    top_left: tuple[int, int]
    bottom_right: tuple[int, int]
    score: float
    method: TemplateMethod


def match_template(
    image: np.ndarray,
    template: np.ndarray,
    *,
    method: TemplateMethod = "ccoeff_normed",
    color_order: ColorOrder = "RGB",
) -> TemplateMatchResult:
    """Locate a template in an image using one of OpenCV's six match methods.

    Both inputs are converted to grayscale, including alpha compositing for
    transparent images. The response map is retained for Lab visualization.
    The default normalized correlation-coefficient method is usually a useful
    starting point when brightness can vary.
    """
    if not isinstance(method, str) or method not in _METHODS:
        choices = ", ".join(_METHODS)
        raise ImageValidationError(f"method must be one of: {choices}.")

    image_gray = to_grayscale(image, color_order=color_order)
    template_gray = to_grayscale(template, color_order=color_order)
    image_height, image_width = image_gray.shape
    template_height, template_width = template_gray.shape
    if template_height > image_height or template_width > image_width:
        raise ImageValidationError("Template dimensions must not exceed image dimensions.")

    response = cv2.matchTemplate(image_gray, template_gray, _METHODS[method])
    min_value, max_value, min_location, max_location = cv2.minMaxLoc(response)
    if method in _LOWER_IS_BETTER:
        location = min_location
        score = min_value
    else:
        location = max_location
        score = max_value

    x, y = location
    return TemplateMatchResult(
        response=response,
        top_left=(x, y),
        bottom_right=(x + template_width, y + template_height),
        score=float(score),
        method=method,
    )
