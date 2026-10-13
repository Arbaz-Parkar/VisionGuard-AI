"""Contour-based candidate-region extraction from binary masks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .preprocessing import ImageValidationError, validate_image


@dataclass(frozen=True)
class CandidateRegion:
    """Geometry and contour points for one connected outer mask region."""

    bounding_box: tuple[int, int, int, int]
    contour_area: float
    centroid: tuple[float, float]
    contour: np.ndarray


def _area_limit(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImageValidationError(f"{name} must be a finite non-negative number or None.")
    if not math.isfinite(value) or value < 0:
        raise ImageValidationError(f"{name} must be a finite non-negative number or None.")
    return float(value)


def extract_candidate_regions(
    mask: np.ndarray,
    *,
    min_area: float = 0.0,
    max_area: float | None = None,
) -> list[CandidateRegion]:
    """Extract external contours and their bounds from a 2D binary mask.

    ``contour_area`` is OpenCV's geometric contour area, which can be smaller
    than the number of foreground pixels. Holes inside an outer contour are not
    returned as separate regions. Results are sorted from top to bottom, then
    left to right for stable display and testing.
    """
    pixels = validate_image(mask)
    if pixels.ndim != 2:
        raise ImageValidationError("Candidate extraction expects a 2D binary mask.")
    if not np.isin(pixels, (0, 255)).all():
        raise ImageValidationError("Binary masks must contain only 0 and 255 values.")
    min_area_value = _area_limit("min_area", min_area)
    max_area_value = _area_limit("max_area", max_area)
    assert min_area_value is not None
    if max_area_value is not None and max_area_value < min_area_value:
        raise ImageValidationError("max_area must be greater than or equal to min_area.")

    contours, _ = cv2.findContours(
        pixels.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates: list[CandidateRegion] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area_value or (max_area_value is not None and area > max_area_value):
            continue

        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            centroid = (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
        else:
            centroid = (x + (width - 1) / 2.0, y + (height - 1) / 2.0)
        candidates.append(
            CandidateRegion(
                bounding_box=(x, y, width, height),
                contour_area=area,
                centroid=centroid,
                contour=contour.copy(),
            )
        )

    return sorted(candidates, key=lambda region: (region.bounding_box[1], region.bounding_box[0]))
