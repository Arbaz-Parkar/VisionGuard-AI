"""Watershed segmentation for simple foreground/background experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .preprocessing import ColorOrder, ImageValidationError, to_grayscale, to_rgb


@dataclass(frozen=True)
class WatershedResult:
    visualization: np.ndarray
    labels: np.ndarray
    region_count: int


def watershed_segmentation(
    image: np.ndarray,
    *,
    foreground: Literal["bright", "dark"] = "bright",
    distance_ratio: float = 0.4,
    color_order: ColorOrder = "RGB",
) -> WatershedResult:
    """Segment foreground regions and highlight watershed boundaries in red.

    Otsu thresholding creates the initial foreground, then a distance transform
    supplies sure-foreground markers to OpenCV's watershed algorithm.
    """
    if foreground not in ("bright", "dark"):
        raise ImageValidationError("foreground must be either 'bright' or 'dark'.")
    if isinstance(distance_ratio, bool) or not isinstance(distance_ratio, (int, float)) or not np.isfinite(distance_ratio) or not 0 < distance_ratio < 1:
        raise ImageValidationError("distance_ratio must be a finite number between 0 and 1.")
    gray = to_grayscale(image, color_order=color_order)
    threshold_kind = cv2.THRESH_BINARY if foreground == "bright" else cv2.THRESH_BINARY_INV
    _, foreground_mask = cv2.threshold(gray, 0, 255, threshold_kind | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_background = cv2.dilate(foreground_mask, kernel, iterations=2)
    distance = cv2.distanceTransform(foreground_mask, cv2.DIST_L2, 5)
    maximum = float(distance.max(initial=0.0))
    if maximum == 0:
        empty_labels = np.zeros(gray.shape, dtype=np.int32)
        return WatershedResult(to_rgb(image, color_order=color_order), empty_labels, 0)
    _, sure_foreground = cv2.threshold(distance, float(distance_ratio) * maximum, 255, cv2.THRESH_BINARY)
    sure_foreground = sure_foreground.astype(np.uint8)
    unknown = cv2.subtract(sure_background, sure_foreground)
    count, markers = cv2.connectedComponents(sure_foreground)
    markers = markers.astype(np.int32) + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(to_rgb(image, color_order=color_order).copy(), markers)
    boundaries = markers == -1
    visualization = to_rgb(image, color_order=color_order)
    visualization[boundaries] = (255, 0, 0)
    labels = markers.copy()
    labels[labels <= 1] = 0
    labels[labels > 1] -= 1
    return WatershedResult(visualization, labels, max(0, count - 1))
