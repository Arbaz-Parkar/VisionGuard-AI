"""Rule-based classical anomaly detection baseline (Baseline A)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .contours import CandidateRegion, extract_candidate_regions
from .filtering import apply_clahe, gaussian_blur
from .morphology import morphology
from .preprocessing import ColorOrder, ImageValidationError, to_grayscale
from .thresholding import otsu_threshold


@dataclass(frozen=True)
class ClassicalDetectionResult:
    """Continuous anomaly scores, an Otsu mask, and cleaned candidate regions."""

    image_score: float
    score_map: np.ndarray
    mask: np.ndarray
    threshold: float
    regions: tuple[CandidateRegion, ...]


def _positive_odd(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 2 == 0:
        raise ImageValidationError(f"{name} must be a positive odd integer.")
    return value


def run_rule_based_baseline(
    image: np.ndarray,
    *,
    blur_kernel_size: int = 3,
    local_blur_kernel_size: int = 31,
    morphology_kernel_size: int = 3,
    min_region_area: float = 20.0,
    color_order: ColorOrder = "RGB",
) -> ClassicalDetectionResult:
    """Run the transparent local-contrast anomaly baseline on one image.

    The pipeline converts to grayscale, reduces noise, applies CLAHE, and
    compares the result with a strongly blurred copy. Otsu thresholding and
    opening/closing produce candidate regions. The continuous absolute-
    difference map is retained for threshold-free metrics such as AUROC.
    """
    blur_kernel_size = _positive_odd("blur_kernel_size", blur_kernel_size)
    local_blur_kernel_size = _positive_odd(
        "local_blur_kernel_size", local_blur_kernel_size
    )
    morphology_kernel_size = _positive_odd(
        "morphology_kernel_size", morphology_kernel_size
    )
    if local_blur_kernel_size <= blur_kernel_size:
        raise ImageValidationError(
            "local_blur_kernel_size must be larger than blur_kernel_size."
        )
    if (
        isinstance(min_region_area, bool)
        or not isinstance(min_region_area, (int, float))
        or not math.isfinite(min_region_area)
        or min_region_area < 0
    ):
        raise ImageValidationError("min_region_area must be a finite non-negative number.")

    gray = to_grayscale(image, color_order=color_order)
    denoised = gaussian_blur(gray, kernel_size=blur_kernel_size)
    enhanced = apply_clahe(denoised)
    local_background = gaussian_blur(enhanced, kernel_size=local_blur_kernel_size)
    difference = cv2.absdiff(enhanced, local_background)
    score_map = difference.astype(np.float32)

    threshold, threshold_mask = otsu_threshold(difference)
    cleaned = morphology(
        threshold_mask, "open", kernel_size=morphology_kernel_size
    )
    cleaned = morphology(cleaned, "close", kernel_size=morphology_kernel_size)
    regions = tuple(extract_candidate_regions(cleaned, min_area=min_region_area))

    final_mask = np.zeros_like(cleaned)
    for region in regions:
        cv2.drawContours(final_mask, [region.contour], -1, 255, thickness=cv2.FILLED)

    return ClassicalDetectionResult(
        image_score=float(np.max(score_map)),
        score_map=score_map,
        mask=final_mask,
        threshold=float(threshold),
        regions=regions,
    )
