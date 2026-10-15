"""Per-pixel statistical reference model (Classical Baseline B)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np

from .baselines import ClassicalDetectionResult
from .contours import extract_candidate_regions
from .preprocessing import ColorOrder, ImageValidationError, to_rgb


@dataclass(frozen=True)
class PerPixelReferenceModel:
    """RGB mean and population standard deviation learned from normal images."""

    mean: np.ndarray
    standard_deviation: np.ndarray
    color_order: ColorOrder
    sample_count: int

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean)
        standard_deviation = np.asarray(self.standard_deviation)
        if (
            mean.ndim != 3
            or mean.shape[2] != 3
            or not np.issubdtype(mean.dtype, np.number)
            or not np.isfinite(mean).all()
        ):
            raise ImageValidationError("mean must be a finite numeric HxWx3 array.")
        if (
            standard_deviation.shape != mean.shape
            or not np.issubdtype(standard_deviation.dtype, np.number)
            or not np.isfinite(standard_deviation).all()
            or np.any(standard_deviation < 0)
        ):
            raise ImageValidationError(
                "standard_deviation must be finite, non-negative, and match mean's shape."
            )
        if self.color_order not in ("RGB", "BGR"):
            raise ImageValidationError("color_order must be either 'RGB' or 'BGR'.")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count <= 0:
            raise ImageValidationError("sample_count must be a positive integer.")

        mean_copy = np.array(mean, dtype=np.float32, copy=True)
        deviation_copy = np.array(standard_deviation, dtype=np.float32, copy=True)
        mean_copy.setflags(write=False)
        deviation_copy.setflags(write=False)
        object.__setattr__(self, "mean", mean_copy)
        object.__setattr__(self, "standard_deviation", deviation_copy)


def _positive_finite(name: str, value: float, *, allow_zero: bool = False) -> float:
    lower_ok = value >= 0 if allow_zero else value > 0
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not lower_ok
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ImageValidationError(f"{name} must be a finite {qualifier} number.")
    return float(value)


def _model_rgb(image: np.ndarray, model: PerPixelReferenceModel) -> np.ndarray:
    rgb = to_rgb(image, color_order=model.color_order)
    if rgb.shape != model.mean.shape:
        raise ImageValidationError(
            f"Image shape {rgb.shape} does not match model shape {model.mean.shape}."
        )
    return rgb


def _score_map(
    image: np.ndarray,
    model: PerPixelReferenceModel,
    *,
    standard_deviation_floor: float,
) -> np.ndarray:
    rgb = _model_rgb(image, model).astype(np.float32)
    denominator = np.maximum(model.standard_deviation, standard_deviation_floor)
    channel_scores = np.abs(rgb - model.mean) / denominator
    return np.max(channel_scores, axis=2).astype(np.float32)


def fit_per_pixel_reference(
    normal_training_images: Iterable[np.ndarray],
    *,
    color_order: ColorOrder = "RGB",
) -> PerPixelReferenceModel:
    """Fit RGB per-pixel statistics using normal training images only.

    Pass only ``train/good`` images from the training portion of the seeded
    split. Calibration images and all test images must be excluded. Welford's
    online update keeps the fit to a few image-sized arrays instead of stacking
    every source image in memory.
    """
    try:
        iterator = iter(normal_training_images)
    except TypeError as exc:
        raise ImageValidationError("normal_training_images must be an iterable of images.") from exc

    count = 0
    mean: np.ndarray | None = None
    sum_squared_differences: np.ndarray | None = None
    expected_shape: tuple[int, ...] | None = None
    for image in iterator:
        rgb = to_rgb(image, color_order=color_order)
        if expected_shape is None:
            expected_shape = rgb.shape
            mean = np.zeros(expected_shape, dtype=np.float64)
            sum_squared_differences = np.zeros(expected_shape, dtype=np.float64)
        elif rgb.shape != expected_shape:
            raise ImageValidationError(
                "All normal training images must have the same height and width."
            )

        values = rgb.astype(np.float64)
        count += 1
        assert mean is not None and sum_squared_differences is not None
        difference = values - mean
        mean += difference / count
        sum_squared_differences += difference * (values - mean)

    if count == 0 or mean is None or sum_squared_differences is None:
        raise ImageValidationError("At least one normal training image is required.")

    variance = np.maximum(sum_squared_differences / count, 0.0)
    return PerPixelReferenceModel(
        mean=mean.astype(np.float32),
        standard_deviation=np.sqrt(variance).astype(np.float32),
        color_order=color_order,
        sample_count=count,
    )


def calibrate_reference_threshold(
    model: PerPixelReferenceModel,
    normal_calibration_images: Iterable[np.ndarray],
    *,
    percentile: float = 99.5,
    standard_deviation_floor: float = 1.0,
) -> float:
    """Choose a pixel threshold from held-out normal calibration score maps."""
    percentile_value = _positive_finite("percentile", percentile)
    if percentile_value > 100:
        raise ImageValidationError("percentile must be greater than 0 and at most 100.")
    floor = _positive_finite("standard_deviation_floor", standard_deviation_floor)
    if floor < 1e-6:
        raise ImageValidationError("standard_deviation_floor must be at least 1e-6.")

    try:
        iterator = iter(normal_calibration_images)
    except TypeError as exc:
        raise ImageValidationError("normal_calibration_images must be an iterable of images.") from exc

    flattened_maps: list[np.ndarray] = []
    for image in iterator:
        flattened_maps.append(
            _score_map(image, model, standard_deviation_floor=floor).ravel()
        )
    if not flattened_maps:
        raise ImageValidationError("At least one normal calibration image is required.")
    return float(np.percentile(np.concatenate(flattened_maps), percentile_value))


def run_reference_baseline(
    image: np.ndarray,
    model: PerPixelReferenceModel,
    *,
    pixel_threshold: float,
    standard_deviation_floor: float = 1.0,
    min_region_area: float = 20.0,
) -> ClassicalDetectionResult:
    """Score one image against a fitted reference and apply its calibrated cutoff.

    Per-pixel absolute z-scores are computed for each RGB channel; the largest
    channel score becomes the continuous map value. The image score is the map
    maximum. Supply ``pixel_threshold`` from normal held-out calibration data.
    """
    threshold = _positive_finite("pixel_threshold", pixel_threshold, allow_zero=True)
    floor = _positive_finite("standard_deviation_floor", standard_deviation_floor)
    if floor < 1e-6:
        raise ImageValidationError("standard_deviation_floor must be at least 1e-6.")
    if (
        isinstance(min_region_area, bool)
        or not isinstance(min_region_area, (int, float))
        or not math.isfinite(min_region_area)
        or min_region_area < 0
    ):
        raise ImageValidationError("min_region_area must be a finite non-negative number.")

    score_map = _score_map(image, model, standard_deviation_floor=floor)
    threshold_mask = np.where(score_map > threshold, 255, 0).astype(np.uint8)
    regions = tuple(
        extract_candidate_regions(threshold_mask, min_area=float(min_region_area))
    )
    final_mask = np.zeros_like(threshold_mask)
    for region in regions:
        cv2.drawContours(final_mask, [region.contour], -1, 255, thickness=cv2.FILLED)

    return ClassicalDetectionResult(
        image_score=float(np.max(score_map)),
        score_map=score_map,
        mask=final_mask,
        threshold=threshold,
        regions=regions,
    )
