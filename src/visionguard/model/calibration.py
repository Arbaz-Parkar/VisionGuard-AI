"""Leakage-safe threshold calibration for the patch anomaly scorer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from visionguard.data import CalibrationSplit, DatasetError
from visionguard.processing import load_image

from .inference import PatchAnomalyScorer


@dataclass(frozen=True)
class PatchThresholdCalibration:
    """Image and pixel cutoffs estimated from held-out normal images."""

    pixel_threshold: float
    image_threshold: float
    pixel_false_positive_rate: float
    image_false_positive_rate: float
    pixel_percentile: float
    image_percentile: float
    calibration_image_count: int
    calibration_pixel_count: int


def _calibration_paths(split: CalibrationSplit) -> tuple[Path, ...]:
    if not isinstance(split, CalibrationSplit):
        raise DatasetError("split must be a CalibrationSplit from normal training data.")

    memory_paths = tuple(Path(path) for path in split.memory_bank_images)
    calibration_paths = tuple(Path(path) for path in split.calibration_images)
    if not memory_paths:
        raise DatasetError("The memory-bank training split contains no images.")
    if not calibration_paths:
        raise DatasetError("The calibration split contains no normal images.")

    memory_resolved = tuple(path.resolve() for path in memory_paths)
    calibration_resolved = tuple(path.resolve() for path in calibration_paths)
    if len(set(memory_resolved)) != len(memory_resolved):
        raise DatasetError("Memory-bank image paths must be unique.")
    if len(set(calibration_resolved)) != len(calibration_resolved):
        raise DatasetError("Calibration image paths must be unique.")
    if set(memory_resolved) & set(calibration_resolved):
        raise DatasetError("Memory-bank and calibration images must be disjoint.")

    for path in (*memory_paths, *calibration_paths):
        parts = tuple(part.casefold() for part in path.parts)
        if not any(
            parts[index : index + 2] == ("train", "good")
            for index in range(len(parts) - 1)
        ):
            raise DatasetError(f"Normal split images must come from train/good: {path}")
    return calibration_paths


def _false_positive_rate(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value < 1
    ):
        raise DatasetError(f"{name} must be finite and in the range [0, 1).")
    return float(value)


def calibrate_patch_thresholds(
    split: CalibrationSplit,
    scorer: PatchAnomalyScorer,
    *,
    pixel_false_positive_rate: float = 0.005,
    image_false_positive_rate: float = 0.05,
) -> PatchThresholdCalibration:
    """Estimate pixel and image cutoffs using only held-out normal images.

    The requested false-positive rates select empirical percentiles of all
    calibration-map pixels and of one image score per calibration image. The
    returned rates are targets; ties and finite calibration samples can make
    the observed rate differ from the requested value.
    """
    calibration_paths = _calibration_paths(split)
    pixel_rate = _false_positive_rate(
        "pixel_false_positive_rate", pixel_false_positive_rate
    )
    image_rate = _false_positive_rate(
        "image_false_positive_rate", image_false_positive_rate
    )
    if not isinstance(scorer, PatchAnomalyScorer):
        raise DatasetError("scorer must be a PatchAnomalyScorer.")

    pixel_percentile = 100.0 * (1.0 - pixel_rate)
    image_percentile = 100.0 * (1.0 - image_rate)
    pixel_maps: list[np.ndarray] = []
    image_scores: list[float] = []
    for path in calibration_paths:
        result = scorer.run(load_image(path), pixel_threshold=None)
        if (
            result.raw_map.ndim != 2
            or result.raw_map.size == 0
            or not np.isfinite(result.raw_map).all()
            or not math.isfinite(result.image_score)
        ):
            raise DatasetError(f"Scorer returned invalid calibration scores for {path}.")
        pixel_maps.append(result.raw_map.reshape(-1))
        image_scores.append(result.image_score)

    pixel_values = np.concatenate(pixel_maps)
    return PatchThresholdCalibration(
        pixel_threshold=float(np.percentile(pixel_values, pixel_percentile)),
        image_threshold=float(np.percentile(image_scores, image_percentile)),
        pixel_false_positive_rate=pixel_rate,
        image_false_positive_rate=image_rate,
        pixel_percentile=pixel_percentile,
        image_percentile=image_percentile,
        calibration_image_count=len(calibration_paths),
        calibration_pixel_count=int(pixel_values.size),
    )
