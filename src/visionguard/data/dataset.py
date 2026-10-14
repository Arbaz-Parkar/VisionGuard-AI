"""MVTec image discovery and leakage-safe calibration splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

from .validation import IMAGE_SUFFIXES


@dataclass(frozen=True)
class EvaluationSample:
    """One test image and the expected mask path for anomalous samples."""

    image_path: Path
    defect_type: str
    is_anomaly: bool
    expected_mask_path: Path | None

    @property
    def mask_available(self) -> bool:
        """Whether this anomalous sample's expected mask currently exists."""
        return self.expected_mask_path is not None and self.expected_mask_path.is_file()


@dataclass(frozen=True)
class CalibrationSplit:
    """Disjoint normal-only images for bank construction and calibration."""

    memory_bank_images: tuple[Path, ...]
    calibration_images: tuple[Path, ...]


class DatasetError(ValueError):
    """Raised when a requested dataset split cannot be constructed safely."""


def _category_directory(dataset_root: str | Path, category: str) -> Path:
    if (
        not isinstance(category, str)
        or not category
        or category in {".", ".."}
        or Path(category).name != category
    ):
        raise DatasetError("Category must be a single directory name.")
    return Path(dataset_root).expanduser() / category


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def discover_normal_images(dataset_root: str | Path, category: str) -> list[Path]:
    """Return only supported images under ``category/train/good``."""
    category_root = _category_directory(dataset_root, category)
    return _image_files(category_root / "train" / "good")


def discover_evaluation_images(
    dataset_root: str | Path, category: str
) -> list[EvaluationSample]:
    """List test images and retain expected mask paths even when masks are missing."""
    category_root = _category_directory(dataset_root, category)
    test_root = category_root / "test"
    ground_truth_root = category_root / "ground_truth"
    samples: list[EvaluationSample] = []
    for image_path in _image_files(test_root):
        defect_type = image_path.parent.name
        is_anomaly = defect_type != "good"
        expected_mask = (
            ground_truth_root / defect_type / f"{image_path.stem}_mask.png"
            if is_anomaly
            else None
        )
        samples.append(
            EvaluationSample(
                image_path=image_path,
                defect_type=defect_type,
                is_anomaly=is_anomaly,
                expected_mask_path=expected_mask,
            )
        )
    return samples


def split_normal_images(
    dataset_root: str | Path,
    category: str,
    *,
    calibration_fraction: float = 0.15,
    seed: int = 42,
) -> CalibrationSplit:
    """Split only ``train/good`` images into a bank and held-out calibration set.

    A seeded random sample determines the calibration set. The count is rounded
    to the nearest image and clamped so both returned subsets are non-empty.
    Test images and masks are never read by this function.
    """
    if (
        isinstance(calibration_fraction, bool)
        or not isinstance(calibration_fraction, (int, float))
        or not 0 < calibration_fraction < 1
    ):
        raise DatasetError("calibration_fraction must be greater than 0 and less than 1.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DatasetError("seed must be a non-negative integer.")

    images = discover_normal_images(dataset_root, category)
    if len(images) < 2:
        raise DatasetError(
            "At least two normal train/good images are required for disjoint "
            "bank and calibration sets."
        )

    calibration_count = max(1, min(len(images) - 1, round(len(images) * calibration_fraction)))
    selected_indices = set(random.Random(seed).sample(range(len(images)), calibration_count))
    bank_images = tuple(path for index, path in enumerate(images) if index not in selected_indices)
    calibration_images = tuple(
        path for index, path in enumerate(images) if index in selected_indices
    )
    return CalibrationSplit(
        memory_bank_images=bank_images,
        calibration_images=calibration_images,
    )
