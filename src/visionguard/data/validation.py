"""Validation helpers for the MVTec AD directory layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})


def _files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file())


def discover_categories(dataset_root: str | Path) -> list[str]:
    """Find dataset categories that contain train and test split directories."""
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        return []
    return sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
    )


def validate_category(dataset_root: str | Path, category: str) -> dict[str, Any]:
    """Check normal training images, test images, and expected defect masks."""
    root = Path(dataset_root).expanduser()
    errors: list[str] = []
    warnings: list[str] = []
    missing_masks: list[str] = []
    unsupported_files: list[str] = []
    available_categories = discover_categories(root)

    safe_category = (
        isinstance(category, str)
        and bool(category)
        and category not in {".", ".."}
        and Path(category).name == category
    )
    if not safe_category:
        errors.append("Category must be a single directory name.")
        category_root = root
    else:
        category_root = root / category

    if not root.is_dir():
        errors.append(f"Dataset root does not exist: {root}")
    elif safe_category and category not in available_categories:
        errors.append(
            f"Category '{category}' is unavailable under {root}. "
            f"Available categories: {available_categories or 'none'}"
        )

    training_dir = category_root / "train" / "good"
    test_dir = category_root / "test"
    ground_truth_dir = category_root / "ground_truth"
    training_files = _files(training_dir)
    training_images = [p for p in training_files if p.suffix.lower() in IMAGE_SUFFIXES]
    unsupported_files.extend(
        str(p) for p in training_files if p.suffix.lower() not in IMAGE_SUFFIXES
    )
    test_files = _files(test_dir)
    test_images = [p for p in test_files if p.suffix.lower() in IMAGE_SUFFIXES]
    unsupported_files.extend(
        str(p) for p in test_files if p.suffix.lower() not in IMAGE_SUFFIXES
    )

    if root.is_dir() and safe_category and category in available_categories:
        if not training_dir.is_dir():
            errors.append(f"Normal training directory is missing: {training_dir}")
        elif not training_images:
            errors.append(f"No supported normal training images found in: {training_dir}")
        if not test_dir.is_dir():
            errors.append(f"Test directory is missing: {test_dir}")
        elif not test_images:
            errors.append(f"No supported test images found in: {test_dir}")

    for image_path in test_images:
        defect_type = image_path.parent.name
        if defect_type == "good":
            continue
        expected_mask = ground_truth_dir / defect_type / f"{image_path.stem}_mask.png"
        if not expected_mask.is_file():
            missing_masks.append(str(expected_mask))

    if missing_masks:
        warnings.append(
            f"{len(missing_masks)} test anomaly image(s) have no matching ground-truth mask."
        )
    if unsupported_files:
        warnings.append(
            f"{len(unsupported_files)} unsupported file(s) were found in train/good or test."
        )
    if any(p.parent.name != "good" for p in test_images) and not ground_truth_dir.is_dir():
        warnings.append(f"Ground-truth directory is missing: {ground_truth_dir}")

    return {
        "valid": not errors,
        "dataset_root": str(root),
        "category": category,
        "available_categories": available_categories,
        "normal_training_image_count": len(training_images),
        "test_image_count": len(test_images),
        "missing_mask_count": len(missing_masks),
        "mask_coverage_complete": bool(test_images) and not missing_masks,
        "missing_masks": missing_masks,
        "unsupported_files": unsupported_files,
        "errors": errors,
        "warnings": warnings,
    }
