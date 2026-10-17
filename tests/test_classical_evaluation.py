import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from visionguard.classical_evaluation import evaluate_classical_baselines
from visionguard.evaluation import EvaluationError


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _normal_image(size: int = 24) -> np.ndarray:
    return np.full((size, size, 3), 100, dtype=np.uint8)


def _defect_image(size: int = 24) -> tuple[np.ndarray, np.ndarray]:
    image = _normal_image(size)
    mask = np.zeros((size, size), dtype=np.uint8)
    image[8:16, 8:16] = 240
    mask[8:16, 8:16] = 255
    return image, mask


def _dataset(root: Path) -> None:
    category = root / "bottle"
    for index in range(4):
        _save_image(category / "train" / "good" / f"normal_{index}.png", _normal_image())
    _save_image(category / "test" / "good" / "normal.png", _normal_image())
    defect, mask = _defect_image()
    missing_mask_defect, _ = _defect_image()
    _save_image(category / "test" / "crack" / "defect.png", defect)
    _save_image(category / "test" / "crack" / "missing_mask.png", missing_mask_defect)
    _save_image(category / "ground_truth" / "crack" / "defect_mask.png", mask)


def test_classical_evaluation_uses_disjoint_normal_calibration_and_saves_reports(
    tmp_path: Path,
):
    dataset_root = tmp_path / "dataset"
    _dataset(dataset_root)
    output_dir = tmp_path / "outputs"

    report = evaluate_classical_baselines(
        dataset_root,
        "bottle",
        calibration_fraction=0.25,
        calibration_seed=7,
        image_percentile=95,
        pixel_percentile=99.5,
        blur_kernel_size=1,
        local_blur_kernel_size=7,
        min_region_area=0,
        output_dir=output_dir,
    )

    assert report.normal_training_image_count == 3
    assert report.calibration_image_count == 1
    assert report.evaluated_image_count == 3
    assert report.normal_image_count == 1
    assert report.anomalous_image_count == 2
    assert report.missing_mask_count == 1
    assert report.pixel_metric_image_count == 2
    assert report.ground_truth_anomaly_pixel_count == 64
    assert report.rule_based.image_metrics.auroc == pytest.approx(1.0)
    assert report.reference.image_metrics.auroc == pytest.approx(1.0)
    assert report.rule_based.pixel_threshold is None
    assert report.reference.pixel_threshold is not None
    assert len(report.per_image) == 3
    assert {row.mask_status for row in report.per_image} == {
        "normal",
        "available",
        "missing",
    }

    summary = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    with (output_dir / "per_image.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert summary["rule_based"]["pixel_threshold_method"].startswith("Per-image Otsu")
    assert summary["reference"]["pixel_threshold_percentile"] == 99.5
    assert len(rows) == 3
    assert "rule_based_predicted_anomaly" in rows[0]
    assert "reference_pixel_threshold" in rows[0]


def test_classical_evaluation_rejects_test_mask_shape_mismatch(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _dataset(dataset_root)
    _save_image(
        dataset_root / "bottle" / "ground_truth" / "crack" / "defect_mask.png",
        np.zeros((23, 24), dtype=np.uint8),
    )

    with pytest.raises(EvaluationError, match="does not match test image shape"):
        evaluate_classical_baselines(
            dataset_root,
            "bottle",
            calibration_fraction=0.25,
            image_percentile=95,
            pixel_percentile=99.5,
            blur_kernel_size=1,
            local_blur_kernel_size=7,
            min_region_area=0,
        )


def test_classical_evaluation_requires_a_test_split(tmp_path: Path):
    category = tmp_path / "bottle"
    for index in range(2):
        _save_image(category / "train" / "good" / f"normal_{index}.png", _normal_image())

    with pytest.raises(EvaluationError, match="No supported test images"):
        evaluate_classical_baselines(
            tmp_path,
            "bottle",
            calibration_fraction=0.5,
            blur_kernel_size=1,
            local_blur_kernel_size=7,
        )


