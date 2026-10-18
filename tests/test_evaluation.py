import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from visionguard.evaluation import (
    EvaluationError,
    _binary_metrics,
    _pixel_metric_arrays,
    evaluate_patch_detector,
)
from visionguard.model import (
    MemoryBank,
    PatchAnomalyScorer,
    PatchDetectorArtifact,
    PatchThresholdCalibration,
)


class PixelFeatureExtractor:
    device = torch.device("cpu")

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return images.permute(0, 2, 3, 1).contiguous()


def _artifact() -> PatchDetectorArtifact:
    bank = MemoryBank(
        embeddings=torch.zeros((1, 3), dtype=torch.float32),
        source_image_count=1,
        candidate_patch_count=1,
        sampling_strategy="random",
        sampling_ratio=1.0,
        seed=1,
        input_size=(8, 8),
        preserve_aspect_ratio=False,
    )
    scorer = PatchAnomalyScorer(
        PixelFeatureExtractor(),
        bank,
        input_size=(8, 8),
        preserve_aspect_ratio=False,
        smoothing_sigma=0,
    )
    thresholds = PatchThresholdCalibration(
        pixel_threshold=0.5,
        image_threshold=0.2,
        pixel_false_positive_rate=0.005,
        image_false_positive_rate=0.05,
        pixel_percentile=99.5,
        image_percentile=95.0,
        calibration_image_count=2,
        calibration_pixel_count=128,
    )
    return PatchDetectorArtifact.from_components(
        category="bottle",
        memory_bank=bank,
        thresholds=thresholds,
        scorer=scorer,
    )


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def test_evaluation_reports_image_and_pixel_metrics_and_missing_masks(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    category_root = dataset_root / "bottle"
    _save_rgb(
        category_root / "test" / "good" / "normal_black.png",
        np.zeros((8, 8, 3), dtype=np.uint8),
    )
    _save_rgb(
        category_root / "test" / "good" / "normal_dim.png",
        np.full((8, 8, 3), 20, dtype=np.uint8),
    )
    localized_defect = np.zeros((8, 8, 3), dtype=np.uint8)
    localized_defect[3:5, 4:6] = 255
    _save_rgb(
        category_root / "test" / "broken" / "localized.png", localized_defect
    )
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[3:5, 4:6] = 255
    mask_path = category_root / "ground_truth" / "broken" / "localized_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(mask_path)
    _save_rgb(
        category_root / "test" / "broken" / "missing_mask.png",
        np.full((8, 8, 3), 100, dtype=np.uint8),
    )

    output_dir = tmp_path / "outputs"
    report = evaluate_patch_detector(
        dataset_root,
        _artifact(),
        PixelFeatureExtractor(),
        output_dir=output_dir,
        pixel_metric_size=(8, 8),
    )

    assert report.evaluated_image_count == 4
    assert report.normal_image_count == 2
    assert report.anomalous_image_count == 2
    assert report.missing_mask_count == 1
    assert report.pixel_metric_image_count == 3
    assert report.ground_truth_anomaly_pixel_count == 4
    assert report.image_metrics.auroc == pytest.approx(1.0)
    assert report.image_metrics.f1 == pytest.approx(1.0)
    assert report.pixel_metrics.auroc == pytest.approx(1.0)
    # The four-pixel candidate is removed by the default 20-pixel region filter.
    assert report.pixel_metrics.f1 == pytest.approx(0.0)
    assert report.pixel_metrics.sample_count == 3 * 8 * 8
    assert [row.mask_status for row in report.per_image].count("missing") == 1
    assert report.per_image[0].image_path.startswith("bottle/test/")

    saved_summary = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    with (output_dir / "per_image.csv").open(encoding="utf-8", newline="") as stream:
        saved_rows = list(csv.DictReader(stream))
    assert saved_summary["missing_mask_count"] == 1
    assert len(saved_rows) == 4
    assert {row["mask_status"] for row in saved_rows} == {
        "normal",
        "available",
        "missing",
    }


def test_evaluation_reports_undefined_auc_when_labels_have_one_class(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _save_rgb(
        dataset_root / "bottle" / "test" / "good" / "normal.png",
        np.zeros((8, 8, 3), dtype=np.uint8),
    )

    report = evaluate_patch_detector(
        dataset_root,
        _artifact(),
        PixelFeatureExtractor(),
        pixel_metric_size=(8, 8),
    )

    assert report.image_metrics.auroc is None
    assert "both normal and anomalous" in report.image_metrics.auroc_unavailable_reason
    assert report.pixel_metrics.auroc is None
    assert report.pixel_metrics.auroc_unavailable_reason is not None


def test_evaluation_rejects_missing_test_images_and_wrong_mask_shapes(tmp_path: Path):
    with pytest.raises(EvaluationError, match="No supported test images"):
        evaluate_patch_detector(tmp_path / "missing", _artifact(), PixelFeatureExtractor())

    category_root = tmp_path / "dataset" / "bottle"
    _save_rgb(
        category_root / "test" / "crack" / "part.png",
        np.full((8, 8, 3), 255, dtype=np.uint8),
    )
    mask_path = category_root / "ground_truth" / "crack" / "part_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((7, 8), dtype=np.uint8)).save(mask_path)

    with pytest.raises(EvaluationError, match="does not match image/map shape"):
        evaluate_patch_detector(
            tmp_path / "dataset",
            _artifact(),
            PixelFeatureExtractor(),
            pixel_metric_size=(8, 8),
        )


def test_pixel_metric_arrays_use_bilinear_scores_and_nearest_neighbor_masks():
    truth = np.zeros((2, 2), dtype=bool)
    truth[0, 0] = True
    scores = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    predictions = truth.copy()

    resized_truth, resized_scores, resized_predictions = _pixel_metric_arrays(
        truth, scores, predictions, (4, 4)
    )

    assert resized_truth.shape == (16,)
    assert resized_scores.shape == (16,)
    assert resized_predictions.shape == (16,)
    assert np.count_nonzero(resized_truth) == 4
    assert np.array_equal(resized_truth, resized_predictions.astype(np.uint8))


def test_binary_metrics_calculates_iou_from_the_final_prediction_mask():
    summary = _binary_metrics(
        np.asarray([1, 1, 0], dtype=np.uint8),
        np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        threshold=0.5,
        predictions=np.asarray([True, False, True]),
    )

    assert summary.true_positive == 1
    assert summary.false_positive == 1
    assert summary.false_negative == 1
    assert summary.iou == pytest.approx(1.0 / 3.0)
