from dataclasses import replace
import json

import pytest
import torch

from visionguard.classical_evaluation import (
    ClassicalBaselineMetrics,
    ClassicalEvaluationImageResult,
    ClassicalEvaluationReport,
)
from visionguard.analysis_outputs import (
    build_experiment_metadata,
    save_comparison_plots,
    save_experiment_metadata,
    save_sweep_report,
)
from visionguard.comparison import combine_evaluation_reports, save_comparison_report
from visionguard.evaluation import (
    BinaryMetricSummary,
    EvaluationError,
    EvaluationImageResult,
    EvaluationReport,
)
from visionguard.model import (
    MemoryBank,
    PatchAnomalyScorer,
    PatchDetectorArtifact,
    PatchThresholdCalibration,
)
from visionguard.sweep import MemoryBankSweepPoint


class _DummyExtractor:
    device = torch.device("cpu")


def _metric_summary(sample_count: int) -> BinaryMetricSummary:
    return BinaryMetricSummary(
        auroc=1.0,
        auroc_unavailable_reason=None,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        iou=1.0,
        true_negative=2,
        false_positive=0,
        false_negative=0,
        true_positive=1,
        sample_count=sample_count,
    )


def _reports():
    image_rows = (
        EvaluationImageResult(
            image_path="bottle/test/good/normal_a.png",
            defect_type="good",
            actual_anomaly=False,
            image_score=0.1,
            image_threshold=0.5,
            predicted_anomaly=False,
            image_inference_ms=2.0,
            preprocessing_ms=0.5,
            model_inference_ms=1.0,
            total_ms=2.0,
            mask_status="normal",
            image_height=2,
            image_width=2,
            pixel_count=4,
            ground_truth_anomaly_pixel_count=0,
            predicted_anomaly_pixel_count=0,
        ),
        EvaluationImageResult(
            image_path="bottle/test/good/normal_b.png",
            defect_type="good",
            actual_anomaly=False,
            image_score=0.2,
            image_threshold=0.5,
            predicted_anomaly=False,
            image_inference_ms=3.0,
            preprocessing_ms=0.6,
            model_inference_ms=1.5,
            total_ms=3.0,
            mask_status="normal",
            image_height=2,
            image_width=2,
            pixel_count=4,
            ground_truth_anomaly_pixel_count=0,
            predicted_anomaly_pixel_count=0,
        ),
        EvaluationImageResult(
            image_path="bottle/test/crack/part.png",
            defect_type="crack",
            actual_anomaly=True,
            image_score=0.8,
            image_threshold=0.5,
            predicted_anomaly=True,
            image_inference_ms=5.0,
            preprocessing_ms=0.7,
            model_inference_ms=4.0,
            total_ms=5.0,
            mask_status="available",
            image_height=2,
            image_width=2,
            pixel_count=4,
            ground_truth_anomaly_pixel_count=1,
            predicted_anomaly_pixel_count=1,
        ),
    )
    classical_rows = (
        ClassicalEvaluationImageResult(
            image_path="bottle/test/good/normal_a.png",
            defect_type="good",
            actual_anomaly=False,
            mask_status="normal",
            pixel_count=4,
            ground_truth_anomaly_pixel_count=0,
            rule_based_image_score=0.1,
            rule_based_image_threshold=0.5,
            rule_based_predicted_anomaly=False,
            rule_based_otsu_threshold=0.2,
            rule_based_predicted_anomaly_pixel_count=0,
            rule_based_inference_ms=4.0,
            rule_based_preprocessing_ms=0.4,
            rule_based_model_inference_ms=3.4,
            rule_based_total_ms=4.0,
            image_height=2,
            image_width=2,
            reference_image_score=0.1,
            reference_image_threshold=0.5,
            reference_predicted_anomaly=False,
            reference_pixel_threshold=1.0,
            reference_predicted_anomaly_pixel_count=0,
            reference_inference_ms=5.0,
            reference_preprocessing_ms=0.4,
            reference_model_inference_ms=4.4,
            reference_total_ms=5.0,
        ),
        ClassicalEvaluationImageResult(
            image_path="bottle/test/good/normal_b.png",
            defect_type="good",
            actual_anomaly=False,
            mask_status="normal",
            pixel_count=4,
            ground_truth_anomaly_pixel_count=0,
            rule_based_image_score=0.2,
            rule_based_image_threshold=0.5,
            rule_based_predicted_anomaly=False,
            rule_based_otsu_threshold=0.2,
            rule_based_predicted_anomaly_pixel_count=0,
            rule_based_inference_ms=5.0,
            rule_based_preprocessing_ms=0.4,
            rule_based_model_inference_ms=4.4,
            rule_based_total_ms=5.0,
            image_height=2,
            image_width=2,
            reference_image_score=0.2,
            reference_image_threshold=0.5,
            reference_predicted_anomaly=False,
            reference_pixel_threshold=1.0,
            reference_predicted_anomaly_pixel_count=0,
            reference_inference_ms=6.0,
            reference_preprocessing_ms=0.4,
            reference_model_inference_ms=5.4,
            reference_total_ms=6.0,
        ),
        ClassicalEvaluationImageResult(
            image_path="bottle/test/crack/part.png",
            defect_type="crack",
            actual_anomaly=True,
            mask_status="available",
            pixel_count=4,
            ground_truth_anomaly_pixel_count=1,
            rule_based_image_score=0.8,
            rule_based_image_threshold=0.5,
            rule_based_predicted_anomaly=True,
            rule_based_otsu_threshold=0.3,
            rule_based_predicted_anomaly_pixel_count=1,
            rule_based_inference_ms=6.0,
            rule_based_preprocessing_ms=0.4,
            rule_based_model_inference_ms=5.4,
            rule_based_total_ms=6.0,
            image_height=2,
            image_width=2,
            reference_image_score=0.7,
            reference_image_threshold=0.5,
            reference_predicted_anomaly=True,
            reference_pixel_threshold=1.0,
            reference_predicted_anomaly_pixel_count=1,
            reference_inference_ms=7.0,
            reference_preprocessing_ms=0.4,
            reference_model_inference_ms=6.4,
            reference_total_ms=7.0,
        ),
    )
    patch = EvaluationReport(
        category="bottle",
        evaluated_image_count=3,
        normal_image_count=2,
        anomalous_image_count=1,
        missing_mask_count=0,
        pixel_metric_image_count=3,
        pixel_metric_size=(2, 2),
        warmup_image_count=2,
        device="cpu",
        ground_truth_anomaly_pixel_count=1,
        pixel_threshold=0.4,
        image_threshold=0.5,
        pixel_false_positive_rate_target=0.005,
        image_false_positive_rate_target=0.05,
        image_metrics=_metric_summary(3),
        pixel_metrics=_metric_summary(12),
        per_defect_image_metrics={"crack": _metric_summary(3)},
        per_defect_pixel_metrics={"crack": _metric_summary(8)},
        per_image=image_rows,
    )
    rule_metrics = ClassicalBaselineMetrics(
        image_threshold=0.5,
        image_threshold_method="95th percentile",
        image_threshold_percentile=95.0,
        pixel_threshold=None,
        pixel_threshold_percentile=None,
        pixel_threshold_method="Per-image Otsu threshold and morphology.",
        image_metrics=_metric_summary(3),
        pixel_metrics=_metric_summary(12),
    )
    reference_metrics = ClassicalBaselineMetrics(
        image_threshold=0.5,
        image_threshold_method="95th percentile",
        image_threshold_percentile=95.0,
        pixel_threshold=1.0,
        pixel_threshold_percentile=99.5,
        pixel_threshold_method="99.5th percentile",
        image_metrics=_metric_summary(3),
        pixel_metrics=_metric_summary(12),
    )
    classical = ClassicalEvaluationReport(
        category="bottle",
        evaluated_image_count=3,
        normal_image_count=2,
        anomalous_image_count=1,
        normal_training_image_count=2,
        calibration_image_count=1,
        calibration_image_threshold_method="95th percentile",
        calibration_image_threshold_percentile=95.0,
        reference_pixel_threshold_method="99.5th percentile",
        reference_pixel_threshold_percentile=99.5,
        missing_mask_count=0,
        pixel_metric_image_count=3,
        pixel_metric_size=(2, 2),
        warmup_image_count=2,
        ground_truth_anomaly_pixel_count=1,
        rule_based=rule_metrics,
        reference=reference_metrics,
        per_defect_pixel_metrics={
            "crack": {
                "rule_based": _metric_summary(8),
                "reference": _metric_summary(8),
            }
        },
        per_image=classical_rows,
    )
    memory_bank = MemoryBank(
        embeddings=torch.zeros((1, 2), dtype=torch.float32),
        source_image_count=2,
        candidate_patch_count=1,
        sampling_strategy="random",
        sampling_ratio=1.0,
        seed=1,
        input_size=(8, 8),
    )
    scorer = PatchAnomalyScorer(_DummyExtractor(), memory_bank, input_size=(8, 8))
    thresholds = PatchThresholdCalibration(
        pixel_threshold=0.4,
        image_threshold=0.5,
        pixel_false_positive_rate=0.005,
        image_false_positive_rate=0.05,
        pixel_percentile=99.5,
        image_percentile=95.0,
        calibration_image_count=1,
        calibration_pixel_count=4,
    )
    artifact = PatchDetectorArtifact.from_components(
        category="bottle",
        memory_bank=memory_bank,
        thresholds=thresholds,
        scorer=scorer,
    )
    return patch, classical, artifact


def test_combined_report_aligns_methods_and_saves_outputs(tmp_path):
    patch, classical, artifact = _reports()

    report = combine_evaluation_reports(patch, classical, artifact)
    json_path, csv_path = save_comparison_report(report, tmp_path)

    assert report.evaluated_image_count == 3
    assert set(report.methods) == {"patch_detector", "rule_based", "reference"}
    assert report.methods["patch_detector"].mean_inference_ms == pytest.approx(10 / 3)
    assert report.methods["patch_detector"].mean_preprocessing_ms == pytest.approx(0.6)
    assert report.methods["patch_detector"].mean_total_ms == pytest.approx(10 / 3)
    assert report.per_defect_image_metrics["crack"]["patch_detector"].auroc == 1.0
    assert report.per_defect_pixel_metrics["crack"]["reference"].sample_count == 8
    assert len(report.per_image) == 3
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(saved["per_image"]) == 3
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 4


def test_combined_report_rejects_different_test_image_coverage():
    patch, classical, artifact = _reports()
    wrong_classical_row = replace(classical.per_image[0], image_path="bottle/test/good/other.png")
    classical = replace(classical, per_image=(wrong_classical_row, *classical.per_image[1:]))

    with pytest.raises(EvaluationError, match="identical test images"):
        combine_evaluation_reports(patch, classical, artifact)


def test_analysis_outputs_write_plots_and_sweep_tables(tmp_path):
    patch, classical, artifact = _reports()
    report = combine_evaluation_reports(patch, classical, artifact)
    points = (
        MemoryBankSweepPoint(
            sampling_strategy="random",
            requested_ratio=0.5,
            effective_ratio=0.5,
            sampling_seed=42,
            candidate_pool_seed=43,
            source_candidate_patch_count=4,
            candidate_patch_count=4,
            stored_patch_count=2,
            stored_megabytes=0.01,
            image_auroc=0.9,
            auroc_unavailable_reason=None,
            mean_inference_ms=2.0,
            median_inference_ms=2.0,
            evaluated_image_count=3,
        ),
        MemoryBankSweepPoint(
            sampling_strategy="greedy",
            requested_ratio=1.0,
            effective_ratio=1.0,
            sampling_seed=42,
            candidate_pool_seed=43,
            source_candidate_patch_count=4,
            candidate_patch_count=4,
            stored_patch_count=4,
            stored_megabytes=0.02,
            image_auroc=1.0,
            auroc_unavailable_reason=None,
            mean_inference_ms=3.0,
            median_inference_ms=3.0,
            evaluated_image_count=3,
        ),
    )

    plots = save_comparison_plots(report, points, tmp_path / "plots")
    sweep_json, sweep_csv = save_sweep_report(points, tmp_path / "metrics")
    metadata = build_experiment_metadata(
        config={"dataset": {"calibration_fraction": 0.15, "calibration_seed": 42}},
        report=report,
        artifact=artifact,
        dataset_root=tmp_path / "dataset",
        sweep_points=points,
    )
    metadata_path = save_experiment_metadata(metadata, tmp_path / "metrics")

    assert {path.name for path in plots} == {
        "roc_curve.png",
        "confusion_matrix.png",
        "score_distributions.png",
        "memory_bank_sweep.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in plots)
    assert sweep_json.is_file()
    assert len(sweep_csv.read_text(encoding="utf-8").splitlines()) == 3
    assert metadata_path.is_file()
    assert metadata["sweep_design"]["test_labels_used_for_selection"] is False
