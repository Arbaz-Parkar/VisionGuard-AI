"""Combine the three detector evaluations into a directly comparable report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from io import StringIO
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from visionguard.classical_evaluation import (
    ClassicalEvaluationReport,
    ClassicalEvaluationImageResult,
)
from visionguard.evaluation import (
    BinaryMetricSummary,
    EvaluationError,
    EvaluationImageResult,
    EvaluationReport,
    _binary_metrics,
)
from visionguard.model import PatchDetectorArtifact


@dataclass(frozen=True)
class ComparisonMethodSummary:
    """Metrics, fixed thresholds, and observed inference times for one method."""

    label: str
    image_metrics: BinaryMetricSummary
    pixel_metrics: BinaryMetricSummary
    image_threshold: float
    image_threshold_method: str
    pixel_threshold: float | None
    pixel_threshold_method: str
    mean_inference_ms: float
    median_inference_ms: float
    mean_preprocessing_ms: float
    median_preprocessing_ms: float
    mean_model_inference_ms: float
    median_model_inference_ms: float
    mean_total_ms: float
    median_total_ms: float


@dataclass(frozen=True)
class ComparisonImageResult:
    """Aligned per-image scores and final mask counts for all approaches."""

    image_path: str
    defect_type: str
    actual_anomaly: bool
    mask_status: str
    image_height: int
    image_width: int
    pixel_count: int
    ground_truth_anomaly_pixel_count: int | None
    patch_image_score: float
    patch_image_threshold: float
    patch_predicted_anomaly: bool
    patch_pixel_threshold: float
    patch_predicted_anomaly_pixel_count: int
    patch_inference_ms: float
    patch_preprocessing_ms: float
    patch_model_inference_ms: float
    patch_total_ms: float
    rule_based_image_score: float
    rule_based_image_threshold: float
    rule_based_predicted_anomaly: bool
    rule_based_otsu_threshold: float
    rule_based_predicted_anomaly_pixel_count: int
    rule_based_inference_ms: float
    rule_based_preprocessing_ms: float
    rule_based_model_inference_ms: float
    rule_based_total_ms: float
    reference_image_score: float
    reference_image_threshold: float
    reference_predicted_anomaly: bool
    reference_pixel_threshold: float
    reference_predicted_anomaly_pixel_count: int
    reference_inference_ms: float
    reference_preprocessing_ms: float
    reference_model_inference_ms: float
    reference_total_ms: float


@dataclass(frozen=True)
class ComparisonReport:
    """Summary for three methods evaluated against the same test images."""

    category: str
    evaluated_image_count: int
    normal_image_count: int
    anomalous_image_count: int
    missing_mask_count: int
    pixel_metric_image_count: int
    pixel_metric_size: tuple[int, int]
    warmup_image_count: int
    smoke_test: bool
    model_input_size: tuple[int, int]
    normal_training_image_count: int
    calibration_image_count: int
    devices: dict[str, str]
    image_metrics_note: str
    latency_note: str
    pixel_metrics_note: str
    methods: dict[str, ComparisonMethodSummary]
    per_defect_image_metrics: dict[str, dict[str, BinaryMetricSummary]]
    per_defect_pixel_metrics: dict[str, dict[str, BinaryMetricSummary]]
    per_image: tuple[ComparisonImageResult, ...]


def _method_values(
    patch: EvaluationImageResult, classical: ClassicalEvaluationImageResult
) -> dict[str, tuple[float, float, bool]]:
    return {
        "patch_detector": (
            patch.image_score,
            patch.image_threshold,
            patch.predicted_anomaly,
        ),
        "rule_based": (
            classical.rule_based_image_score,
            classical.rule_based_image_threshold,
            classical.rule_based_predicted_anomaly,
        ),
        "reference": (
            classical.reference_image_score,
            classical.reference_image_threshold,
            classical.reference_predicted_anomaly,
        ),
    }


def _latency_summary(values: list[float]) -> tuple[float, float]:
    if not values or not np.isfinite(values).all():
        raise EvaluationError("Inference timing values must be finite and non-empty.")
    return float(np.mean(values)), float(np.median(values))


def combine_evaluation_reports(
    patch: EvaluationReport,
    classical: ClassicalEvaluationReport,
    artifact: PatchDetectorArtifact,
    *,
    patch_device: str = "unknown",
    smoke_test: bool = False,
) -> ComparisonReport:
    """Validate test-set parity and assemble comparable method summaries."""
    if not isinstance(patch, EvaluationReport):
        raise EvaluationError("patch must be an EvaluationReport.")
    if not isinstance(classical, ClassicalEvaluationReport):
        raise EvaluationError("classical must be a ClassicalEvaluationReport.")
    if not isinstance(artifact, PatchDetectorArtifact):
        raise EvaluationError("artifact must be a PatchDetectorArtifact.")
    if patch.category != classical.category or patch.category != artifact.category:
        raise EvaluationError("Patch, classical, and artifact categories must match.")
    if patch.pixel_metric_size != classical.pixel_metric_size:
        raise EvaluationError("Patch and classical pixel metric sizes must match.")
    if patch.warmup_image_count != classical.warmup_image_count:
        raise EvaluationError("Patch and classical warm-up counts must match.")

    patch_rows = {row.image_path: row for row in patch.per_image}
    classical_rows = {row.image_path: row for row in classical.per_image}
    if len(patch_rows) != len(patch.per_image) or len(classical_rows) != len(classical.per_image):
        raise EvaluationError("Per-image reports contain duplicate image paths.")
    if patch_rows.keys() != classical_rows.keys():
        raise EvaluationError("Patch and classical reports do not contain identical test images.")

    merged_rows: list[ComparisonImageResult] = []
    for image_path in sorted(patch_rows):
        patch_row = patch_rows[image_path]
        classical_row = classical_rows[image_path]
        shared_patch = (
            patch_row.defect_type,
            patch_row.actual_anomaly,
            patch_row.mask_status,
            patch_row.image_height,
            patch_row.image_width,
            patch_row.pixel_count,
            patch_row.ground_truth_anomaly_pixel_count,
        )
        shared_classical = (
            classical_row.defect_type,
            classical_row.actual_anomaly,
            classical_row.mask_status,
            classical_row.image_height,
            classical_row.image_width,
            classical_row.pixel_count,
            classical_row.ground_truth_anomaly_pixel_count,
        )
        if shared_patch != shared_classical:
            raise EvaluationError(f"Ground-truth coverage differs for {image_path}.")

        merged_rows.append(
            ComparisonImageResult(
                image_path=image_path,
                defect_type=patch_row.defect_type,
                actual_anomaly=patch_row.actual_anomaly,
                mask_status=patch_row.mask_status,
                image_height=patch_row.image_height,
                image_width=patch_row.image_width,
                pixel_count=patch_row.pixel_count,
                ground_truth_anomaly_pixel_count=(
                    patch_row.ground_truth_anomaly_pixel_count
                ),
                patch_image_score=patch_row.image_score,
                patch_image_threshold=patch_row.image_threshold,
                patch_predicted_anomaly=patch_row.predicted_anomaly,
                patch_pixel_threshold=artifact.thresholds.pixel_threshold,
                patch_predicted_anomaly_pixel_count=(
                    patch_row.predicted_anomaly_pixel_count
                ),
                patch_inference_ms=patch_row.image_inference_ms,
                patch_preprocessing_ms=patch_row.preprocessing_ms,
                patch_model_inference_ms=patch_row.model_inference_ms,
                patch_total_ms=patch_row.total_ms,
                rule_based_image_score=classical_row.rule_based_image_score,
                rule_based_image_threshold=classical_row.rule_based_image_threshold,
                rule_based_predicted_anomaly=classical_row.rule_based_predicted_anomaly,
                rule_based_otsu_threshold=classical_row.rule_based_otsu_threshold,
                rule_based_predicted_anomaly_pixel_count=(
                    classical_row.rule_based_predicted_anomaly_pixel_count
                ),
                rule_based_inference_ms=classical_row.rule_based_inference_ms,
                rule_based_preprocessing_ms=classical_row.rule_based_preprocessing_ms,
                rule_based_model_inference_ms=classical_row.rule_based_model_inference_ms,
                rule_based_total_ms=classical_row.rule_based_total_ms,
                reference_image_score=classical_row.reference_image_score,
                reference_image_threshold=classical_row.reference_image_threshold,
                reference_predicted_anomaly=classical_row.reference_predicted_anomaly,
                reference_pixel_threshold=classical_row.reference_pixel_threshold,
                reference_predicted_anomaly_pixel_count=(
                    classical_row.reference_predicted_anomaly_pixel_count
                ),
                reference_inference_ms=classical_row.reference_inference_ms,
                reference_preprocessing_ms=classical_row.reference_preprocessing_ms,
                reference_model_inference_ms=classical_row.reference_model_inference_ms,
                reference_total_ms=classical_row.reference_total_ms,
            )
        )

    if (
        patch.evaluated_image_count != classical.evaluated_image_count
        or patch.normal_image_count != classical.normal_image_count
        or patch.anomalous_image_count != classical.anomalous_image_count
        or patch.missing_mask_count != classical.missing_mask_count
        or patch.pixel_metric_image_count != classical.pixel_metric_image_count
        or patch.pixel_metric_size != classical.pixel_metric_size
        or patch.image_metrics.sample_count != classical.rule_based.image_metrics.sample_count
        or patch.pixel_metrics.sample_count != classical.rule_based.pixel_metrics.sample_count
        or patch.pixel_metrics.sample_count != classical.reference.pixel_metrics.sample_count
    ):
        raise EvaluationError("Patch and classical evaluation coverage does not match.")

    method_series = {
        "patch_detector": {
            "label": "PatchCore-inspired detector",
            "image_metrics": patch.image_metrics,
            "pixel_metrics": patch.pixel_metrics,
            "image_threshold": patch.image_threshold,
            "image_threshold_method": (
                f"{artifact.thresholds.image_percentile:g}th percentile of held-out "
                "normal image scores."
            ),
            "pixel_threshold": patch.pixel_threshold,
            "pixel_threshold_method": (
                f"{artifact.thresholds.pixel_percentile:g}th percentile of held-out "
                "normal pixel scores."
            ),
            "latency": [row.patch_inference_ms for row in merged_rows],
            "preprocessing": [row.patch_preprocessing_ms for row in merged_rows],
            "model_inference": [row.patch_model_inference_ms for row in merged_rows],
            "total": [row.patch_total_ms for row in merged_rows],
        },
        "rule_based": {
            "label": "Rule-based classical baseline",
            "image_metrics": classical.rule_based.image_metrics,
            "pixel_metrics": classical.rule_based.pixel_metrics,
            "image_threshold": classical.rule_based.image_threshold,
            "image_threshold_method": classical.rule_based.image_threshold_method,
            "pixel_threshold": None,
            "pixel_threshold_method": classical.rule_based.pixel_threshold_method,
            "latency": [row.rule_based_inference_ms for row in merged_rows],
            "preprocessing": [row.rule_based_preprocessing_ms for row in merged_rows],
            "model_inference": [row.rule_based_model_inference_ms for row in merged_rows],
            "total": [row.rule_based_total_ms for row in merged_rows],
        },
        "reference": {
            "label": "Per-pixel reference baseline",
            "image_metrics": classical.reference.image_metrics,
            "pixel_metrics": classical.reference.pixel_metrics,
            "image_threshold": classical.reference.image_threshold,
            "image_threshold_method": classical.reference.image_threshold_method,
            "pixel_threshold": classical.reference.pixel_threshold,
            "pixel_threshold_method": classical.reference.pixel_threshold_method,
            "latency": [row.reference_inference_ms for row in merged_rows],
            "preprocessing": [row.reference_preprocessing_ms for row in merged_rows],
            "model_inference": [row.reference_model_inference_ms for row in merged_rows],
            "total": [row.reference_total_ms for row in merged_rows],
        },
    }

    methods: dict[str, ComparisonMethodSummary] = {}
    for key, values in method_series.items():
        mean_ms, median_ms = _latency_summary(values["latency"])
        mean_preprocessing_ms, median_preprocessing_ms = _latency_summary(values["preprocessing"])
        mean_model_inference_ms, median_model_inference_ms = _latency_summary(values["model_inference"])
        mean_total_ms, median_total_ms = _latency_summary(values["total"])
        methods[key] = ComparisonMethodSummary(
            label=values["label"],
            image_metrics=values["image_metrics"],
            pixel_metrics=values["pixel_metrics"],
            image_threshold=values["image_threshold"],
            image_threshold_method=values["image_threshold_method"],
            pixel_threshold=values["pixel_threshold"],
            pixel_threshold_method=values["pixel_threshold_method"],
            mean_inference_ms=mean_ms,
            median_inference_ms=median_ms,
            mean_preprocessing_ms=mean_preprocessing_ms,
            median_preprocessing_ms=median_preprocessing_ms,
            mean_model_inference_ms=mean_model_inference_ms,
            median_model_inference_ms=median_model_inference_ms,
            mean_total_ms=mean_total_ms,
            median_total_ms=median_total_ms,
        )

    good_rows = [row for row in merged_rows if not row.actual_anomaly]
    defect_types = sorted({row.defect_type for row in merged_rows if row.actual_anomaly})
    per_defect: dict[str, dict[str, BinaryMetricSummary]] = {}
    for defect_type in defect_types:
        subset = good_rows + [
            row
            for row in merged_rows
            if row.actual_anomaly and row.defect_type == defect_type
        ]
        per_defect[defect_type] = {}
        for method_name in method_series:
            values = [
                _method_values(
                    patch_rows[row.image_path], classical_rows[row.image_path]
                )[method_name]
                for row in subset
            ]
            per_defect[defect_type][method_name] = _binary_metrics(
                np.asarray([row.actual_anomaly for row in subset], dtype=np.uint8),
                np.asarray([value[0] for value in values], dtype=np.float64),
                methods[method_name].image_threshold,
            )

    pixel_defect_types = sorted(
        set(patch.per_defect_pixel_metrics)
        | set(classical.per_defect_pixel_metrics)
    )
    if set(patch.per_defect_pixel_metrics) != set(classical.per_defect_pixel_metrics):
        raise EvaluationError("Patch and classical per-defect pixel coverage does not match.")
    per_defect_pixel = {
        defect_type: {
            "patch_detector": patch.per_defect_pixel_metrics[defect_type],
            "rule_based": classical.per_defect_pixel_metrics[defect_type]["rule_based"],
            "reference": classical.per_defect_pixel_metrics[defect_type]["reference"],
        }
        for defect_type in pixel_defect_types
    }

    return ComparisonReport(
        category=patch.category,
        evaluated_image_count=patch.evaluated_image_count,
        normal_image_count=patch.normal_image_count,
        anomalous_image_count=patch.anomalous_image_count,
        missing_mask_count=patch.missing_mask_count,
        pixel_metric_image_count=patch.pixel_metric_image_count,
        pixel_metric_size=patch.pixel_metric_size,
        warmup_image_count=patch.warmup_image_count,
        smoke_test=smoke_test,
        model_input_size=artifact.memory_bank.input_size,
        normal_training_image_count=classical.normal_training_image_count,
        calibration_image_count=classical.calibration_image_count,
        devices={
            "patch_detector": patch_device,
            "rule_based": "cpu",
            "reference": "cpu",
        },
        image_metrics_note=(
            "Image-level AUROC is the headline ranking metric. Accuracy is omitted because "
            "class imbalance can make it misleading."
        ),
        latency_note=(
            f"{patch.warmup_image_count} warm-up images are scored before timing. "
            "Preprocessing includes image loading "
            "and model input preparation; model inference includes feature/model scoring; "
            "total is wall time from image loading through final mask creation."
        ),
        pixel_metrics_note=(
            "Maps and masks are resized to the common metric grid using bilinear "
            "interpolation for scores and nearest-neighbor interpolation for masks. "
            "AUROC uses continuous scores; threshold metrics use each method's final mask. "
            "Normal test images use empty masks; missing anomalous masks are excluded."
        ),
        methods=methods,
        per_defect_image_metrics=per_defect,
        per_defect_pixel_metrics=per_defect_pixel,
        per_image=tuple(merged_rows),
    )


def _atomic_write(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise EvaluationError(f"Unable to write comparison report {path}: {exc}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def save_comparison_report(
    report: ComparisonReport, output_dir: str | Path
) -> tuple[Path, Path]:
    """Write the combined JSON summary and aligned per-image CSV atomically."""
    if not isinstance(report, ComparisonReport):
        raise EvaluationError("report must be a ComparisonReport.")
    destination = Path(output_dir).expanduser()
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EvaluationError(
            f"Unable to create comparison output directory {destination}: {exc}"
        ) from exc

    json_path = destination / "metrics.json"
    csv_path = destination / "per_image.csv"
    json_text = json.dumps(asdict(report), indent=2, allow_nan=False) + "\n"
    fieldnames = list(ComparisonImageResult.__dataclass_fields__)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(asdict(row) for row in report.per_image)
    _atomic_write(json_path, json_text)
    _atomic_write(csv_path, buffer.getvalue())
    return json_path, csv_path
