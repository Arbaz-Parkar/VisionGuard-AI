"""Test-set evaluation and report export for the patch detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from visionguard.data import DatasetError, discover_evaluation_images
from visionguard.processing import ImageValidationError, load_image, to_grayscale
from visionguard.model.artifacts import PatchDetectorArtifact
from visionguard.model.memory_bank import PatchFeatureExtractor


class EvaluationError(ValueError):
    """Raised when test data cannot be evaluated safely."""


@dataclass(frozen=True)
class BinaryMetricSummary:
    """Threshold metrics and AUROC for image labels or pixel labels."""

    auroc: float | None
    auroc_unavailable_reason: str | None
    precision: float
    recall: float
    f1: float
    iou: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    sample_count: int


@dataclass(frozen=True)
class EvaluationImageResult:
    """One test image's classification, score, timing, and mask coverage."""

    image_path: str
    defect_type: str
    actual_anomaly: bool
    image_score: float
    image_threshold: float
    predicted_anomaly: bool
    image_inference_ms: float
    preprocessing_ms: float
    model_inference_ms: float
    total_ms: float
    mask_status: str
    image_height: int
    image_width: int
    pixel_count: int
    ground_truth_anomaly_pixel_count: int | None
    predicted_anomaly_pixel_count: int


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate image/pixel metrics plus per-image results."""

    category: str
    evaluated_image_count: int
    normal_image_count: int
    anomalous_image_count: int
    missing_mask_count: int
    pixel_metric_image_count: int
    pixel_metric_size: tuple[int, int]
    warmup_image_count: int
    device: str
    ground_truth_anomaly_pixel_count: int
    pixel_threshold: float
    image_threshold: float
    pixel_false_positive_rate_target: float
    image_false_positive_rate_target: float
    image_metrics: BinaryMetricSummary
    pixel_metrics: BinaryMetricSummary
    per_defect_image_metrics: dict[str, BinaryMetricSummary]
    per_defect_pixel_metrics: dict[str, BinaryMetricSummary]
    per_image: tuple[EvaluationImageResult, ...]


def _binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    predictions: np.ndarray | None = None,
) -> BinaryMetricSummary:
    truth = np.asarray(labels, dtype=np.uint8).reshape(-1)
    values = np.asarray(scores).reshape(-1)
    if truth.size != values.size:
        raise EvaluationError("Metric labels and scores must have the same number of values.")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise EvaluationError("Metric scores must contain only finite numeric values.")
    if not truth.size:
        return BinaryMetricSummary(
            auroc=None,
            auroc_unavailable_reason="No labeled samples are available.",
            precision=0.0,
            recall=0.0,
            f1=0.0,
            iou=0.0,
            true_negative=0,
            false_positive=0,
            false_negative=0,
            true_positive=0,
            sample_count=0,
        )

    if predictions is None:
        predicted = values > threshold
    else:
        predicted = np.asarray(predictions, dtype=bool).reshape(-1)
        if predicted.size != truth.size:
            raise EvaluationError(
                "Metric predictions and labels must have the same number of values."
            )
    positive = truth == 1
    negative = ~positive
    true_positive = int(np.count_nonzero(predicted & positive))
    false_positive = int(np.count_nonzero(predicted & negative))
    false_negative = int(np.count_nonzero(~predicted & positive))
    true_negative = int(np.count_nonzero(~predicted & negative))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    intersection_over_union = (
        true_positive / (true_positive + false_positive + false_negative)
        if true_positive + false_positive + false_negative
        else 0.0
    )

    classes = np.unique(truth)
    if classes.size < 2:
        auroc = None
        reason = "AUROC requires both normal and anomalous labels."
    else:
        auroc = float(roc_auc_score(truth, values))
        reason = None
    return BinaryMetricSummary(
        auroc=auroc,
        auroc_unavailable_reason=reason,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        iou=float(intersection_over_union),
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
        sample_count=int(truth.size),
    )


def _relative_image_path(image_path: Path, dataset_root: Path) -> str:
    try:
        return image_path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError as exc:
        raise EvaluationError(f"Test image is outside the dataset root: {image_path}") from exc


def _pixel_metric_arrays(
    truth_mask: np.ndarray,
    score_map: np.ndarray,
    predicted_mask: np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize pixel labels, scores, and final masks to a common metric grid."""
    if (
        not isinstance(target_size, tuple)
        or len(target_size) != 2
        or any(type(value) is not int or value <= 0 for value in target_size)
    ):
        raise EvaluationError("pixel_metric_size must contain two positive integers.")
    height, width = target_size
    if truth_mask.shape != score_map.shape or predicted_mask.shape != score_map.shape:
        raise EvaluationError("Pixel masks and score maps must have matching shapes.")
    if score_map.shape != target_size:
        scores = cv2.resize(
            np.asarray(score_map, dtype=np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        truth = cv2.resize(
            np.asarray(truth_mask, dtype=np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        predictions = cv2.resize(
            np.asarray(predicted_mask, dtype=np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        scores = np.asarray(score_map, dtype=np.float32)
        truth = np.asarray(truth_mask, dtype=np.uint8)
        predictions = np.asarray(predicted_mask, dtype=np.uint8)
    return truth.reshape(-1), scores.reshape(-1), predictions.reshape(-1).astype(bool)


def evaluate_patch_detector(
    dataset_root: str | Path,
    artifact: PatchDetectorArtifact,
    feature_extractor: PatchFeatureExtractor,
    *,
    output_dir: str | Path | None = None,
    pixel_metric_size: tuple[int, int] = (256, 256),
    warmup_count: int = 3,
    sample_limit: int | None = None,
) -> EvaluationReport:
    """Evaluate an artifact on category test images, never on calibration data.

    Image metrics include every test image. Pixel metrics include normal test
    images as all-normal masks and anomalous images only when their ground-truth
    masks exist. Missing masks are counted and excluded from pixel metrics.
    """
    if not isinstance(artifact, PatchDetectorArtifact):
        raise EvaluationError("artifact must be a PatchDetectorArtifact.")
    root = Path(dataset_root).expanduser()
    try:
        samples = discover_evaluation_images(root, artifact.category)
        scorer = artifact.build_scorer(feature_extractor)
    except (DatasetError, ValueError) as exc:
        raise EvaluationError(str(exc)) from exc
    if sample_limit is not None:
        if isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or sample_limit <= 0:
            raise EvaluationError("sample_limit must be a positive integer when provided.")
        samples = samples[:sample_limit]
    if (
        isinstance(warmup_count, bool)
        or not isinstance(warmup_count, int)
        or warmup_count < 0
    ):
        raise EvaluationError("warmup_count must be a non-negative integer.")
    if not samples:
        raise EvaluationError(
            f"No supported test images were found for category '{artifact.category}' under {root}."
        )

    image_labels: list[int] = []
    image_scores: list[float] = []
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    pixel_predictions: list[np.ndarray] = []
    rows: list[EvaluationImageResult] = []
    missing_mask_count = 0
    pixel_metric_image_count = 0
    ground_truth_anomaly_pixel_count = 0

    warmup_images = samples[: min(warmup_count, len(samples))]
    for sample in warmup_images:
        try:
            scorer.run(load_image(sample.image_path), pixel_threshold=artifact.thresholds.pixel_threshold)
        except (ImageValidationError, ValueError, RuntimeError) as exc:
            raise EvaluationError(
                f"Unable to warm up patch detector with {sample.image_path}: {exc}"
            ) from exc

    per_defect_pixel_arrays: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    for sample in samples:
        try:
            total_start = time.perf_counter()
            preprocessing_start = total_start
            image = load_image(sample.image_path)
            input_loaded_ms = (time.perf_counter() - preprocessing_start) * 1000.0
            result = scorer.run(image, pixel_threshold=artifact.thresholds.pixel_threshold)
            elapsed_total_ms = (time.perf_counter() - total_start) * 1000.0
        except (ImageValidationError, ValueError, RuntimeError) as exc:
            raise EvaluationError(f"Unable to score test image {sample.image_path}: {exc}") from exc

        if result.raw_map.shape != image.shape[:2] or not np.isfinite(result.raw_map).all():
            raise EvaluationError(f"Scorer returned an invalid map for {sample.image_path}.")
        if result.mask.shape != result.raw_map.shape:
            raise EvaluationError(f"Scorer returned an invalid mask for {sample.image_path}.")
        predicted_anomaly = result.image_score > artifact.thresholds.image_threshold
        image_labels.append(int(sample.is_anomaly))
        image_scores.append(result.image_score)

        relative_path = _relative_image_path(sample.image_path, root)
        if not sample.is_anomaly:
            truth_mask = np.zeros(result.raw_map.shape, dtype=bool)
            mask_status = "normal"
        elif not sample.mask_available:
            missing_mask_count += 1
            truth_mask = None
            mask_status = "missing"
        else:
            assert sample.expected_mask_path is not None
            try:
                mask = to_grayscale(load_image(sample.expected_mask_path))
            except ImageValidationError as exc:
                raise EvaluationError(
                    f"Unable to load ground-truth mask {sample.expected_mask_path}: {exc}"
                ) from exc
            if mask.shape != result.raw_map.shape:
                raise EvaluationError(
                    f"Ground-truth mask shape {mask.shape} does not match image/map "
                    f"shape {result.raw_map.shape}: {sample.expected_mask_path}"
                )
            truth_mask = mask > 0
            mask_status = "available"

        if truth_mask is not None:
            metric_truth, metric_scores, metric_predictions = _pixel_metric_arrays(
                truth_mask, result.raw_map, result.mask > 0, pixel_metric_size
            )
            pixel_labels.append(metric_truth)
            pixel_scores.append(metric_scores)
            pixel_predictions.append(metric_predictions)
            per_defect_pixel_arrays.setdefault(sample.defect_type, []).append(
                (metric_truth, metric_scores, metric_predictions)
            )
            pixel_metric_image_count += 1
            ground_truth_pixels = int(np.count_nonzero(truth_mask))
            ground_truth_anomaly_pixel_count += ground_truth_pixels
            predicted_pixels: int | None = int(np.count_nonzero(result.mask))
        else:
            ground_truth_pixels = None
            predicted_pixels = int(np.count_nonzero(result.mask))

        rows.append(
            EvaluationImageResult(
                image_path=relative_path,
                defect_type=sample.defect_type,
                actual_anomaly=sample.is_anomaly,
                image_score=float(result.image_score),
                image_threshold=artifact.thresholds.image_threshold,
                predicted_anomaly=bool(predicted_anomaly),
                image_inference_ms=float(result.timings.total_ms),
                preprocessing_ms=float(input_loaded_ms + result.timings.preprocessing_ms),
                model_inference_ms=float(
                    result.timings.feature_extraction_ms
                    + result.timings.nearest_neighbor_ms
                ),
                total_ms=float(elapsed_total_ms),
                mask_status=mask_status,
                image_height=int(result.raw_map.shape[0]),
                image_width=int(result.raw_map.shape[1]),
                pixel_count=int(result.raw_map.size),
                ground_truth_anomaly_pixel_count=ground_truth_pixels,
                predicted_anomaly_pixel_count=predicted_pixels,
            )
        )

    image_metrics = _binary_metrics(
        np.asarray(image_labels, dtype=np.uint8),
        np.asarray(image_scores, dtype=np.float64),
        artifact.thresholds.image_threshold,
    )
    pixel_truth = (
        np.concatenate(pixel_labels)
        if pixel_labels
        else np.empty(0, dtype=np.uint8)
    )
    pixel_score_values = (
        np.concatenate(pixel_scores)
        if pixel_scores
        else np.empty(0, dtype=np.float32)
    )
    pixel_prediction_values = (
        np.concatenate(pixel_predictions)
        if pixel_predictions
        else np.empty(0, dtype=bool)
    )
    pixel_labels.clear()
    pixel_scores.clear()
    pixel_predictions.clear()
    pixel_metrics = _binary_metrics(
        pixel_truth,
        pixel_score_values,
        artifact.thresholds.pixel_threshold,
        predictions=pixel_prediction_values,
    )
    per_defect_image_metrics: dict[str, BinaryMetricSummary] = {}
    defect_types = sorted({row.defect_type for row in rows if row.actual_anomaly})
    for defect_type in defect_types:
        subset = [row for row in rows if not row.actual_anomaly or row.defect_type == defect_type]
        per_defect_image_metrics[defect_type] = _binary_metrics(
            np.asarray([row.actual_anomaly for row in subset], dtype=np.uint8),
            np.asarray([row.image_score for row in subset], dtype=np.float64),
            artifact.thresholds.image_threshold,
        )
    per_defect_pixel_metrics: dict[str, BinaryMetricSummary] = {}
    for defect_type in defect_types:
        grouped = per_defect_pixel_arrays.get("good", []) + per_defect_pixel_arrays.get(defect_type, [])
        if grouped:
            per_defect_pixel_metrics[defect_type] = _binary_metrics(
                np.concatenate([item[0] for item in grouped]),
                np.concatenate([item[1] for item in grouped]),
                artifact.thresholds.pixel_threshold,
                predictions=np.concatenate([item[2] for item in grouped]),
            )
        else:
            per_defect_pixel_metrics[defect_type] = _binary_metrics(
                np.empty(0, dtype=np.uint8),
                np.empty(0, dtype=np.float32),
                artifact.thresholds.pixel_threshold,
            )
    report = EvaluationReport(
        category=artifact.category,
        evaluated_image_count=len(samples),
        normal_image_count=sum(not sample.is_anomaly for sample in samples),
        anomalous_image_count=sum(sample.is_anomaly for sample in samples),
        missing_mask_count=missing_mask_count,
        pixel_metric_image_count=pixel_metric_image_count,
        pixel_metric_size=pixel_metric_size,
        warmup_image_count=len(warmup_images),
        device=str(getattr(feature_extractor, "device", "unknown")),
        ground_truth_anomaly_pixel_count=ground_truth_anomaly_pixel_count,
        pixel_threshold=artifact.thresholds.pixel_threshold,
        image_threshold=artifact.thresholds.image_threshold,
        pixel_false_positive_rate_target=artifact.thresholds.pixel_false_positive_rate,
        image_false_positive_rate_target=artifact.thresholds.image_false_positive_rate,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        per_defect_image_metrics=per_defect_image_metrics,
        per_defect_pixel_metrics=per_defect_pixel_metrics,
        per_image=tuple(rows),
    )
    if output_dir is not None:
        save_evaluation_report(report, output_dir)
    return report


def _atomic_text_write(path: Path, text: str) -> None:
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
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise EvaluationError(f"Unable to write evaluation output {path}: {exc}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def save_evaluation_report(
    report: EvaluationReport, output_dir: str | Path
) -> tuple[Path, Path]:
    """Write a JSON summary and a per-image CSV report atomically."""
    if not isinstance(report, EvaluationReport):
        raise EvaluationError("report must be an EvaluationReport.")
    destination = Path(output_dir).expanduser()
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EvaluationError(
            f"Unable to create evaluation output directory {destination}: {exc}"
        ) from exc
    json_path = destination / "metrics.json"
    csv_path = destination / "per_image.csv"
    json_text = json.dumps(asdict(report), indent=2, allow_nan=False) + "\n"
    fieldnames = list(EvaluationImageResult.__dataclass_fields__)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(asdict(row) for row in report.per_image)
    _atomic_text_write(json_path, json_text)
    _atomic_text_write(csv_path, buffer.getvalue())
    return json_path, csv_path
