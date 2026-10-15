"""Test-set evaluation and report export for the patch detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from io import StringIO
import json
import os
from pathlib import Path
import tempfile

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
    mask_status: str
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
    ground_truth_anomaly_pixel_count: int
    pixel_threshold: float
    image_threshold: float
    pixel_false_positive_rate_target: float
    image_false_positive_rate_target: float
    image_metrics: BinaryMetricSummary
    pixel_metrics: BinaryMetricSummary
    per_image: tuple[EvaluationImageResult, ...]


def _binary_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
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
            true_negative=0,
            false_positive=0,
            false_negative=0,
            true_positive=0,
            sample_count=0,
        )

    predicted = values > threshold
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


def evaluate_patch_detector(
    dataset_root: str | Path,
    artifact: PatchDetectorArtifact,
    feature_extractor: PatchFeatureExtractor,
    *,
    output_dir: str | Path | None = None,
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
    if not samples:
        raise EvaluationError(
            f"No supported test images were found for category '{artifact.category}' under {root}."
        )

    image_labels: list[int] = []
    image_scores: list[float] = []
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    rows: list[EvaluationImageResult] = []
    missing_mask_count = 0
    pixel_metric_image_count = 0
    ground_truth_anomaly_pixel_count = 0

    for sample in samples:
        try:
            image = load_image(sample.image_path)
            result = scorer.run(image, pixel_threshold=artifact.thresholds.pixel_threshold)
        except (ImageValidationError, ValueError, RuntimeError) as exc:
            raise EvaluationError(f"Unable to score test image {sample.image_path}: {exc}") from exc

        if result.raw_map.shape != image.shape[:2] or not np.isfinite(result.raw_map).all():
            raise EvaluationError(f"Scorer returned an invalid map for {sample.image_path}.")
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
            predicted_mask = result.raw_map > artifact.thresholds.pixel_threshold
            pixel_labels.append(truth_mask.reshape(-1).astype(np.uint8, copy=False))
            pixel_scores.append(result.raw_map.reshape(-1))
            pixel_metric_image_count += 1
            ground_truth_pixels = int(np.count_nonzero(truth_mask))
            ground_truth_anomaly_pixel_count += ground_truth_pixels
            predicted_pixels: int | None = int(np.count_nonzero(predicted_mask))
        else:
            ground_truth_pixels = None
            predicted_pixels = int(
                np.count_nonzero(result.raw_map > artifact.thresholds.pixel_threshold)
            )

        rows.append(
            EvaluationImageResult(
                image_path=relative_path,
                defect_type=sample.defect_type,
                actual_anomaly=sample.is_anomaly,
                image_score=float(result.image_score),
                image_threshold=artifact.thresholds.image_threshold,
                predicted_anomaly=bool(predicted_anomaly),
                image_inference_ms=float(result.timings.total_ms),
                mask_status=mask_status,
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
    pixel_labels.clear()
    pixel_scores.clear()
    pixel_metrics = _binary_metrics(
        pixel_truth,
        pixel_score_values,
        artifact.thresholds.pixel_threshold,
    )
    report = EvaluationReport(
        category=artifact.category,
        evaluated_image_count=len(samples),
        normal_image_count=sum(not sample.is_anomaly for sample in samples),
        anomalous_image_count=sum(sample.is_anomaly for sample in samples),
        missing_mask_count=missing_mask_count,
        pixel_metric_image_count=pixel_metric_image_count,
        ground_truth_anomaly_pixel_count=ground_truth_anomaly_pixel_count,
        pixel_threshold=artifact.thresholds.pixel_threshold,
        image_threshold=artifact.thresholds.image_threshold,
        pixel_false_positive_rate_target=artifact.thresholds.pixel_false_positive_rate,
        image_false_positive_rate_target=artifact.thresholds.image_false_positive_rate,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
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
