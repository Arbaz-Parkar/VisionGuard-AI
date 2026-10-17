"""Leakage-safe test evaluation for the classical anomaly baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable

import numpy as np

from visionguard.data import (
    DatasetError,
    EvaluationSample,
    discover_evaluation_images,
    split_normal_images,
)
from visionguard.evaluation import (
    BinaryMetricSummary,
    EvaluationError,
    _binary_metrics,
    _relative_image_path,
)
from visionguard.processing import (
    ClassicalDetectionResult,
    ImageValidationError,
    calibrate_reference_threshold,
    fit_per_pixel_reference,
    load_image,
    run_reference_baseline,
    run_rule_based_baseline,
    to_grayscale,
)


@dataclass(frozen=True)
class ClassicalBaselineMetrics:
    """Metrics and cutoff rules for one classical detector."""

    image_threshold: float
    image_threshold_method: str
    image_threshold_percentile: float | None
    pixel_threshold: float | None
    pixel_threshold_percentile: float | None
    pixel_threshold_method: str
    image_metrics: BinaryMetricSummary
    pixel_metrics: BinaryMetricSummary


@dataclass(frozen=True)
class ClassicalEvaluationImageResult:
    """Per-image results for both classical detectors."""

    image_path: str
    defect_type: str
    actual_anomaly: bool
    mask_status: str
    pixel_count: int
    ground_truth_anomaly_pixel_count: int | None
    rule_based_image_score: float
    rule_based_image_threshold: float
    rule_based_predicted_anomaly: bool
    rule_based_otsu_threshold: float
    rule_based_predicted_anomaly_pixel_count: int
    rule_based_inference_ms: float
    reference_image_score: float
    reference_image_threshold: float
    reference_predicted_anomaly: bool
    reference_pixel_threshold: float
    reference_predicted_anomaly_pixel_count: int
    reference_inference_ms: float


@dataclass(frozen=True)
class ClassicalEvaluationReport:
    """Aggregate test metrics and cutoffs calibrated from normal train images."""

    category: str
    evaluated_image_count: int
    normal_image_count: int
    anomalous_image_count: int
    normal_training_image_count: int
    calibration_image_count: int
    calibration_image_threshold_method: str
    calibration_image_threshold_percentile: float | None
    reference_pixel_threshold_method: str
    reference_pixel_threshold_percentile: float | None
    missing_mask_count: int
    pixel_metric_image_count: int
    ground_truth_anomaly_pixel_count: int
    rule_based: ClassicalBaselineMetrics
    reference: ClassicalBaselineMetrics
    per_image: tuple[ClassicalEvaluationImageResult, ...]


def _loaded_images(image_paths: tuple[Path, ...]):
    for image_path in image_paths:
        yield load_image(image_path)


def _image_threshold(
    scores: list[float], rule: str, percentile: float, method: str
) -> float:
    if rule not in {"percentile", "mean_plus_3std"}:
        raise EvaluationError(f"Unsupported {method} image threshold rule: {rule}.")
    values = np.asarray(scores, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise EvaluationError(f"{method} calibration did not produce finite image scores.")
    if rule == "percentile":
        if (
            isinstance(percentile, bool)
            or not isinstance(percentile, (int, float))
            or not np.isfinite(percentile)
            or not 0 < percentile <= 100
        ):
            raise EvaluationError(
                f"{method} image percentile must be greater than 0 and at most 100."
            )
        return float(np.percentile(values, percentile))
    return float(np.mean(values, dtype=np.float64) + 3.0 * np.std(values, dtype=np.float64))


def _threshold_method(rule: str, percentile: float) -> tuple[str, float | None]:
    if rule == "percentile":
        return f"{percentile:g}th percentile of held-out normal calibration scores.", float(percentile)
    if rule == "mean_plus_3std":
        return "Mean plus three population standard deviations of held-out normal scores.", None
    raise EvaluationError(f"Unsupported threshold rule: {rule}.")


def _ground_truth_mask(
    sample: EvaluationSample, expected_shape: tuple[int, int]
) -> tuple[np.ndarray | None, str]:
    if not sample.is_anomaly:
        return np.zeros(expected_shape, dtype=bool), "normal"
    if not sample.mask_available:
        return None, "missing"
    assert sample.expected_mask_path is not None
    try:
        mask = to_grayscale(load_image(sample.expected_mask_path))
    except ImageValidationError as exc:
        raise EvaluationError(
            f"Unable to load ground-truth mask {sample.expected_mask_path}: {exc}"
        ) from exc
    if mask.shape != expected_shape:
        raise EvaluationError(
            f"Ground-truth mask shape {mask.shape} does not match test image shape "
            f"{expected_shape}: {sample.expected_mask_path}"
        )
    return mask > 0, "available"


def _evaluate_method(
    samples: list[EvaluationSample],
    dataset_root: Path,
    method_name: str,
    image_threshold: float,
    scorer: Callable[[np.ndarray], ClassicalDetectionResult],
    rows: dict[str, dict[str, object]],
) -> tuple[BinaryMetricSummary, BinaryMetricSummary, int, int]:
    image_labels: list[int] = []
    image_scores: list[float] = []
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    pixel_predictions: list[np.ndarray] = []
    metric_image_count = 0
    ground_truth_pixel_count = 0

    for sample in samples:
        try:
            image = load_image(sample.image_path)
            started = time.perf_counter()
            result = scorer(image)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        except (ImageValidationError, ValueError, RuntimeError) as exc:
            raise EvaluationError(
                f"Unable to score {method_name} test image {sample.image_path}: {exc}"
            ) from exc

        if (
            result.score_map.shape != image.shape[:2]
            or not np.isfinite(result.score_map).all()
            or not np.isfinite(result.image_score)
        ):
            raise EvaluationError(
                f"{method_name} returned invalid scores for {sample.image_path}."
            )
        image_labels.append(int(sample.is_anomaly))
        image_scores.append(float(result.image_score))
        image_predicted = bool(result.image_score > image_threshold)
        relative_path = _relative_image_path(sample.image_path, dataset_root)
        truth_mask, mask_status = _ground_truth_mask(sample, result.score_map.shape)
        predicted_pixels = int(np.count_nonzero(result.mask))

        row = rows.setdefault(
            relative_path,
            {
                "image_path": relative_path,
                "defect_type": sample.defect_type,
                "actual_anomaly": sample.is_anomaly,
                "mask_status": mask_status,
                "pixel_count": int(result.score_map.size),
                "ground_truth_anomaly_pixel_count": (
                    int(np.count_nonzero(truth_mask)) if truth_mask is not None else None
                ),
            },
        )
        prefix = "rule_based" if method_name == "rule-based" else "reference"
        row[f"{prefix}_image_score"] = float(result.image_score)
        row[f"{prefix}_image_threshold"] = float(image_threshold)
        row[f"{prefix}_predicted_anomaly"] = image_predicted
        row[f"{prefix}_predicted_anomaly_pixel_count"] = predicted_pixels
        row[f"{prefix}_inference_ms"] = float(elapsed_ms)
        if prefix == "rule_based":
            row["rule_based_otsu_threshold"] = float(result.threshold)
        else:
            row["reference_pixel_threshold"] = float(result.threshold)

        if truth_mask is not None:
            pixel_labels.append(truth_mask.reshape(-1).astype(np.uint8, copy=False))
            pixel_scores.append(result.score_map.reshape(-1))
            pixel_predictions.append(result.mask.reshape(-1) > 0)
            metric_image_count += 1
            if prefix == "rule_based":
                ground_truth_pixel_count += int(np.count_nonzero(truth_mask))

    image_metrics = _binary_metrics(
        np.asarray(image_labels, dtype=np.uint8),
        np.asarray(image_scores, dtype=np.float64),
        image_threshold,
    )
    truth_values = (
        np.concatenate(pixel_labels) if pixel_labels else np.empty(0, dtype=np.uint8)
    )
    score_values = (
        np.concatenate(pixel_scores) if pixel_scores else np.empty(0, dtype=np.float32)
    )
    prediction_values = (
        np.concatenate(pixel_predictions)
        if pixel_predictions
        else np.empty(0, dtype=bool)
    )
    pixel_labels.clear()
    pixel_scores.clear()
    pixel_predictions.clear()
    pixel_metrics = _binary_metrics(
        truth_values,
        score_values,
        threshold=0.0,
        predictions=prediction_values,
    )
    return image_metrics, pixel_metrics, metric_image_count, ground_truth_pixel_count


def _atomic_text_write(path: Path, content: str) -> None:
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
        raise EvaluationError(f"Unable to write evaluation output {path}: {exc}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def save_classical_evaluation_report(
    report: ClassicalEvaluationReport, output_dir: str | Path
) -> tuple[Path, Path]:
    """Atomically write classical-baseline summary JSON and per-image CSV."""
    if not isinstance(report, ClassicalEvaluationReport):
        raise EvaluationError("report must be a ClassicalEvaluationReport.")
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
    fieldnames = list(ClassicalEvaluationImageResult.__dataclass_fields__)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(asdict(row) for row in report.per_image)
    _atomic_text_write(json_path, json_text)
    _atomic_text_write(csv_path, buffer.getvalue())
    return json_path, csv_path


def evaluate_classical_baselines(
    dataset_root: str | Path,
    category: str,
    *,
    calibration_fraction: float = 0.15,
    calibration_seed: int = 42,
    image_percentile: float = 99.5,
    pixel_percentile: float = 99.5,
    image_rule: str = "percentile",
    pixel_rule: str = "percentile",
    standard_deviation_floor: float = 1.0,
    blur_kernel_size: int = 3,
    local_blur_kernel_size: int = 31,
    min_region_area: float = 20.0,
    output_dir: str | Path | None = None,
) -> ClassicalEvaluationReport:
    """Evaluate both classical baselines without using test data for calibration.

    The per-pixel reference model is fitted on the seeded normal training subset.
    Both image thresholds and the reference pixel threshold are calibrated on the
    disjoint normal holdout. Test masks are read only for final pixel metrics.
    """
    root = Path(dataset_root).expanduser()
    try:
        split = split_normal_images(
            root,
            category,
            calibration_fraction=calibration_fraction,
            seed=calibration_seed,
        )
        samples = discover_evaluation_images(root, category)
        if not samples:
            raise EvaluationError(
                f"No supported test images were found for category '{category}' under {root}."
            )

        reference_model = fit_per_pixel_reference(_loaded_images(split.memory_bank_images))
        reference_pixel_threshold = calibrate_reference_threshold(
            reference_model,
            _loaded_images(split.calibration_images),
            percentile=pixel_percentile,
            standard_deviation_floor=standard_deviation_floor,
            rule=pixel_rule,
        )

        rule_calibration_scores: list[float] = []
        reference_calibration_scores: list[float] = []
        for image in _loaded_images(split.calibration_images):
            rule_calibration_scores.append(
                run_rule_based_baseline(
                    image,
                    blur_kernel_size=blur_kernel_size,
                    local_blur_kernel_size=local_blur_kernel_size,
                    min_region_area=min_region_area,
                ).image_score
            )
            reference_calibration_scores.append(
                run_reference_baseline(
                    image,
                    reference_model,
                    pixel_threshold=reference_pixel_threshold,
                    standard_deviation_floor=standard_deviation_floor,
                    min_region_area=min_region_area,
                ).image_score
            )
        rule_image_threshold = _image_threshold(
            rule_calibration_scores, image_rule, image_percentile, "Rule-based"
        )
        reference_image_threshold = _image_threshold(
            reference_calibration_scores, image_rule, image_percentile, "Reference"
        )

        rows: dict[str, dict[str, object]] = {}
        rule_image_metrics, rule_pixel_metrics, pixel_image_count, gt_pixel_count = (
            _evaluate_method(
                samples,
                root,
                "rule-based",
                rule_image_threshold,
                lambda image: run_rule_based_baseline(
                    image,
                    blur_kernel_size=blur_kernel_size,
                    local_blur_kernel_size=local_blur_kernel_size,
                    min_region_area=min_region_area,
                ),
                rows,
            )
        )
        reference_image_metrics, reference_pixel_metrics, ref_pixel_image_count, _ = (
            _evaluate_method(
                samples,
                root,
                "reference",
                reference_image_threshold,
                lambda image: run_reference_baseline(
                    image,
                    reference_model,
                    pixel_threshold=reference_pixel_threshold,
                    standard_deviation_floor=standard_deviation_floor,
                    min_region_area=min_region_area,
                ),
                rows,
            )
        )
        if ref_pixel_image_count != pixel_image_count:
            raise EvaluationError("Pixel metric image coverage changed during evaluation.")

        image_threshold_method, image_threshold_percentile = _threshold_method(
            image_rule, image_percentile
        )
        if pixel_rule == "percentile":
            reference_pixel_threshold_method = (
                f"{pixel_percentile:g}th percentile of held-out normal calibration pixels."
            )
            reference_pixel_percentile: float | None = float(pixel_percentile)
        elif pixel_rule == "mean_plus_3std":
            reference_pixel_threshold_method = (
                "Mean plus three population standard deviations of held-out normal pixels."
            )
            reference_pixel_percentile = None
        else:
            raise EvaluationError(f"Unsupported reference pixel threshold rule: {pixel_rule}.")
        report = ClassicalEvaluationReport(
            category=category,
            evaluated_image_count=len(samples),
            normal_image_count=sum(not sample.is_anomaly for sample in samples),
            anomalous_image_count=sum(sample.is_anomaly for sample in samples),
            normal_training_image_count=len(split.memory_bank_images),
            calibration_image_count=len(split.calibration_images),
            calibration_image_threshold_method=image_threshold_method,
            calibration_image_threshold_percentile=image_threshold_percentile,
            reference_pixel_threshold_method=reference_pixel_threshold_method,
            reference_pixel_threshold_percentile=reference_pixel_percentile,
            missing_mask_count=sum(
                sample.is_anomaly and not sample.mask_available for sample in samples
            ),
            pixel_metric_image_count=pixel_image_count,
            ground_truth_anomaly_pixel_count=gt_pixel_count,
            rule_based=ClassicalBaselineMetrics(
                image_threshold=rule_image_threshold,
                image_threshold_method=image_threshold_method,
                image_threshold_percentile=image_threshold_percentile,
                pixel_threshold=None,
                pixel_threshold_percentile=None,
                pixel_threshold_method="Per-image Otsu threshold followed by opening and closing morphology.",
                image_metrics=rule_image_metrics,
                pixel_metrics=rule_pixel_metrics,
            ),
            reference=ClassicalBaselineMetrics(
                image_threshold=reference_image_threshold,
                image_threshold_method=image_threshold_method,
                image_threshold_percentile=image_threshold_percentile,
                pixel_threshold=reference_pixel_threshold,
                pixel_threshold_percentile=reference_pixel_percentile,
                pixel_threshold_method=reference_pixel_threshold_method,
                image_metrics=reference_image_metrics,
                pixel_metrics=reference_pixel_metrics,
            ),
            per_image=tuple(
                ClassicalEvaluationImageResult(**rows[_relative_image_path(sample.image_path, root)])
                for sample in samples
            ),
        )
    except EvaluationError:
        raise
    except (DatasetError, ImageValidationError, OSError, ValueError, RuntimeError) as exc:
        raise EvaluationError(str(exc)) from exc

    if output_dir is not None:
        save_classical_evaluation_report(report, output_dir)
    return report
