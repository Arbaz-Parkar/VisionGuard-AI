"""Plots, memory-bank sweep files, metadata, and failure figures."""

from __future__ import annotations

from dataclasses import asdict
import csv
from importlib import metadata as package_metadata
import json
from pathlib import Path
import platform
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

from visionguard.classical_evaluation import _loaded_images
from visionguard.comparison import ComparisonReport
from visionguard.data import DatasetError, discover_evaluation_images, split_normal_images
from visionguard.evaluation import EvaluationError
from visionguard.model import PatchDetectorArtifact, PatchFeatureExtractor
from visionguard.processing import (
    ImageValidationError,
    calibrate_reference_threshold,
    fit_per_pixel_reference,
    load_image,
    run_reference_baseline,
    run_rule_based_baseline,
    to_grayscale,
    to_rgb,
)
from visionguard.sweep import MemoryBankSweepPoint


_METHOD_FIELDS = {
    "patch_detector": ("PatchCore-inspired", "patch_image_score", "patch_image_threshold"),
    "rule_based": ("Rule-based", "rule_based_image_score", "rule_based_image_threshold"),
    "reference": ("Per-pixel reference", "reference_image_score", "reference_image_threshold"),
}


def save_comparison_plots(
    report: ComparisonReport,
    sweep_points: tuple[MemoryBankSweepPoint, ...],
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Write the image ROC, confusion matrices, score distributions, and sweep plot."""
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    smoke_prefix = "SMOKE TEST — NOT A RESEARCH RESULT — " if report.smoke_test else ""
    labels = np.asarray([row.actual_anomaly for row in report.per_image], dtype=np.uint8)

    fig, axis = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    if np.unique(labels).size < 2:
        axis.text(0.5, 0.5, "ROC unavailable: both image classes are required", ha="center", va="center")
        axis.set_axis_off()
    else:
        for key, (label, score_field, _) in _METHOD_FIELDS.items():
            scores = np.asarray([getattr(row, score_field) for row in report.per_image], dtype=np.float64)
            fpr, tpr, _ = roc_curve(labels, scores)
            auroc = report.methods[key].image_metrics.auroc
            axis.plot(fpr, tpr, label=f"{label} (AUROC {auroc:.3f})")
        axis.plot((0, 1), (0, 1), linestyle="--", color="#777777", linewidth=1)
        axis.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{smoke_prefix}Image-level ROC")
        axis.legend(loc="lower right")
        axis.grid(alpha=0.2)
    roc_path = destination / "roc_curve.png"
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)
    paths.append(roc_path)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    for axis, (key, (label, _, _)) in zip(axes, _METHOD_FIELDS.items()):
        metric = report.methods[key].image_metrics
        matrix = np.asarray(
            [[metric.true_negative, metric.false_positive],
             [metric.false_negative, metric.true_positive]],
            dtype=np.int64,
        )
        axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set_xticks((0, 1), labels=("normal", "anomaly"))
        axis.set_yticks((0, 1), labels=("normal", "anomaly"))
        axis.set(xlabel="Predicted", ylabel="Actual", title=label)
        for (row_index, column_index), value in np.ndenumerate(matrix):
            axis.text(column_index, row_index, str(value), ha="center", va="center", color="#111111")
    fig.suptitle(f"{smoke_prefix}Image-level confusion matrices")
    confusion_path = destination / "confusion_matrix.png"
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)
    paths.append(confusion_path)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    for axis, (key, (label, score_field, _)) in zip(axes, _METHOD_FIELDS.items()):
        normal = [getattr(row, score_field) for row in report.per_image if not row.actual_anomaly]
        anomalous = [getattr(row, score_field) for row in report.per_image if row.actual_anomaly]
        bins = 20 if len(report.per_image) >= 20 else max(3, len(report.per_image))
        plotted = False
        if normal:
            axis.hist(normal, bins=bins, alpha=0.65, label="normal", color="#4C78A8")
            plotted = True
        if anomalous:
            axis.hist(anomalous, bins=bins, alpha=0.6, label="anomalous", color="#E45756")
            plotted = True
        if plotted:
            axis.axvline(report.methods[key].image_threshold, color="#222222", linestyle="--", label="calibrated threshold")
            axis.legend(fontsize=8)
        axis.set(title=label, xlabel="Image anomaly score", ylabel="Images")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{smoke_prefix}Image score distributions")
    scores_path = destination / "score_distributions.png"
    fig.savefig(scores_path, dpi=160)
    plt.close(fig)
    paths.append(scores_path)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    strategies = sorted({point.sampling_strategy for point in sweep_points})
    for strategy in strategies:
        points = [point for point in sweep_points if point.sampling_strategy == strategy and point.image_auroc is not None]
        points.sort(key=lambda point: point.stored_patch_count)
        if not points:
            continue
        label = "Greedy coreset" if strategy == "greedy" else "Random sampling"
        axes[0].plot(
            [point.stored_patch_count for point in points],
            [point.image_auroc for point in points],
            marker="o",
            label=label,
        )
        axes[1].plot(
            [point.mean_inference_ms for point in points],
            [point.image_auroc for point in points],
            marker="o",
            label=label,
        )
        for axis, x_field in ((axes[0], "stored_patch_count"), (axes[1], "mean_inference_ms")):
            for point in points:
                axis.annotate(
                    f"{point.requested_ratio:g}",
                    (getattr(point, x_field), point.image_auroc),
                    xytext=(3, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
    axes[0].set(xlabel="Stored memory-bank patches", ylabel="Image AUROC", title="AUROC vs bank size")
    axes[1].set(xlabel="Mean detector inference (ms/image)", ylabel="Image AUROC", title="AUROC vs inference time")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle(f"{smoke_prefix}Memory-bank sweep (labels show requested ratio)")
    sweep_path = destination / "memory_bank_sweep.png"
    fig.savefig(sweep_path, dpi=160)
    plt.close(fig)
    paths.append(sweep_path)
    return tuple(paths)


def save_sweep_report(
    points: tuple[MemoryBankSweepPoint, ...], output_dir: str | Path
) -> tuple[Path, Path]:
    """Write memory-bank sweep records as JSON and CSV."""
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "memory_bank_sweep.json"
    csv_path = destination / "memory_bank_sweep.csv"
    rows = [asdict(point) for point in points]
    json_path.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def build_experiment_metadata(
    *,
    config: dict[str, Any],
    report: ComparisonReport,
    artifact: PatchDetectorArtifact,
    dataset_root: str | Path,
    sweep_points: tuple[MemoryBankSweepPoint, ...],
) -> dict[str, Any]:
    """Capture the settings and local library versions needed to describe a run."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for package, display_name in (
        ("numpy", "numpy"),
        ("opencv-python", "opencv"),
        ("Pillow", "pillow"),
        ("matplotlib", "matplotlib"),
        ("scikit-learn", "scikit_learn"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
    ):
        try:
            versions[display_name] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            versions[display_name] = "unavailable"
    return {
        "run_type": "smoke_test" if report.smoke_test else "research_evaluation",
        "category": report.category,
        "dataset_root": str(Path(dataset_root).expanduser()),
        "evaluation_split": "official test split",
        "evaluated_image_count": report.evaluated_image_count,
        "normal_image_count": report.normal_image_count,
        "anomalous_image_count": report.anomalous_image_count,
        "missing_mask_count": report.missing_mask_count,
        "pixel_metric_image_count": report.pixel_metric_image_count,
        "normal_training_image_count": report.normal_training_image_count,
        "calibration_image_count": report.calibration_image_count,
        "calibration_fraction": config["dataset"]["calibration_fraction"],
        "calibration_seed": config["dataset"]["calibration_seed"],
        "model_input_size": list(artifact.memory_bank.input_size),
        "pixel_metric_size": list(report.pixel_metric_size),
        "device_by_method": report.devices,
        "warmup_image_count": report.warmup_image_count,
        "memory_bank": {
            "strategy": artifact.memory_bank.sampling_strategy,
            "sampling_ratio": artifact.memory_bank.sampling_ratio,
            "seed": artifact.memory_bank.seed,
            "stored_patch_count": int(artifact.memory_bank.embeddings.shape[0]),
            "candidate_patch_count": artifact.memory_bank.candidate_patch_count,
        },
        "feature_extractor": {
            "architecture": artifact.architecture,
            "feature_layers": list(artifact.feature_layers),
            "pretrained_weights": artifact.pretrained_weights,
        },
        "scorer": {
            "smoothing_sigma": artifact.smoothing_sigma,
            "aggregation": artifact.aggregation,
            "score_percentile": artifact.score_percentile,
            "nearest_neighbor_chunk_size": artifact.nearest_neighbor_chunk_size,
        },
        "calibrated_thresholds": asdict(artifact.thresholds),
        "sweep_design": {
            "ratios": sorted({point.requested_ratio for point in sweep_points}),
            "strategies": sorted({point.sampling_strategy for point in sweep_points}),
            "candidate_pool_size": sweep_points[0].candidate_patch_count if sweep_points else 0,
            "available_candidate_patch_count": sweep_points[0].source_candidate_patch_count if sweep_points else 0,
            "candidate_pool_sampling": "seeded random subset when the available pool exceeds the configured pool limit",
            "candidate_pool_seed": sweep_points[0].candidate_pool_seed if sweep_points else None,
            "bank_sampling_seed": sweep_points[0].sampling_seed if sweep_points else None,
            "test_labels_used_for_selection": False,
            "default_changed_by_sweep": False,
        },
        "config": config,
        "library_versions": versions,
    }


def save_experiment_metadata(metadata: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "experiment_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _error_candidates(report: ComparisonReport, per_method_limit: int) -> dict[str, dict[str, list[tuple[Any, float]]]]:
    if isinstance(per_method_limit, bool) or not isinstance(per_method_limit, int) or per_method_limit <= 0:
        raise EvaluationError("per_method_limit must be a positive integer.")
    selected: dict[str, dict[str, list[tuple[Any, float]]]] = {}
    for method, (_, score_field, threshold_field) in _METHOD_FIELDS.items():
        false_positives: list[tuple[Any, float]] = []
        false_negatives: list[tuple[Any, float]] = []
        for row in report.per_image:
            score = float(getattr(row, score_field))
            threshold = float(getattr(row, threshold_field))
            predicted = score > threshold
            if not row.actual_anomaly and predicted:
                false_positives.append((row, score - threshold))
            elif row.actual_anomaly and not predicted:
                false_negatives.append((row, threshold - score))
        selected[method] = {
            "false_positives": sorted(false_positives, key=lambda item: item[1], reverse=True)[:per_method_limit],
            "false_negatives": sorted(false_negatives, key=lambda item: item[1], reverse=True)[:per_method_limit],
        }
    return selected


def save_failure_examples(
    dataset_root: str | Path,
    report: ComparisonReport,
    artifact: PatchDetectorArtifact,
    feature_extractor: PatchFeatureExtractor,
    *,
    calibration_fraction: float,
    calibration_seed: int,
    pixel_rule: str,
    pixel_percentile: float,
    blur_kernel_size: int,
    local_blur_kernel_size: int,
    min_region_area: float,
    output_dir: str | Path,
    per_method_limit: int = 5,
) -> tuple[Path, ...]:
    """Render top false positives/negatives for each method for human review."""
    root = Path(dataset_root).expanduser()
    destination = Path(output_dir).expanduser()
    selected = _error_candidates(report, per_method_limit)
    selected_by_path: dict[str, list[tuple[str, str, int, float]]] = {}
    for method, error_groups in selected.items():
        for error_type, entries in error_groups.items():
            for rank, (row, margin) in enumerate(entries, start=1):
                selected_by_path.setdefault(row.image_path, []).append(
                    (method, error_type, rank, margin)
                )
    if not selected_by_path:
        destination.mkdir(parents=True, exist_ok=True)
        note = destination / "failure_analysis.md"
        smoke_note = "**Smoke test only; not a research result.** " if report.smoke_test else ""
        note.write_text(
            "# Failure analysis\n\n"
            + smoke_note
            + "No false positives or false negatives were found in this run. "
            "This is a result for the current evaluation split only.\n",
            encoding="utf-8",
        )
        return (note,)

    try:
        samples = discover_evaluation_images(root, report.category)
        sample_by_path = {
            sample.image_path.resolve().relative_to(root.resolve()).as_posix(): sample
            for sample in samples
        }
        split = split_normal_images(
            root,
            report.category,
            calibration_fraction=calibration_fraction,
            seed=calibration_seed,
        )
        reference_model = fit_per_pixel_reference(_loaded_images(split.memory_bank_images))
        reference_pixel_threshold = calibrate_reference_threshold(
            reference_model,
            _loaded_images(split.calibration_images),
            percentile=pixel_percentile,
            rule=pixel_rule,
        )
        patch_scorer = artifact.build_scorer(feature_extractor)
    except (DatasetError, ImageValidationError, ValueError, RuntimeError) as exc:
        raise EvaluationError(f"Unable to prepare failure example generation: {exc}") from exc

    actual_paths = {row.image_path: row for row in report.per_image}
    generated: list[Path] = []
    markdown = [
        "# Failure analysis",
        "",
        "**Smoke check, not a research result.**" if report.smoke_test else "Full official test split.",
        "",
        "These figures are ranked by distance past the fixed image-level threshold. "
        "Review each image before recording a visual explanation; no cause is inferred automatically.",
        "",
        "| Method | Case | Rank | Image | Score | Threshold | Plausible visual reason (human review) |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for relative_path, cases in selected_by_path.items():
        sample = sample_by_path.get(relative_path)
        if sample is None:
            raise EvaluationError(f"Selected failure image is missing from the test split: {relative_path}")
        try:
            image = load_image(sample.image_path)
            results = {
                "patch_detector": patch_scorer.run(
                    image, pixel_threshold=artifact.thresholds.pixel_threshold
                ),
                "rule_based": run_rule_based_baseline(
                    image,
                    blur_kernel_size=blur_kernel_size,
                    local_blur_kernel_size=local_blur_kernel_size,
                    min_region_area=min_region_area,
                ),
                "reference": run_reference_baseline(
                    image,
                    reference_model,
                    pixel_threshold=reference_pixel_threshold,
                    min_region_area=min_region_area,
                ),
            }
            if not sample.is_anomaly:
                truth = np.zeros(image.shape[:2], dtype=np.uint8)
            elif sample.mask_available and sample.expected_mask_path is not None:
                truth = (to_grayscale(load_image(sample.expected_mask_path)) > 0).astype(np.uint8) * 255
            else:
                truth = None
        except (ImageValidationError, ValueError, RuntimeError) as exc:
            raise EvaluationError(f"Unable to render failure example {relative_path}: {exc}") from exc

        for method, error_type, rank, margin in cases:
            report_row = actual_paths[relative_path]
            score_field = _METHOD_FIELDS[method][1]
            threshold_field = _METHOD_FIELDS[method][2]
            score = float(getattr(report_row, score_field))
            threshold = float(getattr(report_row, threshold_field))
            result = results[method]
            score_map = result.raw_map if method == "patch_detector" else result.score_map
            predicted_mask = result.mask
            rgb = to_rgb(image)
            scaled = score_map.astype(np.float32)
            low, high = float(scaled.min()), float(scaled.max())
            heat = np.zeros_like(scaled, dtype=np.uint8)
            if high > low:
                heat = np.rint((scaled - low) / (high - low) * 255.0).astype(np.uint8)
            color_heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            color_heat_rgb = cv2.cvtColor(color_heat, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(rgb, 0.58, color_heat_rgb, 0.42, 0)
            figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
            panels = (
                (rgb, "Input image", None),
                (heat, "Processed score map", "gray"),
                (color_heat_rgb, "Heatmap", None),
                (overlay, "Heatmap overlay", None),
                (predicted_mask, "Predicted mask", "gray"),
                (truth, "Ground truth mask", "gray"),
            )
            for axis, (pixels, title, cmap) in zip(axes.flat, panels):
                axis.set_title(title)
                axis.axis("off")
                if pixels is None:
                    axis.text(0.5, 0.5, "Ground-truth mask unavailable", ha="center", va="center", transform=axis.transAxes)
                else:
                    axis.imshow(pixels, cmap=cmap)
            figure.suptitle(
                f"{'SMOKE TEST — NOT A RESEARCH RESULT — ' if report.smoke_test else ''}"
                f"{method}: {error_type.replace('_', ' ')} #{rank} | score={score:.4g}, "
                f"threshold={threshold:.4g}, margin={margin:.4g}"
            )
            base = Path(relative_path).stem
            output = destination / method / error_type / f"{rank:02d}_{base}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=150)
            plt.close(figure)
            generated.append(output)
            markdown.append(
                f"| {method} | {error_type.replace('_', ' ')} | {rank} | `{relative_path}` | "
                f"{score:.6g} | {threshold:.6g} | Pending human visual review. |"
            )

    note = destination / "failure_analysis.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    generated.append(note)
    return tuple(generated)
