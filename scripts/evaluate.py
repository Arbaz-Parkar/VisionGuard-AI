"""Compare the PatchCore-inspired detector and both classical baselines."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys

from visionguard.analysis_outputs import (
    build_experiment_metadata,
    save_comparison_plots,
    save_experiment_metadata,
    save_failure_examples,
    save_sweep_report,
)
from visionguard.classical_evaluation import evaluate_classical_baselines
from visionguard.comparison import combine_evaluation_reports, save_comparison_report
from visionguard.config import ConfigError, load_config, resolve_project_path
from visionguard.data import DatasetError, split_normal_images
from visionguard.evaluation import EvaluationError, evaluate_patch_detector
from visionguard.model import (
    ArtifactError,
    ModelSetupError,
    ResNet18PatchFeatureExtractor,
    extract_patch_embeddings,
    load_patch_artifact,
)
from visionguard.sweep import run_memory_bank_sweep, validate_sweep_ratios


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        help="MVTec AD root (defaults to the project configuration).",
    )
    parser.add_argument(
        "--category",
        help="Category to compare (defaults to the loaded artifact category).",
    )
    parser.add_argument(
        "--artifact",
        help="Patch-detector artifact (defaults to models/<category>_patch_detector.npz).",
    )
    parser.add_argument(
        "--output-dir",
        help="Comparison report directory (defaults to outputs/metrics/<category>).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a clearly labeled smoke check on a few test images; never treat it as a result.",
    )
    parser.add_argument("--smoke-images", type=int)
    parser.add_argument("--warmup-images", type=int)
    parser.add_argument("--pixel-metric-size", type=int)
    parser.add_argument(
        "--sweep-ratios",
        type=float,
        nargs="+",
    )
    parser.add_argument("--sweep-pool-size", type=int)
    parser.add_argument("--failure-examples", type=int)
    args = parser.parse_args()

    try:
        config = load_config()
        evaluation_config = config["evaluation"]
        args.smoke_images = (
            evaluation_config["smoke_test_image_count"]
            if args.smoke_images is None
            else args.smoke_images
        )
        args.warmup_images = (
            evaluation_config["warmup_images"]
            if args.warmup_images is None
            else args.warmup_images
        )
        pixel_metric_size = (
            tuple(evaluation_config["pixel_metric_size"])
            if args.pixel_metric_size is None
            else (args.pixel_metric_size, args.pixel_metric_size)
        )
        args.sweep_ratios = (
            evaluation_config["sweep_ratios"]
            if args.sweep_ratios is None
            else args.sweep_ratios
        )
        args.sweep_pool_size = (
            evaluation_config["sweep_candidate_pool_size"]
            if args.sweep_pool_size is None
            else args.sweep_pool_size
        )
        args.failure_examples = (
            evaluation_config["failure_examples_per_method"]
            if args.failure_examples is None
            else args.failure_examples
        )
        if args.smoke_images <= 0:
            raise EvaluationError("--smoke-images must be positive.")
        if args.warmup_images < 0:
            raise EvaluationError("--warmup-images must be non-negative.")
        if any(size <= 0 for size in pixel_metric_size):
            raise EvaluationError("The pixel metric size must be positive.")
        if args.failure_examples <= 0:
            raise EvaluationError("--failure-examples must be positive.")
        if args.sweep_pool_size <= 0:
            raise EvaluationError("--sweep-pool-size must be positive.")
        sweep_ratios = validate_sweep_ratios(args.sweep_ratios)
        configured_category = args.category or config["dataset"]["category"]
        artifact_path = (
            resolve_project_path(args.artifact)
            if args.artifact
            else resolve_project_path(config["paths"]["artifacts"])
            / f"{configured_category}_patch_detector.npz"
        )
        artifact = load_patch_artifact(artifact_path)
        if args.category and artifact.category != args.category:
            raise EvaluationError(
                f"Artifact category '{artifact.category}' does not match requested "
                f"category '{args.category}'."
            )
        category = artifact.category
        dataset_root = resolve_project_path(args.dataset_root or config["dataset"]["root"])
        normal_split = split_normal_images(
            dataset_root,
            category,
            calibration_fraction=config["dataset"]["calibration_fraction"],
            seed=config["dataset"]["calibration_seed"],
        )
        if (
            artifact.memory_bank.source_image_count != len(normal_split.memory_bank_images)
            or artifact.thresholds.calibration_image_count
            != len(normal_split.calibration_images)
        ):
            raise EvaluationError(
                "Artifact and current configuration use different normal split sizes. "
                "Rebuild the artifact with scripts/build_memory_bank.py before comparing."
            )
        output_dir = (
            resolve_project_path(args.output_dir)
            if args.output_dir
            else resolve_project_path(config["paths"]["outputs"])
            / "metrics"
            / category
        )
        if args.smoke:
            output_dir = output_dir / "smoke_test"

        feature_extractor = ResNet18PatchFeatureExtractor(
            device_preference=config["model"]["device"]
        )
        sample_limit = args.smoke_images if args.smoke else None
        print(
            "Running method comparison"
            + (
                f" in smoke mode ({sample_limit} test images)."
                if args.smoke
                else " on the full test split."
            ),
            flush=True,
        )
        patch_report = evaluate_patch_detector(
            dataset_root,
            artifact,
            feature_extractor,
            pixel_metric_size=pixel_metric_size,
            warmup_count=args.warmup_images,
            sample_limit=sample_limit,
        )
        classical_report = evaluate_classical_baselines(
            dataset_root,
            category,
            calibration_fraction=config["dataset"]["calibration_fraction"],
            calibration_seed=config["dataset"]["calibration_seed"],
            image_rule=config["thresholds"]["image_rule"],
            pixel_rule=config["thresholds"]["pixel_rule"],
            image_percentile=config["thresholds"]["image_percentile"],
            pixel_percentile=config["thresholds"]["pixel_percentile"],
            blur_kernel_size=config["classical"]["blur_kernel_size"],
            local_blur_kernel_size=config["classical"]["local_blur_kernel_size"],
            min_region_area=config["classical"]["min_region_area"],
            pixel_metric_size=pixel_metric_size,
            warmup_count=args.warmup_images,
            sample_limit=sample_limit,
        )
        if (
            len(normal_split.memory_bank_images)
            != classical_report.normal_training_image_count
            or len(normal_split.calibration_images)
            != classical_report.calibration_image_count
        ):
            raise EvaluationError("Normal split changed during the evaluation run.")

        report = combine_evaluation_reports(
            patch_report,
            classical_report,
            artifact,
            patch_device=str(feature_extractor.device),
            smoke_test=args.smoke,
        )
        print("Extracting the normal-only patch pool for the memory-bank sweep.", flush=True)
        candidate_embeddings = extract_patch_embeddings(
            normal_split,
            feature_extractor,
            input_size=tuple(config["preprocessing"]["input_size"]),
            preserve_aspect_ratio=config["preprocessing"]["preserve_aspect_ratio"],
            batch_size=config["model"]["batch_size"],
        )
        sweep_points = run_memory_bank_sweep(
            dataset_root,
            artifact,
            feature_extractor,
            candidate_embeddings,
            ratios=sweep_ratios,
            seed=config["model"]["random_seed"],
            warmup_count=args.warmup_images,
            sample_limit=sample_limit,
            candidate_pool_limit=args.sweep_pool_size,
            progress_callback=lambda message: print(message, flush=True),
        )
        json_path, csv_path = save_comparison_report(report, output_dir)
        output_root = resolve_project_path(config["paths"]["outputs"])
        sweep_json_path, sweep_csv_path = save_sweep_report(sweep_points, output_dir)
        plot_paths = save_comparison_plots(
            report,
            sweep_points,
            output_root / "plots" / category / ("smoke_test" if args.smoke else "research"),
        )
        example_paths = save_failure_examples(
            dataset_root,
            report,
            artifact,
            feature_extractor,
            calibration_fraction=config["dataset"]["calibration_fraction"],
            calibration_seed=config["dataset"]["calibration_seed"],
            pixel_rule=config["thresholds"]["pixel_rule"],
            pixel_percentile=config["thresholds"]["pixel_percentile"],
            blur_kernel_size=config["classical"]["blur_kernel_size"],
            local_blur_kernel_size=config["classical"]["local_blur_kernel_size"],
            min_region_area=config["classical"]["min_region_area"],
            output_dir=output_root / "examples" / category / ("smoke_test" if args.smoke else "research"),
            per_method_limit=args.failure_examples,
        )
        metadata = build_experiment_metadata(
            config=config,
            report=report,
            artifact=artifact,
            dataset_root=dataset_root,
            sweep_points=sweep_points,
        )
        metadata["warmup_image_count_requested"] = args.warmup_images
        metadata["sweep_warmup_image_count"] = min(
            args.warmup_images, report.evaluated_image_count
        )
        metadata["sweep_candidate_pool_limit"] = args.sweep_pool_size
        metadata["failure_examples_per_method"] = args.failure_examples
        metadata["sweep_ratios_requested"] = list(sweep_ratios)
        metadata_path = save_experiment_metadata(metadata, output_dir)
    except (
        ConfigError,
        DatasetError,
        ArtifactError,
        EvaluationError,
        ModelSetupError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Three-method evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "category": report.category,
                "evaluated_image_count": report.evaluated_image_count,
                "normal_image_count": report.normal_image_count,
                "anomalous_image_count": report.anomalous_image_count,
                "missing_mask_count": report.missing_mask_count,
                "pixel_metric_image_count": report.pixel_metric_image_count,
                "methods": {
                    name: {
                        "image_metrics": asdict(method.image_metrics),
                        "pixel_metrics": asdict(method.pixel_metrics),
                        "image_threshold": method.image_threshold,
                        "image_threshold_method": method.image_threshold_method,
                        "pixel_threshold": method.pixel_threshold,
                        "pixel_threshold_method": method.pixel_threshold_method,
                        "mean_inference_ms": method.mean_inference_ms,
                        "median_inference_ms": method.median_inference_ms,
                        "mean_preprocessing_ms": method.mean_preprocessing_ms,
                        "median_preprocessing_ms": method.median_preprocessing_ms,
                        "mean_model_inference_ms": method.mean_model_inference_ms,
                        "median_model_inference_ms": method.median_model_inference_ms,
                        "mean_total_ms": method.mean_total_ms,
                        "median_total_ms": method.median_total_ms,
                    }
                    for name, method in report.methods.items()
                },
                "metrics_json": str(json_path),
                "per_image_csv": str(csv_path),
                "memory_bank_sweep_json": str(sweep_json_path),
                "memory_bank_sweep_csv": str(sweep_csv_path),
                "plots": [str(path) for path in plot_paths],
                "examples": [str(path) for path in example_paths],
                "experiment_metadata": str(metadata_path),
                "smoke_test": report.smoke_test,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
