"""Compare the PatchCore-inspired detector and both classical baselines."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys

from visionguard.classical_evaluation import evaluate_classical_baselines
from visionguard.comparison import combine_evaluation_reports, save_comparison_report
from visionguard.config import ConfigError, load_config, resolve_project_path
from visionguard.data import DatasetError, split_normal_images
from visionguard.evaluation import EvaluationError, evaluate_patch_detector
from visionguard.model import (
    ArtifactError,
    ModelSetupError,
    ResNet18PatchFeatureExtractor,
    load_patch_artifact,
)


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
    args = parser.parse_args()

    try:
        config = load_config()
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

        feature_extractor = ResNet18PatchFeatureExtractor(
            device_preference=config["model"]["device"]
        )
        patch_report = evaluate_patch_detector(
            dataset_root,
            artifact,
            feature_extractor,
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
        )
        json_path, csv_path = save_comparison_report(report, output_dir)
    except (
        ConfigError,
        DatasetError,
        ArtifactError,
        EvaluationError,
        ModelSetupError,
        OSError,
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
                    }
                    for name, method in report.methods.items()
                },
                "metrics_json": str(json_path),
                "per_image_csv": str(csv_path),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
