"""Evaluate a saved patch detector on the configured MVTec test split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys

from visionguard.config import ConfigError, load_config, resolve_project_path
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
        "--artifact",
        help="Patch-detector .npz artifact (defaults to models/<configured-category>_patch_detector.npz).",
    )
    parser.add_argument(
        "--output-dir",
        help="Report directory (defaults to outputs/<artifact-category>).",
    )
    args = parser.parse_args()

    try:
        config = load_config()
        dataset_root = resolve_project_path(args.dataset_root or config["dataset"]["root"])
        default_artifact = (
            resolve_project_path(config["paths"]["artifacts"])
            / f"{config['dataset']['category']}_patch_detector.npz"
        )
        artifact_path = (
            resolve_project_path(args.artifact) if args.artifact else default_artifact
        )
        artifact = load_patch_artifact(artifact_path)
        output_dir = (
            resolve_project_path(args.output_dir)
            if args.output_dir
            else resolve_project_path(config["paths"]["outputs"]) / artifact.category
        )
        feature_extractor = ResNet18PatchFeatureExtractor(
            device_preference=config["model"]["device"]
        )
        report = evaluate_patch_detector(
            dataset_root,
            artifact,
            feature_extractor,
            output_dir=output_dir,
        )
    except (ConfigError, ArtifactError, EvaluationError, ModelSetupError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
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
                "image_metrics": asdict(report.image_metrics),
                "pixel_metrics": asdict(report.pixel_metrics),
                "metrics_json": str(output_dir / "metrics.json"),
                "per_image_csv": str(output_dir / "per_image.csv"),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
