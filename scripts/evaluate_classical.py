"""Evaluate both classical baselines on the configured MVTec test split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys

from visionguard.classical_evaluation import evaluate_classical_baselines
from visionguard.config import ConfigError, load_config, resolve_project_path
from visionguard.evaluation import EvaluationError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        help="MVTec AD root (defaults to the project configuration).",
    )
    parser.add_argument(
        "--category",
        help="Category name (defaults to the project configuration).",
    )
    parser.add_argument(
        "--output-dir",
        help="Report directory (defaults to outputs/classical/<category>).",
    )
    args = parser.parse_args()

    try:
        config = load_config()
        category = args.category or config["dataset"]["category"]
        dataset_root = resolve_project_path(args.dataset_root or config["dataset"]["root"])
        output_dir = (
            resolve_project_path(args.output_dir)
            if args.output_dir
            else resolve_project_path(config["paths"]["outputs"])
            / "classical"
            / category
        )
        report = evaluate_classical_baselines(
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
            output_dir=output_dir,
        )
    except (ConfigError, EvaluationError) as exc:
        print(f"Classical evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "category": report.category,
                "evaluated_image_count": report.evaluated_image_count,
                "normal_image_count": report.normal_image_count,
                "anomalous_image_count": report.anomalous_image_count,
                "normal_training_image_count": report.normal_training_image_count,
                "calibration_image_count": report.calibration_image_count,
                "missing_mask_count": report.missing_mask_count,
                "rule_based": {
                    "image_metrics": asdict(report.rule_based.image_metrics),
                    "pixel_metrics": asdict(report.rule_based.pixel_metrics),
                    "image_threshold": report.rule_based.image_threshold,
                    "image_threshold_method": report.rule_based.image_threshold_method,
                    "pixel_threshold_method": report.rule_based.pixel_threshold_method,
                },
                "reference": {
                    "image_metrics": asdict(report.reference.image_metrics),
                    "pixel_metrics": asdict(report.reference.pixel_metrics),
                    "image_threshold": report.reference.image_threshold,
                    "image_threshold_method": report.reference.image_threshold_method,
                    "pixel_threshold": report.reference.pixel_threshold,
                    "pixel_threshold_method": report.reference.pixel_threshold_method,
                },
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
