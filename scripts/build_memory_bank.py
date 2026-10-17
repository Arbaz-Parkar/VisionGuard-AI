"""Build and calibrate a normal-only PatchCore-inspired detector artifact."""

from __future__ import annotations

import argparse
import json
import sys

from visionguard.config import ConfigError, load_config, resolve_project_path
from visionguard.data import DatasetError, split_normal_images
from visionguard.model import (
    ArtifactError,
    ModelSetupError,
    PatchAnomalyScorer,
    PatchDetectorArtifact,
    ResNet18PatchFeatureExtractor,
    build_memory_bank,
    calibrate_patch_thresholds,
    save_patch_artifact,
)


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
        "--output",
        help="Artifact path (defaults to models/<category>_patch_detector.npz).",
    )
    args = parser.parse_args()

    try:
        config = load_config()
        category = args.category or config["dataset"]["category"]
        model_config = config["model"]
        if model_config["architecture"] != "resnet18" or tuple(
            model_config["feature_layers"]
        ) != ("layer2", "layer3"):
            raise ConfigError(
                "The current detector builder supports ResNet18 with layer2 and layer3 features."
            )

        dataset_root = resolve_project_path(args.dataset_root or config["dataset"]["root"])
        artifact_path = (
            resolve_project_path(args.output)
            if args.output
            else resolve_project_path(config["paths"]["artifacts"])
            / f"{category}_patch_detector.npz"
        )
        split = split_normal_images(
            dataset_root,
            category,
            calibration_fraction=config["dataset"]["calibration_fraction"],
            seed=config["dataset"]["calibration_seed"],
        )
        feature_extractor = ResNet18PatchFeatureExtractor(
            device_preference=model_config["device"]
        )
        memory_bank = build_memory_bank(
            split,
            feature_extractor,
            input_size=tuple(config["preprocessing"]["input_size"]),
            preserve_aspect_ratio=config["preprocessing"]["preserve_aspect_ratio"],
            batch_size=model_config["batch_size"],
            sampling_strategy=model_config["sampling_strategy"],
            sampling_ratio=model_config["sampling_ratio"],
            max_patches=model_config["max_memory_bank_patches"],
            seed=model_config["random_seed"],
        )
        scorer = PatchAnomalyScorer(
            feature_extractor,
            memory_bank,
            input_size=memory_bank.input_size,
            preserve_aspect_ratio=memory_bank.preserve_aspect_ratio,
            smoothing_sigma=model_config["smoothing_sigma"],
            aggregation=model_config["aggregation"],
            score_percentile=model_config["score_percentile"],
        )
        thresholds = calibrate_patch_thresholds(
            split,
            scorer,
            pixel_false_positive_rate=(
                1.0 - config["thresholds"]["pixel_percentile"] / 100.0
            ),
            image_false_positive_rate=(
                1.0 - config["thresholds"]["image_percentile"] / 100.0
            ),
        )
        artifact = PatchDetectorArtifact.from_components(
            category=category,
            memory_bank=memory_bank,
            thresholds=thresholds,
            scorer=scorer,
        )
        save_patch_artifact(artifact_path, artifact)
    except (
        ConfigError,
        DatasetError,
        ArtifactError,
        ModelSetupError,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"Detector build failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "category": category,
                "artifact": str(artifact_path),
                "device": str(feature_extractor.device),
                "memory_bank_source_images": memory_bank.source_image_count,
                "calibration_images": len(split.calibration_images),
                "candidate_patches": memory_bank.candidate_patch_count,
                "stored_patches": int(memory_bank.embeddings.shape[0]),
                "sampling_strategy": memory_bank.sampling_strategy,
                "sampling_ratio": memory_bank.sampling_ratio,
                "thresholds": {
                    "image_value": thresholds.image_threshold,
                    "image_percentile": thresholds.image_percentile,
                    "image_false_positive_rate_target": (
                        thresholds.image_false_positive_rate
                    ),
                    "pixel_value": thresholds.pixel_threshold,
                    "pixel_percentile": thresholds.pixel_percentile,
                    "pixel_false_positive_rate_target": (
                        thresholds.pixel_false_positive_rate
                    ),
                    "calibration_image_count": thresholds.calibration_image_count,
                    "calibration_pixel_count": thresholds.calibration_pixel_count,
                },
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
