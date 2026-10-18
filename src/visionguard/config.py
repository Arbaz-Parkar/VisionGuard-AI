"""Load and validate the project's YAML configuration."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when project configuration is missing or invalid."""


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_config(config: Any) -> dict[str, Any]:
    """Validate project, dataset, model, threshold, and output settings."""
    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a YAML mapping.")

    for section in (
        "project",
        "dataset",
        "preprocessing",
        "model",
        "thresholds",
        "classical",
        "evaluation",
        "paths",
        "logging",
    ):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"Configuration section '{section}' must be a mapping.")

    project = config["project"]
    project_fields_valid = all(
        isinstance(project.get(key), str) and project[key].strip()
        for key in ("name", "version")
    )
    if not project_fields_valid:
        raise ConfigError("project.name and project.version must be non-empty strings.")

    dataset = config["dataset"]
    if not isinstance(dataset.get("root"), str) or not dataset["root"].strip():
        raise ConfigError("dataset.root must be a non-empty path string.")
    if not isinstance(dataset.get("category"), str) or not dataset["category"].strip():
        raise ConfigError("dataset.category must be a non-empty string.")
    if dataset.get("normal_class") != "good":
        raise ConfigError("dataset.normal_class must be 'good' for the MVTec AD layout.")
    calibration_fraction = dataset.get("calibration_fraction")
    if (
        isinstance(calibration_fraction, bool)
        or not isinstance(calibration_fraction, (int, float))
        or not math.isfinite(calibration_fraction)
        or not 0 < calibration_fraction < 1
    ):
        raise ConfigError("dataset.calibration_fraction must be greater than 0 and less than 1.")
    calibration_seed = dataset.get("calibration_seed")
    if (
        isinstance(calibration_seed, bool)
        or not isinstance(calibration_seed, int)
        or calibration_seed < 0
    ):
        raise ConfigError("dataset.calibration_seed must be a non-negative integer.")

    input_size = config["preprocessing"].get("input_size")
    if (not isinstance(input_size, list) or len(input_size) != 2
            or any(type(value) is not int or value <= 0 for value in input_size)):
        raise ConfigError("preprocessing.input_size must contain two positive integers.")
    if not isinstance(config["preprocessing"].get("preserve_aspect_ratio"), bool):
        raise ConfigError("preprocessing.preserve_aspect_ratio must be a boolean.")

    model = config["model"]
    if not isinstance(model.get("architecture"), str) or not model["architecture"].strip():
        raise ConfigError("model.architecture must be a non-empty string.")
    feature_layers = model.get("feature_layers")
    if (
        not isinstance(feature_layers, list)
        or not feature_layers
        or any(
            not isinstance(layer, str)
            or layer not in {"layer1", "layer2", "layer3", "layer4"}
            for layer in feature_layers
        )
        or len(set(feature_layers)) != len(feature_layers)
    ):
        raise ConfigError("model.feature_layers must be a non-empty unique list of ResNet stages.")
    if model.get("device") not in {"auto", "cpu", "cuda"}:
        raise ConfigError("model.device must be one of: auto, cpu, cuda.")
    for key in ("batch_size", "max_memory_bank_patches"):
        value = model.get(key)
        if type(value) is not int or value <= 0:
            raise ConfigError(f"model.{key} must be a positive integer.")
    if model.get("sampling_strategy") not in {"random", "greedy"}:
        raise ConfigError("model.sampling_strategy must be 'random' or 'greedy'.")
    sampling_ratio = model.get("sampling_ratio")
    if (
        isinstance(sampling_ratio, bool)
        or not isinstance(sampling_ratio, (int, float))
        or not math.isfinite(sampling_ratio)
        or not 0 < sampling_ratio <= 1
    ):
        raise ConfigError("model.sampling_ratio must be greater than 0 and at most 1.")
    if model.get("aggregation") not in {"max", "percentile"}:
        raise ConfigError("model.aggregation must be 'max' or 'percentile'.")
    smoothing_sigma = model.get("smoothing_sigma")
    if (
        isinstance(smoothing_sigma, bool)
        or not isinstance(smoothing_sigma, (int, float))
        or not math.isfinite(smoothing_sigma)
        or smoothing_sigma < 0
    ):
        raise ConfigError("model.smoothing_sigma must be a finite non-negative number.")
    score_percentile = model.get("score_percentile")
    if (
        isinstance(score_percentile, bool)
        or not isinstance(score_percentile, (int, float))
        or not 0 < score_percentile <= 100
    ):
        raise ConfigError("model.score_percentile must be greater than 0 and at most 100.")
    random_seed = model.get("random_seed")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise ConfigError("model.random_seed must be a non-negative integer.")
    threshold = model.get("mask_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ConfigError("model.mask_threshold must be between 0 and 1.")

    thresholds = config["thresholds"]
    if thresholds.get("image_rule") not in {"mean_plus_3std", "percentile"}:
        raise ConfigError("thresholds.image_rule must be 'mean_plus_3std' or 'percentile'.")
    if thresholds.get("pixel_rule") not in {"mean_plus_3std", "percentile"}:
        raise ConfigError("thresholds.pixel_rule must be 'mean_plus_3std' or 'percentile'.")
    for key in ("image_percentile", "pixel_percentile"):
        percentile = thresholds.get(key)
        if (
            isinstance(percentile, bool)
            or not isinstance(percentile, (int, float))
            or not math.isfinite(percentile)
            or not 0 < percentile <= 100
        ):
            raise ConfigError(f"thresholds.{key} must be greater than 0 and at most 100.")

    classical = config["classical"]
    for key in ("blur_kernel_size", "local_blur_kernel_size"):
        value = classical.get(key)
        if type(value) is not int or value <= 0 or value % 2 == 0:
            raise ConfigError(f"classical.{key} must be a positive odd integer.")
    min_region_area = classical.get("min_region_area")
    if (
        isinstance(min_region_area, bool)
        or not isinstance(min_region_area, (int, float))
        or not math.isfinite(min_region_area)
        or min_region_area < 0
    ):
        raise ConfigError("classical.min_region_area must be non-negative.")

    evaluation = config["evaluation"]
    pixel_metric_size = evaluation.get("pixel_metric_size")
    if (
        not isinstance(pixel_metric_size, list)
        or len(pixel_metric_size) != 2
        or any(type(value) is not int or value <= 0 for value in pixel_metric_size)
    ):
        raise ConfigError("evaluation.pixel_metric_size must contain two positive integers.")
    warmup_images = evaluation.get("warmup_images")
    if type(warmup_images) is not int or warmup_images < 0:
        raise ConfigError("evaluation.warmup_images must be a non-negative integer.")
    for key in ("smoke_test_image_count", "sweep_candidate_pool_size", "failure_examples_per_method"):
        value = evaluation.get(key)
        if type(value) is not int or value <= 0:
            raise ConfigError(f"evaluation.{key} must be a positive integer.")
    sweep_ratios = evaluation.get("sweep_ratios")
    if (
        not isinstance(sweep_ratios, list)
        or not sweep_ratios
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 1
            for value in sweep_ratios
        )
        or len(set(sweep_ratios)) != len(sweep_ratios)
    ):
        raise ConfigError("evaluation.sweep_ratios must be unique values in (0, 1].")

    for key in ("artifacts", "outputs"):
        value = config["paths"].get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"paths.{key} must be a non-empty path string.")
    level = config["logging"].get("level")
    if not isinstance(level, str) or not isinstance(logging.getLevelName(level.upper()), int):
        raise ConfigError("logging.level must be a standard logging level name.")
    return config


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the specified YAML file, or the repository default, safely."""
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    return validate_config(config)
