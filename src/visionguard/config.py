"""Load and validate the project's YAML configuration."""

from __future__ import annotations

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
    """Validate required sections and frequently changed settings."""
    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a YAML mapping.")

    for section in ("dataset", "preprocessing", "model", "paths", "logging"):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"Configuration section '{section}' must be a mapping.")

    dataset = config["dataset"]
    if not isinstance(dataset.get("root"), str) or not dataset["root"].strip():
        raise ConfigError("dataset.root must be a non-empty path string.")
    if not isinstance(dataset.get("category"), str) or not dataset["category"].strip():
        raise ConfigError("dataset.category must be a non-empty string.")

    input_size = config["preprocessing"].get("input_size")
    if (not isinstance(input_size, list) or len(input_size) != 2
            or any(not isinstance(value, int) or value <= 0 for value in input_size)):
        raise ConfigError("preprocessing.input_size must contain two positive integers.")

    model = config["model"]
    if model.get("device") not in {"auto", "cpu", "cuda"}:
        raise ConfigError("model.device must be one of: auto, cpu, cuda.")
    for key in ("batch_size", "max_memory_bank_patches"):
        value = model.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(f"model.{key} must be a positive integer.")
    threshold = model.get("mask_threshold")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ConfigError("model.mask_threshold must be between 0 and 1.")

    if not isinstance(config["paths"].get("artifacts"), str):
        raise ConfigError("paths.artifacts must be a path string.")
    if not isinstance(config["paths"].get("outputs"), str):
        raise ConfigError("paths.outputs must be a path string.")
    if not isinstance(config["logging"].get("level"), str):
        raise ConfigError("logging.level must be a string.")
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
