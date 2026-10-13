from pathlib import Path

import pytest
import yaml

from visionguard.config import ConfigError, DEFAULT_CONFIG_PATH, load_config


def test_default_configuration_loads() -> None:
    config = load_config()
    assert config["dataset"]["category"] == "bottle"
    assert config["model"]["architecture"] == "resnet18"


def test_invalid_mask_threshold_is_rejected(tmp_path: Path) -> None:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["model"]["mask_threshold"] = 1.5
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="mask_threshold"):
        load_config(invalid_config)
