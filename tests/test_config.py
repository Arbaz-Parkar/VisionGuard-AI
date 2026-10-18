from pathlib import Path

import pytest
import yaml

from visionguard.config import ConfigError, DEFAULT_CONFIG_PATH, load_config


def test_default_configuration_loads() -> None:
    config = load_config()
    assert config["dataset"]["category"] == "bottle"
    assert config["model"]["architecture"] == "resnet18"
    assert config["dataset"]["calibration_fraction"] == 0.15
    assert config["thresholds"]["image_rule"] == "mean_plus_3std"
    assert config["evaluation"]["pixel_metric_size"] == [256, 256]


def test_invalid_mask_threshold_is_rejected(tmp_path: Path) -> None:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["model"]["mask_threshold"] = 1.5
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="mask_threshold"):
        load_config(invalid_config)


@pytest.mark.parametrize(
    "section, key, value, message",
    [
        ("dataset", "calibration_fraction", 1.0, "calibration_fraction"),
        ("model", "feature_layers", ["layer2", "layer2"], "feature_layers"),
        ("model", "sampling_ratio", 0.0, "sampling_ratio"),
        ("thresholds", "pixel_rule", "test_labels", "pixel_rule"),
        ("classical", "local_blur_kernel_size", 4, "local_blur_kernel_size"),
        ("evaluation", "sweep_ratios", [0.1, 0.1], "sweep_ratios"),
    ],
)
def test_invalid_experiment_settings_are_rejected(
    tmp_path: Path, section: str, key: str, value: object, message: str
) -> None:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config[section][key] = value
    invalid_config = tmp_path / f"invalid_{key}.yaml"
    invalid_config.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(invalid_config)
