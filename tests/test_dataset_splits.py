from pathlib import Path

import pytest

from visionguard.data import (
    DatasetError,
    discover_evaluation_images,
    discover_normal_images,
    split_normal_images,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic layout fixture")


def test_evaluation_discovery_keeps_unmatched_mask_explicit(tmp_path: Path) -> None:
    category = tmp_path / "bottle"
    _touch(category / "test" / "good" / "normal.png")
    _touch(category / "test" / "broken_large" / "defect.png")

    samples = discover_evaluation_images(tmp_path, "bottle")

    assert len(samples) == 2
    normal = next(sample for sample in samples if not sample.is_anomaly)
    anomaly = next(sample for sample in samples if sample.is_anomaly)
    assert normal.defect_type == "good"
    assert normal.expected_mask_path is None
    assert anomaly.expected_mask_path == (
        category / "ground_truth" / "broken_large" / "defect_mask.png"
    )
    assert not anomaly.mask_available


def test_calibration_split_is_seeded_disjoint_and_train_good_only(tmp_path: Path) -> None:
    category = tmp_path / "bottle"
    for index in range(10):
        _touch(category / "train" / "good" / f"normal_{index:02}.png")
    _touch(category / "test" / "good" / "held_out_normal.png")
    _touch(category / "test" / "crack" / "test_defect.png")
    normal_images = set(discover_normal_images(tmp_path, "bottle"))

    first = split_normal_images(tmp_path, "bottle", calibration_fraction=0.2, seed=7)
    second = split_normal_images(tmp_path, "bottle", calibration_fraction=0.2, seed=7)
    bank = set(first.memory_bank_images)
    calibration = set(first.calibration_images)

    assert first == second
    assert len(bank) == 8
    assert len(calibration) == 2
    assert bank.isdisjoint(calibration)
    assert bank | calibration == normal_images
    assert all("train" in path.parts and "good" in path.parts for path in bank | calibration)


@pytest.mark.parametrize(
    "fraction, seed",
    [(0, 42), (1, 42), (0.2, -1)],
)
def test_calibration_split_rejects_invalid_settings(
    tmp_path: Path, fraction: float, seed: int
) -> None:
    with pytest.raises(DatasetError):
        split_normal_images(tmp_path, "bottle", calibration_fraction=fraction, seed=seed)


def test_calibration_split_requires_two_normal_images(tmp_path: Path) -> None:
    _touch(tmp_path / "bottle" / "train" / "good" / "only.png")

    with pytest.raises(DatasetError, match="At least two"):
        split_normal_images(tmp_path, "bottle")
