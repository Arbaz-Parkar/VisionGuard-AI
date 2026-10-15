from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from visionguard.data import CalibrationSplit, DatasetError
from visionguard.model import (
    MemoryBank,
    PatchAnomalyScorer,
    calibrate_patch_thresholds,
)


class PixelFeatureExtractor:
    device = torch.device("cpu")

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return images.permute(0, 2, 3, 1).contiguous()


def _scorer() -> PatchAnomalyScorer:
    bank = MemoryBank(
        embeddings=torch.zeros((1, 3), dtype=torch.float32),
        source_image_count=1,
        candidate_patch_count=1,
        sampling_strategy="random",
        sampling_ratio=1.0,
        seed=42,
        input_size=(8, 8),
        preserve_aspect_ratio=False,
    )
    return PatchAnomalyScorer(
        PixelFeatureExtractor(),
        bank,
        input_size=(8, 8),
        preserve_aspect_ratio=False,
        smoothing_sigma=0,
    )


def _save_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)


def test_thresholds_use_only_disjoint_held_out_normal_calibration_images(tmp_path: Path):
    memory_path = tmp_path / "bottle" / "train" / "good" / "bank_invalid.png"
    memory_path.parent.mkdir(parents=True)
    memory_path.write_bytes(b"must not be opened during calibration")
    calibration_paths = tuple(
        tmp_path / "bottle" / "train" / "good" / f"calibration_{value}.png"
        for value in (0, 64, 128)
    )
    for path, value in zip(calibration_paths, (0, 64, 128)):
        _save_rgb(path, value)
    split = CalibrationSplit(
        memory_bank_images=(memory_path,), calibration_images=calibration_paths
    )

    calibration = calibrate_patch_thresholds(split, _scorer())

    scores = np.sqrt(3.0) * np.array([0, 64, 128], dtype=np.float64) / 255.0
    pixels = np.repeat(scores, 8 * 8)
    assert calibration.pixel_threshold == pytest.approx(np.percentile(pixels, 99.5))
    assert calibration.image_threshold == pytest.approx(np.percentile(scores, 95.0))
    assert calibration.pixel_false_positive_rate == 0.005
    assert calibration.image_false_positive_rate == 0.05
    assert calibration.pixel_percentile == 99.5
    assert calibration.image_percentile == 95.0
    assert calibration.calibration_image_count == 3
    assert calibration.calibration_pixel_count == 3 * 8 * 8


def test_scorer_can_return_continuous_scores_without_a_threshold():
    image = np.full((8, 8, 3), 64, dtype=np.uint8)

    result = _scorer().run(image)

    assert result.pixel_threshold is None
    assert result.image_score > 0
    assert np.isfinite(result.raw_map).all()
    assert not result.mask.any()
    assert result.regions == ()


def test_calibration_rejects_overlapping_and_non_train_good_splits(tmp_path: Path):
    shared_path = tmp_path / "bottle" / "train" / "good" / "shared.png"
    overlap = CalibrationSplit(
        memory_bank_images=(shared_path,), calibration_images=(shared_path,)
    )
    with pytest.raises(DatasetError, match="disjoint"):
        calibrate_patch_thresholds(overlap, _scorer())

    test_path = tmp_path / "bottle" / "test" / "good" / "sample.png"
    invalid = CalibrationSplit(
        memory_bank_images=(shared_path,), calibration_images=(test_path,)
    )
    with pytest.raises(DatasetError, match="train/good"):
        calibrate_patch_thresholds(invalid, _scorer())


@pytest.mark.parametrize(
    "rate", [-0.1, 1.0, float("nan"), float("inf")]
)
def test_calibration_rejects_invalid_false_positive_rates(tmp_path: Path, rate: float):
    memory_path = tmp_path / "bottle" / "train" / "good" / "bank.png"
    calibration_path = tmp_path / "bottle" / "train" / "good" / "calibration.png"
    _save_rgb(memory_path, 0)
    _save_rgb(calibration_path, 10)
    split = CalibrationSplit(
        memory_bank_images=(memory_path,), calibration_images=(calibration_path,)
    )

    with pytest.raises(DatasetError, match="false_positive_rate"):
        calibrate_patch_thresholds(
            split, _scorer(), pixel_false_positive_rate=rate
        )
