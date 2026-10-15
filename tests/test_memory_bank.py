from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

from visionguard.data import CalibrationSplit, DatasetError
from visionguard.model import build_memory_bank, sample_patch_embeddings


class TinyFeatureExtractor:
    device = torch.device("cpu")

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(images, kernel_size=4, stride=4)
        return pooled.permute(0, 2, 3, 1).contiguous()


def _save_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((8, 8, 3), value, dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def test_random_patch_sampling_is_seeded_unique_and_capped():
    patches = torch.arange(200, dtype=torch.float32).reshape(100, 2)

    first = sample_patch_embeddings(
        patches, sampling_ratio=0.5, max_patches=9, seed=17
    )
    second = sample_patch_embeddings(
        patches, sampling_ratio=0.5, max_patches=9, seed=17
    )

    assert first.shape == (9, 2)
    assert torch.equal(first, second)
    assert torch.unique(first, dim=0).shape[0] == 9


def test_greedy_coreset_sampling_is_seeded_and_selects_distinct_patches():
    patches = torch.tensor(
        [[0, 0], [0.1, 0], [1, 1], [0.9, 1], [0.5, 0.5], [10, 10]],
        dtype=torch.float32,
    )

    first = sample_patch_embeddings(
        patches, sampling_strategy="greedy", sampling_ratio=0.5, seed=3
    )
    second = sample_patch_embeddings(
        patches, sampling_strategy="greedy", sampling_ratio=0.5, seed=3
    )

    assert first.shape == (3, 2)
    assert torch.equal(first, second)
    assert torch.unique(first, dim=0).shape[0] == 3


def test_builder_uses_only_disjoint_train_good_memory_bank_paths(tmp_path: Path):
    first_path = tmp_path / "bottle" / "train" / "good" / "normal_a.png"
    second_path = tmp_path / "bottle" / "train" / "good" / "normal_b.png"
    calibration_path = tmp_path / "bottle" / "train" / "good" / "calibration.png"
    _save_rgb(first_path, 20)
    _save_rgb(second_path, 40)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_bytes(b"must not be loaded")
    split = CalibrationSplit(
        memory_bank_images=(first_path, second_path),
        calibration_images=(calibration_path,),
    )

    bank = build_memory_bank(
        split,
        TinyFeatureExtractor(),
        input_size=(8, 8),
        batch_size=1,
        sampling_ratio=1.0,
        max_patches=20,
    )

    assert bank.source_image_count == 2
    assert bank.candidate_patch_count == 8
    assert bank.embeddings.shape == (8, 3)
    assert bank.embeddings.max().item() < 0.2
    assert bank.sampling_strategy == "random"
    assert bank.input_size == (8, 8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sampling_strategy": "unknown"},
        {"sampling_ratio": 0},
        {"max_patches": 0},
        {"seed": -1},
    ],
)
def test_sampler_rejects_invalid_settings(kwargs):
    with pytest.raises(DatasetError):
        sample_patch_embeddings(torch.ones((4, 2)), **kwargs)


def test_builder_rejects_test_paths_and_overlapping_calibration_paths(tmp_path: Path):
    test_image = tmp_path / "bottle" / "test" / "good" / "test.png"
    split_with_test = CalibrationSplit(memory_bank_images=(test_image,), calibration_images=())
    with pytest.raises(DatasetError, match="train/good"):
        build_memory_bank(split_with_test, TinyFeatureExtractor(), input_size=(8, 8))

    normal_image = tmp_path / "bottle" / "train" / "good" / "same.png"
    overlap = CalibrationSplit(
        memory_bank_images=(normal_image,), calibration_images=(normal_image,)
    )
    with pytest.raises(DatasetError, match="disjoint"):
        build_memory_bank(overlap, TinyFeatureExtractor(), input_size=(8, 8))
