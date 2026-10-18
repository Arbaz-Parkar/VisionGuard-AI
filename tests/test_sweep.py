from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from visionguard.evaluation import EvaluationError
from visionguard.model import (
    MemoryBank,
    PatchAnomalyScorer,
    PatchDetectorArtifact,
    PatchThresholdCalibration,
)
from visionguard.sweep import run_memory_bank_sweep, validate_sweep_ratios


class _PixelExtractor:
    device = torch.device("cpu")

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return images.permute(0, 2, 3, 1).contiguous()


def _artifact() -> PatchDetectorArtifact:
    extractor = _PixelExtractor()
    bank = MemoryBank(
        embeddings=torch.zeros((1, 3), dtype=torch.float32),
        source_image_count=1,
        candidate_patch_count=1,
        sampling_strategy="random",
        sampling_ratio=1.0,
        seed=3,
        input_size=(8, 8),
        preserve_aspect_ratio=False,
    )
    scorer = PatchAnomalyScorer(
        extractor,
        bank,
        input_size=(8, 8),
        preserve_aspect_ratio=False,
        smoothing_sigma=0,
    )
    thresholds = PatchThresholdCalibration(
        pixel_threshold=0.5,
        image_threshold=0.5,
        pixel_false_positive_rate=0.01,
        image_false_positive_rate=0.05,
        pixel_percentile=99.5,
        image_percentile=95,
        calibration_image_count=1,
        calibration_pixel_count=64,
    )
    return PatchDetectorArtifact.from_components(
        category="bottle",
        memory_bank=bank,
        thresholds=thresholds,
        scorer=scorer,
    )


def _save(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)


def test_sweep_ratios_are_sorted_and_reject_duplicates_or_invalid_values():
    assert validate_sweep_ratios([1.0, 0.05, 0.25]) == (0.05, 0.25, 1.0)
    with pytest.raises(EvaluationError, match="duplicate"):
        validate_sweep_ratios([0.1, 0.1])
    with pytest.raises(EvaluationError, match=r"in \(0, 1\]"):
        validate_sweep_ratios([0.0])


def test_memory_bank_sweep_uses_capped_seeded_candidate_pool(tmp_path: Path):
    root = tmp_path / "dataset"
    _save(root / "bottle" / "test" / "good" / "normal.png", 0)
    _save(root / "bottle" / "test" / "crack" / "defect.png", 255)
    candidates = torch.zeros((4, 3), dtype=torch.float32)

    points = run_memory_bank_sweep(
        root,
        _artifact(),
        _PixelExtractor(),
        candidates,
        ratios=(0.5, 1.0),
        seed=4,
        warmup_count=1,
        candidate_pool_limit=3,
    )

    assert len(points) == 4
    assert {point.sampling_strategy for point in points} == {"random", "greedy"}
    assert all(point.source_candidate_patch_count == 4 for point in points)
    assert all(point.candidate_patch_count == 3 for point in points)
    assert all(point.evaluated_image_count == 2 for point in points)
    assert all(point.image_auroc == pytest.approx(1.0) for point in points)
