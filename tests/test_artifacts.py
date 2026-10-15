import json
from pathlib import Path

import numpy as np
import pytest
import torch

from visionguard.model import (
    ArtifactError,
    MemoryBank,
    PatchAnomalyScorer,
    PatchDetectorArtifact,
    PatchThresholdCalibration,
    load_patch_artifact,
    save_patch_artifact,
)


class PixelFeatureExtractor:
    device = torch.device("cpu")

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return images.permute(0, 2, 3, 1).contiguous()


def _bank() -> MemoryBank:
    return MemoryBank(
        embeddings=torch.zeros((2, 3), dtype=torch.float32),
        source_image_count=2,
        candidate_patch_count=64,
        sampling_strategy="greedy",
        sampling_ratio=0.25,
        seed=21,
        input_size=(8, 8),
        preserve_aspect_ratio=False,
    )


def _thresholds() -> PatchThresholdCalibration:
    return PatchThresholdCalibration(
        pixel_threshold=0.1,
        image_threshold=0.2,
        pixel_false_positive_rate=0.005,
        image_false_positive_rate=0.05,
        pixel_percentile=99.5,
        image_percentile=95.0,
        calibration_image_count=3,
        calibration_pixel_count=192,
    )


def _scorer(bank: MemoryBank) -> PatchAnomalyScorer:
    return PatchAnomalyScorer(
        PixelFeatureExtractor(),
        bank,
        input_size=(8, 8),
        preserve_aspect_ratio=False,
        smoothing_sigma=0,
        aggregation="percentile",
        score_percentile=98.0,
        nearest_neighbor_chunk_size=5,
    )


def _artifact() -> PatchDetectorArtifact:
    bank = _bank()
    return PatchDetectorArtifact.from_components(
        category="bottle",
        memory_bank=bank,
        thresholds=_thresholds(),
        scorer=_scorer(bank),
    )


def test_save_and_load_round_trip_recreates_scoring(tmp_path: Path):
    artifact = _artifact()
    path = save_patch_artifact(tmp_path / "bottle.npz", artifact)

    loaded = load_patch_artifact(path)
    scorer = loaded.build_scorer(PixelFeatureExtractor())
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2:6, 3:5] = 255
    original_result = _scorer(artifact.memory_bank).run(
        image, pixel_threshold=artifact.thresholds.pixel_threshold
    )
    loaded_result = scorer.run(
        image, pixel_threshold=loaded.thresholds.pixel_threshold
    )

    assert loaded.category == "bottle"
    assert loaded.architecture == "resnet18"
    assert loaded.feature_layers == ("layer2", "layer3")
    assert loaded.pretrained_weights == "torchvision.models.ResNet18_Weights.IMAGENET1K_V1"
    assert loaded.memory_bank.sampling_strategy == "greedy"
    assert loaded.memory_bank.source_image_count == 2
    assert loaded.memory_bank.candidate_patch_count == 64
    assert loaded.memory_bank.input_size == (8, 8)
    assert loaded.thresholds == artifact.thresholds
    assert loaded.smoothing_sigma == 0
    assert loaded.aggregation == "percentile"
    assert loaded.score_percentile == 98.0
    assert loaded.nearest_neighbor_chunk_size == 5
    assert torch.equal(loaded.memory_bank.embeddings, artifact.memory_bank.embeddings)
    np.testing.assert_array_equal(loaded_result.raw_map, original_result.raw_map)
    np.testing.assert_array_equal(loaded_result.mask, original_result.mask)
    assert loaded_result.image_score == pytest.approx(original_result.image_score)


def test_artifact_does_not_record_source_image_paths(tmp_path: Path):
    path = save_patch_artifact(tmp_path / "model.npz", _artifact())

    contents = path.read_bytes()

    assert b"train" not in contents
    assert b"bank.png" not in contents
    assert b"calibration.png" not in contents


def test_loader_rejects_unknown_schema_and_malformed_archives(tmp_path: Path):
    unsupported = tmp_path / "unsupported.npz"
    metadata = json.dumps(
        {"format": "visionguard.patch_detector", "schema_version": 999}
    ).encode("utf-8")
    np.savez_compressed(
        unsupported,
        embeddings=np.zeros((1, 3), dtype=np.float32),
        metadata=np.frombuffer(metadata, dtype=np.uint8),
    )
    with pytest.raises(ArtifactError, match="schema version"):
        load_patch_artifact(unsupported)

    malformed = tmp_path / "malformed.npz"
    np.savez_compressed(malformed, unrelated=np.ones(1))
    with pytest.raises(ArtifactError, match="embeddings and metadata"):
        load_patch_artifact(malformed)


def test_save_rejects_invalid_category_mismatched_bank_and_wrong_suffix(tmp_path: Path):
    bank = _bank()
    with pytest.raises(ArtifactError, match="category"):
        PatchDetectorArtifact.from_components(
            category="../bottle",
            memory_bank=bank,
            thresholds=_thresholds(),
            scorer=_scorer(bank),
        )

    with pytest.raises(ArtifactError, match="same embeddings"):
        PatchDetectorArtifact.from_components(
            category="bottle",
            memory_bank=MemoryBank(
                embeddings=torch.ones((2, 3)),
                source_image_count=2,
                candidate_patch_count=64,
                sampling_strategy="greedy",
                sampling_ratio=0.25,
                seed=21,
                input_size=(8, 8),
                preserve_aspect_ratio=False,
            ),
            thresholds=_thresholds(),
            scorer=_scorer(bank),
        )

    with pytest.raises(ArtifactError, match=".npz"):
        save_patch_artifact(tmp_path / "model.pt", _artifact())
