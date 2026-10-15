import numpy as np
import pytest
import torch

from visionguard.model import MemoryBank, PatchAnomalyScorer
from visionguard.processing import ImageValidationError


class PixelFeatureExtractor:
    """Deterministic stand-in for the pretrained extractor in unit tests."""

    device = torch.device("cpu")

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return images.permute(0, 2, 3, 1).contiguous()


def _bank(
    feature_dim: int = 3,
    input_size: tuple[int, int] = (8, 8),
    preserve_aspect_ratio: bool = False,
) -> MemoryBank:
    return MemoryBank(
        embeddings=torch.zeros((1, feature_dim), dtype=torch.float32),
        source_image_count=1,
        candidate_patch_count=1,
        sampling_strategy="random",
        sampling_ratio=1.0,
        seed=42,
        input_size=input_size,
        preserve_aspect_ratio=preserve_aspect_ratio,
    )


def _scorer(bank: MemoryBank | None = None, **kwargs) -> PatchAnomalyScorer:
    return PatchAnomalyScorer(
        PixelFeatureExtractor(),
        bank or _bank(),
        input_size=(8, 8),
        preserve_aspect_ratio=False,
        smoothing_sigma=0,
        nearest_neighbor_chunk_size=3,
        **kwargs,
    )


def test_patch_scorer_builds_distance_map_mask_regions_and_timings():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[3:5, 4:6] = 255

    result = _scorer().run(image, pixel_threshold=1.0, min_region_area=0)

    assert result.raw_map.shape == (8, 8)
    assert result.raw_map.dtype == np.float32
    assert result.raw_map[3:5, 4:6] == pytest.approx(np.sqrt(3.0))
    assert result.raw_map[0, 0] == pytest.approx(0.0)
    assert result.image_score == pytest.approx(np.sqrt(3.0))
    assert set(np.unique(result.display_map)) == {0.0, 1.0}
    assert set(np.unique(result.mask)) == {0, 255}
    assert len(result.regions) == 1
    assert result.regions[0].bounding_box == (4, 3, 2, 2)
    assert result.pixel_threshold == 1.0
    assert result.device == "cpu"
    assert all(
        value >= 0
        for value in (
            result.timings.preprocessing_ms,
            result.timings.feature_extraction_ms,
            result.timings.nearest_neighbor_ms,
            result.timings.postprocessing_ms,
            result.timings.total_ms,
        )
    )


def test_constant_normal_image_has_zero_map_and_no_regions():
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    result = _scorer().run(image, pixel_threshold=0.0, min_region_area=0)

    assert result.image_score == 0.0
    assert not result.raw_map.any()
    assert not result.display_map.any()
    assert not result.mask.any()
    assert result.regions == ()


def test_scorer_restores_letterboxed_map_to_original_image_shape():
    scorer = PatchAnomalyScorer(
        PixelFeatureExtractor(),
        _bank(preserve_aspect_ratio=True),
        input_size=(8, 8),
        preserve_aspect_ratio=True,
        smoothing_sigma=0,
    )

    result = scorer.run(np.full((4, 8, 3), 255, dtype=np.uint8), pixel_threshold=1.0)

    assert result.raw_map.shape == (4, 8)
    assert result.raw_map.min() == pytest.approx(np.sqrt(3.0))
    assert result.raw_map.max() == pytest.approx(np.sqrt(3.0))


def test_scorer_rejects_non_uint8_input_and_invalid_threshold():
    scorer = _scorer()

    with pytest.raises(ImageValidationError, match="uint8"):
        scorer.run(np.zeros((8, 8, 3), dtype=np.uint16), pixel_threshold=1.0)
    with pytest.raises(ValueError, match="pixel_threshold"):
        scorer.run(np.zeros((8, 8, 3), dtype=np.uint8), pixel_threshold=-1.0)


def test_scorer_requires_matching_bank_input_size_and_embedding_dimensions():
    with pytest.raises(ValueError, match="must match the size"):
        PatchAnomalyScorer(PixelFeatureExtractor(), _bank(input_size=(4, 4)))
    with pytest.raises(ValueError, match="preserve_aspect_ratio"):
        PatchAnomalyScorer(
            PixelFeatureExtractor(),
            _bank(preserve_aspect_ratio=False),
            input_size=(8, 8),
        )

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match the memory bank"):
        _scorer(_bank(feature_dim=2)).run(image, pixel_threshold=0.1)
