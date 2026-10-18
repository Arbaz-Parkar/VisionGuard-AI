"""Normal-only patch memory-bank creation and reduction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch

from visionguard.data import CalibrationSplit, DatasetError
from visionguard.processing import load_image, resize_image, to_rgb

SamplingStrategy = Literal["random", "greedy"]


class PatchFeatureExtractor(Protocol):
    """Small protocol implemented by the frozen ResNet patch extractor."""

    def __call__(self, images: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class MemoryBank:
    """Sampled patch embeddings and basic provenance for one bank build."""

    embeddings: torch.Tensor
    source_image_count: int
    candidate_patch_count: int
    sampling_strategy: SamplingStrategy
    sampling_ratio: float
    seed: int
    input_size: tuple[int, int]
    preserve_aspect_ratio: bool = True


def _validate_sampling(
    strategy: str, sampling_ratio: float, max_patches: int, seed: int
) -> tuple[SamplingStrategy, float, int, int]:
    if not isinstance(strategy, str) or strategy not in {"random", "greedy"}:
        raise DatasetError("sampling_strategy must be either 'random' or 'greedy'.")
    if (
        isinstance(sampling_ratio, bool)
        or not isinstance(sampling_ratio, (int, float))
        or not math.isfinite(sampling_ratio)
        or not 0 < sampling_ratio <= 1
    ):
        raise DatasetError("sampling_ratio must be greater than 0 and at most 1.")
    if isinstance(max_patches, bool) or not isinstance(max_patches, int) or max_patches <= 0:
        raise DatasetError("max_patches must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DatasetError("seed must be a non-negative integer.")
    return strategy, float(sampling_ratio), max_patches, seed  # type: ignore[return-value]


def _greedy_indices(
    patches: torch.Tensor, count: int, *, seed: int, chunk_size: int = 8192
) -> torch.Tensor:
    """Approximate k-center greedy selection using chunked squared distances."""
    candidate_count, _ = patches.shape
    if count >= candidate_count:
        return torch.arange(candidate_count)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(candidate_count, (1,), generator=generator).item())
    selected = [first]
    selected_mask = torch.zeros(candidate_count, dtype=torch.bool)
    selected_mask[first] = True
    nearest_distances = torch.full((candidate_count,), float("inf"), dtype=torch.float32)
    squared_norms = torch.sum(patches * patches, dim=1)

    for _ in range(1, count):
        center = patches[selected[-1]]
        center_norm = torch.dot(center, center)
        for start in range(0, candidate_count, chunk_size):
            stop = min(start + chunk_size, candidate_count)
            block = patches[start:stop]
            distances = squared_norms[start:stop] + center_norm - 2.0 * torch.mv(block, center)
            nearest_distances[start:stop] = torch.minimum(
                nearest_distances[start:stop], distances.clamp_min_(0.0)
            )
        nearest_distances[selected_mask] = -1.0
        next_index = int(torch.argmax(nearest_distances).item())
        selected.append(next_index)
        selected_mask[next_index] = True

    return torch.tensor(selected, dtype=torch.int64)


def sample_patch_embeddings(
    patches: torch.Tensor,
    *,
    sampling_strategy: SamplingStrategy = "random",
    sampling_ratio: float = 0.1,
    max_patches: int = 10_000,
    seed: int = 42,
) -> torch.Tensor:
    """Reduce an ``[patch_count, feature_dim]`` tensor reproducibly.

    Random sampling uses a seeded permutation. Greedy coreset sampling chooses
    farthest points under Euclidean distance, updating candidate distances in
    chunks so it does not allocate a full pairwise distance matrix.
    """
    strategy, ratio, maximum, seed_value = _validate_sampling(
        sampling_strategy, sampling_ratio, max_patches, seed
    )
    if (
        not isinstance(patches, torch.Tensor)
        or patches.ndim != 2
        or patches.shape[0] == 0
        or patches.shape[1] == 0
        or not patches.is_floating_point()
        or not torch.isfinite(patches).all().item()
    ):
        raise DatasetError("patches must be a non-empty finite floating-point [N, D] tensor.")

    candidates = patches.detach().to(device="cpu", dtype=torch.float32).contiguous()
    target_count = min(
        candidates.shape[0], maximum, max(1, math.ceil(candidates.shape[0] * ratio))
    )
    if target_count == candidates.shape[0]:
        return candidates.clone()

    if strategy == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed_value)
        selected = torch.randperm(candidates.shape[0], generator=generator)[:target_count]
    else:
        selected = _greedy_indices(candidates, target_count, seed=seed_value)
    return candidates.index_select(0, selected).contiguous()


def _validate_training_split(split: CalibrationSplit) -> tuple[Path, ...]:
    if not isinstance(split, CalibrationSplit):
        raise DatasetError("split must be a CalibrationSplit from the normal training data.")
    bank_paths = tuple(Path(path) for path in split.memory_bank_images)
    calibration_paths = {Path(path).resolve() for path in split.calibration_images}
    if not bank_paths:
        raise DatasetError("The memory-bank training split contains no images.")
    if {path.resolve() for path in bank_paths} & calibration_paths:
        raise DatasetError("Memory-bank and calibration images must be disjoint.")
    for path in bank_paths:
        parts = tuple(part.casefold() for part in path.parts)
        if not any(parts[index : index + 2] == ("train", "good") for index in range(len(parts) - 1)):
            raise DatasetError(f"Memory-bank image must come from train/good: {path}")
    return bank_paths


def _load_preprocessed(path: Path, input_size: tuple[int, int], preserve_aspect_ratio: bool) -> torch.Tensor:
    pixels = load_image(path)
    resized, _ = resize_image(
        pixels,
        input_size,
        preserve_aspect_ratio=preserve_aspect_ratio,
        pad_value=0,
    )
    rgb = to_rgb(resized)
    array = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def extract_patch_embeddings(
    split: CalibrationSplit,
    feature_extractor: PatchFeatureExtractor,
    *,
    input_size: tuple[int, int] = (256, 256),
    preserve_aspect_ratio: bool = True,
    batch_size: int = 8,
) -> torch.Tensor:
    """Extract the complete normal-only candidate pool on CPU for analysis runs."""
    bank_paths = _validate_training_split(split)
    if (
        not isinstance(input_size, tuple)
        or len(input_size) != 2
        or any(type(value) is not int or value <= 0 for value in input_size)
    ):
        raise DatasetError("input_size must contain two positive (height, width) integers.")
    if not isinstance(preserve_aspect_ratio, bool):
        raise DatasetError("preserve_aspect_ratio must be a boolean.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise DatasetError("batch_size must be a positive integer.")

    feature_batches: list[torch.Tensor] = []
    expected_patch_shape: tuple[int, int, int] | None = None
    for start in range(0, len(bank_paths), batch_size):
        paths = bank_paths[start : start + batch_size]
        images = torch.stack(
            [_load_preprocessed(path, input_size, preserve_aspect_ratio) for path in paths]
        )
        patch_features = feature_extractor(images)
        if (
            not isinstance(patch_features, torch.Tensor)
            or patch_features.ndim != 4
            or patch_features.shape[0] != len(paths)
            or any(size <= 0 for size in patch_features.shape[1:])
            or not torch.isfinite(patch_features).all().item()
        ):
            raise DatasetError(
                "feature_extractor must return finite [batch, patch_height, patch_width, channels] features."
            )
        patch_shape = tuple(int(size) for size in patch_features.shape[1:])
        if expected_patch_shape is None:
            expected_patch_shape = patch_shape
        elif patch_shape != expected_patch_shape:
            raise DatasetError("Feature extractor returned inconsistent patch shapes.")
        feature_batches.append(
            patch_features.detach().to(device="cpu", dtype=torch.float32).reshape(
                -1, patch_features.shape[-1]
            )
        )
    return torch.cat(feature_batches, dim=0).contiguous()


def build_memory_bank(
    split: CalibrationSplit,
    feature_extractor: PatchFeatureExtractor,
    *,
    input_size: tuple[int, int] = (256, 256),
    preserve_aspect_ratio: bool = True,
    batch_size: int = 8,
    sampling_strategy: SamplingStrategy = "random",
    sampling_ratio: float = 0.1,
    max_patches: int = 10_000,
    seed: int = 42,
) -> MemoryBank:
    """Extract and reduce patches using only ``split.memory_bank_images``.

    The calibration side of ``CalibrationSplit`` is checked for overlap but is
    never loaded or sent through the extractor. All bank source paths must be
    under ``train/good``; test and defect images are rejected.
    """
    bank_paths = _validate_training_split(split)
    strategy, ratio, maximum, seed_value = _validate_sampling(
        sampling_strategy, sampling_ratio, max_patches, seed
    )
    if (
        not isinstance(input_size, tuple)
        or len(input_size) != 2
        or any(type(value) is not int or value <= 0 for value in input_size)
    ):
        raise DatasetError("input_size must contain two positive (height, width) integers.")
    if not isinstance(preserve_aspect_ratio, bool):
        raise DatasetError("preserve_aspect_ratio must be a boolean.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise DatasetError("batch_size must be a positive integer.")

    all_patches = extract_patch_embeddings(
        split,
        feature_extractor,
        input_size=input_size,
        preserve_aspect_ratio=preserve_aspect_ratio,
        batch_size=batch_size,
    )
    sampled = sample_patch_embeddings(
        all_patches,
        sampling_strategy=strategy,
        sampling_ratio=ratio,
        max_patches=maximum,
        seed=seed_value,
    )
    return MemoryBank(
        embeddings=sampled,
        source_image_count=len(bank_paths),
        candidate_patch_count=all_patches.shape[0],
        sampling_strategy=strategy,
        sampling_ratio=ratio,
        seed=seed_value,
        input_size=input_size,
        preserve_aspect_ratio=preserve_aspect_ratio,
    )
