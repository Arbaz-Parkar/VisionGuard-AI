"""Memory-bank size experiment for the PatchCore-inspired detector."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from visionguard.data import DatasetError, discover_evaluation_images
from visionguard.evaluation import EvaluationError
from visionguard.model import (
    MemoryBank,
    PatchDetectorArtifact,
    PatchFeatureExtractor,
    sample_patch_embeddings,
)
from visionguard.processing import ImageValidationError, load_image


@dataclass(frozen=True)
class MemoryBankSweepPoint:
    """One strategy/ratio result measured on the same held-out test images."""

    sampling_strategy: str
    requested_ratio: float
    effective_ratio: float
    sampling_seed: int
    candidate_pool_seed: int
    source_candidate_patch_count: int
    candidate_patch_count: int
    stored_patch_count: int
    stored_megabytes: float
    image_auroc: float | None
    auroc_unavailable_reason: str | None
    mean_inference_ms: float
    median_inference_ms: float
    evaluated_image_count: int


def validate_sweep_ratios(ratios: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Validate, de-duplicate, and sort the requested non-empty sweep ratios."""
    if not isinstance(ratios, (tuple, list)) or not ratios:
        raise EvaluationError("sweep_ratios must be a non-empty list of values in (0, 1].")
    values: list[float] = []
    for ratio in ratios:
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not np.isfinite(ratio)
            or not 0 < ratio <= 1
        ):
            raise EvaluationError("Each memory-bank sweep ratio must be in (0, 1].")
        values.append(float(ratio))
    if len(set(values)) != len(values):
        raise EvaluationError("sweep_ratios must not contain duplicate values.")
    return tuple(sorted(values))


def run_memory_bank_sweep(
    dataset_root: str | Path,
    artifact: PatchDetectorArtifact,
    feature_extractor: PatchFeatureExtractor,
    candidate_embeddings: torch.Tensor,
    *,
    ratios: tuple[float, ...] | list[float] = (0.01, 0.05, 0.10, 0.25, 1.0),
    seed: int = 42,
    warmup_count: int = 3,
    sample_limit: int | None = None,
    candidate_pool_limit: int = 2048,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[MemoryBankSweepPoint, ...]:
    """Compare random and greedy banks without tuning on test-set outcomes.

    AUROC is threshold-free. The test set is used only to measure each fixed
    design point; this function never selects a new default or changes the
    saved production artifact.
    """
    if not isinstance(artifact, PatchDetectorArtifact):
        raise EvaluationError("artifact must be a PatchDetectorArtifact.")
    if (
        not isinstance(candidate_embeddings, torch.Tensor)
        or candidate_embeddings.ndim != 2
        or candidate_embeddings.shape[0] == 0
        or candidate_embeddings.shape[1] == 0
        or not candidate_embeddings.is_floating_point()
        or not torch.isfinite(candidate_embeddings).all().item()
    ):
        raise EvaluationError("candidate_embeddings must be a finite [N, D] tensor.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvaluationError("seed must be a non-negative integer.")
    if isinstance(warmup_count, bool) or not isinstance(warmup_count, int) or warmup_count < 0:
        raise EvaluationError("warmup_count must be a non-negative integer.")
    if isinstance(candidate_pool_limit, bool) or not isinstance(candidate_pool_limit, int) or candidate_pool_limit <= 0:
        raise EvaluationError("candidate_pool_limit must be a positive integer.")
    if progress_callback is not None and not callable(progress_callback):
        raise EvaluationError("progress_callback must be callable when provided.")
    ratios = validate_sweep_ratios(ratios)

    root = Path(dataset_root).expanduser()
    try:
        samples = discover_evaluation_images(root, artifact.category)
    except DatasetError as exc:
        raise EvaluationError(str(exc)) from exc
    if sample_limit is not None:
        if isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or sample_limit <= 0:
            raise EvaluationError("sample_limit must be a positive integer when provided.")
        samples = samples[:sample_limit]
    if not samples:
        raise EvaluationError("The memory-bank sweep found no test images.")

    full_candidate_count = int(candidate_embeddings.shape[0])
    features = candidate_embeddings.detach().to(device="cpu", dtype=torch.float32).contiguous()
    pool_seed = seed + 1
    if features.shape[0] > candidate_pool_limit:
        generator = torch.Generator(device="cpu").manual_seed(pool_seed)
        pool_indices = torch.randperm(features.shape[0], generator=generator)[:candidate_pool_limit]
        features = features.index_select(0, pool_indices).contiguous()
    candidate_count = int(features.shape[0])
    points: list[MemoryBankSweepPoint] = []
    for strategy in ("random", "greedy"):
        for ratio in ratios:
            if progress_callback is not None:
                progress_callback(
                    f"Sweeping {strategy} memory bank at ratio {ratio:g} "
                    f"({candidate_count:,} candidate patches)."
                )
            try:
                selected = sample_patch_embeddings(
                    features,
                    sampling_strategy=strategy,
                    sampling_ratio=ratio,
                    max_patches=candidate_count,
                    seed=seed,
                )
            except DatasetError as exc:
                raise EvaluationError(f"Unable to sample {strategy} memory bank: {exc}") from exc
            bank = MemoryBank(
                embeddings=selected,
                source_image_count=artifact.memory_bank.source_image_count,
                candidate_patch_count=candidate_count,
                sampling_strategy=strategy,
                sampling_ratio=ratio,
                seed=seed,
                input_size=artifact.memory_bank.input_size,
                preserve_aspect_ratio=artifact.memory_bank.preserve_aspect_ratio,
            )
            candidate_artifact = replace(artifact, memory_bank=bank)
            scorer = candidate_artifact.build_scorer(feature_extractor)

            warmup_samples = samples[: min(warmup_count, len(samples))]
            for sample in warmup_samples:
                try:
                    scorer.run(load_image(sample.image_path), pixel_threshold=None)
                except (ImageValidationError, ValueError, RuntimeError) as exc:
                    raise EvaluationError(
                        f"Unable to warm up {strategy} memory-bank sweep: {exc}"
                    ) from exc

            labels: list[int] = []
            scores: list[float] = []
            inference_ms: list[float] = []
            for sample in samples:
                try:
                    image = load_image(sample.image_path)
                    started = time.perf_counter()
                    result = scorer.run(image, pixel_threshold=None)
                    inference_ms.append((time.perf_counter() - started) * 1000.0)
                except (ImageValidationError, ValueError, RuntimeError) as exc:
                    raise EvaluationError(
                        f"Unable to score {sample.image_path} with the {strategy} bank: {exc}"
                    ) from exc
                if not np.isfinite(result.image_score) or not np.isfinite(result.raw_map).all():
                    raise EvaluationError("Sweep scorer returned non-finite anomaly scores.")
                labels.append(int(sample.is_anomaly))
                scores.append(float(result.image_score))

            if len(set(labels)) < 2:
                auroc = None
                reason = "AUROC requires both normal and anomalous labels."
            else:
                auroc = float(roc_auc_score(labels, scores))
                reason = None
            points.append(
                MemoryBankSweepPoint(
                    sampling_strategy=strategy,
                    requested_ratio=ratio,
                    effective_ratio=float(selected.shape[0] / candidate_count),
                    sampling_seed=seed,
                    candidate_pool_seed=pool_seed,
                    source_candidate_patch_count=full_candidate_count,
                    candidate_patch_count=candidate_count,
                    stored_patch_count=int(selected.shape[0]),
                    stored_megabytes=float(
                        selected.numel() * selected.element_size() / (1024.0 * 1024.0)
                    ),
                    image_auroc=auroc,
                    auroc_unavailable_reason=reason,
                    mean_inference_ms=float(np.mean(inference_ms)),
                    median_inference_ms=float(np.median(inference_ms)),
                    evaluated_image_count=len(samples),
                )
            )
    return tuple(points)
