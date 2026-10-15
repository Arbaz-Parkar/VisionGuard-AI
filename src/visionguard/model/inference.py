"""Nearest-neighbor patch scoring and anomaly-map construction."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from visionguard.processing import (
    CandidateRegion,
    extract_candidate_regions,
    resize_image,
    restore_map,
    to_rgb,
    validate_image,
)

from .memory_bank import MemoryBank, PatchFeatureExtractor


@dataclass(frozen=True)
class PatchInferenceTimings:
    """Elapsed milliseconds for the main inference stages."""

    preprocessing_ms: float
    feature_extraction_ms: float
    nearest_neighbor_ms: float
    postprocessing_ms: float
    total_ms: float


@dataclass(frozen=True)
class PatchDetectionResult:
    """Image score, continuous anomaly maps, thresholded mask, and regions."""

    image_score: float
    raw_map: np.ndarray
    display_map: np.ndarray
    mask: np.ndarray
    pixel_threshold: float
    regions: tuple[CandidateRegion, ...]
    timings: PatchInferenceTimings
    device: str


class PatchAnomalyScorer:
    """Score patch embeddings by chunked distance to a normal-only memory bank."""

    def __init__(
        self,
        feature_extractor: PatchFeatureExtractor,
        memory_bank: MemoryBank,
        *,
        input_size: tuple[int, int] = (256, 256),
        preserve_aspect_ratio: bool = True,
        smoothing_sigma: float = 4.0,
        aggregation: str = "max",
        score_percentile: float = 99.5,
        nearest_neighbor_chunk_size: int = 512,
    ) -> None:
        if (
            not isinstance(memory_bank, MemoryBank)
            or not isinstance(memory_bank.embeddings, torch.Tensor)
            or memory_bank.embeddings.ndim != 2
            or memory_bank.embeddings.shape[0] == 0
            or memory_bank.embeddings.shape[1] == 0
            or not memory_bank.embeddings.is_floating_point()
            or not torch.isfinite(memory_bank.embeddings).all().item()
        ):
            raise ValueError("memory_bank must contain finite [patch_count, feature_dim] embeddings.")
        if (
            not isinstance(input_size, tuple)
            or len(input_size) != 2
            or any(type(value) is not int or value <= 0 for value in input_size)
        ):
            raise ValueError("input_size must contain two positive (height, width) integers.")
        if input_size != memory_bank.input_size:
            raise ValueError(
                "input_size must match the size used to build the memory bank "
                f"({input_size} versus {memory_bank.input_size})."
            )
        if not isinstance(preserve_aspect_ratio, bool):
            raise ValueError("preserve_aspect_ratio must be a boolean.")
        if preserve_aspect_ratio != memory_bank.preserve_aspect_ratio:
            raise ValueError(
                "preserve_aspect_ratio must match the setting used to build the memory bank."
            )
        if (
            isinstance(smoothing_sigma, bool)
            or not isinstance(smoothing_sigma, (int, float))
            or not math.isfinite(smoothing_sigma)
            or smoothing_sigma < 0
        ):
            raise ValueError("smoothing_sigma must be a finite non-negative number.")
        if not isinstance(aggregation, str) or aggregation not in {"max", "percentile"}:
            raise ValueError("aggregation must be either 'max' or 'percentile'.")
        if (
            isinstance(score_percentile, bool)
            or not isinstance(score_percentile, (int, float))
            or not math.isfinite(score_percentile)
            or not 0 < score_percentile <= 100
        ):
            raise ValueError("score_percentile must be greater than 0 and at most 100.")
        if (
            isinstance(nearest_neighbor_chunk_size, bool)
            or not isinstance(nearest_neighbor_chunk_size, int)
            or nearest_neighbor_chunk_size <= 0
        ):
            raise ValueError("nearest_neighbor_chunk_size must be a positive integer.")

        extractor_device = getattr(feature_extractor, "device", "cpu")
        self.device = torch.device(extractor_device)
        self.feature_extractor = feature_extractor
        self.memory_embeddings = (
            memory_bank.embeddings.detach()
            .to(device=self.device, dtype=torch.float32)
            .contiguous()
            .clone()
        )
        self.input_size = input_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.smoothing_sigma = float(smoothing_sigma)
        self.aggregation = aggregation
        self.score_percentile = float(score_percentile)
        self.nearest_neighbor_chunk_size = nearest_neighbor_chunk_size

    def _synchronize(self) -> None:
        """Wait for queued CUDA work so reported stage timings are wall-clock accurate."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _nearest_distances(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.shape[1] != self.memory_embeddings.shape[1]:
            raise ValueError(
                "Feature extractor dimension does not match the memory bank "
                f"({patches.shape[1]} versus {self.memory_embeddings.shape[1]})."
            )
        distances = []
        for start in range(0, patches.shape[0], self.nearest_neighbor_chunk_size):
            query = patches[start : start + self.nearest_neighbor_chunk_size]
            chunk_distances = torch.cdist(query, self.memory_embeddings, p=2)
            distances.append(torch.min(chunk_distances, dim=1).values)
        result = torch.cat(distances)
        if not torch.isfinite(result).all().item():
            raise ValueError("Nearest-neighbor scoring produced non-finite distances.")
        return result

    @torch.inference_mode()
    def run(
        self,
        image: np.ndarray,
        *,
        pixel_threshold: float,
        min_region_area: float = 20.0,
    ) -> PatchDetectionResult:
        """Score one image and apply a pixel threshold calibrated on normal data."""
        if (
            isinstance(pixel_threshold, bool)
            or not isinstance(pixel_threshold, (int, float))
            or not math.isfinite(pixel_threshold)
            or pixel_threshold < 0
        ):
            raise ValueError("pixel_threshold must be a finite non-negative number.")
        if (
            isinstance(min_region_area, bool)
            or not isinstance(min_region_area, (int, float))
            or not math.isfinite(min_region_area)
            or min_region_area < 0
        ):
            raise ValueError("min_region_area must be a finite non-negative number.")

        total_start = time.perf_counter()
        preprocess_start = total_start
        pixels = validate_image(image)
        resized, transform = resize_image(
            pixels,
            self.input_size,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
            pad_value=0,
        )
        rgb = to_rgb(resized)
        batch_array = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        batch = torch.from_numpy(batch_array).unsqueeze(0)
        preprocessing_ms = (time.perf_counter() - preprocess_start) * 1000.0

        self._synchronize()
        feature_start = time.perf_counter()
        feature_map = self.feature_extractor(batch)
        self._synchronize()
        if (
            not isinstance(feature_map, torch.Tensor)
            or feature_map.ndim != 4
            or feature_map.shape[0] != 1
            or any(size <= 0 for size in feature_map.shape[1:])
            or not torch.isfinite(feature_map).all().item()
        ):
            raise ValueError(
                "feature_extractor must return finite [1, patch_height, patch_width, channels] features."
            )
        patch_height, patch_width, feature_dim = feature_map.shape[1:]
        patch_vectors = feature_map.reshape(-1, feature_dim).to(
            device=self.device, dtype=torch.float32
        )
        feature_extraction_ms = (time.perf_counter() - feature_start) * 1000.0

        nearest_start = time.perf_counter()
        patch_scores = self._nearest_distances(patch_vectors)
        self._synchronize()
        nearest_neighbor_ms = (time.perf_counter() - nearest_start) * 1000.0

        postprocess_start = time.perf_counter()
        score_tensor = patch_scores.reshape(1, 1, patch_height, patch_width)
        upsampled = F.interpolate(
            score_tensor,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        input_map = upsampled.detach().to(device="cpu", dtype=torch.float32).numpy()
        if self.smoothing_sigma > 0:
            input_map = cv2.GaussianBlur(
                input_map, (0, 0), sigmaX=self.smoothing_sigma
            )
        raw_map = restore_map(input_map, transform).astype(np.float32, copy=False)
        raw_map = np.maximum(raw_map, 0.0)
        minimum, maximum = float(raw_map.min()), float(raw_map.max())
        if maximum > minimum:
            display_map = ((raw_map - minimum) / (maximum - minimum)).astype(np.float32)
        else:
            display_map = np.zeros_like(raw_map, dtype=np.float32)

        threshold_mask = np.where(raw_map > pixel_threshold, 255, 0).astype(np.uint8)
        regions = tuple(
            extract_candidate_regions(threshold_mask, min_area=float(min_region_area))
        )
        mask = np.zeros_like(threshold_mask)
        for region in regions:
            cv2.drawContours(mask, [region.contour], -1, 255, thickness=cv2.FILLED)

        image_score = (
            float(np.max(raw_map))
            if self.aggregation == "max"
            else float(np.percentile(raw_map, self.score_percentile))
        )
        postprocessing_ms = (time.perf_counter() - postprocess_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        timings = PatchInferenceTimings(
            preprocessing_ms=preprocessing_ms,
            feature_extraction_ms=feature_extraction_ms,
            nearest_neighbor_ms=nearest_neighbor_ms,
            postprocessing_ms=postprocessing_ms,
            total_ms=total_ms,
        )
        return PatchDetectionResult(
            image_score=image_score,
            raw_map=raw_map,
            display_map=display_map,
            mask=mask,
            pixel_threshold=float(pixel_threshold),
            regions=regions,
            timings=timings,
            device=str(self.device),
        )
