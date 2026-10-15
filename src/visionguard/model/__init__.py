"""Patch-based anomaly detector components."""

from .features import ModelSetupError, ResNet18PatchFeatureExtractor
from .memory_bank import MemoryBank, build_memory_bank, sample_patch_embeddings
from .inference import PatchAnomalyScorer, PatchDetectionResult, PatchInferenceTimings
from .calibration import PatchThresholdCalibration, calibrate_patch_thresholds
from .artifacts import (
    ArtifactError,
    PatchDetectorArtifact,
    load_patch_artifact,
    save_patch_artifact,
)

__all__ = [
    "MemoryBank",
    "ArtifactError",
    "ModelSetupError",
    "PatchAnomalyScorer",
    "PatchDetectionResult",
    "PatchDetectorArtifact",
    "PatchInferenceTimings",
    "PatchThresholdCalibration",
    "ResNet18PatchFeatureExtractor",
    "build_memory_bank",
    "calibrate_patch_thresholds",
    "load_patch_artifact",
    "save_patch_artifact",
    "sample_patch_embeddings",
]
