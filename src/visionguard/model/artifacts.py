"""Safe save/load helpers for patch-detector artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch

from .calibration import PatchThresholdCalibration
from .features import RESNET18_PRETRAINED_WEIGHTS
from .inference import PatchAnomalyScorer
from .memory_bank import MemoryBank, PatchFeatureExtractor

ARTIFACT_FORMAT = "visionguard.patch_detector"
ARTIFACT_SCHEMA_VERSION = 1
SUPPORTED_ARCHITECTURE = "resnet18"
SUPPORTED_FEATURE_LAYERS = ("layer2", "layer3")
PRETRAINED_WEIGHTS_ID = RESNET18_PRETRAINED_WEIGHTS


class ArtifactError(ValueError):
    """Raised when a patch-detector artifact is invalid or unsupported."""


@dataclass(frozen=True)
class PatchDetectorArtifact:
    """Portable bank, calibration, and runtime settings for one category."""

    category: str
    memory_bank: MemoryBank
    thresholds: PatchThresholdCalibration
    smoothing_sigma: float
    aggregation: str
    score_percentile: float
    nearest_neighbor_chunk_size: int
    architecture: str = SUPPORTED_ARCHITECTURE
    feature_layers: tuple[str, ...] = SUPPORTED_FEATURE_LAYERS
    pretrained_weights: str = PRETRAINED_WEIGHTS_ID

    @classmethod
    def from_components(
        cls,
        *,
        category: str,
        memory_bank: MemoryBank,
        thresholds: PatchThresholdCalibration,
        scorer: PatchAnomalyScorer,
    ) -> PatchDetectorArtifact:
        """Capture a scorer's settings alongside its source bank and cutoffs."""
        if not isinstance(scorer, PatchAnomalyScorer):
            raise ArtifactError("scorer must be a PatchAnomalyScorer.")
        if not isinstance(memory_bank, MemoryBank):
            raise ArtifactError("memory_bank must be a MemoryBank.")
        if (
            not isinstance(memory_bank.embeddings, torch.Tensor)
            or memory_bank.embeddings.ndim != 2
            or memory_bank.embeddings.shape[0] == 0
            or memory_bank.embeddings.shape[1] == 0
            or not memory_bank.embeddings.is_floating_point()
        ):
            raise ArtifactError("memory_bank must contain a non-empty floating-point tensor.")
        if not isinstance(thresholds, PatchThresholdCalibration):
            raise ArtifactError("thresholds must be a PatchThresholdCalibration.")
        bank_values = memory_bank.embeddings.detach().to(device="cpu", dtype=torch.float32)
        scorer_values = scorer.memory_embeddings.detach().to(device="cpu", dtype=torch.float32)
        if bank_values.shape != scorer_values.shape or not torch.equal(bank_values, scorer_values):
            raise ArtifactError("scorer and memory_bank must contain the same embeddings.")
        artifact = cls(
            category=category,
            memory_bank=memory_bank,
            thresholds=thresholds,
            smoothing_sigma=scorer.smoothing_sigma,
            aggregation=scorer.aggregation,
            score_percentile=scorer.score_percentile,
            nearest_neighbor_chunk_size=scorer.nearest_neighbor_chunk_size,
        )
        _validate_artifact(artifact)
        return artifact

    def build_scorer(self, feature_extractor: PatchFeatureExtractor) -> PatchAnomalyScorer:
        """Recreate runtime scoring with the supplied pretrained extractor."""
        _validate_artifact(self)
        return PatchAnomalyScorer(
            feature_extractor,
            self.memory_bank,
            input_size=self.memory_bank.input_size,
            preserve_aspect_ratio=self.memory_bank.preserve_aspect_ratio,
            smoothing_sigma=self.smoothing_sigma,
            aggregation=self.aggregation,
            score_percentile=self.score_percentile,
            nearest_neighbor_chunk_size=self.nearest_neighbor_chunk_size,
        )


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ArtifactError(f"{name} must be an integer greater than or equal to {minimum}.")
    return value


def _require_number(
    name: str,
    value: object,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    try:
        finite = math.isfinite(value) if isinstance(value, (int, float)) else False
    except OverflowError:
        finite = False
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not finite
    ):
        raise ArtifactError(f"{name} must be a finite number.")
    valid_minimum = value >= minimum if minimum_inclusive else value > minimum
    if not valid_minimum or (maximum is not None and value > maximum):
        upper = f" and at most {maximum}" if maximum is not None else ""
        qualifier = "at least" if minimum_inclusive else "greater than"
        raise ArtifactError(f"{name} must be finite, {qualifier} {minimum}{upper}.")
    return float(value)


def _validate_artifact(artifact: PatchDetectorArtifact) -> None:
    if not isinstance(artifact, PatchDetectorArtifact):
        raise ArtifactError("artifact must be a PatchDetectorArtifact.")
    if (
        not isinstance(artifact.category, str)
        or not artifact.category.strip()
        or artifact.category in {".", ".."}
        or Path(artifact.category).name != artifact.category
    ):
        raise ArtifactError("category must be a non-empty single directory name.")
    if artifact.architecture != SUPPORTED_ARCHITECTURE:
        raise ArtifactError(f"Only {SUPPORTED_ARCHITECTURE} artifacts are supported.")
    if (
        not isinstance(artifact.feature_layers, tuple)
        or artifact.feature_layers != SUPPORTED_FEATURE_LAYERS
    ):
        raise ArtifactError(
            "feature_layers must be exactly ('layer2', 'layer3') for this extractor."
        )
    if artifact.pretrained_weights != PRETRAINED_WEIGHTS_ID:
        raise ArtifactError("pretrained_weights does not match the supported extractor.")

    bank = artifact.memory_bank
    if (
        not isinstance(bank, MemoryBank)
        or not isinstance(bank.embeddings, torch.Tensor)
        or bank.embeddings.ndim != 2
        or bank.embeddings.shape[0] == 0
        or bank.embeddings.shape[1] == 0
        or not bank.embeddings.is_floating_point()
        or not torch.isfinite(bank.embeddings).all().item()
    ):
        raise ArtifactError(
            "memory_bank must contain finite [patch_count, feature_dim] embeddings."
        )
    if (
        not isinstance(bank.sampling_strategy, str)
        or bank.sampling_strategy not in {"random", "greedy"}
    ):
        raise ArtifactError("memory_bank.sampling_strategy must be 'random' or 'greedy'.")
    _require_number(
        "memory_bank.sampling_ratio",
        bank.sampling_ratio,
        minimum=0,
        maximum=1,
        minimum_inclusive=False,
    )
    _require_int("memory_bank.seed", bank.seed)
    _require_int("memory_bank.source_image_count", bank.source_image_count, minimum=1)
    _require_int(
        "memory_bank.candidate_patch_count", bank.candidate_patch_count, minimum=1
    )
    if bank.candidate_patch_count < bank.embeddings.shape[0]:
        raise ArtifactError("candidate_patch_count cannot be less than the sampled bank size.")
    if (
        not isinstance(bank.input_size, tuple)
        or len(bank.input_size) != 2
        or any(type(size) is not int or size <= 0 for size in bank.input_size)
    ):
        raise ArtifactError("memory_bank.input_size must contain two positive integers.")
    if not isinstance(bank.preserve_aspect_ratio, bool):
        raise ArtifactError("memory_bank.preserve_aspect_ratio must be a boolean.")

    if not isinstance(artifact.thresholds, PatchThresholdCalibration):
        raise ArtifactError("thresholds must be a PatchThresholdCalibration.")
    _require_number("thresholds.pixel_threshold", artifact.thresholds.pixel_threshold)
    _require_number("thresholds.image_threshold", artifact.thresholds.image_threshold)
    _require_number(
        "thresholds.pixel_false_positive_rate",
        artifact.thresholds.pixel_false_positive_rate,
        maximum=1,
        minimum_inclusive=True,
    )
    if artifact.thresholds.pixel_false_positive_rate >= 1:
        raise ArtifactError("thresholds.pixel_false_positive_rate must be less than 1.")
    _require_number(
        "thresholds.image_false_positive_rate",
        artifact.thresholds.image_false_positive_rate,
        maximum=1,
        minimum_inclusive=True,
    )
    if artifact.thresholds.image_false_positive_rate >= 1:
        raise ArtifactError("thresholds.image_false_positive_rate must be less than 1.")
    _require_number(
        "thresholds.pixel_percentile",
        artifact.thresholds.pixel_percentile,
        minimum=0,
        maximum=100,
        minimum_inclusive=False,
    )
    _require_number(
        "thresholds.image_percentile",
        artifact.thresholds.image_percentile,
        minimum=0,
        maximum=100,
        minimum_inclusive=False,
    )
    _require_int(
        "thresholds.calibration_image_count",
        artifact.thresholds.calibration_image_count,
        minimum=1,
    )
    _require_int(
        "thresholds.calibration_pixel_count",
        artifact.thresholds.calibration_pixel_count,
        minimum=1,
    )

    _require_number("smoothing_sigma", artifact.smoothing_sigma)
    if (
        not isinstance(artifact.aggregation, str)
        or artifact.aggregation not in {"max", "percentile"}
    ):
        raise ArtifactError("aggregation must be either 'max' or 'percentile'.")
    _require_number(
        "score_percentile",
        artifact.score_percentile,
        minimum=0,
        maximum=100,
        minimum_inclusive=False,
    )
    _require_int("nearest_neighbor_chunk_size", artifact.nearest_neighbor_chunk_size, minimum=1)


def _metadata(artifact: PatchDetectorArtifact) -> dict[str, object]:
    bank = artifact.memory_bank
    thresholds = artifact.thresholds
    return {
        "format": ARTIFACT_FORMAT,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "category": artifact.category,
        "feature_extractor": {
            "architecture": artifact.architecture,
            "feature_layers": list(artifact.feature_layers),
            "pretrained_weights": artifact.pretrained_weights,
            "embedding_dim": int(bank.embeddings.shape[1]),
        },
        "memory_bank": {
            "patch_count": int(bank.embeddings.shape[0]),
            "source_image_count": bank.source_image_count,
            "candidate_patch_count": bank.candidate_patch_count,
            "sampling_strategy": bank.sampling_strategy,
            "sampling_ratio": float(bank.sampling_ratio),
            "seed": bank.seed,
            "input_size": list(bank.input_size),
            "preserve_aspect_ratio": bank.preserve_aspect_ratio,
        },
        "scorer": {
            "smoothing_sigma": artifact.smoothing_sigma,
            "aggregation": artifact.aggregation,
            "score_percentile": artifact.score_percentile,
            "nearest_neighbor_chunk_size": artifact.nearest_neighbor_chunk_size,
        },
        "thresholds": {
            "pixel_threshold": thresholds.pixel_threshold,
            "image_threshold": thresholds.image_threshold,
            "pixel_false_positive_rate": thresholds.pixel_false_positive_rate,
            "image_false_positive_rate": thresholds.image_false_positive_rate,
            "pixel_percentile": thresholds.pixel_percentile,
            "image_percentile": thresholds.image_percentile,
            "calibration_image_count": thresholds.calibration_image_count,
            "calibration_pixel_count": thresholds.calibration_pixel_count,
        },
    }


def save_patch_artifact(path: str | Path, artifact: PatchDetectorArtifact) -> Path:
    """Atomically save one self-contained NPZ artifact without pickle data."""
    _validate_artifact(artifact)
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".npz":
        raise ArtifactError("Artifact path must use the .npz extension.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    metadata_bytes = json.dumps(
        _metadata(artifact), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    embeddings = (
        artifact.memory_bank.embeddings.detach().to(device="cpu", dtype=torch.float32).numpy()
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(
                stream,
                embeddings=embeddings,
                metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Unable to save artifact to {destination}: {exc}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass
    return destination


def _object(mapping: object, name: str) -> dict[str, object]:
    if not isinstance(mapping, dict):
        raise ArtifactError(f"Artifact field '{name}' must be an object.")
    return mapping


def _from_metadata(metadata: object, embeddings: np.ndarray) -> PatchDetectorArtifact:
    root = _object(metadata, "root")
    if root.get("format") != ARTIFACT_FORMAT:
        raise ArtifactError("File is not a VisionGuard patch-detector artifact.")
    schema_version = root.get("schema_version")
    if type(schema_version) is not int or schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError(f"Unsupported artifact schema version: {root.get('schema_version')}.")

    extractor = _object(root.get("feature_extractor"), "feature_extractor")
    bank_metadata = _object(root.get("memory_bank"), "memory_bank")
    scorer_metadata = _object(root.get("scorer"), "scorer")
    threshold_metadata = _object(root.get("thresholds"), "thresholds")

    if (
        embeddings.ndim != 2
        or embeddings.dtype != np.float32
        or not np.isfinite(embeddings).all()
    ):
        raise ArtifactError(
            "Artifact embeddings must be finite float32 [patch_count, feature_dim]."
        )
    patch_count = _require_int(
        "memory_bank.patch_count", bank_metadata.get("patch_count"), minimum=1
    )
    embedding_dim = _require_int(
        "feature_extractor.embedding_dim", extractor.get("embedding_dim"), minimum=1
    )
    if patch_count != embeddings.shape[0] or embedding_dim != embeddings.shape[1]:
        raise ArtifactError("Artifact embedding dimensions do not match the metadata.")

    size_value = bank_metadata.get("input_size")
    if not isinstance(size_value, list) or len(size_value) != 2:
        raise ArtifactError("memory_bank.input_size metadata must contain two integers.")
    input_size = tuple(
        _require_int("memory_bank.input_size item", item, minimum=1)
        for item in size_value
    )
    feature_layers_value = extractor.get("feature_layers")
    if not isinstance(feature_layers_value, list) or any(
        not isinstance(item, str) for item in feature_layers_value
    ):
        raise ArtifactError("feature_extractor.feature_layers metadata must be a string list.")

    memory_bank = MemoryBank(
        embeddings=torch.from_numpy(np.array(embeddings, dtype=np.float32, copy=True)),
        source_image_count=_require_int(
            "memory_bank.source_image_count",
            bank_metadata.get("source_image_count"),
            minimum=1,
        ),
        candidate_patch_count=_require_int(
            "memory_bank.candidate_patch_count",
            bank_metadata.get("candidate_patch_count"),
            minimum=1,
        ),
        sampling_strategy=bank_metadata.get("sampling_strategy"),  # validated below
        sampling_ratio=_require_number(
            "memory_bank.sampling_ratio",
            bank_metadata.get("sampling_ratio"),
            minimum=0,
            maximum=1,
            minimum_inclusive=False,
        ),
        seed=_require_int("memory_bank.seed", bank_metadata.get("seed")),
        input_size=input_size,
        preserve_aspect_ratio=bank_metadata.get("preserve_aspect_ratio"),
    )
    thresholds = PatchThresholdCalibration(
        pixel_threshold=_require_number(
            "thresholds.pixel_threshold", threshold_metadata.get("pixel_threshold")
        ),
        image_threshold=_require_number(
            "thresholds.image_threshold", threshold_metadata.get("image_threshold")
        ),
        pixel_false_positive_rate=_require_number(
            "thresholds.pixel_false_positive_rate",
            threshold_metadata.get("pixel_false_positive_rate"),
            maximum=1,
        ),
        image_false_positive_rate=_require_number(
            "thresholds.image_false_positive_rate",
            threshold_metadata.get("image_false_positive_rate"),
            maximum=1,
        ),
        pixel_percentile=_require_number(
            "thresholds.pixel_percentile",
            threshold_metadata.get("pixel_percentile"),
            minimum=0,
            maximum=100,
            minimum_inclusive=False,
        ),
        image_percentile=_require_number(
            "thresholds.image_percentile",
            threshold_metadata.get("image_percentile"),
            minimum=0,
            maximum=100,
            minimum_inclusive=False,
        ),
        calibration_image_count=_require_int(
            "thresholds.calibration_image_count",
            threshold_metadata.get("calibration_image_count"),
            minimum=1,
        ),
        calibration_pixel_count=_require_int(
            "thresholds.calibration_pixel_count",
            threshold_metadata.get("calibration_pixel_count"),
            minimum=1,
        ),
    )
    artifact = PatchDetectorArtifact(
        category=root.get("category"),
        memory_bank=memory_bank,
        thresholds=thresholds,
        smoothing_sigma=_require_number(
            "scorer.smoothing_sigma", scorer_metadata.get("smoothing_sigma")
        ),
        aggregation=scorer_metadata.get("aggregation"),
        score_percentile=_require_number(
            "scorer.score_percentile",
            scorer_metadata.get("score_percentile"),
            minimum=0,
            maximum=100,
            minimum_inclusive=False,
        ),
        nearest_neighbor_chunk_size=_require_int(
            "scorer.nearest_neighbor_chunk_size",
            scorer_metadata.get("nearest_neighbor_chunk_size"),
            minimum=1,
        ),
        architecture=extractor.get("architecture"),
        feature_layers=tuple(feature_layers_value),
        pretrained_weights=extractor.get("pretrained_weights"),
    )
    _validate_artifact(artifact)
    return artifact


def load_patch_artifact(path: str | Path) -> PatchDetectorArtifact:
    """Load and validate a project NPZ artifact with pickle disabled."""
    source = Path(path).expanduser()
    if source.suffix.lower() != ".npz":
        raise ArtifactError("Artifact path must use the .npz extension.")
    if not source.is_file():
        raise ArtifactError(f"Artifact file does not exist: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != {"embeddings", "metadata"} or len(archive.files) != 2:
                raise ArtifactError("Artifact archive must contain embeddings and metadata only.")
            metadata_bytes = archive["metadata"]
            if (
                metadata_bytes.ndim != 1
                or metadata_bytes.dtype != np.uint8
                or metadata_bytes.size > 1_000_000
            ):
                raise ArtifactError("Artifact metadata must be a uint8 vector no larger than 1 MB.")
            metadata = json.loads(metadata_bytes.tobytes().decode("utf-8"))
            embeddings = np.array(archive["embeddings"], copy=True)
        return _from_metadata(metadata, embeddings)
    except ArtifactError:
        raise
    except (
        OSError,
        ValueError,
        KeyError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        raise ArtifactError(f"Unable to load artifact from {source}: {exc}") from exc
