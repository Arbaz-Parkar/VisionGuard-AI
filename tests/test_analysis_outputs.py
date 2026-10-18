from pathlib import Path

import numpy as np
from PIL import Image
import torch

from visionguard.analysis_outputs import save_failure_examples
from visionguard.classical_evaluation import evaluate_classical_baselines
from visionguard.comparison import combine_evaluation_reports
from visionguard.evaluation import evaluate_patch_detector
from visionguard.model import (
    MemoryBank,
    PatchAnomalyScorer,
    PatchDetectorArtifact,
    PatchThresholdCalibration,
)


class _PixelExtractor:
    device = torch.device("cpu")

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return images.permute(0, 2, 3, 1).contiguous()


def _save_image(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def test_failure_export_saves_ranked_synthetic_example_for_human_review(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    normal = np.full((8, 8, 3), 10, dtype=np.uint8)
    for index in range(4):
        _save_image(dataset_root / "bottle" / "train" / "good" / f"normal_{index}.png", normal)
    _save_image(dataset_root / "bottle" / "test" / "good" / "normal.png", normal)
    defect = normal.copy()
    defect[2:6, 2:6] = 255
    _save_image(dataset_root / "bottle" / "test" / "crack" / "part.png", defect)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    _save_image(dataset_root / "bottle" / "ground_truth" / "crack" / "part_mask.png", mask)

    extractor = _PixelExtractor()
    bank = MemoryBank(
        embeddings=torch.zeros((1, 3), dtype=torch.float32),
        source_image_count=3,
        candidate_patch_count=64,
        sampling_strategy="random",
        sampling_ratio=0.1,
        seed=11,
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
    artifact = PatchDetectorArtifact.from_components(
        category="bottle",
        memory_bank=bank,
        thresholds=PatchThresholdCalibration(
            pixel_threshold=0.5,
            image_threshold=1000.0,
            pixel_false_positive_rate=0.005,
            image_false_positive_rate=0.05,
            pixel_percentile=99.5,
            image_percentile=99.5,
            calibration_image_count=1,
            calibration_pixel_count=64,
        ),
        scorer=scorer,
    )
    patch_report = evaluate_patch_detector(
        dataset_root,
        artifact,
        extractor,
        pixel_metric_size=(8, 8),
        warmup_count=0,
    )
    classical_report = evaluate_classical_baselines(
        dataset_root,
        "bottle",
        calibration_fraction=0.25,
        calibration_seed=11,
        image_percentile=100,
        pixel_percentile=99.5,
        blur_kernel_size=1,
        local_blur_kernel_size=7,
        min_region_area=0,
        pixel_metric_size=(8, 8),
        warmup_count=0,
    )
    comparison = combine_evaluation_reports(
        patch_report,
        classical_report,
        artifact,
        patch_device="cpu",
    )

    files = save_failure_examples(
        dataset_root,
        comparison,
        artifact,
        extractor,
        calibration_fraction=0.25,
        calibration_seed=11,
        pixel_rule="percentile",
        pixel_percentile=99.5,
        blur_kernel_size=1,
        local_blur_kernel_size=7,
        min_region_area=0,
        output_dir=tmp_path / "examples",
        per_method_limit=2,
    )

    figures = [path for path in files if path.suffix == ".png"]
    assert figures
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures)
    review_notes = next(path for path in files if path.name == "failure_analysis.md")
    assert "Pending human visual review" in review_notes.read_text(encoding="utf-8")
