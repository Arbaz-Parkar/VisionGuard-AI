from pathlib import Path

from visionguard.data.validation import discover_categories, validate_category


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def test_valid_mvtec_layout_and_mask_pairing(tmp_path: Path) -> None:
    category_root = tmp_path / "bottle"
    _touch(category_root / "train" / "good" / "normal.png")
    _touch(category_root / "test" / "good" / "normal.png")
    _touch(category_root / "test" / "crack" / "defect.png")
    _touch(category_root / "ground_truth" / "crack" / "defect_mask.png")
    assert discover_categories(tmp_path) == ["bottle"]
    report = validate_category(tmp_path, "bottle")
    assert report["valid"]
    assert report["normal_training_image_count"] == 1
    assert report["test_image_count"] == 2
    assert report["missing_mask_count"] == 0
    assert report["mask_coverage_complete"]


def test_missing_ground_truth_mask_is_reported(tmp_path: Path) -> None:
    category_root = tmp_path / "bottle"
    _touch(category_root / "train" / "good" / "normal.png")
    _touch(category_root / "test" / "good" / "normal.png")
    _touch(category_root / "test" / "crack" / "defect.png")
    report = validate_category(tmp_path, "bottle")
    assert report["valid"]
    assert report["missing_mask_count"] == 1
    assert not report["mask_coverage_complete"]
    assert report["warnings"]


def test_missing_dataset_is_reported(tmp_path: Path) -> None:
    report = validate_category(tmp_path / "missing", "bottle")
    assert not report["valid"]
    assert not report["mask_coverage_complete"]
    assert report["errors"]


def test_unsafe_category_is_rejected(tmp_path: Path) -> None:
    report = validate_category(tmp_path, "../outside")
    assert not report["valid"]
    assert "single directory name" in report["errors"][0]
