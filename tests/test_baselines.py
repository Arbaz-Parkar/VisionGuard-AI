import numpy as np
import pytest

from visionguard.processing import ImageValidationError, run_rule_based_baseline


def test_rule_based_baseline_returns_continuous_map_mask_and_regions():
    image = np.full((96, 96), 110, dtype=np.uint8)
    image[36:60, 38:62] = 245

    result = run_rule_based_baseline(image)

    assert result.score_map.shape == image.shape
    assert result.score_map.dtype == np.float32
    assert np.isfinite(result.score_map).all()
    assert result.image_score == float(result.score_map.max())
    assert result.image_score > 0
    assert result.mask.shape == image.shape
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)).issubset({0, 255})
    assert len(result.regions) >= 1
    assert result.threshold >= 0


def test_rule_based_baseline_handles_constant_image_without_nan():
    image = np.full((32, 40), 90, dtype=np.uint8)

    result = run_rule_based_baseline(image)

    assert result.image_score == 0.0
    assert not result.score_map.any()
    assert not result.mask.any()
    assert result.regions == ()
    assert np.isfinite(result.score_map).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"blur_kernel_size": 2},
        {"local_blur_kernel_size": 3},
        {"morphology_kernel_size": 0},
        {"min_region_area": -1},
        {"min_region_area": float("nan")},
        {"color_order": "XYZ"},
    ],
)
def test_rule_based_baseline_rejects_invalid_parameters(kwargs):
    with pytest.raises(ImageValidationError):
        run_rule_based_baseline(np.zeros((32, 32), dtype=np.uint8), **kwargs)
