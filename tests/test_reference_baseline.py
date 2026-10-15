import numpy as np
import pytest

from visionguard.processing import (
    ImageValidationError,
    calibrate_reference_threshold,
    fit_per_pixel_reference,
    run_reference_baseline,
)


def test_fit_uses_population_statistics_from_training_images():
    first = np.full((4, 5), 10, dtype=np.uint8)
    second = np.full((4, 5), 14, dtype=np.uint8)

    model = fit_per_pixel_reference([first, second])

    assert model.sample_count == 2
    assert model.mean.shape == (4, 5, 3)
    assert model.mean.dtype == np.float32
    assert model.standard_deviation.dtype == np.float32
    np.testing.assert_allclose(model.mean, 12)
    np.testing.assert_allclose(model.standard_deviation, 2)


def test_calibrated_reference_detects_deviation_with_zero_training_std():
    normal = np.full((12, 13, 3), 80, dtype=np.uint8)
    defect = normal.copy()
    defect[5:8, 7:10, 0] = 200
    model = fit_per_pixel_reference([normal])
    threshold = calibrate_reference_threshold(model, [normal, normal.copy()])

    result = run_reference_baseline(
        defect, model, pixel_threshold=threshold, min_region_area=0
    )

    assert threshold == 0
    assert result.score_map.shape == normal.shape[:2]
    assert result.score_map.dtype == np.float32
    assert np.isfinite(result.score_map).all()
    assert result.image_score == 120
    assert np.all(result.mask[5:8, 7:10] == 255)
    assert result.mask.sum() == 9 * 255
    assert len(result.regions) == 1


def test_reference_model_respects_training_color_order():
    rgb = np.zeros((8, 9, 3), dtype=np.uint8)
    rgb[..., 0] = 30
    bgr = rgb[..., ::-1].copy()

    model = fit_per_pixel_reference([bgr], color_order="BGR")
    result = run_reference_baseline(
        bgr, model, pixel_threshold=1, min_region_area=0
    )

    assert result.image_score == 0
    assert not result.mask.any()
    np.testing.assert_allclose(model.mean, rgb)


@pytest.mark.parametrize(
    "images",
    [[], [np.zeros((4, 4), dtype=np.uint8), np.zeros((3, 4), dtype=np.uint8)]],
)
def test_fit_rejects_empty_or_mismatched_training_images(images):
    with pytest.raises(ImageValidationError):
        fit_per_pixel_reference(images)


def test_calibration_requires_normal_images_and_valid_percentile():
    model = fit_per_pixel_reference([np.zeros((4, 4), dtype=np.uint8)])

    with pytest.raises(ImageValidationError, match="At least one"):
        calibrate_reference_threshold(model, [])
    with pytest.raises(ImageValidationError, match="percentile"):
        calibrate_reference_threshold(model, [np.zeros((4, 4), dtype=np.uint8)], percentile=0)


def test_reference_scoring_rejects_shape_mismatch_and_invalid_threshold():
    model = fit_per_pixel_reference([np.zeros((4, 4), dtype=np.uint8)])

    with pytest.raises(ImageValidationError, match="does not match"):
        run_reference_baseline(np.zeros((5, 4), dtype=np.uint8), model, pixel_threshold=1)
    with pytest.raises(ImageValidationError, match="pixel_threshold"):
        run_reference_baseline(np.zeros((4, 4), dtype=np.uint8), model, pixel_threshold=-1)
