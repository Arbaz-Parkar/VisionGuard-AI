import numpy as np
import pytest

from visionguard.processing import ImageValidationError, morphology


def test_dilation_expands_a_single_foreground_pixel():
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[3, 3] = 255

    dilated = morphology(mask, "dilate", kernel_size=3)

    assert np.count_nonzero(dilated) == 9
    assert set(np.unique(dilated)).issubset({0, 255})


def test_erosion_shrinks_a_foreground_region():
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[2:7, 2:7] = 255

    eroded = morphology(mask, "erode", kernel_size=3)

    assert np.count_nonzero(eroded) == 9
    assert eroded[4, 4] == 255


def test_opening_removes_small_specks_and_closing_fills_small_holes():
    mask_with_speck = np.zeros((11, 11), dtype=np.uint8)
    mask_with_speck[2:9, 2:9] = 255
    mask_with_speck[0, 0] = 255
    mask_with_hole = mask_with_speck.copy()
    mask_with_hole[5, 5] = 0

    opened = morphology(mask_with_speck, "open", kernel_size=3)
    closed = morphology(mask_with_hole, "close", kernel_size=3)

    assert opened[0, 0] == 0
    assert opened[5, 5] == 255
    assert closed[5, 5] == 255


def test_gradient_returns_boundary_pixels():
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[2:7, 2:7] = 255

    gradient = morphology(mask, "gradient", kernel_size=3)

    assert gradient.shape == mask.shape
    assert np.count_nonzero(gradient) > 0
    assert gradient[2, 4] == 255
    assert gradient[4, 4] == 0
    assert set(np.unique(gradient)).issubset({0, 255})


def test_top_hat_extracts_a_bright_detail_from_grayscale_image():
    image = np.full((15, 15), 30, dtype=np.uint8)
    image[7, 7] = 220

    result = morphology(image, "tophat", kernel_size=3)

    assert result.shape == image.shape
    assert result[7, 7] == 190
    assert np.count_nonzero(result) == 1


@pytest.mark.parametrize(
    "mask, operation, kwargs",
    [
        (np.zeros((4, 4, 3), dtype=np.uint8), "open", {}),
        (np.full((4, 4), 1, dtype=np.uint8), "open", {}),
        (np.zeros((4, 4), dtype=np.uint8), "unknown", {}),
        (np.zeros((4, 4), dtype=np.uint8), "erode", {"kernel_size": 2}),
        (np.zeros((4, 4), dtype=np.uint8), "dilate", {"kernel_shape": "diamond"}),
        (np.zeros((4, 4), dtype=np.uint8), "close", {"iterations": 0}),
    ],
)
def test_morphology_rejects_invalid_mask_or_parameters(mask, operation, kwargs):
    with pytest.raises(ImageValidationError):
        morphology(mask, operation, **kwargs)
