import cv2
import numpy as np
import pytest

from visionguard.processing import (
    ImageValidationError,
    flip_image,
    rotate_image,
    translate_image,
)


def test_flip_image_supports_each_direction():
    image = np.arange(6, dtype=np.uint8).reshape(2, 3)

    np.testing.assert_array_equal(
        flip_image(image), np.array([[2, 1, 0], [5, 4, 3]], dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        flip_image(image, direction="vertical"),
        np.array([[3, 4, 5], [0, 1, 2]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        flip_image(image, direction="both"),
        np.array([[5, 4, 3], [2, 1, 0]], dtype=np.uint8),
    )


def test_rotate_image_can_expand_canvas_for_right_angle():
    image = np.arange(6, dtype=np.uint8).reshape(2, 3)

    rotated = rotate_image(
        image, 90, expand=True, interpolation=cv2.INTER_NEAREST
    )

    assert rotated.shape == (3, 2)
    np.testing.assert_array_equal(
        rotated, np.array([[2, 5], [1, 4], [0, 3]], dtype=np.uint8)
    )


def test_rotation_preserves_canvas_by_default_and_single_channel_shape():
    image = np.zeros((5, 7, 1), dtype=np.uint8)
    image[2, 5, 0] = 255

    rotated = rotate_image(image, 45, interpolation=cv2.INTER_NEAREST)

    assert rotated.shape == image.shape
    assert rotated.dtype == image.dtype


def test_translation_shifts_pixels_and_fills_empty_area():
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)

    shifted = translate_image(
        image,
        1,
        -1,
        interpolation=cv2.INTER_NEAREST,
        border_value=99,
    )

    assert shifted.shape == image.shape
    np.testing.assert_array_equal(shifted[0], [99, 4, 5, 6])
    np.testing.assert_array_equal(shifted[1], [99, 8, 9, 10])
    np.testing.assert_array_equal(shifted[2], [99, 99, 99, 99])


@pytest.mark.parametrize(
    "call",
    [
        lambda image: flip_image(image, direction="diagonal"),
        lambda image: rotate_image(image, float("inf")),
        lambda image: rotate_image(image, 20, expand=1),
        lambda image: rotate_image(image, 20, interpolation=999),
        lambda image: translate_image(image, float("nan"), 0),
        lambda image: translate_image(image, 0, 0, border_value=256),
    ],
)
def test_transform_helpers_reject_invalid_options(call):
    with pytest.raises(ImageValidationError):
        call(np.zeros((5, 6), dtype=np.uint8))
