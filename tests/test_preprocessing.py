from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from visionguard.processing import (
    ImageValidationError,
    load_image,
    restore_map,
    resize_image,
    resize_mask,
    to_grayscale,
    to_rgb,
    validate_image,
)


def test_load_image_from_bytes_preserves_rgb_and_alpha():
    source = np.array([[[12, 34, 56, 128]]], dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(source).save(buffer, format="PNG")

    loaded = load_image(buffer.getvalue())

    assert loaded.shape == (1, 1, 4)
    np.testing.assert_array_equal(loaded, source)


def test_load_image_rejects_corrupt_and_oversized_inputs():
    with pytest.raises(ImageValidationError, match="Unable to load image"):
        load_image(b"not an image")

    image = Image.new("RGB", (3, 2))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    with pytest.raises(ImageValidationError, match="pixels"):
        load_image(buffer.getvalue(), max_pixels=5)


def test_validate_image_checks_shape_and_dtype():
    with pytest.raises(ImageValidationError, match="shaped"):
        validate_image(np.zeros((4, 4, 2), dtype=np.uint8))
    with pytest.raises(ImageValidationError, match="uint8"):
        validate_image(np.zeros((4, 4), dtype=np.float32))


def test_grayscale_and_rgb_conversions_respect_declared_channel_order():
    rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
    bgr = rgb[..., ::-1].copy()

    gray_rgb = to_grayscale(rgb)
    gray_bgr = to_grayscale(bgr, color_order="BGR")
    np.testing.assert_array_equal(gray_rgb, gray_bgr)
    np.testing.assert_array_equal(to_rgb(bgr, color_order="BGR"), rgb)


def test_resize_image_tracks_letterbox_geometry():
    image = np.full((2, 4, 3), 90, dtype=np.uint8)

    resized, transform = resize_image(image, (4, 4))

    assert resized.shape == (4, 4, 3)
    assert (transform.resized_height, transform.resized_width) == (2, 4)
    assert (transform.offset_y, transform.offset_x) == (1, 0)
    assert np.all(resized[0] == 0) and np.all(resized[-1] == 0)
    assert np.all(resized[1:3] == 90)


def test_resize_preserves_single_channel_array_shape():
    image = np.full((2, 4, 1), 40, dtype=np.uint8)

    resized, _ = resize_image(image, (4, 4))

    assert resized.shape == (4, 4, 1)


def test_mask_resize_uses_nearest_neighbor_and_map_restores_original_size():
    image = np.zeros((2, 4), dtype=np.uint8)
    _, transform = resize_image(image, (4, 4))
    mask = np.array([[0, 255, 255, 0], [0, 0, 255, 0]], dtype=np.uint8)

    resized_mask = resize_mask(mask, transform)
    restored = restore_map(np.full((4, 4), 0.75, dtype=np.float32), transform)

    assert resized_mask.shape == (4, 4)
    assert not resized_mask[0].any() and not resized_mask[-1].any()
    assert set(np.unique(resized_mask)).issubset({0, 255})
    assert restored.shape == image.shape
    np.testing.assert_allclose(restored, 0.75)
