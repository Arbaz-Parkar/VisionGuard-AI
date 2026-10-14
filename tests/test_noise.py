import numpy as np
import pytest

from visionguard.processing import (
    ImageValidationError,
    add_gaussian_noise,
    add_salt_pepper_noise,
)


def test_gaussian_noise_is_seeded_clipped_and_keeps_alpha():
    image = np.full((12, 10, 4), 120, dtype=np.uint8)
    image[..., 3] = 173

    first = add_gaussian_noise(image, sigma=30, seed=19)
    second = add_gaussian_noise(image, sigma=30, seed=19)

    np.testing.assert_array_equal(first, second)
    assert first.shape == image.shape
    assert first.dtype == image.dtype
    assert first[..., :3].min() >= 0 and first[..., :3].max() <= 255
    np.testing.assert_array_equal(first[..., 3], image[..., 3])


def test_zero_gaussian_noise_returns_an_unchanged_copy():
    image = np.arange(20, dtype=np.uint8).reshape(4, 5)

    result = add_gaussian_noise(image, sigma=0, seed=1)

    np.testing.assert_array_equal(result, image)
    assert result is not image


def test_salt_and_pepper_noise_changes_whole_color_pixels_reproducibly():
    image = np.full((5, 6, 4), 100, dtype=np.uint8)
    image[..., 3] = 210

    noisy = add_salt_pepper_noise(
        image, amount=1.0, salt_probability=0.5, seed=23
    )

    colors = noisy[..., :3]
    assert set(np.unique(colors)).issubset({0, 255})
    assert np.any(np.all(colors == 0, axis=2))
    assert np.any(np.all(colors == 255, axis=2))
    assert np.all((colors == 0) | (colors == 255))
    np.testing.assert_array_equal(noisy[..., 3], image[..., 3])


def test_zero_salt_pepper_amount_returns_unchanged_pixels():
    image = np.full((4, 4), 128, dtype=np.uint8)

    noisy = add_salt_pepper_noise(image, amount=0, seed=5)

    np.testing.assert_array_equal(noisy, image)


@pytest.mark.parametrize(
    "call",
    [
        lambda image: add_gaussian_noise(image, sigma=-1),
        lambda image: add_gaussian_noise(image, sigma=256),
        lambda image: add_gaussian_noise(image, seed=-1),
        lambda image: add_salt_pepper_noise(image, amount=1.1),
        lambda image: add_salt_pepper_noise(image, salt_probability=float("nan")),
    ],
)
def test_noise_helpers_reject_invalid_parameters(call):
    with pytest.raises(ImageValidationError):
        call(np.zeros((5, 5), dtype=np.uint8))
