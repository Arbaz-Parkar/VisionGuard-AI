"""Reproducible Gaussian and impulse-noise helpers for Lab experiments."""

from __future__ import annotations

import math

import numpy as np

from .preprocessing import ImageValidationError, validate_image


def _rng(seed: int | None) -> np.random.Generator:
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ImageValidationError("seed must be a non-negative integer or None.")
    return np.random.default_rng(seed)


def add_gaussian_noise(
    image: np.ndarray,
    *,
    sigma: float = 20.0,
    seed: int | None = None,
) -> np.ndarray:
    """Add independent Gaussian noise, clip to uint8, and preserve alpha.

    ``sigma`` is the noise standard deviation in the 0–255 intensity scale.
    A seed makes the generated noise repeatable for a Lab demonstration.
    """
    pixels = validate_image(image)
    if (
        isinstance(sigma, bool)
        or not isinstance(sigma, (int, float))
        or not math.isfinite(sigma)
        or not 0 <= sigma <= 255
    ):
        raise ImageValidationError("sigma must be a finite number from 0 through 255.")
    rng = _rng(seed)
    result = pixels.copy()
    target = result[..., :3] if result.ndim == 3 and result.shape[2] == 4 else result
    noise = rng.normal(0.0, float(sigma), size=target.shape)
    target[...] = np.clip(np.rint(target.astype(np.float32) + noise), 0, 255).astype(np.uint8)
    return result


def add_salt_pepper_noise(
    image: np.ndarray,
    *,
    amount: float = 0.05,
    salt_probability: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Replace a random fraction of pixels with salt or pepper values.

    The same selected pixels are changed across color channels. Four-channel
    images retain their original alpha values.
    """
    pixels = validate_image(image)
    for name, value in (("amount", amount), ("salt_probability", salt_probability)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ImageValidationError(f"{name} must be a finite number from 0 through 1.")
    rng = _rng(seed)
    result = pixels.copy()
    target = result[..., :3] if result.ndim == 3 and result.shape[2] == 4 else result

    random_values = rng.random(target.shape[:2])
    salt_mask = random_values < amount * salt_probability
    pepper_mask = (random_values >= amount * salt_probability) & (random_values < amount)
    target[salt_mask] = 255
    target[pepper_mask] = 0
    return result
