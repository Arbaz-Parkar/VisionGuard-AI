import cv2
import numpy as np
import pytest

from visionguard.processing import (
    ImageValidationError,
    fft_magnitude_spectrum,
    frequency_filter,
)


def _opencv_frequency_reference(image, pass_type, filter_type, cutoff, order):
    height, width = image.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fx**2 + fy**2)
    if filter_type == "ideal":
        response = (radius <= cutoff).astype(np.float64)
    elif filter_type == "gaussian":
        response = np.exp(-0.5 * (radius / cutoff) ** 2)
    else:
        response = 1.0 / (1.0 + (radius / cutoff) ** (2 * order))
    if pass_type == "highpass":
        response = 1.0 - response

    spectrum = cv2.dft(image.astype(np.float64), flags=cv2.DFT_COMPLEX_OUTPUT)
    spectrum[..., 0] *= response
    spectrum[..., 1] *= response
    return cv2.idft(
        spectrum,
        flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT,
    )


def test_log_scaled_fft_spectrum_is_centered_shaped_and_finite():
    image = np.zeros((16, 20), dtype=np.float32)
    image[4:12, 6:14] = 1.0

    spectrum = fft_magnitude_spectrum(image)
    raw_spectrum = fft_magnitude_spectrum(image, log_scale=False)

    assert spectrum.shape == image.shape
    assert np.isfinite(spectrum).all()
    assert np.all(spectrum <= raw_spectrum)
    assert np.unravel_index(np.argmax(spectrum), spectrum.shape) == (8, 10)


@pytest.mark.parametrize(
    "pass_type, filter_type, cutoff, order",
    [
        ("lowpass", "gaussian", 0.12, 2),
        ("highpass", "butterworth", 0.18, 3),
        ("lowpass", "ideal", 0.2, 2),
    ],
)
def test_numpy_frequency_filter_matches_opencv_dft(
    pass_type, filter_type, cutoff, order
):
    rng = np.random.default_rng(17)
    image = rng.normal(size=(15, 18)).astype(np.float64)

    actual = frequency_filter(
        image,
        pass_type=pass_type,
        filter_type=filter_type,
        cutoff_frequency=cutoff,
        order=order,
    )
    expected = _opencv_frequency_reference(
        image, pass_type, filter_type, cutoff, order
    )

    assert actual.shape == image.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_highpass_removes_constant_image_and_lowpass_keeps_it():
    image = np.full((12, 14), 8.0, dtype=np.float64)

    highpass = frequency_filter(image, pass_type="highpass")
    lowpass = frequency_filter(image, pass_type="lowpass")

    np.testing.assert_allclose(highpass, 0.0, atol=1e-12)
    np.testing.assert_allclose(lowpass, image, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pass_type": "bandpass"},
        {"filter_type": "median"},
        {"cutoff_frequency": 0},
        {"cutoff_frequency": 0.6},
        {"order": 0},
    ],
)
def test_frequency_filter_rejects_invalid_options(kwargs):
    with pytest.raises(ImageValidationError):
        frequency_filter(np.zeros((8, 8)), **kwargs)


def test_fft_helpers_reject_color_or_non_finite_images():
    with pytest.raises(ImageValidationError, match="2D"):
        fft_magnitude_spectrum(np.zeros((4, 4, 3)))
    with pytest.raises(ImageValidationError, match="finite"):
        frequency_filter(np.full((4, 4), np.nan))
