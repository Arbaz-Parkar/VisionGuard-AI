import numpy as np
import pytest

from visionguard.processing import (
    ImageValidationError,
    TemplateMatchResult,
    match_template,
)


@pytest.mark.parametrize(
    "method",
    [
        "sqdiff",
        "sqdiff_normed",
        "ccorr",
        "ccorr_normed",
        "ccoeff",
        "ccoeff_normed",
    ],
)
def test_template_matching_finds_exact_patch_and_returns_response(method):
    rng = np.random.default_rng(14)
    template = rng.integers(20, 240, size=(4, 5), dtype=np.uint8)
    image = np.zeros((11, 13), dtype=np.uint8)
    image[3:7, 7:12] = template

    result = match_template(image, template, method=method)

    assert isinstance(result, TemplateMatchResult)
    assert result.response.shape == (8, 9)
    assert result.top_left == (7, 3)
    assert result.bottom_right == (12, 7)
    assert result.method == method
    if method.startswith("sqdiff"):
        assert result.score == pytest.approx(0.0)
    elif method.endswith("normed"):
        assert result.score == pytest.approx(1.0)


def test_template_matching_accepts_rgb_and_declared_bgr_order():
    rng = np.random.default_rng(5)
    template = rng.integers(0, 256, size=(3, 4, 3), dtype=np.uint8)
    image = np.zeros((9, 10, 3), dtype=np.uint8)
    image[4:7, 2:6] = template

    rgb_result = match_template(image, template)
    bgr_result = match_template(
        image[..., ::-1].copy(), template[..., ::-1].copy(), color_order="BGR"
    )

    assert rgb_result.top_left == (2, 4)
    assert bgr_result.top_left == rgb_result.top_left


@pytest.mark.parametrize("method", ["unknown", None, []])
def test_template_matching_rejects_invalid_method(method):
    image = np.zeros((4, 5), dtype=np.uint8)

    with pytest.raises(ImageValidationError, match="method"):
        match_template(image, image[:2, :2], method=method)


def test_template_matching_rejects_oversized_template():
    image = np.zeros((4, 5), dtype=np.uint8)

    with pytest.raises(ImageValidationError, match="Template dimensions"):
        match_template(image, np.zeros((5, 2), dtype=np.uint8))
