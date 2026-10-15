import cv2
import numpy as np
import pytest

from visionguard.processing import ImageValidationError, detect_orb_features


def test_orb_returns_keypoints_with_aligned_binary_descriptors():
    image = np.random.default_rng(37).integers(0, 256, size=(128, 128), dtype=np.uint8)

    features = detect_orb_features(image, n_features=150)

    assert 0 < len(features.keypoints) <= 150
    assert features.descriptors is not None
    assert features.descriptors.shape == (len(features.keypoints), 32)
    assert features.descriptors.dtype == np.uint8


def test_orb_returns_empty_result_for_uniform_image():
    image = np.full((80, 80), 127, dtype=np.uint8)

    features = detect_orb_features(image)

    assert features.keypoints == ()
    assert features.descriptors is None


def test_orb_rgb_and_equivalent_bgr_inputs_produce_same_features():
    gray = np.random.default_rng(91).integers(0, 256, size=(96, 96), dtype=np.uint8)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    rgb_features = detect_orb_features(rgb, color_order="RGB")
    bgr_features = detect_orb_features(bgr, color_order="BGR")

    assert [point.pt for point in rgb_features.keypoints] == [
        point.pt for point in bgr_features.keypoints
    ]
    np.testing.assert_array_equal(rgb_features.descriptors, bgr_features.descriptors)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_features": 0},
        {"n_features": True},
        {"fast_threshold": -1},
        {"fast_threshold": 256},
        {"fast_threshold": True},
        {"color_order": "XYZ"},
    ],
)
def test_orb_rejects_invalid_options(kwargs):
    with pytest.raises(ImageValidationError):
        detect_orb_features(np.zeros((32, 32), dtype=np.uint8), **kwargs)
