"""Local ORB keypoint and binary descriptor extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocessing import ColorOrder, ImageValidationError, to_grayscale


@dataclass(frozen=True)
class ORBFeatures:
    """Keypoints and their aligned 32-byte ORB descriptors.

    ``descriptors`` is ``None`` when no keypoints were detected. Otherwise,
    descriptor row ``i`` belongs to ``keypoints[i]``.
    """

    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray | None


def detect_orb_features(
    image: np.ndarray,
    *,
    n_features: int = 500,
    fast_threshold: int = 20,
    color_order: ColorOrder = "RGB",
) -> ORBFeatures:
    """Detect ORB keypoints and binary descriptors in an image.

    RGB/BGR and transparent images use the same grayscale conversion as the
    other processing helpers. ``fast_threshold`` controls how strongly a point
    must stand out from its surroundings before ORB considers it a candidate.
    """
    if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features <= 0:
        raise ImageValidationError("n_features must be a positive integer.")
    if (
        isinstance(fast_threshold, bool)
        or not isinstance(fast_threshold, int)
        or not 0 <= fast_threshold <= 255
    ):
        raise ImageValidationError("fast_threshold must be an integer from 0 through 255.")

    gray = to_grayscale(image, color_order=color_order)
    detector = cv2.ORB_create(nfeatures=n_features, fastThreshold=fast_threshold)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return ORBFeatures(
        keypoints=tuple(keypoints or ()),
        descriptors=None if descriptors is None else descriptors.copy(),
    )
