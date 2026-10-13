"""Classical image-processing helpers used by VisionGuard AI."""

from .filtering import apply_clahe, gaussian_blur, median_blur
from .histograms import compute_histograms, plot_histograms
from .edges import canny_edges
from .preprocessing import (
    ImageValidationError,
    ResizeTransform,
    load_image,
    restore_map,
    resize_image,
    resize_mask,
    to_grayscale,
    to_rgb,
    validate_image,
)

__all__ = [
    "ImageValidationError",
    "ResizeTransform",
    "apply_clahe",
    "canny_edges",
    "compute_histograms",
    "gaussian_blur",
    "load_image",
    "median_blur",
    "plot_histograms",
    "restore_map",
    "resize_image",
    "resize_mask",
    "to_grayscale",
    "to_rgb",
    "validate_image",
]
