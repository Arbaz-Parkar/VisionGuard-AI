"""Classical image-processing helpers used by VisionGuard AI."""

from .filtering import apply_clahe, gaussian_blur, median_blur
from .histograms import compute_histograms, plot_histograms
from .edges import canny_edges
from .thresholding import adaptive_threshold, global_threshold, otsu_threshold
from .morphology import morphology
from .contours import CandidateRegion, extract_candidate_regions
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
    "CandidateRegion",
    "ResizeTransform",
    "apply_clahe",
    "adaptive_threshold",
    "canny_edges",
    "extract_candidate_regions",
    "compute_histograms",
    "gaussian_blur",
    "global_threshold",
    "load_image",
    "median_blur",
    "morphology",
    "plot_histograms",
    "otsu_threshold",
    "restore_map",
    "resize_image",
    "resize_mask",
    "to_grayscale",
    "to_rgb",
    "validate_image",
]
