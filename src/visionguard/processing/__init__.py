"""Classical image-processing helpers used by VisionGuard AI."""

from .filtering import apply_clahe, gaussian_blur, median_blur, sharpen_image
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
from .convolution import convolve2d
from .frequency import fft_magnitude_spectrum, frequency_filter
from .noise import add_gaussian_noise, add_salt_pepper_noise
from .template_matching import TemplateMatchResult, TemplateMethod, match_template
from .transformations import FlipDirection, flip_image, rotate_image, translate_image

__all__ = [
    "ImageValidationError",
    "CandidateRegion",
    "ResizeTransform",
    "TemplateMatchResult",
    "TemplateMethod",
    "FlipDirection",
    "apply_clahe",
    "adaptive_threshold",
    "add_gaussian_noise",
    "add_salt_pepper_noise",
    "canny_edges",
    "convolve2d",
    "extract_candidate_regions",
    "fft_magnitude_spectrum",
    "flip_image",
    "frequency_filter",
    "compute_histograms",
    "gaussian_blur",
    "global_threshold",
    "load_image",
    "median_blur",
    "match_template",
    "morphology",
    "plot_histograms",
    "otsu_threshold",
    "restore_map",
    "rotate_image",
    "resize_image",
    "resize_mask",
    "sharpen_image",
    "to_grayscale",
    "to_rgb",
    "translate_image",
    "validate_image",
]
