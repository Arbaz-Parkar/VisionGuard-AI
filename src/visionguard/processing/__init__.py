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
from .features import ORBFeatures, detect_orb_features
from .baselines import ClassicalDetectionResult, run_rule_based_baseline
from .reference_baseline import (
    PerPixelReferenceModel,
    calibrate_reference_threshold,
    fit_per_pixel_reference,
    run_reference_baseline,
)

__all__ = [
    "ImageValidationError",
    "CandidateRegion",
    "ClassicalDetectionResult",
    "PerPixelReferenceModel",
    "ResizeTransform",
    "TemplateMatchResult",
    "TemplateMethod",
    "FlipDirection",
    "ORBFeatures",
    "apply_clahe",
    "adaptive_threshold",
    "add_gaussian_noise",
    "add_salt_pepper_noise",
    "canny_edges",
    "calibrate_reference_threshold",
    "convolve2d",
    "extract_candidate_regions",
    "detect_orb_features",
    "fft_magnitude_spectrum",
    "flip_image",
    "frequency_filter",
    "fit_per_pixel_reference",
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
    "run_rule_based_baseline",
    "run_reference_baseline",
    "sharpen_image",
    "to_grayscale",
    "to_rgb",
    "translate_image",
    "validate_image",
]
