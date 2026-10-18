"""Streamlit page for hands-on classical image-processing experiments."""

from __future__ import annotations

import hashlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from visionguard.processing import (
    ImageValidationError,
    adaptive_threshold,
    add_gaussian_noise,
    add_salt_pepper_noise,
    apply_clahe,
    canny_edges,
    compute_histograms,
    contrast_stretch,
    convolve2d,
    detect_orb_features,
    equalize_histogram,
    extract_candidate_regions,
    fft_magnitude_spectrum,
    frequency_filter,
    gamma_transform,
    gaussian_blur,
    global_threshold,
    harris_corners,
    hog_features,
    hough_circles,
    hough_lines,
    laplacian_edges,
    laplacian_of_gaussian,
    laplacian_sharpen,
    log_transform,
    match_template,
    median_blur,
    morphology,
    otsu_threshold,
    sharpen_image,
    sobel_edges,
    to_grayscale,
    to_hsv,
    to_lab,
    to_rgb,
    watershed_segmentation,
)


_SECTIONS = {
    "Color spaces": ("Grayscale", "RGB conversion", "HSV channels", "Lab channels"),
    "Intensity and histograms": (
        "Log transform", "Gamma transform", "Contrast stretch",
        "Histogram equalization", "CLAHE", "Histogram",
    ),
    "Smoothing and sharpening": (
        "Gaussian smoothing", "Median filtering", "Gaussian denoising", "Median denoising",
        "Laplacian sharpening", "Unsharp sharpening", "Convolution",
    ),
    "Noise and frequency": (
        "Gaussian noise", "Salt-and-pepper noise", "FFT spectrum", "FFT filter",
    ),
    "Template matching": ("Template matching",),
    "Edges and morphology": (
        "Sobel edges", "Laplacian edges", "LoG edges", "Canny edges",
        "Erode", "Dilate", "Open", "Close", "Top-hat",
    ),
    "Thresholding and segmentation": (
        "Global threshold", "Otsu threshold", "Adaptive threshold",
        "Connected regions", "Watershed",
    ),
    "Features": ("Harris corners", "HOG features", "Hough lines", "Hough circles", "ORB features"),
}

_DESCRIPTIONS = {
    "Grayscale": "Convert the image to one intensity channel using its declared RGB/BGR order.",
    "RGB conversion": "Convert grayscale or declared BGR/RGB channels into a displayable RGB image.",
    "HSV channels": "Convert to HSV and inspect one encoded hue, saturation, or value channel.",
    "Lab channels": "Convert to CIELAB and inspect one encoded lightness or chroma channel.",
    "Log transform": "Expand dark intensity values with a logarithmic point transform.",
    "Gamma transform": "Adjust brightness with a power-law intensity curve.",
    "Contrast stretch": "Map a selected intensity interval linearly to the full 0–255 range.",
    "Histogram equalization": "Equalize grayscale intensity or color-image lightness in CIELAB.",
    "CLAHE": "Enhance local contrast while limiting noise amplification in small regions.",
    "Histogram": "Count pixels at each intensity value for every image channel.",
    "Gaussian smoothing": "Reduce high-frequency detail with a Gaussian-weighted neighborhood.",
    "Median filtering": "Suppress isolated impulse noise with a neighborhood median.",
    "Gaussian denoising": "Reduce additive Gaussian noise by averaging nearby intensities with Gaussian weights.",
    "Median denoising": "Reduce salt-and-pepper noise by replacing each pixel with its neighborhood median.",
    "Laplacian sharpening": "Emphasize second-order detail by subtracting a scaled Laplacian response.",
    "Unsharp sharpening": "Add a scaled high-frequency residual back to the original image.",
    "Convolution": "Apply a spatial kernel to grayscale pixels using same-size padding.",
    "Gaussian noise": "Add repeatable zero-mean Gaussian noise to the image.",
    "Salt-and-pepper noise": "Replace a selected fraction of pixels with black or white impulses.",
    "FFT spectrum": "Display the centered log magnitude of the two-dimensional Fourier transform.",
    "FFT filter": "Keep or suppress spatial frequencies with a radial frequency-domain filter.",
    "Template matching": "Search for a selected image patch and outline its best match.",
    "Sobel edges": "Estimate horizontal and vertical intensity gradients with Sobel kernels.",
    "Laplacian edges": "Highlight rapid second-order intensity changes with the Laplacian.",
    "LoG edges": "Smooth with a Gaussian before measuring second-order intensity changes.",
    "Canny edges": "Find thin edges using gradients, non-maximum suppression, and hysteresis.",
    "Erode": "Shrink foreground regions in a binary mask using a structuring element.",
    "Dilate": "Expand foreground regions in a binary mask using a structuring element.",
    "Open": "Remove small foreground specks with erosion followed by dilation.",
    "Close": "Fill small mask gaps with dilation followed by erosion.",
    "Top-hat": "Extract bright details smaller than the selected structuring element.",
    "Global threshold": "Classify pixels above or below one fixed intensity cutoff.",
    "Otsu threshold": "Choose one global intensity cutoff by minimizing within-class variance.",
    "Adaptive threshold": "Choose a local cutoff from a neighborhood around each pixel.",
    "Connected regions": "Threshold the image, then report external contour bounds and areas.",
    "Watershed": "Grow foreground markers into surrounding regions and mark their boundaries.",
    "Harris corners": "Detect points with strong intensity changes in two directions.",
    "HOG features": "Describe local gradient orientations with normalized cell-block histograms.",
    "Hough lines": "Vote for straight-line candidates in an edge map.",
    "Hough circles": "Vote for circular shapes in a smoothed grayscale edge map.",
    "ORB features": "Detect oriented keypoints and compact binary descriptors.",
}


def _controls(operation: str, image: np.ndarray) -> dict[str, object]:
    params: dict[str, object] = {}
    if operation in ("HSV channels", "Lab channels"):
        channels = ("H", "S", "V") if operation == "HSV channels" else ("L", "a", "b")
        params["channel"] = st.selectbox("Channel to display", channels, key="lab_channel")
    elif operation == "Gamma transform":
        params["gamma"] = st.slider("Gamma", 0.1, 4.0, 1.4, 0.1, key="lab_gamma")
    elif operation == "Contrast stretch":
        lower, upper = st.columns(2)
        minimum = lower.slider("Input minimum", 0, 254, 20, key="lab_stretch_min")
        maximum = upper.slider("Input maximum", minimum + 1, 255, max(235, minimum + 1), key="lab_stretch_max")
        params.update(input_min=minimum, input_max=maximum)
    elif operation == "CLAHE":
        params["clip_limit"] = st.slider("Clip limit", 0.5, 8.0, 2.0, 0.5, key="lab_clahe_clip")
        params["tile"] = st.slider("Tiles per side", 2, 16, 8, key="lab_clahe_tile")
    elif operation in ("Gaussian smoothing", "Gaussian denoising"):
        key_suffix = "denoise" if operation == "Gaussian denoising" else "smooth"
        params["kernel_size"] = st.select_slider("Kernel size", options=[3, 5, 7, 9, 11], value=5, key=f"lab_gaussian_kernel_{key_suffix}")
        params["sigma"] = st.slider("Sigma (0 = automatic)", 0.0, 5.0, 0.0, 0.1, key=f"lab_gaussian_sigma_{key_suffix}")
    elif operation in ("Median filtering", "Median denoising"):
        key_suffix = "denoise" if operation == "Median denoising" else "filter"
        params["kernel_size"] = st.select_slider("Kernel size", options=[3, 5, 7, 9, 11], value=5, key=f"lab_median_kernel_{key_suffix}")
    elif operation == "Unsharp sharpening":
        params["amount"] = st.slider("Amount", 0.0, 5.0, 1.0, 0.1, key="lab_sharp_amount")
        params["sigma"] = st.slider("Blur sigma", 0.1, 5.0, 1.0, 0.1, key="lab_sharp_sigma")
    elif operation == "Laplacian sharpening":
        params["amount"] = st.slider("Amount", 0.0, 2.0, 0.5, 0.1, key="lab_laplacian_amount")
        params["aperture_size"] = st.selectbox("Kernel aperture", [1, 3, 5, 7], index=1, key="lab_laplacian_sharp_aperture")
    elif operation == "Convolution":
        presets = {
            "Box blur": np.ones((3, 3), dtype=np.float32) / 9,
            "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
            "Sobel X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
            "Laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
        }
        params["kernel"] = presets[st.selectbox("3×3 kernel", list(presets), key="lab_kernel")]
    elif operation == "Gaussian noise":
        params["sigma"] = st.slider("Noise sigma", 0.0, 80.0, 20.0, 1.0, key="lab_noise_sigma")
        params["seed"] = st.number_input("Random seed", min_value=0, max_value=1000000, value=7, key="lab_noise_seed")
    elif operation == "Salt-and-pepper noise":
        params["amount"] = st.slider("Noisy pixel fraction", 0.0, 0.5, 0.05, 0.01, key="lab_noise_amount")
        params["seed"] = st.number_input("Random seed", min_value=0, max_value=1000000, value=7, key="lab_impulse_seed")
    elif operation == "FFT filter":
        params["pass_type"] = st.selectbox("Pass type", ["lowpass", "highpass"], key="lab_fft_pass")
        params["filter_type"] = st.selectbox("Filter shape", ["ideal", "gaussian", "butterworth"], key="lab_fft_shape")
        params["cutoff_frequency"] = st.slider("Cutoff (cycles per pixel)", 0.01, 0.5, 0.1, 0.01, key="lab_fft_cutoff")
    elif operation == "Template matching":
        height, width = image.shape[:2]
        max_x, max_y = max(0, width - 1), max(0, height - 1)
        x = st.slider("Template left edge", 0, max_x, 0, key="lab_template_x")
        y = st.slider("Template top edge", 0, max_y, 0, key="lab_template_y")
        template_w = st.slider("Template width", 1, width, min(32, width), key="lab_template_w")
        template_h = st.slider("Template height", 1, height, min(32, height), key="lab_template_h")
        params.update(x=min(x, width - template_w), y=min(y, height - template_h), width=template_w, height=template_h)
        params["method"] = st.selectbox("Matching method", ["ccoeff_normed", "sqdiff_normed", "ccorr_normed"], key="lab_template_method")
    elif operation in ("Sobel edges", "Laplacian edges"):
        params["aperture_size"] = st.selectbox("Kernel aperture", [3, 5, 7], key="lab_edge_aperture")
    elif operation == "LoG edges":
        params["sigma"] = st.slider("Gaussian sigma", 0.2, 5.0, 1.0, 0.1, key="lab_log_sigma")
    elif operation == "Canny edges":
        low, high = st.columns(2)
        low_value = low.slider("Low threshold", 0, 254, 50, key="lab_canny_low")
        high_value = high.slider("High threshold", low_value + 1, 255, max(150, low_value + 1), key="lab_canny_high")
        params.update(low_threshold=low_value, high_threshold=high_value)
    elif operation in ("Erode", "Dilate", "Open", "Close", "Top-hat"):
        params["kernel_size"] = st.select_slider("Structuring element size", options=[3, 5, 7, 9], value=3, key="lab_morph_kernel")
        params["kernel_shape"] = st.selectbox("Structuring element", ["rect", "ellipse", "cross"], key="lab_morph_shape")
        if operation != "Top-hat":
            params["iterations"] = st.slider("Iterations", 1, 5, 1, key="lab_morph_iterations")
    elif operation == "Global threshold":
        params["threshold"] = st.slider("Cutoff", 0, 255, 127, key="lab_threshold")
        params["invert"] = st.checkbox("Invert foreground", key="lab_threshold_invert")
    elif operation == "Otsu threshold":
        params["invert"] = st.checkbox("Invert foreground", key="lab_otsu_invert")
    elif operation == "Adaptive threshold":
        params["block_size"] = st.select_slider("Neighborhood size", options=[3, 5, 7, 11, 15, 21], value=11, key="lab_adaptive_block")
        params["c"] = st.slider("Offset C", -10.0, 10.0, 2.0, 0.5, key="lab_adaptive_c")
        params["method"] = st.selectbox("Local statistic", ["mean", "gaussian"], key="lab_adaptive_method")
        params["invert"] = st.checkbox("Invert foreground", key="lab_adaptive_invert")
    elif operation == "Connected regions":
        params["threshold"] = st.slider("Intensity cutoff", 0, 255, 127, key="lab_regions_threshold")
        params["min_area"] = st.number_input("Minimum contour area", min_value=0.0, value=0.0, step=5.0, key="lab_regions_area")
        params["invert"] = st.checkbox("Dark objects on bright background", key="lab_regions_invert")
    elif operation == "Watershed":
        params["foreground"] = st.selectbox("Foreground polarity", ["bright", "dark"], key="lab_watershed_polarity")
        params["distance_ratio"] = st.slider("Marker distance ratio", 0.1, 0.9, 0.4, 0.05, key="lab_watershed_ratio")
    elif operation == "Harris corners":
        params["threshold"] = st.slider("Relative response threshold", 0.001, 0.2, 0.01, 0.001, format="%.3f", key="lab_harris_threshold")
    elif operation == "HOG features":
        params["cell_size"] = st.selectbox("Cell size", [4, 8, 16], index=1, key="lab_hog_cell")
        params["orientations"] = st.slider("Orientation bins", 2, 18, 9, key="lab_hog_bins")
    elif operation == "Hough lines":
        params["threshold"] = st.slider("Vote threshold", 1, 200, 60, key="lab_hough_line_threshold")
        params["min_line_length"] = st.slider("Minimum line length", 1, max(image.shape[:2]), min(30, max(image.shape[:2])), key="lab_hough_min_line")
    elif operation == "Hough circles":
        params["param2"] = st.slider("Circle vote threshold", 5.0, 100.0, 30.0, 1.0, key="lab_hough_circle_threshold")
        params["min_distance"] = st.slider("Minimum center distance", 1.0, float(max(image.shape[:2])), float(max(1, min(20, max(image.shape[:2])))), 1.0, key="lab_hough_circle_distance")
    return params


def _float_preview(values: np.ndarray, *, colormap: bool = False) -> np.ndarray:
    pixels = values.astype(np.float32)
    minimum, maximum = float(np.min(pixels)), float(np.max(pixels))
    normalized = np.zeros(pixels.shape, dtype=np.uint8) if maximum <= minimum else np.clip(np.rint((pixels - minimum) * 255 / (maximum - minimum)), 0, 255).astype(np.uint8)
    if colormap:
        return cv2.cvtColor(cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return normalized


def _run(operation: str, image: np.ndarray, params: dict[str, object], color_order: str):
    note = ""
    if operation == "Grayscale":
        return to_grayscale(image, color_order=color_order), note
    if operation == "RGB conversion":
        return to_rgb(image, color_order=color_order), note
    if operation in ("HSV channels", "Lab channels"):
        converted = to_hsv(image, color_order=color_order) if operation == "HSV channels" else to_lab(image, color_order=color_order)
        index = {"H": 0, "S": 1, "V": 2, "L": 0, "a": 1, "b": 2}[str(params["channel"])]
        channel = str(params["channel"])
        detail = "HSV hue is 0–179; saturation and value are 0–255." if operation == "HSV channels" else "Lab lightness is 0–255; a and b are encoded with 128 as neutral."
        return converted[..., index], f"Encoded {channel} channel. {detail}"
    if operation == "Log transform": return log_transform(image), note
    if operation == "Gamma transform": return gamma_transform(image, gamma=float(params["gamma"])), note
    if operation == "Contrast stretch": return contrast_stretch(image, input_min=int(params["input_min"]), input_max=int(params["input_max"])), note
    if operation == "Histogram equalization": return equalize_histogram(image, color_order=color_order), note
    if operation == "CLAHE":
        tile = int(params["tile"])
        return apply_clahe(image, clip_limit=float(params["clip_limit"]), tile_grid_size=(tile, tile), color_order=color_order), note
    if operation == "Histogram": return compute_histograms(image, color_order=color_order), note
    if operation in ("Gaussian smoothing", "Gaussian denoising"): return gaussian_blur(image, kernel_size=int(params["kernel_size"]), sigma=float(params["sigma"])), note
    if operation in ("Median filtering", "Median denoising"): return median_blur(image, kernel_size=int(params["kernel_size"])), note
    if operation == "Unsharp sharpening": return sharpen_image(image, amount=float(params["amount"]), sigma=float(params["sigma"])), note
    if operation == "Laplacian sharpening": return laplacian_sharpen(image, amount=float(params["amount"]), aperture_size=int(params["aperture_size"])), note
    if operation == "Convolution": return _float_preview(convolve2d(to_grayscale(image, color_order=color_order), params["kernel"]), colormap=True), note
    if operation == "Gaussian noise": return add_gaussian_noise(image, sigma=float(params["sigma"]), seed=int(params["seed"])), note
    if operation == "Salt-and-pepper noise": return add_salt_pepper_noise(image, amount=float(params["amount"]), seed=int(params["seed"])), note
    if operation == "FFT spectrum": return _float_preview(fft_magnitude_spectrum(to_grayscale(image, color_order=color_order)), colormap=True), note
    if operation == "FFT filter": return _float_preview(frequency_filter(to_grayscale(image, color_order=color_order), pass_type=params["pass_type"], filter_type=params["filter_type"], cutoff_frequency=float(params["cutoff_frequency"])), colormap=True), note
    if operation == "Template matching":
        x, y, width, height = (int(params[key]) for key in ("x", "y", "width", "height"))
        template = image[y : y + height, x : x + width]
        match = match_template(image, template, method=params["method"], color_order=color_order)
        result = to_rgb(image, color_order=color_order)
        cv2.rectangle(result, match.top_left, match.bottom_right, (255, 0, 0), 2)
        return result, f"Best match at {match.top_left}; score {match.score:.4f}."
    if operation == "Sobel edges": return sobel_edges(image, aperture_size=int(params["aperture_size"]), color_order=color_order), note
    if operation == "Laplacian edges": return laplacian_edges(image, aperture_size=int(params["aperture_size"]), color_order=color_order), note
    if operation == "LoG edges": return laplacian_of_gaussian(image, sigma=float(params["sigma"]), color_order=color_order), note
    if operation == "Canny edges": return canny_edges(image, low_threshold=int(params["low_threshold"]), high_threshold=int(params["high_threshold"]), color_order=color_order), note
    if operation in ("Erode", "Dilate", "Open", "Close", "Top-hat"):
        operation_name = operation.lower().replace("-", "")
        pixels = to_grayscale(image, color_order=color_order)
        if operation_name != "tophat":
            _, pixels = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return morphology(pixels, operation_name, kernel_size=int(params["kernel_size"]), kernel_shape=params["kernel_shape"], iterations=int(params.get("iterations", 1))), note
    if operation == "Global threshold": return global_threshold(image, int(params["threshold"]), invert=bool(params["invert"]), color_order=color_order), note
    if operation == "Otsu threshold":
        cutoff, mask = otsu_threshold(image, invert=bool(params["invert"]), color_order=color_order)
        return mask, f"Otsu selected cutoff {cutoff}."
    if operation == "Adaptive threshold": return adaptive_threshold(image, block_size=int(params["block_size"]), c=float(params["c"]), method=params["method"], invert=bool(params["invert"]), color_order=color_order), note
    if operation == "Connected regions":
        gray = to_grayscale(image, color_order=color_order)
        _, mask = cv2.threshold(gray, int(params["threshold"]), 255, cv2.THRESH_BINARY_INV if params["invert"] else cv2.THRESH_BINARY)
        regions = extract_candidate_regions(mask, min_area=float(params["min_area"]))
        result = to_rgb(image, color_order=color_order)
        rows = []
        for index, region in enumerate(regions, start=1):
            cv2.drawContours(result, [region.contour], -1, (255, 0, 0), 2)
            rows.append({"Region": index, "Bounding box (x, y, width, height)": region.bounding_box, "Contour area": round(region.contour_area, 2), "Centroid (x, y)": tuple(round(value, 2) for value in region.centroid)})
        return result, f"Found {len(regions)} external contours.", rows
    if operation == "Watershed":
        result = watershed_segmentation(image, foreground=params["foreground"], distance_ratio=float(params["distance_ratio"]), color_order=color_order)
        return result.visualization, f"Watershed marker regions: {result.region_count}."
    if operation == "Harris corners":
        result = harris_corners(image, threshold=float(params["threshold"]), color_order=color_order)
        return result.visualization, f"Detected {len(result.coordinates)} local corner maxima."
    if operation == "HOG features":
        result = hog_features(image, cell_size=int(params["cell_size"]), orientations=int(params["orientations"]), color_order=color_order)
        return result.visualization, f"Descriptor length: {result.descriptor.size} values."
    if operation == "Hough lines":
        result = hough_lines(image, threshold=int(params["threshold"]), min_line_length=float(params["min_line_length"]), color_order=color_order)
        return result.visualization, f"Detected {result.detection_count} line segments."
    if operation == "Hough circles":
        result = hough_circles(image, param2=float(params["param2"]), min_distance=float(params["min_distance"]), color_order=color_order)
        return result.visualization, f"Detected {result.detection_count} circles."
    if operation == "ORB features":
        features = detect_orb_features(image, color_order=color_order)
        drawing = cv2.drawKeypoints(cv2.cvtColor(to_rgb(image, color_order=color_order), cv2.COLOR_RGB2BGR), list(features.keypoints), None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return cv2.cvtColor(drawing, cv2.COLOR_BGR2RGB), f"Detected {len(features.keypoints)} keypoints; descriptor array shape: {None if features.descriptors is None else features.descriptors.shape}."
    raise ImageValidationError(f"Unknown lab operation: {operation}")


def render_image_processing_lab(image_bytes: bytes | None, filename: str | None = None) -> None:
    """Render the Lab operation picker, parameter controls, and comparison."""
    st.header("Image Processing Lab")
    st.write("Explore classical image-processing operations on a local image. Choose a section and operation, adjust its parameters, then compare the result with the input.")
    if image_bytes is None:
        st.info("Upload an image above to start the lab.")
        return
    digest = hashlib.sha256(image_bytes).hexdigest()
    try:
        from visionguard.processing import load_image

        image = load_image(image_bytes)
    except ImageValidationError as exc:
        st.error(f"Unable to read this image: {exc}")
        return

    section = st.selectbox("Lab section", list(_SECTIONS), key="lab_section")
    operation = st.selectbox("Operation", _SECTIONS[section], key="lab_operation")
    st.caption(_DESCRIPTIONS[operation])
    color_order = st.selectbox("Interpret color channels as", ["RGB", "BGR"], key="lab_color_order")
    params = _controls(operation, image)
    signature = (digest, section, operation, color_order, tuple(sorted((key, str(value.tolist() if isinstance(value, np.ndarray) else value)) for key, value in params.items())))

    left, right = st.columns(2)
    left.image(image, caption=f"Input{f': {filename}' if filename else ''}", use_container_width=True)
    if st.button("Run operation", type="primary", key="lab_run"):
        try:
            result = _run(operation, image, params, color_order)
            st.session_state["lab_result_signature"] = signature
            st.session_state["lab_result"] = result
        except (ImageValidationError, cv2.error, ValueError, RuntimeError) as exc:
            st.session_state.pop("lab_result", None)
            st.error(f"Operation failed: {exc}")

    if st.session_state.get("lab_result_signature") != signature:
        st.session_state.pop("lab_result", None)
        right.info("Adjust the operation controls and select **Run operation** to see its output.")
        return
    stored = st.session_state.get("lab_result")
    if stored is None:
        return
    if operation == "Histogram":
        histograms, note = stored
        figure, axis = plt.subplots(figsize=(8, 3.5))
        colors = {"gray": "dimgray", "red": "red", "green": "green", "blue": "blue", "alpha": "purple"}
        for channel, counts in histograms.items():
            axis.plot(np.arange(256), counts, color=colors[channel], label=channel.title())
        axis.set(xlim=(0, 255), xlabel="Intensity", ylabel="Pixel count", title="Intensity histogram")
        axis.legend()
        right.pyplot(figure, use_container_width=True)
        plt.close(figure)
    else:
        output, note, *extra = stored
        right.image(output, caption="Processed output", use_container_width=True)
        if extra:
            st.dataframe(extra[0], hide_index=True, use_container_width=True)
    if note:
        st.caption(note)
