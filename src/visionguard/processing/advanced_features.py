"""Harris, HOG, and Hough feature visualizations for the Lab."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .preprocessing import ColorOrder, ImageValidationError, to_grayscale, to_rgb


@dataclass(frozen=True)
class HarrisResult:
    visualization: np.ndarray
    coordinates: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class HOGResult:
    descriptor: np.ndarray
    visualization: np.ndarray


@dataclass(frozen=True)
class HoughResult:
    visualization: np.ndarray
    detection_count: int


def harris_corners(
    image: np.ndarray,
    *,
    block_size: int = 2,
    aperture_size: int = 3,
    k: float = 0.04,
    threshold: float = 0.01,
    color_order: ColorOrder = "RGB",
) -> HarrisResult:
    """Detect Harris corners and mark local response maxima in green."""
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ImageValidationError("block_size must be a positive integer.")
    if isinstance(aperture_size, bool) or not isinstance(aperture_size, int) or aperture_size not in (3, 5, 7):
        raise ImageValidationError("aperture_size must be one of 3, 5, or 7.")
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or not 0 < k < 1:
        raise ImageValidationError("k must be a finite number between 0 and 1.")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ImageValidationError("threshold must be a finite number greater than 0 and at most 1.")
    gray = to_grayscale(image, color_order=color_order)
    response = cv2.cornerHarris(gray.astype(np.float32) / 255.0, block_size, aperture_size, float(k))
    maximum = float(response.max(initial=0.0))
    if maximum <= 0:
        return HarrisResult(to_rgb(image, color_order=color_order), ())
    local_max = response == cv2.dilate(response, np.ones((3, 3), dtype=np.uint8))
    ys, xs = np.nonzero(local_max & (response >= threshold * maximum) & (response > 0))
    visualization = to_rgb(image, color_order=color_order)
    points = tuple((int(x), int(y)) for y, x in zip(ys, xs))
    for x, y in points:
        cv2.circle(visualization, (x, y), 3, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    return HarrisResult(visualization, points)


def hog_features(
    image: np.ndarray,
    *,
    cell_size: int = 8,
    block_size: int = 2,
    orientations: int = 9,
    color_order: ColorOrder = "RGB",
) -> HOGResult:
    """Compute cell histograms, L2-Hys block descriptors, and a HOG drawing."""
    for name, value in (("cell_size", cell_size), ("block_size", block_size), ("orientations", orientations)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ImageValidationError(f"{name} must be a positive integer.")
    if orientations < 2 or orientations > 36:
        raise ImageValidationError("orientations must be from 2 through 36.")
    gray = to_grayscale(image, color_order=color_order)
    minimum = cell_size * block_size
    height = max(minimum, (gray.shape[0] // cell_size) * cell_size)
    width = max(minimum, (gray.shape[1] // cell_size) * cell_size)
    if (height, width) != gray.shape:
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    angle %= 180.0
    cell_rows, cell_columns = height // cell_size, width // cell_size
    cell_histograms = np.zeros((cell_rows, cell_columns, orientations), dtype=np.float32)
    bin_width = 180.0 / orientations
    for row in range(cell_rows):
        for column in range(cell_columns):
            y0, x0 = row * cell_size, column * cell_size
            cell_histograms[row, column], _ = np.histogram(
                angle[y0 : y0 + cell_size, x0 : x0 + cell_size],
                bins=orientations,
                range=(0.0, 180.0),
                weights=magnitude[y0 : y0 + cell_size, x0 : x0 + cell_size],
            )

    descriptor_parts = []
    for row in range(cell_rows - block_size + 1):
        for column in range(cell_columns - block_size + 1):
            block = cell_histograms[row : row + block_size, column : column + block_size].ravel()
            normalized = block / math.sqrt(float(np.dot(block, block)) + 1e-12)
            normalized = np.minimum(normalized, 0.2)
            normalized /= math.sqrt(float(np.dot(normalized, normalized)) + 1e-12)
            descriptor_parts.append(normalized)
    descriptor = np.concatenate(descriptor_parts).astype(np.float32, copy=False)

    drawing = np.zeros((height, width), dtype=np.uint8)
    scale = max(1.0, float(cell_histograms.max(initial=0.0)))
    center_offset = cell_size // 2
    for row in range(cell_rows):
        for column in range(cell_columns):
            center = (column * cell_size + center_offset, row * cell_size + center_offset)
            for index, strength in enumerate(cell_histograms[row, column]):
                length = int(round((strength / scale) * (cell_size * 0.45)))
                if length <= 0:
                    continue
                radians = math.radians((index + 0.5) * bin_width)
                delta = (int(round(length * math.cos(radians))), int(round(length * math.sin(radians))))
                cv2.line(drawing, (center[0] - delta[0], center[1] - delta[1]), (center[0] + delta[0], center[1] + delta[1]), 255, 1, cv2.LINE_AA)
    return HOGResult(descriptor, cv2.cvtColor(drawing, cv2.COLOR_GRAY2RGB))


def hough_lines(
    image: np.ndarray,
    *,
    threshold: int = 80,
    rho: float = 1.0,
    theta_degrees: float = 1.0,
    min_line_length: float = 30.0,
    max_line_gap: float = 10.0,
    color_order: ColorOrder = "RGB",
) -> HoughResult:
    """Detect probabilistic Hough lines and draw them in red."""
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
        raise ImageValidationError("threshold must be a positive integer.")
    for name, value in (("rho", rho), ("theta_degrees", theta_degrees), ("min_line_length", min_line_length), ("max_line_gap", max_line_gap)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ImageValidationError(f"{name} must be a finite number greater than 0.")
    edges = cv2.Canny(to_grayscale(image, color_order=color_order), 50, 150)
    lines = cv2.HoughLinesP(edges, float(rho), math.radians(float(theta_degrees)), threshold, minLineLength=float(min_line_length), maxLineGap=float(max_line_gap))
    visualization = to_rgb(image, color_order=color_order)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(visualization, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2, cv2.LINE_AA)
    return HoughResult(visualization, 0 if lines is None else len(lines))


def hough_circles(
    image: np.ndarray,
    *,
    dp: float = 1.2,
    min_distance: float = 20.0,
    param1: float = 100.0,
    param2: float = 30.0,
    min_radius: int = 0,
    max_radius: int = 0,
    color_order: ColorOrder = "RGB",
) -> HoughResult:
    """Detect Hough circles and draw each circle and center in red/green."""
    for name, value in (("dp", dp), ("min_distance", min_distance), ("param1", param1), ("param2", param2)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ImageValidationError(f"{name} must be a finite number greater than 0.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (min_radius, max_radius)):
        raise ImageValidationError("circle radii must be non-negative integers.")
    if max_radius and max_radius < min_radius:
        raise ImageValidationError("max_radius must be 0 or greater than or equal to min_radius.")
    gray = cv2.medianBlur(to_grayscale(image, color_order=color_order), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, float(dp), float(min_distance), param1=float(param1), param2=float(param2), minRadius=min_radius, maxRadius=max_radius)
    visualization = to_rgb(image, color_order=color_order)
    count = 0
    if circles is not None:
        rounded = np.rint(circles[0]).astype(int)
        count = len(rounded)
        for x, y, radius in rounded:
            cv2.circle(visualization, (int(x), int(y)), int(radius), (255, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(visualization, (int(x), int(y)), 2, (0, 255, 0), 2, cv2.LINE_AA)
    return HoughResult(visualization, count)
