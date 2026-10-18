"""Local Streamlit interface for image inspection and evaluation reports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import streamlit as st

from visionguard.config import ConfigError, load_config, resolve_project_path
from visionguard.data import DatasetError, discover_normal_images, split_normal_images
from visionguard.evaluation import EvaluationError
from visionguard.model import (
    ArtifactError,
    ModelSetupError,
    PatchDetectorArtifact,
    PatchAnomalyScorer,
    ResNet18PatchFeatureExtractor,
    load_patch_artifact,
)
from visionguard.processing import (
    ImageValidationError,
    calibrate_reference_threshold,
    fit_per_pixel_reference,
    load_image,
    run_reference_baseline,
    run_rule_based_baseline,
    to_rgb,
)
from visionguard.lab_ui import render_image_processing_lab


@st.cache_resource(show_spinner=False)
def _load_detector(
    artifact_path: str,
    artifact_modified_ns: int,
    device_preference: str,
) -> tuple[PatchDetectorArtifact, PatchAnomalyScorer]:
    """Cache the loaded artifact, frozen extractor, and scorer across reruns."""
    del artifact_modified_ns  # Included in the cache key to notice artifact replacement.
    artifact = load_patch_artifact(artifact_path)
    extractor = ResNet18PatchFeatureExtractor(device_preference=device_preference)
    return artifact, artifact.build_scorer(extractor)


def _heatmap_rgb(display_map: np.ndarray) -> np.ndarray:
    values = np.clip(display_map, 0.0, 1.0)
    heatmap_bgr = cv2.applyColorMap(
        np.rint(values * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def _iter_loaded_images(image_paths):
    for image_path in image_paths:
        yield load_image(image_path)


def _normal_image_fingerprint(dataset_root: Path, category: str) -> tuple[tuple[str, int, int], ...]:
    """Track normal training file metadata so Streamlit invalidates the fit on changes."""
    return tuple(
        (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
        for path in discover_normal_images(dataset_root, category)
    )


@st.cache_resource(show_spinner=False)
def _load_reference_model(
    dataset_root: str,
    category: str,
    calibration_fraction: float,
    calibration_seed: int,
    calibration_percentile: float,
    calibration_rule: str,
    standard_deviation_floor: float,
    normal_fingerprint: tuple[tuple[str, int, int], ...],
):
    """Fit on the bank subset and calibrate only on disjoint normal holdout images."""
    del normal_fingerprint  # Its value is part of the cache key.
    split = split_normal_images(
        dataset_root,
        category,
        calibration_fraction=calibration_fraction,
        seed=calibration_seed,
    )
    model = fit_per_pixel_reference(_iter_loaded_images(split.memory_bank_images))
    threshold = calibrate_reference_threshold(
        model,
        _iter_loaded_images(split.calibration_images),
        percentile=calibration_percentile,
        standard_deviation_floor=standard_deviation_floor,
        rule=calibration_rule,
    )
    return model, threshold, len(split.memory_bank_images), len(split.calibration_images)


def _render_metric_group(label: str, metrics: dict[str, object]) -> None:
    st.subheader(label)
    columns = st.columns(5)
    auroc = metrics.get("auroc")
    columns[0].metric("AUROC", "Unavailable" if auroc is None else f"{float(auroc):.4f}")
    columns[1].metric("Precision", f"{float(metrics.get('precision', 0.0)):.4f}")
    columns[2].metric("Recall", f"{float(metrics.get('recall', 0.0)):.4f}")
    columns[3].metric("F1", f"{float(metrics.get('f1', 0.0)):.4f}")
    columns[4].metric("IoU", f"{float(metrics.get('iou', 0.0)):.4f}")
    if auroc is None and metrics.get("auroc_unavailable_reason"):
        st.caption(str(metrics["auroc_unavailable_reason"]))
    counts = {
        "True negative": metrics.get("true_negative", 0),
        "False positive": metrics.get("false_positive", 0),
        "False negative": metrics.get("false_negative", 0),
        "True positive": metrics.get("true_positive", 0),
        "Samples": metrics.get("sample_count", 0),
    }
    st.caption(" · ".join(f"{name}: {value}" for name, value in counts.items()))


def _render_inspection(
    config: dict[str, object],
    artifact_path_text: str,
    device: str,
    uploaded,
    image_bytes: bytes | None,
) -> None:
    st.header("Inspect an image")
    st.write(
        "Upload one image to view its anomaly score, calibrated decision, heatmap, "
        "and candidate regions. The score is not a defect probability."
    )
    upload_digest = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
    artifact_path = (
        resolve_project_path(artifact_path_text) if artifact_path_text.strip() else None
    )
    artifact_exists = artifact_path is not None and artifact_path.is_file()
    artifact_modified_ns = artifact_path.stat().st_mtime_ns if artifact_exists else 0
    signature = (
        upload_digest,
        str(artifact_path.resolve()) if artifact_path is not None else "",
        artifact_modified_ns,
        device,
    )
    if st.session_state.get("inspection_signature") != signature:
        st.session_state.pop("inspection_result", None)
        st.session_state.pop("inspection_image", None)
        st.session_state["inspection_signature"] = signature

    if not artifact_exists:
        st.info(
            "No detector artifact was found at the selected path. Create and save a "
            "`.npz` artifact before running an inspection."
        )

    if uploaded is not None:
        try:
            preview_image = load_image(image_bytes)
            if st.session_state.get("inspection_result") is None:
                st.image(preview_image, caption=uploaded.name, use_container_width=True)
        except ImageValidationError as exc:
            st.error(f"Unable to read this image: {exc}")
            preview_image = None
    else:
        preview_image = None

    inspect_clicked = st.button(
        "Inspect image",
        type="primary",
        disabled=uploaded is None or not artifact_exists or preview_image is None,
    )
    if inspect_clicked and uploaded is not None and artifact_path is not None:
        try:
            with st.spinner("Loading the detector and scoring the image..."):
                artifact, scorer = _load_detector(
                    str(artifact_path.resolve()), artifact_modified_ns, device
                )
                image = load_image(image_bytes)
                result = scorer.run(
                    image,
                    pixel_threshold=artifact.thresholds.pixel_threshold,
                    min_region_area=float(config["classical"]["min_region_area"]),
                )
            st.session_state["inspection_result"] = (artifact, result)
            st.session_state["inspection_image"] = image
        except (
            ArtifactError,
            ModelSetupError,
            ImageValidationError,
            EvaluationError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            st.error(f"Inspection failed: {exc}")

    stored = st.session_state.get("inspection_result")
    stored_image = st.session_state.get("inspection_image")
    if stored is None or stored_image is None or upload_digest is None:
        return

    artifact, result = stored
    is_anomaly = result.image_score > artifact.thresholds.image_threshold
    st.divider()
    score_column, decision_column, timing_column = st.columns(3)
    score_column.metric(
        "Image anomaly score",
        f"{result.image_score:.5f}",
        help="Continuous model score; it is not a calibrated probability.",
    )
    decision_column.metric(
        "Calibrated decision",
        "Anomaly flagged" if is_anomaly else "Below threshold",
        help=f"Image threshold: {artifact.thresholds.image_threshold:.5f}",
    )
    timing_column.metric("Inference time", f"{result.timings.total_ms:.1f} ms")

    heatmap = _heatmap_rgb(result.display_map)
    overlay = cv2.addWeighted(to_rgb(stored_image), 0.58, heatmap, 0.42, 0)
    image_column, map_column, overlay_column = st.columns(3)
    image_column.image(stored_image, caption="Input", use_container_width=True)
    map_column.image(heatmap, caption="Anomaly heatmap", use_container_width=True)
    overlay_column.image(overlay, caption="Heatmap overlay", use_container_width=True)
    st.caption(
        f"Pixel threshold: {artifact.thresholds.pixel_threshold:.5f} · "
        f"Inference device: {result.device} · "
        f"Regions shown after minimum area filtering: {len(result.regions)}"
    )

    if result.regions:
        region_rows = [
            {
                "Region": index,
                "Bounding box (x, y, width, height)": region.bounding_box,
                "Contour area": round(region.contour_area, 2),
                "Centroid (x, y)": tuple(round(value, 2) for value in region.centroid),
            }
            for index, region in enumerate(result.regions, start=1)
        ]
        st.dataframe(region_rows, hide_index=True, use_container_width=True)
    else:
        st.caption("No regions met the configured minimum area at the calibrated pixel threshold.")
    with st.expander("View thresholded candidate mask"):
        st.image(
            result.mask,
            caption="Regions retained after minimum area filtering",
            use_container_width=True,
        )


def _render_classical_baseline(
    config: dict[str, object], uploaded, image_bytes: bytes | None
) -> None:
    st.header("Rule-based local-contrast baseline")
    st.write(
        "This transparent baseline enhances grayscale local contrast, compares each "
        "pixel with a blurred background, then uses Otsu thresholding and morphology "
        "to find candidate regions."
    )
    st.caption(
        "Its Otsu cutoff is chosen separately for each image. The resulting score is "
        "not calibrated and is not a defect probability."
    )
    digest = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
    if st.session_state.get("classical_upload_digest") != digest:
        st.session_state.pop("classical_result", None)
        st.session_state.pop("classical_image", None)
        st.session_state["classical_upload_digest"] = digest

    if uploaded is None:
        st.info("Upload an image to run the rule-based baseline.")
        return

    try:
        image = load_image(image_bytes)
    except ImageValidationError as exc:
        st.error(f"Unable to read this image: {exc}")
        return

    classical_settings = config["classical"]
    run_clicked = st.button("Run rule-based baseline", key="run_classical_baseline")
    if run_clicked:
        started = time.perf_counter()
        try:
            result = run_rule_based_baseline(
                image,
                blur_kernel_size=int(classical_settings["blur_kernel_size"]),
                local_blur_kernel_size=int(classical_settings["local_blur_kernel_size"]),
                min_region_area=float(classical_settings["min_region_area"]),
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            st.session_state["classical_result"] = result
            st.session_state["classical_image"] = image
            st.session_state["classical_elapsed_ms"] = elapsed_ms
        except (ImageValidationError, ValueError, RuntimeError) as exc:
            st.error(f"Baseline failed: {exc}")

    result = st.session_state.get("classical_result")
    result_image = st.session_state.get("classical_image")
    if result is None or result_image is None or digest is None:
        st.image(image, caption=uploaded.name, use_container_width=True)
        return

    score_maximum = float(np.max(result.score_map))
    normalized_map = (
        result.score_map / score_maximum
        if score_maximum > 0
        else np.zeros_like(result.score_map, dtype=np.float32)
    )
    heatmap = _heatmap_rgb(normalized_map)
    overlay = cv2.addWeighted(to_rgb(result_image), 0.58, heatmap, 0.42, 0)

    st.divider()
    score_column, threshold_column, count_column, timing_column = st.columns(4)
    score_column.metric("Maximum local-contrast score", f"{result.image_score:.1f}")
    threshold_column.metric("Per-image Otsu threshold", f"{result.threshold:.1f}")
    count_column.metric("Candidate regions", len(result.regions))
    timing_column.metric(
        "Processing time", f"{st.session_state.get('classical_elapsed_ms', 0.0):.1f} ms"
    )

    input_column, heatmap_column, overlay_column = st.columns(3)
    input_column.image(result_image, caption="Input", use_container_width=True)
    heatmap_column.image(heatmap, caption="Local-contrast score map", use_container_width=True)
    overlay_column.image(overlay, caption="Score-map overlay", use_container_width=True)

    if result.regions:
        region_rows = [
            {
                "Region": index,
                "Bounding box (x, y, width, height)": region.bounding_box,
                "Contour area": round(region.contour_area, 2),
                "Centroid (x, y)": tuple(round(value, 2) for value in region.centroid),
            }
            for index, region in enumerate(result.regions, start=1)
        ]
        st.dataframe(region_rows, hide_index=True, use_container_width=True)
    else:
        st.caption("No regions met the configured minimum area.")
    with st.expander("View candidate mask"):
        st.image(
            result.mask,
            caption="Candidate regions after opening and closing",
            use_container_width=True,
        )


def _render_reference_baseline(
    config: dict[str, object], uploaded, image_bytes: bytes | None
) -> None:
    st.header("Calibrated per-pixel reference baseline")
    st.write(
        "This baseline compares each pixel with statistics learned from normal "
        "training images. Its pixel threshold is selected using a separate normal "
        "calibration subset."
    )
    st.caption(
        "The seeded split uses only `train/good` images. Test images and defect masks "
        "are not used to fit or calibrate this detector."
    )
    digest = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
    dataset_root = resolve_project_path(config["dataset"]["root"])
    category = str(config["dataset"]["category"])
    calibration_fraction = float(config["dataset"]["calibration_fraction"])
    calibration_seed = int(config["dataset"]["calibration_seed"])
    calibration_percentile = float(config["thresholds"]["pixel_percentile"])
    calibration_rule = str(config["thresholds"]["pixel_rule"])
    standard_deviation_floor = 1.0
    minimum_region_area = float(config["classical"]["min_region_area"])
    signature = (
        digest,
        str(dataset_root),
        category,
        calibration_fraction,
        calibration_seed,
        calibration_percentile,
        calibration_rule,
        standard_deviation_floor,
        minimum_region_area,
    )
    if st.session_state.get("reference_signature") != signature:
        st.session_state.pop("reference_result", None)
        st.session_state.pop("reference_image", None)
        st.session_state["reference_signature"] = signature

    if uploaded is None:
        st.info("Upload an image above to run the calibrated reference baseline.")
        return
    try:
        image = load_image(image_bytes)
    except ImageValidationError as exc:
        st.error(f"Unable to read this image: {exc}")
        return

    normal_root = dataset_root / category / "train" / "good"
    if not normal_root.is_dir():
        st.info(
            f"Normal training images were not found at `{normal_root}`. Set up the "
            "local MVTec category before preparing this baseline."
        )

    run_clicked = st.button(
        "Prepare reference and score image",
        key="run_reference_baseline",
        disabled=not normal_root.is_dir(),
    )
    if run_clicked:
        started = time.perf_counter()
        try:
            fingerprint = _normal_image_fingerprint(dataset_root, category)
            with st.spinner(
                "Fitting normal-pixel statistics and calibrating on held-out normal images..."
            ):
                model, pixel_threshold, training_count, calibration_count = (
                    _load_reference_model(
                        str(dataset_root),
                        category,
                        calibration_fraction,
                        calibration_seed,
                        calibration_percentile,
                        calibration_rule,
                        standard_deviation_floor,
                        fingerprint,
                    )
                )
                if image.shape[:2] != model.mean.shape[:2]:
                    expected_height, expected_width = model.mean.shape[:2]
                    actual_height, actual_width = image.shape[:2]
                    raise ImageValidationError(
                        "Uploaded image size does not match the reference images "
                        f"(expected {expected_width}×{expected_height}, received "
                        f"{actual_width}×{actual_height})."
                    )
                result = run_reference_baseline(
                    image,
                    model,
                    pixel_threshold=pixel_threshold,
                    standard_deviation_floor=standard_deviation_floor,
                    min_region_area=minimum_region_area,
                )
            st.session_state["reference_result"] = (
                result,
                pixel_threshold,
                training_count,
                calibration_count,
            )
            st.session_state["reference_image"] = image
            st.session_state["reference_elapsed_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
        except (
            DatasetError,
            ImageValidationError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            st.error(f"Reference baseline failed: {exc}")

    stored = st.session_state.get("reference_result")
    result_image = st.session_state.get("reference_image")
    if stored is None or result_image is None or digest is None:
        st.image(image, caption=uploaded.name, use_container_width=True)
        return

    result, pixel_threshold, training_count, calibration_count = stored
    maximum_score = float(np.max(result.score_map))
    normalized_map = (
        result.score_map / maximum_score
        if maximum_score > 0
        else np.zeros_like(result.score_map, dtype=np.float32)
    )
    heatmap = _heatmap_rgb(normalized_map)
    overlay = cv2.addWeighted(to_rgb(result_image), 0.58, heatmap, 0.42, 0)

    st.caption(
        f"Reference fit: {training_count} normal training images · "
        f"calibration: {calibration_count} disjoint normal images · "
        f"pixel threshold rule: {calibration_rule}"
    )
    score_column, threshold_column, count_column, timing_column = st.columns(4)
    score_column.metric("Maximum pixel deviation", f"{result.image_score:.3f}")
    threshold_column.metric("Calibrated pixel threshold", f"{pixel_threshold:.3f}")
    count_column.metric("Candidate regions", len(result.regions))
    timing_column.metric(
        "Setup and scoring", f"{st.session_state.get('reference_elapsed_ms', 0.0):.1f} ms"
    )

    input_column, heatmap_column, overlay_column = st.columns(3)
    input_column.image(result_image, caption="Input", use_container_width=True)
    heatmap_column.image(heatmap, caption="Per-pixel deviation map", use_container_width=True)
    overlay_column.image(overlay, caption="Deviation overlay", use_container_width=True)
    st.caption(
        "The calibrated cutoff applies to pixels; this reference baseline does not "
        "provide an image-level anomaly decision."
    )
    if result.regions:
        region_rows = [
            {
                "Region": index,
                "Bounding box (x, y, width, height)": region.bounding_box,
                "Contour area": round(region.contour_area, 2),
                "Centroid (x, y)": tuple(round(value, 2) for value in region.centroid),
            }
            for index, region in enumerate(result.regions, start=1)
        ]
        st.dataframe(region_rows, hide_index=True, use_container_width=True)
    else:
        st.caption("No regions met the configured minimum area.")
    with st.expander("View calibrated candidate mask"):
        st.image(
            result.mask,
            caption="Pixels above threshold, grouped into retained candidate regions",
            use_container_width=True,
        )


def _render_report(config: dict[str, object]) -> None:
    st.header("Evaluation report")
    st.write(
        "View reports generated from the official test split. These metrics describe "
        "the evaluated artifact and dataset; uploading an inspection image does not change them."
    )
    default_report_dir = (
        resolve_project_path(config["paths"]["outputs"])
        / "metrics"
        / str(config["dataset"]["category"])
    )
    report_dir_text = st.text_input(
        "Report directory",
        value=str(default_report_dir),
        help="The directory containing metrics.json and per_image.csv.",
    )
    report_dir = resolve_project_path(report_dir_text)
    summary_path = report_dir / "metrics.json"
    rows_path = report_dir / "per_image.csv"
    if not summary_path.is_file():
        st.info(
            f"No evaluation report exists at `{summary_path}`. Run "
            "`scripts/evaluate.py` after building the detector artifact and setting "
            "up the dataset."
        )
        return

    try:
        with summary_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        st.error(f"Unable to read the evaluation summary: {exc}")
        return
    if not isinstance(summary, dict):
        st.error("The evaluation summary must contain a JSON object.")
        return

    st.subheader(f"Category: {summary.get('category', 'unknown')}")
    if summary.get("smoke_test"):
        st.warning("Smoke test output only. These metrics are not a research result.")
    count_columns = st.columns(4)
    count_columns[0].metric("Evaluated images", summary.get("evaluated_image_count", 0))
    count_columns[1].metric("Normal images", summary.get("normal_image_count", 0))
    count_columns[2].metric("Anomalous images", summary.get("anomalous_image_count", 0))
    count_columns[3].metric("Missing masks", summary.get("missing_mask_count", 0))
    image_metrics = summary.get("image_metrics")
    pixel_metrics = summary.get("pixel_metrics")
    if isinstance(image_metrics, dict) and isinstance(pixel_metrics, dict):
        _render_metric_group("Image-level metrics", image_metrics)
        _render_metric_group("Pixel-level metrics", pixel_metrics)
        st.caption(
            "Thresholds used: "
            f"image {summary.get('image_threshold', 'unknown')} · "
            f"pixel {summary.get('pixel_threshold', 'unknown')}"
        )
    elif isinstance(summary.get("rule_based"), dict) and isinstance(
        summary.get("reference"), dict
    ):
        for method_name, method_label in (
            ("rule_based", "Rule-based baseline"),
            ("reference", "Per-pixel reference baseline"),
        ):
            method = summary[method_name]
            method_image_metrics = method.get("image_metrics")
            method_pixel_metrics = method.get("pixel_metrics")
            if not isinstance(method_image_metrics, dict) or not isinstance(
                method_pixel_metrics, dict
            ):
                st.error(f"The {method_label} report is missing metrics.")
                return
            st.divider()
            st.subheader(method_label)
            _render_metric_group("Image-level metrics", method_image_metrics)
            _render_metric_group("Pixel-level metrics", method_pixel_metrics)
            pixel_threshold = method.get("pixel_threshold")
            pixel_threshold_text = (
                f"{pixel_threshold} ({method.get('pixel_threshold_method', 'unknown')})"
                if pixel_threshold is not None
                else str(method.get("pixel_threshold_method", "unknown"))
            )
            st.caption(
                f"Image threshold: {method.get('image_threshold', 'unknown')} "
                f"({method.get('image_threshold_method', 'unknown')}) · "
                f"Pixel threshold: {pixel_threshold_text}"
            )
    elif isinstance(summary.get("methods"), dict):
        for method_name, method in summary["methods"].items():
            if not isinstance(method, dict):
                st.error(f"The {method_name} report entry is invalid.")
                return
            method_image_metrics = method.get("image_metrics")
            method_pixel_metrics = method.get("pixel_metrics")
            if not isinstance(method_image_metrics, dict) or not isinstance(
                method_pixel_metrics, dict
            ):
                st.error(f"The {method_name} report is missing metrics.")
                return
            st.divider()
            st.subheader(str(method.get("label", method_name)))
            _render_metric_group("Image-level metrics", method_image_metrics)
            _render_metric_group("Pixel-level metrics", method_pixel_metrics)
            pixel_threshold = method.get("pixel_threshold")
            pixel_threshold_text = (
                f"{pixel_threshold} ({method.get('pixel_threshold_method', 'unknown')})"
                if pixel_threshold is not None
                else str(method.get("pixel_threshold_method", "unknown"))
            )
            st.caption(
                f"Image threshold: {method.get('image_threshold', 'unknown')} "
                f"({method.get('image_threshold_method', 'unknown')}) · "
                f"Pixel threshold: {pixel_threshold_text} · "
                f"Mean preprocessing / inference / total: "
                f"{method.get('mean_preprocessing_ms', 'unknown')} / "
                f"{method.get('mean_model_inference_ms', 'unknown')} / "
                f"{method.get('mean_total_ms', 'unknown')} ms · "
                f"Device: {summary.get('devices', {}).get(method_name, 'unknown')}"
            )
        defect_metrics = summary.get("per_defect_image_metrics")
        if isinstance(defect_metrics, dict) and defect_metrics:
            defect_rows = []
            for defect_type, methods in defect_metrics.items():
                if not isinstance(methods, dict):
                    continue
                for method_name, metrics in methods.items():
                    if isinstance(metrics, dict):
                        defect_rows.append(
                            {
                                "Defect type": defect_type,
                                "Method": method_name,
                                "Image AUROC": metrics.get("auroc"),
                                "Precision": metrics.get("precision"),
                                "Recall": metrics.get("recall"),
                                "F1": metrics.get("f1"),
                                "IoU": metrics.get("iou"),
                                "Samples": metrics.get("sample_count"),
                            }
                        )
            st.subheader("Per-defect image metrics")
            st.dataframe(defect_rows, hide_index=True, use_container_width=True)
        defect_pixel_metrics = summary.get("per_defect_pixel_metrics")
        if isinstance(defect_pixel_metrics, dict) and defect_pixel_metrics:
            pixel_rows = []
            for defect_type, methods in defect_pixel_metrics.items():
                if not isinstance(methods, dict):
                    continue
                for method_name, metrics in methods.items():
                    if isinstance(metrics, dict):
                        pixel_rows.append(
                            {
                                "Defect type": defect_type,
                                "Method": method_name,
                                "Pixel AUROC": metrics.get("auroc"),
                                "Precision": metrics.get("precision"),
                                "Recall": metrics.get("recall"),
                                "F1": metrics.get("f1"),
                                "IoU": metrics.get("iou"),
                                "Pixels": metrics.get("sample_count"),
                            }
                        )
            st.subheader("Per-defect pixel metrics")
            st.dataframe(pixel_rows, hide_index=True, use_container_width=True)
        st.caption(
            f"Pixel metrics use a common {summary.get('pixel_metric_size', 'unknown')} grid; "
            f"detector input size is {summary.get('model_input_size', 'unknown')}; "
            f"{summary.get('warmup_image_count', 0)} warm-up images were excluded from timings."
        )
        if summary.get("image_metrics_note"):
            st.caption(str(summary["image_metrics_note"]))
        if summary.get("latency_note"):
            st.caption(str(summary["latency_note"]))
        if summary.get("pixel_metrics_note"):
            st.caption(str(summary["pixel_metrics_note"]))
    else:
        st.error("The evaluation summary is missing image or pixel metrics.")
        return

    if rows_path.is_file():
        try:
            with rows_path.open("r", encoding="utf-8-sig", newline="") as stream:
                image_rows = list(csv.DictReader(stream))
            st.subheader("Per-image results")
            st.dataframe(image_rows, hide_index=True, use_container_width=True)
        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            st.error(f"Unable to read per-image results: {exc}")
    else:
        st.warning(f"Per-image CSV was not found: {rows_path}")


def main() -> None:
    st.set_page_config(
        page_title="VisionGuard AI",
        page_icon="🔎",
        layout="wide",
    )
    st.title("VisionGuard AI")
    st.caption("Local visual inspection with a PatchCore-inspired anomaly detector")

    try:
        config = load_config()
    except ConfigError as exc:
        st.error(f"Unable to load project configuration: {exc}")
        st.stop()

    category = str(config["dataset"]["category"])
    default_artifact = (
        resolve_project_path(config["paths"]["artifacts"])
        / f"{category}_patch_detector.npz"
    )
    available_devices = ["auto", "cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            available_devices.append("cuda")
    except ImportError:
        pass
    configured_device = str(config["model"]["device"])
    default_device = configured_device if configured_device in available_devices else "auto"

    with st.sidebar:
        st.header("Detector settings")
        artifact_path_text = st.text_input(
            "Artifact path",
            value=str(default_artifact),
            help="Path to a VisionGuard .npz artifact. Model files are loaded without pickle.",
        )
        device = st.selectbox(
            "Inference device",
            available_devices,
            index=available_devices.index(default_device),
        )
        st.caption(f"Configured dataset category: {category}")
        st.divider()
        st.caption(
            "This research prototype supports human review. An anomaly flag is not "
            "a manufacturing decision."
        )

    uploaded = st.file_uploader(
        "Image for the Lab or inspection",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="Upload once, then use it in the Image Processing Lab or Inspect page. Images are processed locally.",
        key="inspection_image_upload",
    )
    image_bytes = uploaded.getvalue() if uploaded is not None else None

    lab_tab, inspection_tab, results_tab = st.tabs(
        ["Image Processing Lab", "Inspect", "Results"]
    )
    with lab_tab:
        render_image_processing_lab(image_bytes, uploaded.name if uploaded is not None else None)
    with inspection_tab:
        inspection_method = st.radio(
            "Inspection method",
            ["Patch detector", "Rule-based baseline", "Per-pixel reference baseline"],
            horizontal=True,
            key="inspection_method",
        )
        if inspection_method == "Patch detector":
            _render_inspection(config, artifact_path_text, device, uploaded, image_bytes)
        elif inspection_method == "Rule-based baseline":
            _render_classical_baseline(config, uploaded, image_bytes)
        else:
            _render_reference_baseline(config, uploaded, image_bytes)
    with results_tab:
        _render_report(config)


if __name__ == "__main__":
    main()
