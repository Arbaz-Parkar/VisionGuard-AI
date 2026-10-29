# VisionGuard AI

**Local image-processing lab and visual anomaly-inspection prototype**

VisionGuard AI combines classical digital image processing with a frozen-feature anomaly detector. The primary benchmark is the **screw** category of MVTec AD: a compact, fully framed object with varying in-frame rotations. Training uses only normal examples; inference returns an anomaly score, heatmap, and candidate regions for human review.

This is an academic prototype. It does not identify defect types, produce calibrated defect probabilities, or certify products for manufacturing.

## What it includes

- A Streamlit app with an **Image Processing Lab**, **Inspect**, and **Results** tabs.
- Image operations for color spaces, intensity transforms, histograms, smoothing and sharpening, noise, frequency filtering, template matching, edges, morphology, thresholding, contours, watershed, and Harris/HOG/Hough/ORB features.
- Three inspection methods: a ResNet18 patch-memory detector, a rule-based local-contrast baseline, and a per-pixel normal-reference baseline.
- Dataset validation, reproducible detector building, test-split evaluation, memory-bank sweeps, plots, and ranked error examples.

The app operates on uploaded images locally. A model artifact is category-specific; the default configuration uses screw.

## Measured result snapshot

The full screw test split contains 160 images (41 normal, 119 anomalous) and has complete anomaly-mask coverage. On this split the patch detector reached image AUROC **0.7510** and pixel AUROC **0.9825**. At its calibrated image threshold it detected 19 of 119 anomalous images, so its image-level recall was **0.1597**. The thresholded pixel map had precision **0.1057** and recall **0.6797**. AUROC measures ranking; it does not mean a single threshold is effective.

See [the results report](docs/results.md) for all three methods, the secondary bottle experiment, and plots. These values are copied from local evaluator outputs; they are not official MVTec benchmark scores.

## Setup

Use Python 3.11 on Windows. From PowerShell in the repository root:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
~~~

The first detector build may download the pinned pretrained ResNet18 weights into Torch's local cache. A CPU-only PyTorch build is sufficient.

## Dataset

Dataset images are not included in Git. The local experiment used the foersben/mvtec-ad Hugging Face mirror; it is not MVTec's official download host. MVTec AD is licensed CC BY-NC-SA 4.0. Keep the attribution and license notice, do not use it commercially, and check the terms before redistributing images or derived examples.

Install the optional Hub client and download the screw category:

~~~powershell
.\.venv\Scripts\python.exe -m pip install huggingface_hub
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='foersben/mvtec-ad', repo_type='dataset', allow_patterns=['screw/**'], local_dir='data/mvtec_ad')"
~~~

Validate the local files:

~~~powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py --root data\mvtec_ad --category screw
~~~

The local screw category has 320 normal training images and 160 test images. Its camera is fixed; the screw's pose varies within the image. A bottle category is also present for a secondary comparison. See [dataset notes](data/README.md).

## Build, evaluate, and run

~~~powershell
.\.venv\Scripts\python.exe scripts\build_memory_bank.py --category screw
.\.venv\Scripts\python.exe scripts\evaluate.py --category screw
.\.venv\Scripts\python.exe -m streamlit run app.py
~~~

Open the local URL printed by Streamlit. The app defaults to the screw detector artifact and screw report directory. Upload a product image, select **Inspect** for anomaly detection, **Image Processing Lab** for transformations, or **Results** for the saved evaluation report.

The evaluator writes metrics and per-image rows under outputs/metrics/screw/, plots under outputs/plots/screw/research/, and ranked examples under outputs/examples/screw/research/. These generated files, the dataset, and detector artifacts are ignored by Git. The compact report and selected plots under docs/ are versioned for review.

For a short wiring check, run scripts/evaluate.py --category screw --smoke; smoke metrics are explicitly labeled and are not research results.

## Project map

- app.py: Streamlit interface and report display.
- src/visionguard/lab_ui.py: Image Processing Lab controls and rendering.
- src/visionguard/processing/: color, filtering, transforms, segmentation, features, baselines, and related operations.
- src/visionguard/model/: ResNet features, memory bank, threshold calibration, inference, and safe artifact loading.
- src/visionguard/data/: MVTec discovery, splitting, and validation.
- src/visionguard/evaluation.py, classical_evaluation.py, comparison.py: scoring and method comparison.
- scripts/: dataset checking, model building, and evaluation entry points.
- docs/: methodology, source walkthrough, results, and limitations.

## Reproducibility and license

The configuration, split seed, model settings, and dataset provenance are described in [the methodology](docs/methodology.md). Important constraints are listed in [limitations](docs/limitations.md). This project is for academic demonstration; anomaly flags need human review. The MVTec-derived data remains subject to its own license.