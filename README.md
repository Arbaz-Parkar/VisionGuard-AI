# VisionGuard AI

**Intelligent Visual Inspection & Anomaly Detection**

VisionGuard AI is an MSc Digital Image Processing project exploring how classical image processing and deep feature-based anomaly detection can support industrial visual inspection. The initial research dataset is MVTec AD, starting with the `bottle` category.

## Current status

The foundation includes validated experiment settings, CPU/CUDA device selection, logging, dataset discovery, and a seeded normal-only calibration split. The Image Processing Lab currently includes safe image loading, explicit RGB/BGR conversion, resizing, flips/rotation/translation, Gaussian/median filtering, unsharp-mask sharpening, reproducible noise injection, template matching, ORB keypoints and descriptors, CLAHE, histograms, Canny edges, thresholding, morphology, contour candidates, NumPy spatial convolution, 2D FFT spectra, and frequency-domain low/high-pass filters. Both classical baselines are implemented. A frozen pretrained ResNet18 extractor produces layer2/layer3 patch embeddings, and a normal-only memory-bank builder supports random and greedy coreset sampling. Patch scoring, anomaly-map construction, image/pixel threshold calibration, artifact save/load, test-split evaluation with JSON/CSV reports, and a local Streamlit inspection/report interface are implemented. No model scores or benchmark results are available until an artifact is evaluated on the local dataset.

## Planned approach

The initial AI baseline will use pretrained ResNet18 intermediate feature maps, patch embeddings, and nearest-neighbor distances to a sampled memory bank built only from normal training images. It is an educational PatchCore-inspired approach, not a claim to implement the official PatchCore algorithm. A classical image-processing baseline will provide a transparent comparison.

Evaluation will use the official test split. Test images and their masks will not be used to build the normal reference bank. Results will be reported only after real experiments. Image processing is intended to run locally; inspection outputs are experimental and require human review.

## Environment

Use Python 3.11 on Windows. Create the virtual environment from PowerShell:

```powershell
py -3.11 -m venv .venv
```

If the Python launcher cannot find Python 3.11, use the installed interpreter path:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
```

The checked-in environment uses CPU-only PyTorch, which satisfies the required CPU path. This machine also has an NVIDIA RTX 2050 with 4 GB VRAM, but the current Torch build cannot use it. Choose and install a matching CUDA-enabled Torch/TorchVision build before model experiments if you want GPU acceleration. Install PyTorch and TorchVision first:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

To use the NVIDIA GPU instead, replace `cpu` with `cu132` in the PyTorch command. The CUDA wheel is substantially larger. Keep the matched Torch/TorchVision versions together. Official choices are listed at [PyTorch Get Started](https://pytorch.org/get-started/locally/) and [previous versions](https://pytorch.org/get-started/previous-versions/).

You can use the environment without activating it by running commands through `.venv\Scripts\python.exe`.

## Dataset setup

Obtain MVTec AD from the [official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), review its access and usage conditions, and extract it so the selected category resembles:

```text
data/
└── mvtec_ad/
    └── bottle/
        ├── train/good/
        ├── test/good/
        ├── test/<defect_type>/
        └── ground_truth/<defect_type>/
```

The dataset is not included in this repository. Do not commit dataset images or generated model weights.

MVTec AD is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). This project uses the dataset for non-commercial academic work; see the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) for the current license terms. Dataset credit: MVTec Software GmbH.

After installing the package, validate the local category:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py --root data\mvtec_ad --category bottle
```

## Evaluate a saved patch detector

Once a `.npz` artifact has been created, evaluate it on that category's official `test` split:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_detector.py
```

By default, the command reads the dataset root and category from `configs/default.yaml`, loads `models/<category>_patch_detector.npz`, and writes `outputs/<category>/metrics.json` and `outputs/<category>/per_image.csv`. Pass `--artifact`, `--dataset-root`, or `--output-dir` to override those paths. The artifact determines which category is evaluated. The first run may need to load the pinned pretrained ResNet18 weights.

Image metrics include every test image. Pixel metrics use normal test images as all-normal masks and anomalous images only when their ground-truth masks exist; missing masks are counted and excluded from pixel metrics. AUROC is calculated from continuous scores. Precision, recall, F1, and confusion counts use the calibrated thresholds stored in the artifact. AUC is reported as unavailable when the evaluated labels contain only one class. Test data is used only for final evaluation, never for memory-bank construction or calibration. Reports are generated only by running the command; no example benchmark scores are checked in.

## Run the local dashboard

Launch the Streamlit interface from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The **Inspect image** tab accepts one local image and displays the detector score, calibrated decision, heatmap overlay, and candidate-region details. The **Evaluation report** tab reads the saved JSON and CSV output from the evaluation command above. Use a saved `.npz` detector artifact in the sidebar; the first inspection may load the pinned pretrained ResNet18 weights. Uploaded inspection images are separate from test-split evaluation and do not change its metrics.

## Development checks

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Repository layout

- `configs/`: shared project settings.
- `data/`: local dataset instructions and ignored dataset files.
- `models/`: instructions for local model artifacts.
- `scripts/`: reproducible command-line workflows.
- `src/visionguard/`: configuration, data, processing, model, evaluation, visualization, and UI modules.
- `tests/`: fast tests using small temporary fixtures.
- `docs/`: methodology and experiment documentation as implementation proceeds.
- `outputs/`: generated local artifacts, ignored by Git apart from this README.

## Limitations

The configured mask threshold is an initial setting, not a calibrated decision boundary. Thresholds must be selected using a separate validation procedure, never the final test masks. This is an academic demonstration, not a certified industrial inspection system. An anomaly score is not automatically a calibrated defect probability. A qualified human inspector must verify suspected defects before manufacturing decisions.
