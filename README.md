# VisionGuard AI

**Intelligent Visual Inspection & Anomaly Detection**

VisionGuard AI is an MSc Digital Image Processing project exploring how classical image processing and deep feature-based anomaly detection can support industrial visual inspection. The initial research dataset is MVTec AD, starting with the `bottle` category.

## Current status

The foundation includes Python environment guidance, project configuration, device selection, logging, and dataset-layout validation. Initial image-processing tasks add safe image loading, explicit RGB/BGR conversion, aspect-ratio resizing, mask/map size alignment, Gaussian and median filtering, CLAHE enhancement, channel-aware intensity histograms, Canny edge detection, global, Otsu, and adaptive threshold masks, binary morphology, and contour-based candidate regions. The trained anomaly model, evaluation outputs, and Streamlit workflow are later implementation phases. No model scores or benchmark results are available yet.

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

The initial setup uses the CPU-only PyTorch wheel to keep installation smaller. Install PyTorch and TorchVision first:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

To use the NVIDIA GPU instead, replace `cpu` with `cu132` in the PyTorch command. The CUDA wheel is substantially larger. Keep the matched Torch/TorchVision versions together. Official choices are listed at [PyTorch Get Started](https://pytorch.org/get-started/locally/) and [previous versions](https://pytorch.org/get-started/previous-versions/).

You can use the environment without activating it by running commands through `.venv\Scripts\python.exe`.

## Dataset setup

Obtain MVTec AD from the [official dataset page](https://www.mvtec.com/company/research/datasets/mvtec-ad), review its access and usage conditions, and extract it so the selected category resembles:

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

After installing the package, validate the local category:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py --root data\mvtec_ad --category bottle
```

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
