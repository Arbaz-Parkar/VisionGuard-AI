# VisionGuard AI

**Intelligent Visual Inspection & Anomaly Detection**

VisionGuard AI is an MSc Digital Image Processing project exploring how classical image processing and deep feature-based anomaly detection can support industrial visual inspection. The initial research dataset is MVTec AD, starting with the `bottle` category.

## Current status

The foundation includes validated experiment settings, CPU/CUDA device selection, logging, dataset discovery, and a seeded normal-only calibration split. The Image Processing Lab currently includes safe image loading, explicit RGB/BGR conversion, resizing, flips/rotation/translation, Gaussian/median filtering, unsharp-mask sharpening, reproducible noise injection, template matching, ORB keypoints and descriptors, CLAHE, histograms, Canny edges, thresholding, morphology, contour candidates, NumPy spatial convolution, 2D FFT spectra, and frequency-domain low/high-pass filters. Both classical baselines are implemented. A frozen pretrained ResNet18 extractor produces layer2/layer3 patch embeddings, and a normal-only memory-bank builder supports random and greedy coreset sampling. Patch scoring, anomaly-map construction, image/pixel threshold calibration, artifact save/load, test-split evaluation for the patch detector and classical baselines, and a local Streamlit inspection/report interface are implemented. No benchmark results are available until the evaluators are run on the local dataset.

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

## Build a patch detector artifact

Build a detector from the configured category's normal training images:

```powershell
.\.venv\Scripts\python.exe scripts\build_memory_bank.py
```

The command makes a seeded, disjoint split of `train/good`, builds the sampled patch memory bank from the training portion, and calibrates image and pixel thresholds on the normal holdout. It uses the configured ResNet input size, sampling strategy, seed, and image/pixel percentiles. The resulting `.npz` artifact is saved under `models/`; it contains the memory bank and detector metadata, but not the pretrained ResNet weights. The first run may need to retrieve the pinned TorchVision weights. Use `--dataset-root`, `--category`, or `--output` to override the configured paths.

## Evaluate the saved patch detector

Evaluate the artifact on that category's official `test` split:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_detector.py
```

By default, the command reads the dataset root and category from `configs/default.yaml`, loads `models/<category>_patch_detector.npz`, and writes `outputs/<category>/metrics.json` and `outputs/<category>/per_image.csv`. Pass `--artifact`, `--dataset-root`, or `--output-dir` to override those paths. The artifact determines which category is evaluated. The first run may need to load the pinned pretrained ResNet18 weights.

Image metrics include every test image. Pixel metrics use normal test images as all-normal masks and anomalous images only when their ground-truth masks exist; missing masks are counted and excluded from pixel metrics. AUROC is calculated from continuous scores. Precision, recall, F1, and confusion counts use the calibrated thresholds stored in the artifact. AUC is reported as unavailable when the evaluated labels contain only one class. Test data is used only for final evaluation, never for memory-bank construction or calibration. Reports are generated only by running the command; no example benchmark scores are checked in.

## Evaluate the classical baselines

Run both classical methods on the official test split:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_classical.py
```

The evaluator fits the per-pixel reference model on the seeded normal training subset, calibrates both image thresholds and the reference pixel threshold on the disjoint normal holdout, then evaluates the test images and masks. It uses the configured threshold rules (`percentile` or `mean_plus_3std`); the rule-based method keeps its per-image Otsu pixel mask. Pixel precision/recall/F1 use each method's final mask, while pixel AUROC uses its continuous score map. Reports are written to `outputs/classical/<category>/metrics.json` and `per_image.csv`. Use `--dataset-root`, `--category`, or `--output-dir` to override the configured paths.

## Compare all three methods

After building the patch detector artifact, run the unified evaluation command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py
```

It evaluates the patch detector and both classical baselines on the same official test images and writes aligned metrics and per-image results to `outputs/metrics/<category>/`. The comparison checks that image and mask coverage agrees and requires an artifact built from the current normal split sizes; rebuild it with `scripts/build_memory_bank.py` when the split settings change. Test labels and masks are used only for final evaluation.

Pixel metrics use a common 256 by 256 grid. Score maps use bilinear resizing; ground-truth and predicted masks use nearest-neighbor resizing. Reports include image and pixel AUROC, precision, recall, F1, and IoU, including per-defect metrics. Three warm-up images are excluded from latency summaries. Each method reports preprocessing, inference, and total time with its device and input size. For the learned detector, inference covers feature extraction and nearest-neighbor scoring; for the classical methods, it covers the full baseline pipeline.

Image AUROC is the headline ranking metric. Accuracy is omitted because class imbalance can make it misleading.

The command also runs the random and greedy memory-bank sweep at 1%, 5%, 10%, 25%, and 100%. To keep greedy coreset selection tractable, it creates a seeded candidate pool of up to 2,048 patches from the full normal-only patch pool. The 100% point is the full candidate pool used for this experiment. Metadata records both pool and source candidate counts; sweep test AUROC does not change the configured detector default. Sweep metrics are saved as `memory_bank_sweep.json` and `.csv` under `outputs/metrics/<category>/`.

The evaluator saves ROC, confusion-matrix, score-distribution, and memory-bank-sweep plots under `outputs/plots/<category>/research/`. Ranked false-positive and false-negative figures go under `outputs/examples/<category>/research/`. `failure_analysis.md` lists the cases for human inspection; add plausible visual causes after reviewing the genuine figures. `experiment_metadata.json` records the config, category, split counts, seeds, device, warm-up settings, and library versions.

Run a five-image wiring check with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --smoke
```

Smoke outputs are separated under `smoke_test` folders and labeled as non-research results. Adjust the run with `--warmup-images`, `--pixel-metric-size`, `--sweep-ratios`, `--sweep-pool-size`, or `--failure-examples` as needed.

## Run the local dashboard

Launch the Streamlit interface from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Upload one image above the tabs, then use it in an inspection tab. **Inspect image** displays the detector score, calibrated decision, heatmap overlay, and candidate-region details. **Classical baseline** runs local-contrast detection with per-image Otsu thresholding. **Reference baseline** fits pixel statistics on the seeded normal training split and calibrates its pixel threshold on the disjoint held-out normal split; it does not use test images or masks and requires the uploaded image to match the training-image dimensions. **Evaluation report** displays the combined report from `outputs/metrics/<category>` by default; enter another report directory to view detector-only or classical-only reports. Use a saved `.npz` detector artifact in the sidebar for AI inspection; the first inspection may load the pinned pretrained ResNet18 weights. Uploaded inspection images are separate from test-split evaluation and do not change its metrics.

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

The image and pixel cutoffs are calibrated on held-out normal training images and may not generalize to new production conditions. Test masks are reserved for final evaluation. This is an academic demonstration, not a certified industrial inspection system. An anomaly score is not automatically a calibrated defect probability. A qualified human inspector must verify suspected defects before manufacturing decisions.
