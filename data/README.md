# Dataset setup

Keep MVTec AD outside version control. Obtain it from the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) and follow its current access and usage conditions. MVTec AD is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0), which prohibits commercial use. Credit MVTec when describing results, and do not redistribute the dataset from this repository.

The expected local layout starts with:

```text
data/mvtec_ad/<category>/
├── train/good/
├── test/good/
├── test/<defect_type>/
└── ground_truth/<defect_type>/
```

The initial category is `bottle`. Dataset validation is available with:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py --root data\mvtec_ad --category bottle
```

The checker reports missing directories, unsupported files, and missing test-mask pairs. Evaluation-image discovery retains an expected mask path even when a mask is absent; it does not silently remove that sample. It does not download data.

## Screw category

The local `screw` category provides a more recognizable, fully framed object for the same anomaly-detection workflow. It contains 320 normal training images and 160 test images (41 normal and 119 anomalous), with pixel masks for every anomalous test image. Images are 1024x1024 grayscale frames. The camera remains fixed while the screw's in-frame rotation varies; this is not a multi-camera-view dataset.

The original [MVTec AD dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) is licensed under CC BY-NC-SA 4.0. The project uses the `foersben/mvtec-ad` Hugging Face mirror to download only this category. Keep the downloaded images out of Git; `/data/mvtec_ad/` is ignored.

To download the category into the expected layout:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='foersben/mvtec-ad', repo_type='dataset', allow_patterns=['screw/**'], local_dir='data/mvtec_ad')"
```

Validate it with:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py --root data\mvtec_ad --category screw
```
