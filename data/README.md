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
