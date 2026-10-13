# Dataset setup

Keep MVTec AD outside version control. Obtain it from the [official MVTec AD page](https://www.mvtec.com/company/research/datasets/mvtec-ad) and review its current access and usage conditions before downloading or redistributing it.

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

The checker reports missing directories, unsupported files, and missing test-mask pairs. It does not download data.
