# Dataset setup

Dataset images are local and ignored by Git. The primary experiment uses MVTec AD screw; the optional secondary category is bottle. The local copy used for the recorded runs came from the [foersben/mvtec-ad Hugging Face mirror](https://huggingface.co/datasets/foersben/mvtec-ad), not the official MVTec download host.

MVTec AD is licensed CC BY-NC-SA 4.0. Credit MVTec when describing results, do not use the data commercially, and review the license before sharing data or derived examples.

Expected layout:

~~~text
data/mvtec_ad/<category>/
├── train/good/
├── test/good/
├── test/<defect_type>/
└── ground_truth/<defect_type>/
~~~

Download only the primary category:

~~~powershell
.\.venv\Scripts\python.exe -m pip install huggingface_hub
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='foersben/mvtec-ad', repo_type='dataset', allow_patterns=['screw/**'], local_dir='data/mvtec_ad')"
~~~

Validate it:

~~~powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py --root data\mvtec_ad --category screw
~~~

The local screw category contains 320 normal training images and 160 test images (41 normal, 119 anomalous), with complete anomaly masks. Images are 1024×1024 grayscale frames with a fixed camera and varying in-frame screw pose. The bottle category contains 209 normal training images and 83 test images.

The checker reports missing directories, unsupported files, and missing test-mask pairs. It does not download data.