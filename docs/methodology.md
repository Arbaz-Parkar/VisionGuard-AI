# Methodology

## Study scope

VisionGuard AI is an educational visual-inspection project. This experiment
compares a frozen-feature patch anomaly detector with two classical image
processing baselines on one MVTec AD category: `bottle`. The detector is
PatchCore-inspired; it is not a reproduction or official implementation of
PatchCore.

MVTec AD provides defect-free training images, test images containing normal
and anomalous samples, and pixel-level masks for defects. The original dataset
paper and MVTec's dataset page describe its categories and annotations
([Bergmann et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html),
[MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)).

## Data source and experiment snapshot

The local run used the `bottle` folder from the
[`foersben/mvtec-ad` Hugging Face mirror](https://huggingface.co/datasets/foersben/mvtec-ad/tree/main/bottle),
not a download from MVTec's official download form. The mirror lists
CC BY-NC-SA 4.0 and contains its own `license.txt` and `readme.txt`. MVTec's
dataset page also states CC BY-NC-SA 4.0 for MVTec AD. Keep the mirror's
provenance and attribution with any report, and review the dataset terms before
redistributing images or image-derived examples.

The experiment run on 2026-10-18 reported:

| Split | Count | Use |
| --- | ---: | --- |
| `train/good` | 209 | Seeded split into a memory-bank subset and a normal calibration subset |
| Memory-bank subset | 178 | Fit the patch feature reference |
| Normal calibration subset | 31 | Select image and pixel thresholds |
| Test, `good` | 20 | Final evaluation as normal images with all-zero masks |
| Test, anomalous | 63 | Final evaluation using defect labels and ground-truth masks |
| Missing anomaly masks | 0 | All anomalous test images were usable for pixel metrics |

The 31-image calibration size is the rounded 15% split configured in
`configs/default.yaml`. Split seed is 42. The same seeded normal split is used
when evaluating the classical methods. No test image or mask is used to build
the memory bank or calibrate its thresholds.

## Patch anomaly detector

### Input preparation and features

Images are converted to RGB, resized to 256 by 256 while preserving aspect
ratio, and padded where needed. Pixel values are scaled to `[0, 1]`. The model
then applies the ImageNet channel mean and standard deviation expected by the
pretrained TorchVision ResNet18 weights.

The ResNet18 backbone is frozen and run in evaluation mode. The detector reads
the intermediate `layer2` and `layer3` feature maps, applies 3 by 3 average
pooling, resizes the `layer3` map to the `layer2` grid, and concatenates both
feature maps channel-wise. The backbone weights are the pinned
`ResNet18_Weights.IMAGENET1K_V1` weights, not weights trained on MVTec AD
([ResNet, He et al., CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_for_Image_Recognition_CVPR_2016_paper.html),
[TorchVision ResNet18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)).

### Normal memory bank

Only the 178 images in the seeded `train/good` memory-bank subset are used to
create patch embeddings. This run produced 182,272 candidate embeddings. The
configured seeded random sampler requested 10% of candidates, capped at 10,000
stored patches, so the artifact contains 10,000 embeddings. The artifact also
stores the category, preprocessing settings, sampling metadata, and calibrated
thresholds; the pretrained ResNet weights remain in TorchVision's weight
cache.

At inference, each test patch is scored by its Euclidean distance to the
nearest stored normal patch. Distances are reshaped to a feature grid,
bilinearly upsampled, Gaussian-smoothed with sigma 4, and restored to the input
image geometry. The image score is the maximum value in its continuous anomaly
map. These design choices are inspired by patch-memory anomaly detection, but
the project uses a smaller ResNet18 feature extractor and a random sampled bank
rather than claiming to implement the PatchCore paper's method
([Roth et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)).

### Threshold calibration

The image and pixel cutoffs are the empirical 99.5th percentiles of scores
from the 31 held-out normal calibration images. The requested false-positive
rate target is 0.5%; a finite calibration set, ties, and differences between
normal calibration and test distributions can make the observed test rate
differ from that target. The calibrated values in this run were:

- Image threshold: `1.5593989491`
- Pixel threshold: `1.2359335423`

These thresholds were fixed before test evaluation. Test labels and masks are
used only to calculate final metrics and create diagnostic outputs.

## Classical comparison methods

The unified evaluation includes two transparent baselines:

1. **Rule-based local contrast:** convert to grayscale, blur, apply CLAHE,
   subtract a strongly blurred local background, use Otsu thresholding, then
   clean the binary mask with opening and closing. Its continuous difference
   map supports AUROC; its binary pixel mask comes from the per-image Otsu
   threshold and morphology.
2. **Per-pixel reference:** calculate per-pixel RGB mean and population
   standard deviation from the normal memory-bank subset. A test pixel score
   is the maximum channel-wise absolute deviation divided by a standard
   deviation floor. Image and pixel thresholds are calibrated from held-out
   normal images using the configured rules.

Both baselines use only the same normal training split and calibration split;
their fitting and cutoffs do not use defect images or test masks.

## Evaluation protocol

The full test split is evaluated for all three methods. Image AUROC uses the
continuous image scores. Pixel AUROC uses continuous anomaly maps; pixel
precision, recall, F1, and intersection over union use the fixed binary masks
produced by each method. Normal test images have all-zero ground-truth masks.
For pixel metrics, score maps are resized to a common 256 by 256 grid with
bilinear interpolation; ground-truth and predicted masks use nearest-neighbor
resizing. This project reports these metrics from its own evaluator rather
than claiming the official MVTec evaluation score.

Latency summaries exclude three warm-up images. The recorded total time
includes image preprocessing and the method's inference work. The experiment
metadata records the selected device and software versions so the numbers can
be interpreted in their run environment.

The memory-bank sweep compares random and greedy selection at ratios 1%, 5%,
10%, 25%, and 100%. It first draws a seeded normal-only candidate pool capped
at 2,048 patches from the full candidate pool of 182,272. In this sweep, 100%
means the full 2,048-patch candidate pool, not every source patch. The sweep is
reported as a fixed design comparison; it does not change the saved default
artifact or select a model using test performance.

## Reproduce the run

Run these from the repository root after placing the category at
`data/mvtec_ad/bottle/` and installing the project dependencies:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py --root data\mvtec_ad --category bottle
.\.venv\Scripts\python.exe scripts\build_memory_bank.py
.\.venv\Scripts\python.exe scripts\evaluate.py
```

The first command checks the MVTec folder structure and mask coverage. The
second creates `models/bottle_patch_detector.npz`. The third evaluates the
full test split and produces the unified report.

## Generated experiment files

The full run writes local, generated files under `outputs/`:

- `outputs/metrics/bottle/metrics.json` and `per_image.csv`: aggregate and
  per-image method results.
- `outputs/metrics/bottle/memory_bank_sweep.json` and `.csv`: sweep settings
  and measurements.
- `outputs/metrics/bottle/experiment_metadata.json`: configuration, split,
  device, timing, and library-version details.
- `outputs/plots/bottle/research/`: ROC, confusion matrix, score-distribution,
  and memory-bank sweep plots.
- `outputs/examples/bottle/research/`: ranked false-positive and
  false-negative examples, plus `failure_analysis.md` for qualitative review.

Dataset images, the saved detector artifact, and generated experiment outputs
are local and are not part of the source-code repository.
