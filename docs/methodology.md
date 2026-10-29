# Methodology

## Objective and dataset

The primary experiment evaluates normal-versus-anomalous detection and pixel localization on the MVTec AD screw category. The local data came from the [foersben/mvtec-ad Hugging Face mirror](https://huggingface.co/datasets/foersben/mvtec-ad/tree/main/screw), not MVTec's official download host. Preserve MVTec attribution and the CC BY-NC-SA 4.0 terms; the dataset and derived examples are not for commercial use.

The screw frames are 1024×1024 grayscale images from a fixed camera, with varied in-frame screw poses. This is not a multi-camera or multi-viewpoint collection.

| Partition | Images | Purpose |
| --- | ---: | --- |
| train/good | 320 | Normal-only model development |
| Memory-bank subset | 272 | Fit patch reference and pixel baseline |
| Normal calibration subset | 48 | Select normal-only decision thresholds |
| test/good | 41 | Final normal evaluation |
| Anomalous test images | 119 | Final image and mask evaluation |
| Missing anomaly masks | 0 | All test samples available for pixel evaluation |

The 320 normal training images are split deterministically with seed 42: 85% for fitting and 15% for calibration. The exact split is recorded in outputs/metrics/screw/experiment_metadata.json. Test images and masks do not participate in fitting or threshold selection.

## Patch-memory detector

### Preprocessing and embeddings

Images are converted to RGB, resized to 256×256 while preserving aspect ratio, and padded. Pixel values are normalized with the ImageNet mean and standard deviation. A frozen TorchVision ResNet18 with ResNet18_Weights.IMAGENET1K_V1 produces intermediate layer2 and layer3 feature maps. The aligned maps are pooled and concatenated into spatial patch embeddings. The backbone is pretrained on ImageNet and is not fine-tuned on MVTec.

### Normal memory bank and scoring

Patch embeddings are extracted only from the 272 normal fitting images. The source pool contains 278,528 candidate patches. The saved artifact retains at most 10,000 patches using seeded random sampling (seed 42). At inference, each patch receives its Euclidean distance to the nearest stored normal embedding. The distance map is upsampled, Gaussian-smoothed with sigma 4, and mapped to the image geometry. The image score is the maximum map value.

This is a compact PatchCore-inspired detector, not an official PatchCore implementation or reproduction. The .npz artifact stores the normal bank and settings; the pretrained ResNet weights are cached separately by TorchVision.

### Calibration

The image and pixel thresholds are the empirical 99.5th percentiles of scores from the 48 held-out normal images. For this artifact, the image threshold is 2.19379 and the pixel threshold is 1.72057. The nominal 0.5% calibration false-positive target is not a guarantee on new data.

## Classical comparison methods

1. **Rule-based local contrast:** grayscale conversion, Gaussian filtering, CLAHE, local-background subtraction, Otsu thresholding, and morphological cleanup.
2. **Per-pixel normal reference:** per-pixel RGB mean and standard deviation estimated from the 272 normal fitting images; scores use the maximum channel-wise standardized deviation with a standard-deviation floor.

Both methods use the same normal-only fitting/calibration split. Neither uses defect images or test masks to set parameters or thresholds.

## Evaluation

All methods are evaluated on the same 160 official test images. Image AUROC uses one continuous score per image. Pixel AUROC uses continuous anomaly maps. Precision, recall, F1, and IoU use the method's thresholded prediction. Maps and masks are evaluated on a 256×256 grid; scores use bilinear resizing and masks use nearest-neighbor resizing. Three warm-up images are excluded from latency summaries.

The results show why thresholded metrics and AUROC must be reported together: the patch detector ranks pixel anomalies well, but its calibrated pixel threshold has low precision, and its image threshold misses many anomalous screws. See [results](results.md).

The memory-bank sweep compares random and greedy sampling at 1%, 5%, 10%, 25%, and 100% of a seeded candidate pool capped at 2,048 normal patches. Thus, 100% in that sweep means all 2,048 pool patches, not all 278,528 source candidates. The sweep is descriptive and does not change the default artifact.

## Reproduction

From the repository root with the local dataset present:

~~~powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py --root data\mvtec_ad --category screw
.\.venv\Scripts\python.exe scripts\build_memory_bank.py --category screw
.\.venv\Scripts\python.exe scripts\evaluate.py --category screw
.\.venv\Scripts\python.exe -m streamlit run app.py
~~~

The evaluation writes metrics and metadata to outputs/metrics/screw/, figures to outputs/plots/screw/research/, and ranked failure examples to outputs/examples/screw/research/. These generated files are local and ignored by Git. Selected plots and a measured metric snapshot are included in docs/results.md.

## References

- Bergmann, P. et al. [MVTec AD—A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html), CVPR 2019; [official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
- He, K. et al. [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_for_Image_Recognition_CVPR_2016_paper.html), CVPR 2016; [TorchVision ResNet18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- Roth, K. et al. [Towards Total Recall in Industrial Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html), CVPR 2022. VisionGuard uses an inspired compact patch-memory design and does not claim to reproduce PatchCore.