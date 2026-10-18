# Limitations and interpretation

## Scope of the result

The real-data result currently covers only the MVTec AD `bottle` category and
83 test images. It does not establish performance on the other MVTec
categories, on different cameras, or on production images. The strong bottle
image AUROC is a result for this dataset split and should not be described as
a general industrial-inspection accuracy claim.

The local dataset came from the `foersben/mvtec-ad` Hugging Face mirror rather
than MVTec's official download form. The mirror labels the data CC BY-NC-SA
4.0, and MVTec's dataset page lists the same license. The run validated file
counts and mask coverage, but did not verify the mirror against official
checksums. Preserve the mirror attribution, original MVTec dataset citation,
and license notices. Dataset images and derived failure figures may carry the
dataset's usage conditions; check those terms before sharing them.

## Detector and calibration

The patch detector uses frozen ImageNet-pretrained ResNet18 features. It does
not learn the visual representation from this bottle dataset. Its normal
reference is a random sample of 10,000 patch embeddings from 178 normal images.
This is a deliberately compact PatchCore-inspired baseline, not the official
PatchCore implementation or a reproduction of its reported benchmark.

Image and pixel cutoffs are calibrated using 31 normal images. A 99.5th
percentile cutoff is an empirical estimate from a small held-out image set;
the many pixels within an image are spatially correlated and should not be
treated as millions of independent calibration samples. The requested 0.5%
false-positive target is not a guarantee on new images.

For this test split, the patch detector reached pixel AUROC `0.983`, but its
calibrated pixel mask had precision `0.335` and recall `0.995`. This shows why
ranking quality (AUROC) and the behavior of a single threshold are different:
the selected threshold found most defect pixels while also marking many
normal pixels. The threshold should not be adjusted against this test set. Any
new tuning should use a separate validation set, followed by one final test.

## Dataset and baseline assumptions

MVTec AD is a useful controlled benchmark, but the project tests one category
at a time and does not cover changing illumination, camera drift, different
product poses, unseen materials, or manufacturing-line variation. The
per-pixel reference assumes the product remains spatially aligned and the
normal appearance at each pixel is reasonably stable. The local-contrast
baseline depends on hand-set blur, threshold, and morphology operations and
may miss subtle or low-contrast defects.

Test masks are included in aggregate pixel metrics. Pixel metrics are
calculated on a common 256 by 256 grid, so they are affected by resampling and
by the large number of normal background pixels. The report includes AUROC,
precision, recall, F1, and IoU; it should not be presented as an official MVTec
benchmark score or compared directly with results that use a different
protocol.

The memory-bank sweep evaluates fixed bank designs on the test split for
descriptive comparison. It does not modify the default artifact. If sweep
results are used to choose a future configuration, that choice must be
confirmed on separate validation data before reporting a final test result.

## Runtime and deployment

The recorded comparison ran on CPU. Latency depends on the computer,
Torch/TorchVision build, thread settings, image size, and cache state; it is
not a hardware-independent speed claim. CUDA results would need to be measured
and reported separately.

This project is an academic demonstration, not a certified inspection system.
It does not provide a calibrated probability that a product is defective, a
production monitoring strategy, or a safety guarantee. Suspected defects
require qualified human review before manufacturing decisions.
