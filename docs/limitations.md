# Limitations and interpretation

## Dataset coverage

The primary result is one MVTec AD category, screw, evaluated on 160 test images. The secondary bottle run contains 83 test images. These are controlled benchmark categories from the same dataset source, not evidence of performance across products, factories, cameras, or production conditions. The screw data uses a fixed camera with changing in-frame pose; it does not test multiple camera viewpoints.

The local copy came from the foersben/mvtec-ad Hugging Face mirror, not MVTec's official download form. The dataset is subject to CC BY-NC-SA 4.0. Preserve attribution and license notices, do not use the data commercially, and review the terms before redistributing images or derived examples. The project validated image/mask structure but did not verify the mirror against official checksums.

## Detector and thresholds

The patch detector uses frozen ImageNet-pretrained ResNet18 features and a random sample of up to 10,000 normal patch embeddings. It does not learn its representation from the screw or bottle training images. It is a compact PatchCore-inspired method, not the official PatchCore implementation.

Thresholds are calibrated from 48 normal screw images and 31 normal bottle images using the empirical 99.5th percentile. That small holdout does not guarantee a 0.5% false-positive rate on future data. Pixels within an image are spatially correlated and must not be treated as millions of independent calibration examples.

For screw, image AUROC is 0.7510 but the calibrated image threshold detects only 19 of 119 anomalous test images. Pixel AUROC is 0.9825, while pixel precision is 0.1057 at the selected threshold. Ranking and thresholded operating performance differ substantially. Do not adjust thresholds on this test split and then describe the same split as an untouched final test.

## Baselines and evaluation protocol

The per-pixel reference assumes consistent spatial alignment between the reference images and the image under inspection. The local-contrast baseline depends on hand-set blur, threshold, and morphology parameters. Neither method is expected to transfer automatically to new imaging conditions.

Pixel metrics are computed on a common 256×256 grid, so interpolation affects results and the large normal background can dominate pixel counts. The evaluator's metrics are not official MVTec scores and should not be compared directly with results using a different protocol. The memory-bank sweep is descriptive; selecting a configuration based on its test performance would require a separate validation set and a new untouched test evaluation.

## Use and runtime

The reported runs used CPU. Latency depends on the machine, software build, image size, and cache state. GPU results require separate measurement.

The app provides anomaly scores and candidate regions, not calibrated defect probabilities, defect-type labels, a production monitoring plan, or a safety guarantee. A qualified human must review suspected defects before manufacturing decisions.