# Code walkthrough

## Application flow

app.py is the Streamlit entry point. It loads configs/default.yaml, presents a shared image uploader and category-specific detector settings, then renders three tabs:

- **Image Processing Lab** calls src/visionguard/lab_ui.py and applies an explicitly selected operation to the uploaded image.
- **Inspect** runs the chosen patch detector or classical baseline and displays a score, heatmap, threshold decision, and candidate regions.
- **Results** reads the saved evaluation JSON and CSV; it does not rerun model evaluation.

The dashboard is local. Uploaded images are separate from the benchmark test split and do not change benchmark metrics.

## Inspection pipeline

For the patch detector, model/features.py applies RGB conversion, aspect-preserving resize/padding, and ImageNet normalization before extracting frozen ResNet18 layer2 and layer3 features. model/memory_bank.py holds normal patch embeddings and provides nearest-neighbor distances. model/calibration.py stores the image and pixel thresholds. model/inference.py turns patch distances into a smoothed anomaly map and image score. model/artifacts.py reads and writes the category-specific .npz artifact without pickle.

The processing/ package contains the classical operations and baselines. The rule-based baseline compares local image contrast after filtering and morphology. The per-pixel reference scores deviations from normal pixel statistics and expects matching image alignment.

## Data and experiment flow

data/dataset.py discovers normal training images, test images, and ground-truth masks. data/validation.py checks the MVTec directory layout and mask coverage. split_normal_images makes the seeded, disjoint fitting/calibration split; defect labels and masks are reserved for final evaluation.

scripts/build_memory_bank.py creates the patch detector artifact from normal training data. scripts/evaluate.py evaluates the detector and both baselines on the category's test split, calculates image and pixel metrics, and generates comparison artifacts. scripts/check_dataset.py validates a local category. evaluation.py, classical_evaluation.py, and comparison.py contain metric and report logic. analysis_outputs.py creates plots, run metadata, and ranked error examples; sweep.py evaluates memory-bank sampling designs.

## Image Processing Lab

The Lab dispatches to functions in src/visionguard/processing/. The implemented groups include:

- RGB/HSV/Lab/grayscale transforms and intensity changes.
- Histograms, equalization, and CLAHE.
- Gaussian/median filtering, sharpening, and convolution.
- Noise addition/removal and frequency-domain filtering.
- Template matching, Sobel/Laplacian/LoG/Canny edges, and morphology.
- Thresholding, contours, watershed, Harris corners, HOG, Hough lines/circles, and ORB features.

The Lab result is an image-processing demonstration. It does not train or alter the anomaly detector.