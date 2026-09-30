# Glacial Lake Detection and Semantic Segmentation

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset: GLID](https://img.shields.io/badge/Dataset-GLID%20(Zenodo)-green.svg)](https://doi.org/10.5281/zenodo.14838695)

An end-to-end deep learning system for automated glacial lake detection and high-resolution binary semantic segmentation from optical remote sensing satellite imagery.

---

## 1. Problem Statement & Motivation

Glacial lakes in high-mountain regions (such as the Himalayas, Hindu Kush, and Andes) are expanding rapidly as a direct consequence of global climate change and accelerating glacier retreat. The rapid growth of these glacial lakes poses catastrophic risks of **Glacial Lake Outburst Floods (GLOFs)**, which threaten downstream human settlements, critical infrastructure, agriculture, and hydroelectric installations.

Traditional glacial lake mapping relies on manual digitisation and semi-automated spectral water indices (e.g., NDWI, MNDWI). These classical approaches suffer from severe operational limitations:
- **Mountain Shadows & Topography**: Deep cast shadows in high-relief terrain have spectral signatures near-identical to clear water bodies, causing extensive false-positive detections.
- **Turbidity & Glacial Flour**: High concentrations of suspended glacial silt alter water reflectance, causing spectral index failures.
- **Ice & Snow Cover**: Partial surface freezing and floating ice break lake continuity.
- **Extreme Class Imbalance**: Glacial lake surface area accounts for less than 3% of regional satellite image pixels.

This project implements a reproducible deep learning segmentation framework that performs **pixel-wise binary classification** ($512 \times 512 \times 3 \text{ RGB} \to 512 \times 512 \times 1 \text{ binary mask}$) to delineate glacial lake boundaries across multi-sensor satellite imagery.

---

## 2. Project Objectives

1. **Robust Deep Learning Pipeline**: Build modular, tested pipelines for dataset loading, synchronized spatial augmentations, multi-architecture modeling, loss calculation, evaluation, and checkpoint management.
2. **Architecture Comparison**:
   - **Baseline**: ResNet-34 Fully Convolutional Network (FCN) with progressive upsampling.
   - **Advanced Model**: DeepLabV3+ with ResNet-50 backbone, Atrous Spatial Pyramid Pooling (ASPP), and boundary-aware multi-scale feature fusion.
3. **Rigorous Evaluation**: Benchmark performance across the entire 2,367-sample validation set using global confusion-matrix accumulation (IoU, F1/Dice, Precision, Recall, Accuracy).
4. **Visual Diagnostics & Error Analysis**: Categorize failure modes into systematic subsets (under-segmentation, over-segmentation, difficult environmental edge cases) to guide model refinement.
5. **Reproducibility & Portability**: Provide standalone CLI scripts, Google Colab GPU notebooks, unit tests, and isolated configuration management.

---

## 3. Dataset: GLID (Glacial Lake Image Dataset)

The project utilizes the benchmark **GLID** dataset:
- **Reference**: Ma D, Li J, Jiang L. (2025). *Efficient glacial lake mapping by leveraging deep transfer learning and a new annotated glacial lake dataset.* Journal of Hydrology, 657: 133072.
- **Zenodo Repository**: [DOI: 10.5281/zenodo.14838695](https://doi.org/10.5281/zenodo.14838695)
- **Patch Resolution**: $512 \times 512$ pixels, 3 channels (RGB).
- **Label Encoding**: Binary single-channel PNG ($255 = \text{Glacial Lake}, 0 = \text{Background}$).
- **Sensor Diversity**: Mixed unlabelled patches from WorldView-2 (2 m), Gaofen-2 (4 m), Sentinel-2 (10 m), and Landsat-8 (30 m).

### Dataset Split & Imbalance

| Split | Sample Count | Total Pixels | Lake Pixels (%) | Background Pixels (%) |
|---|---|---|---|---|
| **Training (`GLID`)** | 16,000 | 4,194,304,000 | ~2.87% | ~97.13% |
| **Validation (`val`)** | 2,367 | 620,494,848 | ~2.29% | ~97.71% |
| **Total** | **18,367** | **4,814,798,848** | **~2.79%** | **~97.21%** |

> **Class Imbalance Note**: Background terrain outnumbers lake pixels by approximately **33.9 : 1**, necessitating composite boundary/overlap loss functions.

### Expected Directory Layout

```
data/
└── raw/
    ├── GLID/               # Training split (16,000 samples)
    │   ├── images/         # 1.png, 2.png, ..., 16000.png (RGB)
    │   └── labels/         # 1.png, 2.png, ..., 16000.png (Binary masks)
    └── val/                # Validation split (2,367 samples)
        ├── images/         # 1.png, 2.png, ..., 2367.png (RGB)
        └── labels/         # 1.png, 2.png, ..., 2367.png (Binary masks)
```

*(See `data/README.md` for direct download links and archive extraction instructions).*

---

## 4. Model Architectures

```
                     ┌─────────────────────────────────────────────────────────┐
                     │              Input Image (3 x 512 x 512)                │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         │                                                 │
                         ▼                                                 ▼
        ┌──────────────────────────────────┐             ┌──────────────────────────────────┐
        │       ResNet34-FCN Baseline      │             │       DeepLabV3+ Advanced        │
        ├──────────────────────────────────┤             ├──────────────────────────────────┤
        │ • Encoder: ResNet-34             │             │ • Encoder: ResNet-50 (OS=16)     │
        │ • Params: 24.08M                 │             │ • Params: 40.35M                 │
        │ • Decoder: 5-Stage Progressive   │             │ • Context: ASPP (r=6, 12, 18)    │
        │   Transposed Conv (no skips)     │             │ • Decoder: Low-Level Skip Fusion │
        └────────────────┬─────────────────┘             └────────────────┬─────────────────┘
                         │                                                 │
                         ▼                                                 ▼
        ┌──────────────────────────────────┐             ┌──────────────────────────────────┐
        │  Binary Logits (1 x 512 x 512)   │             │  Binary Logits (1 x 512 x 512)   │
        └──────────────────────────────────┘             └──────────────────────────────────┘
```

### 1. ResNet34-FCN Baseline (`src/models/resnet34_fcn.py`)
- **Encoder**: ResNet-34 backbone with ImageNet-pretrained weights (`ResNet34_Weights.DEFAULT`).
- **Decoder**: Lightweight 5-stage progressive transposed-convolution upsampling head ($512 \to 256 \to 128 \to 64 \to 32 \to 16 \to 1$) with Batch Normalization and ReLU.
- **Parameters**: 24,081,393 total (~21.28M encoder + ~2.79M decoder).

### 2. DeepLabV3+ Advanced Model (`src/models/deeplabv3_plus.py`)
- **Encoder**: ResNet-50 backbone with atrous convolutions in `layer4` maintaining an Output Stride of 16.
- **ASPP Module**: Parallel multi-rate atrous convolutions (dilation rates $r \in \{6, 12, 18\}$), $1\times 1$ convolution, and Global Average Pooling with bilinear upsampling (256 output channels).
- **Decoder**: Low-level feature projection ($256 \to 48$ channels from `layer1`, stride 4) concatenated with $4\times$ upsampled ASPP features, refined through $3\times 3$ separable convolutions and final $4\times$ bilinear upsampling.
- **Parameters**: 40,351,809 total (~23.51M encoder + ~16.84M ASPP and decoder head).

---

## 5. Training Methodology

### Preprocessing & Augmentation
- **Normalization**: Standardized per-channel with ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Augmentation Pipeline**: Synchronized spatial transformations:
  - Random Horizontal Flip ($p=0.5$)
  - Random Vertical Flip ($p=0.5$)
  - Random Orthogonal Rotation ($0^\circ, 90^\circ, 180^\circ, 270^\circ$, $p=0.5$)

### Optimization & Loss Formulation
- **Optimizer**: AdamW ($\text{learning rate} = 1\times 10^{-4}$, $\text{weight decay} = 1\times 10^{-4}$).
- **Scheduler**: Cosine Annealing with linear warmup.
- **Loss Function**: Combined Binary Cross-Entropy and Soft Dice Loss:
  $$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}}(p, y) + 0.5 \cdot \mathcal{L}_{\text{Dice}}(p, y)$$
  $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum p_i y_i + \epsilon}{\sum p_i + \sum y_i + \epsilon}$$

### Evaluation Metrics
Validation metrics are computed over global pixel-level confusion matrix sums:
- **Intersection over Union (IoU)**: $\text{IoU} = \frac{TP}{TP + FP + FN}$
- **Dice / F1-Score**: $\text{F1} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$
- **Precision**: $\text{Precision} = \frac{TP}{TP + FP}$
- **Recall**: $\text{Recall} = \frac{TP}{TP + FN}$
- **Pixel Accuracy**: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$

---

## 6. Project Directory Structure

```
glacial-lake-segmentation/
├── configs/
│   ├── default.yaml                 # Master configuration (paths, DeepLabV3+, hyperparameters)
│   └── resnet34_fcn.yaml            # ResNet34-FCN baseline experiment configuration
├── src/
│   ├── __init__.py
│   ├── dataset.py                   # GLIDDataset, synchronized augmentations, DataLoader factory
│   ├── losses.py                    # CombinedBCEDiceLoss, DiceLoss, BinaryFocalLoss
│   ├── metrics.py                   # Global MetricTracker, confusion matrix, metric math
│   ├── model.py                     # Unified build_model factory function
│   ├── models/
│   │   ├── __init__.py
│   │   ├── deeplabv3_plus.py        # ResNet-50 + ASPP + DeepLabV3+ boundary decoder
│   │   └── resnet34_fcn.py          # ResNet-34 + 5-stage progressive FCN decoder
│   ├── trainer.py                   # Atomic checkpointing, validation, fit loop, scheduling
│   └── utils.py                     # Seed locking, config loader, EDA/normalization utilities
├── scripts/
│   ├── compare_models.py            # Side-by-side architecture comparison generator
│   ├── error_analysis.py            # Visual error analysis & category-based inspector
│   ├── evaluate.py                  # Full-validation evaluation pipeline (supports both models)
│   ├── inspect_dataset.py           # Dataset integrity, file discovery, and EDA statistics
│   ├── predict.py                   # Inference prediction & visual 4-panel figure generator
│   └── train.py                     # Training entry point (with smoke-test & resume support)
├── tests/
│   ├── test_dataset.py              # Unit tests for dataset discovery, loading, augmentations
│   └── test_model.py                # Unit tests for models, losses, metrics, checkpoint resuming
├── notebooks/
│   └── Glacial_Lake_Segmentation.ipynb # Google Colab GPU training & evaluation notebook
├── data/
│   └── README.md                    # Dataset download URLs and directory setup guide
├── requirements.txt                 # Pinned dependencies
├── .gitignore                       # Protection rules for data, weights, checkpoints, caches
└── README.md                        # Academic project documentation
```

---

## 7. Installation & Environment Setup

### Local Installation (macOS / Linux / Windows)

```bash
# 1. Clone repository
git clone https://github.com/ShubhamSnSharma/glacial-lake-segmentation.git
cd glacial-lake-segmentation

# 2. Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Google Colab GPU Setup

1. Open `notebooks/Glacial_Lake_Segmentation.ipynb` in Google Colab.
2. Navigate to **Runtime $\to$ Change runtime type $\to$ T4 GPU**.
3. Follow the guided cells to mount Google Drive, copy dataset archives to local NVMe SSD (`/content/dataset/`), install dependencies, and train with CUDA mixed precision (AMP).

---

## 8. Dataset Setup Instructions

1. Download `GLID.rar` and `val.zip` from Zenodo ([DOI: 10.5281/zenodo.14838695](https://doi.org/10.5281/zenodo.14838695)).
2. Place and extract the archives in `data/raw/`:

```bash
mkdir -p data/raw
# Extract validation split
unzip data/raw/val.zip -d data/raw/
# Extract training split (requires unrar: brew install rar / sudo apt install unrar)
unrar x data/raw/GLID.rar data/raw/
```

3. Verify dataset structure and compute dataset statistics:
```bash
python scripts/inspect_dataset.py --config configs/default.yaml
```

---

## 9. Model Training Commands

### 1. Fast Smoke Test (Pipeline Verification)
Verify model initialization, forward pass, backward pass, loss calculation, and checkpoint saving on a 3-batch sample:
```bash
python scripts/train.py --config configs/default.yaml --model resnet34_fcn --smoke-test
python scripts/train.py --config configs/default.yaml --model deeplabv3plus --smoke-test
```

### 2. Train ResNet34-FCN Baseline
```bash
python scripts/train.py \
    --config configs/resnet34_fcn.yaml \
    --model resnet34_fcn \
    --output-dir resnet34_fcn \
    --epochs 30 \
    --batch-size 8
```

### 3. Train DeepLabV3+ Advanced Model
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --model deeplabv3plus \
    --output-dir . \
    --epochs 50 \
    --batch-size 8
```

---

## 10. Resuming Training from Checkpoints

The training pipeline uses atomic writes (`latest_model.pth` and `best_model.pth`) that preserve model weights, optimizer state, learning rate scheduler state, completed epoch count, and validation metric history.

To resume an interrupted run:
```bash
# Resume DeepLabV3+
python scripts/train.py \
    --config configs/default.yaml \
    --model deeplabv3plus \
    --resume checkpoints/latest_model.pth

# Resume ResNet34-FCN Baseline
python scripts/train.py \
    --config configs/resnet34_fcn.yaml \
    --model resnet34_fcn \
    --output-dir resnet34_fcn \
    --resume resnet34_fcn/checkpoints/latest_model.pth
```

---

## 11. Evaluation, Prediction, Error Analysis & Comparison

### Full-Validation Evaluation (`scripts/evaluate.py`)
Evaluates model checkpoints across all 2,367 validation samples:
```bash
# DeepLabV3+ evaluation
python scripts/evaluate.py \
    --checkpoint checkpoints/best_amp_model.pth \
    --val-dir data/raw/val \
    --output-dir results/evaluation

# ResNet34-FCN baseline evaluation
python scripts/evaluate.py \
    --checkpoint resnet34_fcn/checkpoints/best_model.pth \
    --model resnet34_fcn \
    --val-dir data/raw/val \
    --output-dir resnet34_fcn/results/evaluation
```

### Visual Inference Predictions (`scripts/predict.py`)
Generates 4-panel visual plots (Satellite Image, Ground Truth, Predicted Mask, Probability Heatmap):
```bash
python scripts/predict.py \
    --checkpoint checkpoints/best_amp_model.pth \
    --n-samples 12 \
    --output-dir results/predictions
```

### Visual Error Analysis (`scripts/error_analysis.py`)
Categorizes validation samples into Good Predictions, Under-segmentation (FN), Over-segmentation (FP), and Difficult Environmental Cases:
```bash
python scripts/error_analysis.py \
    --checkpoint checkpoints/best_amp_model.pth \
    --val-dir data/raw/val \
    --n-samples 20 \
    --scan-pool 150 \
    --output-dir results/error_analysis
```

### Model Comparison Table (`scripts/compare_models.py`)
Generates side-by-side Markdown, text, and JSON benchmark comparisons:
```bash
python scripts/compare_models.py \
    --dlv3-checkpoint checkpoints/best_amp_model.pth \
    --dlv3-eval results/evaluation/eval_metrics.json \
    --fcn-checkpoint resnet34_fcn/checkpoints/best_model.pth \
    --fcn-eval resnet34_fcn/results/evaluation/eval_metrics.json \
    --output-dir results/comparison
```

---

## 12. Current Project Benchmarks & Results

All reported metrics reflect evaluation across the full **2,367 validation samples** ($620,494,848$ total pixels):

### Full Validation Benchmark (DeepLabV3+)

| Metric | Measured Value | Confusion Matrix Elements | Count |
|---|---|---|---|
| **Dice / F1-Score** | **93.97%** ($0.939675$) | **True Positives (TP)** | 13,109,933 |
| **Mean IoU (Jaccard)** | **88.62%** ($0.886213$) | **False Positives (FP)** | 602,409 |
| **Precision** | **95.61%** ($0.956068$) | **False Negatives (FN)** | 1,080,860 |
| **Recall** | **92.38%** ($0.923834$) | **True Negatives (TN)** | 605,701,646 |
| **Pixel Accuracy** | **99.73%** ($0.997287$) | **Total Pixels Evaluated** | 620,494,848 |

- **Inference Speed**: ~205.0 ms per $512 \times 512$ image (on Apple Silicon MPS).
- **Error Analysis Findings**: False Negatives ($1.08 \times 10^6$ px) exceed False Positives ($0.60 \times 10^6$ px) by approximately $1.8 : 1$, indicating that subtle boundary omissions in turbid water or shadows are more common than false alarms.

---

## 13. Limitations & Implementation Deviations

1. **Backbone Pretraining Initialization**:
   - *Paper Approach*: Pretrained on Gaofen Image Dataset (GID).
   - *Our Implementation*: Pretrained on standard ImageNet-1k (`torchvision.models`). GID pretraining weights are not publicly accessible.
2. **FCN Decoder Specification**:
   - The original paper outlines a general "ResNet34-FT with FCN decoder" without releasing sub-layer kernel sizes or channel transitions. We implement a structured 5-stage progressive transposed convolution decoder ($512 \to 256 \to 128 \to 64 \to 32 \to 16 \to 1$).
3. **Multi-Sensor Spatial Scale Variations**:
   - GLID aggregates patches ranging from 2 m (WorldView-2) to 30 m (Landsat-8) without sensor metadata tags per sample. Lakes in low-resolution patches span very few pixels, whereas high-resolution patches resolve intricate shoreline details.
4. **Severe Environmental Edge Cases**:
   - Cast terrain shadows from steep mountain ridges, floating ice sheets, and dried lakebeds with sediment residues present visual ambiguities that can challenge pure RGB spectral representations without auxiliary elevation (DEM) data.

---

## 14. Artifact & Data Storage Notice

> **Important Note**:
> In accordance with open-source and version control best practices, raw datasets (`data/raw/`), pre-extracted imagery, and large binary model checkpoints (`.pth`, `.ckpt`) are **not tracked in this GitHub repository**. 
> - Dataset archives are accessible directly from Zenodo ([DOI: 10.5281/zenodo.14838695](https://doi.org/10.5281/zenodo.14838695)).
> - Pre-trained model checkpoints and training logs are archived separately in Google Drive.

---

## 15. Automated Test Suite

Run the full automated test suite (27 unit and integration tests):

```bash
python -m unittest discover -s tests -v
# or using pytest
pytest -v
```

The test suite validates:
- Dataset directory auto-discovery, sample discovery, shape assertions, and duplicate handling.
- ResNet34-FCN and DeepLabV3+ architecture forward passes, shape contracts, and parameter counts.
- Gradient backpropagation flow through backbones and decoder heads.
- Combined BCE + Dice loss numerical stability and reduction.
- MetricTracker precision, recall, F1, IoU, and confusion matrix arithmetic.
- Checkpoint atomic saving, state restoration, and training resume continuation.
