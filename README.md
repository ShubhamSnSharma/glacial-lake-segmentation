# Glacial Lake Detection and Semantic Segmentation

A deep learning–based system for automatic glacial lake detection and high-resolution semantic segmentation from optical remote sensing satellite imagery.

---

## Problem Statement

Glacial lakes in high-mountain regions are expanding rapidly due to climate change and glacial retreat, creating severe risks of Glacial Lake Outburst Floods (GLOFs). Manual mapping from satellite imagery is labor-intensive and challenging due to complex terrain, mountain shadows, turbid waters, and seasonal ice/snow cover. 

This project performs **automatic pixel-wise binary semantic segmentation** of glacial lake water bodies from remote sensing RGB satellite imagery to support automated environmental monitoring, hazard assessment, and climate change studies.

---

## Input and Output Specification

| Component | Dimensions / Format | Description |
|---|---|---|
| **Input** | $512 \times 512 \times 3$ (RGB) | Multi-sensor satellite image patch (normalized to $[0, 1]$ and standardized) |
| **Output** | $512 \times 512 \times 1$ (Binary Mask) | Pixel-level classification ($1.0 = \text{Glacial Lake}, 0.0 = \text{Background / Terrain}$) |

---

## Model Architectures

This project implements and compares two distinct segmentation architectures:

### 1. Baseline Model: ResNet-34 FCN
- **Encoder**: ResNet-34 convolutional backbone (ImageNet-pretrained initialization).
- **Decoder**: Lightweight progressive 5-stage Transposed Convolution decoder ($512 \to 256 \to 128 \to 64 \to 32 \to 16 \to 1$) with Batch Normalization and ReLU activations.
- **Parameters**: ~24.08M total (21.28M encoder + 2.79M decoder).
- **Role**: Serves as our paper-inspired convolutional baseline.

### 2. Advanced Model: DeepLabV3+
- **Encoder**: ResNet-50 backbone with atrous (dilated) convolutions in `layer4` for an output stride of 16.
- **Context Module**: **Atrous Spatial Pyramid Pooling (ASPP)** capturing multi-scale context via parallel atrous convolution branches (dilations: $6, 12, 18$), $1\times 1$ convolution, and global image pooling.
- **Decoder**: Boundary-aware fusion decoder projecting low-level features (stride 4) and concatenating with $4\times$ upsampled ASPP features, followed by $3\times 3$ refinement convolutions.
- **Parameters**: ~40.35M total (23.51M encoder + 16.84M ASPP/decoder head).
- **Role**: Advanced semantic segmentation model designed to improve boundary crispness and multi-scale lake capture.

> **Note on Research Attribution**: DeepLabV3+ is an **advanced extension and comparison model** introduced in our project. It is **not** proposed by the original research paper (*Ma et al., 2025*), which evaluated ResNet-34 FCN. Furthermore, because Gaofen Image Dataset (GID) pretrained weights are not publicly released, our implementations utilize standard ImageNet-pretrained backbones.

---

## Dataset: GLID (Glacial Lake Image Dataset)

The project uses the benchmark **GLID** dataset (*Zenodo DOI: 10.5281/zenodo.14838695*):
- **Total Samples**: 18,367 annotated $512 \times 512$ RGB image and grayscale mask pairs.
- **Sensor Sources**: WorldView-2 (2 m), Gaofen-2 (4 m), Sentinel-2 (10 m), and Landsat-8 (30 m).
- **Official Split**:
  - **Training set (`GLID`)**: 16,000 samples
  - **Validation set (`val`)**: 2,367 samples
- **Class Distribution**: ~2.87% glacial lake pixels vs. ~97.13% background terrain (33.9 : 1 class imbalance, motivating our composite BCE + Dice loss formulation).

*(Note: The actual raw dataset files are stored outside the Git repository and must be downloaded or linked via Google Drive).*

---

## Repository Structure

```
glacial-lake-segmentation/
├── configs/
│   └── default.yaml              # Central configuration (paths, hyperparams, model selection)
├── src/
│   ├── __init__.py
│   ├── dataset.py                # GLIDDataset, synchronized augmentations, DataLoader factory
│   ├── model.py                  # Model builder factory (build_model)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet34_fcn.py       # Baseline: ResNet-34 + progressive FCN decoder
│   │   └── deeplabv3_plus.py     # Advanced: ResNet-50 + ASPP + DeepLabV3+ decoder
│   ├── losses.py                 # Combined BCE + Dice, DiceLoss, BinaryFocalLoss
│   ├── metrics.py                # IoU, F1/Dice, Precision, Recall, Accuracy, MetricTracker
│   ├── trainer.py                # Training/validation loop, scheduling, checkpointing, visualization
│   └── utils.py                  # Seed locking, visualization, normalization computation
├── scripts/
│   ├── inspect_dataset.py        # Dataset verification and EDA script
│   └── train.py                  # Main training execution script
├── tests/
│   ├── test_dataset.py           # Unit tests for dataset, augmentations, and loaders
│   └── test_model.py             # Unit tests for architectures, loss functions, and metrics
├── data/
│   └── README.md                 # Dataset download and extraction instructions
├── requirements.txt              # Python package dependencies
├── .gitignore                    # Excludes dataset, checkpoints, cache, and system files
└── README.md
```

---

## Installation & Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/ShubhamSnSharma/glacial-lake-segmentation.git
cd glacial-lake-segmentation

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Setup

Ensure the extracted dataset adheres to the expected directory structure:

```
data/
└── raw/
    ├── GLID/
    │   ├── images/     # 16,000 .png RGB image patches (512x512)
    │   └── labels/     # 16,000 .png binary masks (0=bg, 255=lake)
    └── val/
        ├── images/     # 2,367 .png RGB image patches (512x512)
        └── labels/     # 2,367 .png binary masks (0=bg, 255=lake)
```

To verify dataset integrity, directory structure, and class distributions:
```bash
PYTHONPATH=. python scripts/inspect_dataset.py --config configs/default.yaml
```

---

## Training

The training pipeline uses the `CombinedBCEDiceLoss` ($\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$), AdamW optimizer, gradient clipping, and Cosine Annealing learning rate scheduling with a 5-epoch linear warmup.

### 1. Train Baseline Model (ResNet-34 FCN)
```bash
PYTHONPATH=. python scripts/train.py \
    --config configs/default.yaml \
    --model resnet34_fcn \
    --epochs 50 \
    --batch-size 8
```

### 2. Train Advanced Model (DeepLabV3+)
```bash
PYTHONPATH=. python scripts/train.py \
    --config configs/default.yaml \
    --model deeplabv3plus \
    --epochs 50 \
    --batch-size 8
```

### 3. Pipeline Smoke Testing (Fast Verification)
To verify forward pass, backward pass, validation, checkpoint saving, and visual predictions on a small 3-batch subset:
```bash
PYTHONPATH=. python scripts/train.py --config configs/default.yaml --model resnet34_fcn --smoke-test
PYTHONPATH=. python scripts/train.py --config configs/default.yaml --model deeplabv3plus --smoke-test
```

---

## Evaluation & Metrics

Model evaluation is performed on the held-out validation set using the global `MetricTracker`:
- **Intersection over Union (IoU / Jaccard Index)**: $\text{IoU} = \frac{TP}{TP + FP + FN}$
- **Dice Coefficient / F1-Score**: $\text{F1} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$
- **Precision**: $\text{Precision} = \frac{TP}{TP + FP}$
- **Recall**: $\text{Recall} = \frac{TP}{TP + FN}$
- **Pixel Accuracy**: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$

---

## Running Unit & Integration Tests

The project includes an automated test suite verifying dataset loading, augmentations, mask binarization, model output shapes, loss functions, and metric tracking:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

---

## Running on Google Colab (GPU Training)

To train on Google Colab with a Tesla T4 / A100 GPU:

1. **Open a new Colab Notebook** and set the runtime to **GPU** (`Runtime -> Change runtime type -> T4 GPU`).
2. **Mount Google Drive** (where your private dataset is stored):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Clone the Repository & Install Dependencies**:
   ```bash
   !git clone https://github.com/ShubhamSnSharma/glacial-lake-segmentation.git
   %cd glacial-lake-segmentation
   !pip install -r requirements.txt
   ```
4. **Link or Symlink the Dataset from Google Drive**:
   ```bash
   mkdir -p data/raw
   # Example: symlink GLID and val directories from your Google Drive
   ln -s "/content/drive/MyDrive/GLID_dataset/GLID" "data/raw/GLID"
   ln -s "/content/drive/MyDrive/GLID_dataset/val" "data/raw/val"
   ```
5. **Run Tests to Verify Setup**:
   ```bash
   !PYTHONPATH=. python -m unittest discover -s tests -v
   ```
6. **Execute Training**:
   ```bash
   # Train ResNet-34 Baseline
   !PYTHONPATH=. python scripts/train.py --config configs/default.yaml --model resnet34_fcn --epochs 50 --batch-size 8

   # Train DeepLabV3+ Advanced Model
   !PYTHONPATH=. python scripts/train.py --config configs/default.yaml --model deeplabv3plus --epochs 50 --batch-size 8
   ```
