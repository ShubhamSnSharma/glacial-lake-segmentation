# Glacial Lake Segmentation using Deep Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: GLID](https://img.shields.io/badge/Dataset-GLID-green.svg)](https://doi.org/10.5281/zenodo.14838695)
[![Google Colab](https://img.shields.io/badge/Compute-Google_Colab_GPU-orange.svg)](https://colab.research.google.com/)

An empirical evaluation and benchmarking of deep-learning semantic segmentation architectures for automated glacial lake delineation from multi-spectral satellite imagery. 

Developed as a **B.Tech. Capstone Project** in the Department of Computer Science and Engineering at **Bennett University**, Greater Noida, India (Academic Year 2024–2025).

---

## Academic & Team Information

- **Institution:** Bennett University
- **Department:** Computer Science and Engineering
- **Programme:** B.Tech. in Computer Science and Engineering
- **Subject:** Capstone Project
- **Project Guide:** Dr. Rakesh Sahu, Associate Professor

### Student Team Members

| Student Name | Enrolment ID |
| :--- | :--- |
| **Shubham Sharma** | `E23CSEU2407` |
| **Abhinav** | `E23CSEU0110` |
| **Pragya** | `E23CSEU0109` |

---

## Executive Summary

Glacial lakes in high-altitude mountain environments (such as the Himalayas and Tibetan Plateau) are vital indicators of cryosphere dynamics and climate change. However, manual delineation from satellite imagery is challenging due to steep mountain shadows, floating ice, turbid water, and massive spatial coverage.

This project implements, evaluates, and compares three deep learning architectures for binary semantic segmentation ($0 = \text{Background}, 1 = \text{Glacial Lake}$) on the **Glacial Lake Image Dataset (GLID)**:
1. **ResNet34-FCN** (Baseline Fully Convolutional Network)
2. **DeepLabV3+** (Multi-scale Atrous Spatial Pyramid Pooling with ResNet-50 backbone)
3. **Attention ResUNet34** (ResNet-34 encoder with Attention Gates on skip connections)

All models were trained independently using GPU compute on **Google Colab** and systematically evaluated on a common validation partition of 2,367 images.

---

## Model Architectures & Parameter Counts

| Architecture | Backbone Encoder | Decoder Mechanism | Exact Parameter Count | Best Checkpoint |
| :--- | :--- | :--- | :---: | :---: |
| **ResNet34-FCN** | ResNet-34 (ImageNet) | 5-stage progressive transposed convolution | **24,079,153** | Epoch 29 |
| **DeepLabV3+** | ResNet-50 (ImageNet) | ASPP (rates: 6, 12, 18) + shallow conv2 projection | **40,347,041** | Epoch 26 (AMP) |
| **Attention ResUNet34** | ResNet-34 (ImageNet) | 4-stage U-Net with additive Attention Gates | **25,313,269** | Epoch 30 |

---

## Experimental Benchmark Results

Each trained checkpoint was evaluated across the complete validation split (**2,367 satellite image patches**, $512 \times 512$ resolution) using a standard binary decision threshold of $0.5$:

### Global Pixel-Level Evaluation

| Model | Dice / F1 Score | IoU (Jaccard) | Precision | Recall | Pixel Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DeepLabV3+ (Best)** | **93.9673%** | **88.6211%** | **95.6100%** | **92.3802%** | **99.7287%** |
| **ResNet34-FCN** | 93.4105% | 87.6358% | 95.2569% | 91.6344% | 99.7043% |
| **Attention ResUNet34** | 91.7360% | 84.7335% | 93.1279% | 90.3850% | 99.6276% |

### Per-Image Distribution & Zero-Dice Analysis

| Model | Mean Per-Image Dice | Mean Per-Image IoU | Zero-Dice Images (out of 2,367) |
| :--- | :---: | :---: | :---: |
| **DeepLabV3+** | **85.5746%** | **79.5614%** | **109** |
| **ResNet34-FCN** | 81.3355% | 74.8501% | 188 |
| **Attention ResUNet34** | 79.7552% | 73.0104% | 188 |

> **Key Takeaway:** DeepLabV3+ demonstrated the strongest overall segmentation performance and highest robustness, with the lowest count of zero-Dice failure cases (109 vs. 188). These metrics reflect validation-set evaluations on the GLID dataset.

---

## Qualitative Overlays & Error Analysis

Inspection of pixel-level error overlays revealed three primary edge cases:
- **Small Lake Regions:** Small water bodies have high perimeter-to-area ratios; minor edge misalignments result in steep Dice penalties.
- **Topographic Mountain Shadows:** Steep ridgelines cast dark terrain shadows that visually and spectrally mimic clear lake water in NIR bands.
- **Turbid / Frozen Surfaces:** Heavy suspended glacial silt or winter ice alters surface reflectance, occasionally leading to false-negative omissions.

Diagnostic figures with True Positive (Green), False Positive (Red), and False Negative (Blue) overlays are available in [`reports/figures/`](reports/figures/).

---

## Repository Structure

This repository follows a clean, notebook-centric layout:

```text
glacial-lake-segmentation/
├── .gitignore
├── README.md                                                   # Main project overview & benchmarks
├── requirements.txt                                            # Environment dependencies
│
├── data/
│   └── README.md                                               # Dataset specifications & Zenodo download guide
│
├── notebooks/                                                  # Google Colab Training & Comparison Notebooks
│   ├── Glacial_Lake_ResNet34_FCN_Training.ipynb                # ResNet34-FCN baseline training
│   ├── Glacial_Lake_DeepLabV3_Training.ipynb                   # DeepLabV3+ training with AMP
│   ├── Glacial_Lake_Attn_ResUNet34_Training.ipynb              # Attention ResUNet34 training
│   └── Glacial_Lake_Three_Model_Comparison.ipynb               # 3-model validation benchmark & visualizations
│
└── reports/                                                    # University Capstone Deliverables
    ├── Glacial_Lake_Segmentation_Capstone_Report.pdf           # Capstone Project Report (PDF)
    ├── Glacial_Lake_Segmentation_Capstone_Report.docx          # Capstone Project Report (DOCX)
    ├── Glacial_Lake_Segmentation_Capstone_Presentation.pdf     # Capstone Presentation Slides (PDF)
    ├── Glacial_Lake_Segmentation_Capstone_Presentation.pptx    # Capstone Presentation Slides (PPTX)
    ├── Capstone_Project_Paper_1.pdf                            # Reference study (Ma et al. 2025)
    └── figures/                                                # High-resolution architectural & comparison figures
        ├── chart_model_comparison.png
        ├── diagram_pipeline.png
        ├── diagram_deeplabv3.png
        ├── diagram_models_2_3.png
        ├── rep_1.png
        ├── diff_105.png
        └── fail_10.png
```

---

## Dataset & Model Checkpoints

### Dataset (GLID)
The **Glacial Lake Image Dataset (GLID)** is openly available on Zenodo:
- **DOI:** [10.5281/zenodo.14838695](https://doi.org/10.5281/zenodo.14838695)
- **Training Set:** 16,000 image-mask pairs ($512 \times 512$)
- **Validation Set:** 2,367 image-mask pairs ($512 \times 512$)
- See [`data/README.md`](data/README.md) for download links and folder layout instructions.

### Model Weights
Trained PyTorch checkpoints (`.pth`) are stored on Google Drive due to file size constraints:
- `selected_model.pth` (ResNet34-FCN, Epoch 29): ~276 MB
- `best_amp_model.pth` (DeepLabV3+, Epoch 26): ~462 MB
- `best_model.pth` (Attention ResUNet34, Epoch 30): ~97 MB

*(To use the checkpoints in the comparison notebook, mount your Google Drive and set the corresponding paths in Section 15 of `Glacial_Lake_Three_Model_Comparison.ipynb`.)*

---

## How to Run on Google Colab

1. **Open Google Colab** and select a GPU runtime (`Runtime -> Change runtime type -> T4 GPU`).
2. **Mount Google Drive** to access the GLID dataset and save checkpoints:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Open any notebook** from the `notebooks/` directory:
   - For individual training: run `Glacial_Lake_ResNet34_FCN_Training.ipynb`, `Glacial_Lake_DeepLabV3_Training.ipynb`, or `Glacial_Lake_Attn_ResUNet34_Training.ipynb`.
   - For the full 3-model benchmark and visual analysis: run `Glacial_Lake_Three_Model_Comparison.ipynb`.

---

## References

1. **Ma, D., Li, J., & Jiang, L. (2025).** *Efficient glacial lake mapping by leveraging deep transfer learning and a new annotated glacial lake dataset.* Journal of Hydrology, 657, 133072. [https://doi.org/10.1016/j.jhydrol.2025.133072](https://doi.org/10.1016/j.jhydrol.2025.133072)
2. **Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018).** *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.* Proceedings of the European Conference on Computer Vision (ECCV).
3. **Oktay, O., Schlemper, J., Le Folgoc, L., et al. (2018).** *Attention U-Net: Learning Where to Look for the Pancreas.* Proceedings of Medical Imaging with Deep Learning (MIDL).
4. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** *Deep Residual Learning for Image Recognition.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
