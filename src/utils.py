"""
utils.py
========
Shared utility functions for the glacial lake mapping project.

Covers:
  - Visualization (image/mask overlays, batch grids)
  - Dataset statistics (class distribution, per-channel mean/std)
  - Reproducibility (seed setting)
  - Config loading

Note: metric computation (Precision, Recall, F1, IoU) lives in evaluate.py (Milestone 2).
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe in scripts and headless envs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch
from torch.utils.data import DataLoader
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy and PyTorch for reproducibility.

    Call this at the start of every script or training run.
    Does NOT guarantee full determinism on GPU — set
    torch.backends.cudnn.deterministic = True if exact reproducibility
    across runs is required (at the cost of speed).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    """Load a YAML configuration file and return as a dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _unnormalize(
    tensor: torch.Tensor,
    mean: List[float] = (0.485, 0.456, 0.406),
    std:  List[float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Reverse ImageNet normalization and convert CHW tensor to HWC uint8 array.

    Args:
        tensor: Float tensor of shape (3, H, W) in normalized form.
        mean, std: Normalization constants used during loading.

    Returns:
        NumPy array of shape (H, W, 3), dtype uint8, values in [0, 255].
    """
    t = tensor.clone().cpu()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    t = t.permute(1, 2, 0).numpy()          # CHW → HWC
    t = np.clip(t * 255, 0, 255).astype(np.uint8)
    return t


def visualize_sample(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    pred_tensor: Optional[torch.Tensor] = None,
    title: str = "",
    save_path: Optional[str] = None,
    normalize_mean: List[float] = (0.485, 0.456, 0.406),
    normalize_std:  List[float] = (0.229, 0.224, 0.225),
) -> plt.Figure:
    """
    Visualize an image, its ground-truth mask, and optionally a predicted mask.

    Layout (2 or 3 columns):
        [Original Image] | [Ground-truth Mask] | [Overlay] | [Prediction*]
        * only when pred_tensor is provided

    Args:
        image_tensor: Float tensor (3, H, W) — normalized.
        mask_tensor:  Float tensor (1, H, W) or (H, W) — values in {0, 1}.
        pred_tensor:  Optional float tensor, same shape as mask_tensor.
        title:        Optional suptitle string.
        save_path:    If given, saves figure to this path.
        normalize_mean, normalize_std: Normalization constants for unnormalization.

    Returns:
        matplotlib Figure object.
    """
    img_np  = _unnormalize(image_tensor, normalize_mean, normalize_std)
    mask_np = mask_tensor.squeeze().cpu().numpy()  # (H, W), values 0/1

    n_cols = 3 if pred_tensor is None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # Panel 1: Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Satellite Image")
    axes[0].axis("off")

    # Panel 2: Ground-truth mask
    axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground-truth Mask")
    axes[1].axis("off")

    # Panel 3: Overlay (image + mask boundary)
    overlay = img_np.copy()
    lake_px = mask_np.astype(bool)
    overlay[lake_px] = (
        0.4 * overlay[lake_px]
        + 0.6 * np.array([0, 120, 255], dtype=np.uint8)
    ).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title("GT Overlay")
    axes[2].axis("off")

    # Panel 4 (optional): Prediction
    if pred_tensor is not None:
        pred_np = pred_tensor.squeeze().cpu().numpy()
        axes[3].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Predicted Mask")
        axes[3].axis("off")

    # Legend
    lake_patch = mpatches.Patch(color=(0 / 255, 120 / 255, 255 / 255), label="Glacial Lake")
    fig.legend(handles=[lake_patch], loc="lower center", ncol=1, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_batch_grid(
    images: torch.Tensor,
    masks: torch.Tensor,
    n_samples: int = 8,
    save_path: Optional[str] = None,
    normalize_mean: List[float] = (0.485, 0.456, 0.406),
    normalize_std:  List[float] = (0.229, 0.224, 0.225),
) -> plt.Figure:
    """
    Display a grid of (image, mask) pairs from a batch tensor.

    Args:
        images:    Float tensor (B, 3, H, W)
        masks:     Float tensor (B, 1, H, W)
        n_samples: How many samples to display (≤ batch size)
        save_path: If given, saves the figure.
    """
    n = min(n_samples, images.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))

    if n == 1:
        axes = axes[:, np.newaxis]  # Ensure 2D indexing

    for i in range(n):
        img_np  = _unnormalize(images[i], normalize_mean, normalize_std)
        mask_np = masks[i].squeeze().cpu().numpy()

        axes[0, i].imshow(img_np)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Image", fontsize=9)

        axes[1, i].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Mask", fontsize=9)

    plt.suptitle(f"Sample batch — {n} of {images.shape[0]} items", fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def compute_class_distribution(
    dataset,
    n_samples: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Estimate the class distribution (lake vs background) over the dataset.

    Iterates over `n_samples` samples from the dataset (or all if None),
    counts lake pixels (value=1) and background pixels (value=0).

    Args:
        dataset:   A GLIDDataset instance (or any dataset returning (image, mask)).
        n_samples: Number of samples to scan. If None, scans entire dataset.
        verbose:   Print progress and results.

    Returns:
        Dict with keys: 'total_pixels', 'lake_pixels', 'bg_pixels',
                        'lake_fraction', 'bg_fraction'
    """
    n = n_samples if n_samples else len(dataset)
    n = min(n, len(dataset))

    indices = list(range(n))
    total_pixels = 0
    lake_pixels  = 0

    for i, idx in enumerate(indices):
        _, mask = dataset[idx]
        mask_np = mask.numpy()
        total_pixels += mask_np.size
        lake_pixels  += int((mask_np > 0.5).sum())

        if verbose and (i + 1) % 500 == 0:
            print(f"  Scanned {i + 1}/{n} samples...")

    bg_pixels    = total_pixels - lake_pixels
    lake_fraction = lake_pixels / total_pixels if total_pixels > 0 else 0.0
    bg_fraction   = bg_pixels   / total_pixels if total_pixels > 0 else 0.0

    stats = {
        "samples_scanned":  n,
        "total_pixels":     total_pixels,
        "lake_pixels":      lake_pixels,
        "bg_pixels":        bg_pixels,
        "lake_fraction":    lake_fraction,
        "bg_fraction":      bg_fraction,
        "imbalance_ratio":  bg_pixels / lake_pixels if lake_pixels > 0 else float("inf"),
    }

    if verbose:
        print(
            f"\nClass distribution ({n} samples):\n"
            f"  Total pixels     : {total_pixels:,}\n"
            f"  Lake pixels      : {lake_pixels:,}  ({lake_fraction * 100:.2f}%)\n"
            f"  Background pixels: {bg_pixels:,}   ({bg_fraction * 100:.2f}%)\n"
            f"  Imbalance ratio  : {stats['imbalance_ratio']:.1f}× more background than lake"
        )

    return stats


def compute_normalization_stats(
    dataset,
    n_samples: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Compute per-channel mean and std over the dataset (on unnormalized pixels).

    Loads images as raw tensors (bypassing the normalize transform). This
    requires the dataset to have been built WITHOUT normalization or the values
    to be un-normalized before accumulating.

    IMPORTANT: This function operates on the raw PIL images read from disk
    to avoid the chicken-and-egg problem of needing stats to normalize but
    needing normalized data to compute stats.

    Args:
        dataset:   A GLIDDataset instance.
        n_samples: Number of samples to use.
        verbose:   Print progress.

    Returns:
        Dict with 'mean' and 'std' as lists of 3 floats (one per RGB channel).
    """
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()

    n = n_samples if n_samples else len(dataset)
    n = min(n, len(dataset))

    channel_sum   = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    pixel_count   = 0

    for i in range(n):
        # Read raw image directly from disk (bypassing dataset normalization)
        fname = dataset.image_files[i]
        img   = Image.open(dataset.image_dir / fname).convert("RGB")
        arr   = np.array(img, dtype=np.float64) / 255.0  # (H, W, 3) in [0, 1]

        channel_sum    += arr.sum(axis=(0, 1))
        channel_sum_sq += (arr ** 2).sum(axis=(0, 1))
        pixel_count    += arr.shape[0] * arr.shape[1]

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Computing stats: {i + 1}/{n} ...")

    mean = (channel_sum / pixel_count).tolist()
    var  = (channel_sum_sq / pixel_count) - np.array(mean) ** 2
    std  = np.sqrt(np.clip(var, 0, None)).tolist()

    if verbose:
        print(
            f"\nDataset normalization stats ({n} samples):\n"
            f"  Mean (R, G, B): {[round(v, 4) for v in mean]}\n"
            f"  Std  (R, G, B): {[round(v, 4) for v in std]}\n"
            f"  (ImageNet mean: [0.485, 0.456, 0.406])\n"
            f"  (ImageNet std:  [0.229, 0.224, 0.225])"
        )

    return {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# Image/mask dimension checker
# ---------------------------------------------------------------------------

def verify_sample_properties(
    dataset,
    n_samples: int = 5,
) -> Dict:
    """
    Load n_samples from the dataset and verify shapes, dtypes, and value ranges.

    Args:
        dataset:   A GLIDDataset instance.
        n_samples: Number of random samples to check.

    Returns:
        Dict summarizing the verified properties.

    Raises:
        AssertionError if any check fails.
    """
    results = {}
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    image_shapes  = []
    mask_shapes   = []
    mask_uniques  = set()
    image_ranges  = []

    for idx in indices:
        img, msk = dataset[idx]

        assert isinstance(img, torch.Tensor), f"Image is not a tensor: {type(img)}"
        assert isinstance(msk, torch.Tensor), f"Mask is not a tensor: {type(msk)}"

        image_shapes.append(tuple(img.shape))
        mask_shapes.append(tuple(msk.shape))

        # Check image is float
        assert img.dtype == torch.float32, f"Image dtype is {img.dtype}, expected float32"
        # Check mask is float
        assert msk.dtype == torch.float32, f"Mask dtype is {msk.dtype}, expected float32"

        # Mask values must be in {0, 1}
        unique_vals = msk.unique().tolist()
        for v in unique_vals:
            assert v in (0.0, 1.0), f"Unexpected mask value: {v}"
        mask_uniques.update([round(v) for v in unique_vals])

        image_ranges.append((float(img.min()), float(img.max())))

    # All shapes must be consistent
    assert len(set(image_shapes)) == 1, f"Inconsistent image shapes: {image_shapes}"
    assert len(set(mask_shapes))  == 1, f"Inconsistent mask shapes: {mask_shapes}"

    results = {
        "image_shape":         image_shapes[0],
        "mask_shape":          mask_shapes[0],
        "image_dtype":         "float32",
        "mask_dtype":          "float32",
        "mask_unique_values":  sorted(mask_uniques),
        "image_min_range":     min(r[0] for r in image_ranges),
        "image_max_range":     max(r[1] for r in image_ranges),
        "n_verified":          len(indices),
        "all_checks_passed":   True,
    }

    print(
        f"\nSample property verification ({n_samples} samples):\n"
        f"  Image shape  : {results['image_shape']}  (C, H, W)\n"
        f"  Mask shape   : {results['mask_shape']}  (1, H, W)\n"
        f"  Image dtype  : {results['image_dtype']}\n"
        f"  Mask dtype   : {results['mask_dtype']}\n"
        f"  Mask values  : {results['mask_unique_values']}  (0=bg, 1=lake) ✓\n"
        f"  Image range  : [{results['image_min_range']:.3f}, {results['image_max_range']:.3f}] (post-normalization)\n"
        f"  All checks   : PASSED ✓"
    )

    return results
