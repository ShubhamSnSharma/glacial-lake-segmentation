"""
predict.py
==========
Standalone inference and prediction visualization for the trained DeepLabV3+ glacial lake
segmentation model.

Usage:
    # Run inference on 12 random validation samples using best Colab checkpoint
    python scripts/predict.py --checkpoint checkpoints/best_amp_model.pth

    # Specify number of samples, batch size, and output directory
    python scripts/predict.py \
        --checkpoint checkpoints/best_amp_model.pth \
        --n-samples 6 \
        --batch-size 4 \
        --output-dir results/predictions/run1

    # Use a different threshold and fixed random seed
    python scripts/predict.py \
        --checkpoint checkpoints/best_amp_model.pth \
        --n-samples 20 \
        --threshold 0.5 \
        --seed 42

Output:
    results/predictions/<run_dir>/
        sample_001_<filename>.png   # 4-panel per-sample figure
        sample_002_<filename>.png
        ...
        summary_grid.png            # Combined grid of all samples
        metrics_summary.txt         # Aggregate IoU / F1 / Precision / Recall / Accuracy
"""

import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Make src importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe in scripts
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.dataset import GLIDDataset
from src.metrics import MetricTracker, compute_confusion_matrix_elements, calculate_metrics
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.utils import ensure_dirs, set_seed


# ---------------------------------------------------------------------------
# Constants — ImageNet normalization used during training
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the trained DeepLabV3+ glacial lake segmentation model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_amp_model.pth",
        help="Path to model checkpoint .pth file (default: checkpoints/best_amp_model.pth)",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="data/raw/val",
        help="Path to validation split directory (default: data/raw/val)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=12,
        help="Number of validation samples to visualize (default: 12)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size (default: 4)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid probability threshold for lake class (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save prediction images. "
            "Defaults to results/predictions/<timestamp>/"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use for inference. Auto-detected if not specified (CUDA -> MPS -> CPU).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample selection (default: 42)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers (default: 0 — safe for MPS/CUDA)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_device(requested: Optional[str]) -> torch.device:
    """Select inference device: CUDA if available, elif MPS if available, else CPU."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """
    Build DeepLabV3+ architecture and load trained weights from checkpoint.

    Supports both:
      - Colab AMP checkpoint format  (keys: model_state_dict, best_f1, epoch, ...)
      - Local Trainer checkpoint format (keys: model_state_dict, best_metric_val, ...)

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        device: Target inference device.

    Returns:
        (model, checkpoint_dict) tuple.
    """
    chk_path = Path(checkpoint_path)
    if not chk_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: '{chk_path}'\n"
            "Download best_amp_model.pth from Google Drive and place it in checkpoints/."
        )

    print(f"Loading checkpoint: {chk_path} ...")
    checkpoint = torch.load(chk_path, map_location="cpu", weights_only=False)

    # Build architecture (pretrained=False — weights come entirely from checkpoint)
    model = DeepLabV3Plus(pretrained=False, num_classes=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Report checkpoint provenance
    epoch      = checkpoint.get("epoch", "unknown")
    best_f1    = checkpoint.get("best_f1", checkpoint.get("best_metric_val", "unknown"))
    model_name = checkpoint.get(
        "model_name",
        checkpoint.get("cfg", {}).get("model", {}).get("name", "deeplabv3plus"),
    )
    print(f"  Checkpoint loaded  — epoch {epoch}, best F1 = {best_f1}")
    print(f"  Model: {model_name}")

    return model, checkpoint


def _unnormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization; return HWC uint8 numpy array."""
    t = tensor.clone().cpu().float()
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        t[c] = t[c] * s + m
    t = t.permute(1, 2, 0).numpy()
    return np.clip(t * 255, 0, 255).astype(np.uint8)


def make_overlay(
    img_np: np.ndarray,
    gt_np: np.ndarray,
    pred_np: np.ndarray,
) -> np.ndarray:
    """
    Build a colour overlay showing GT and prediction on the satellite image.

    Colour coding:
      Cyan  (#00E5FF) — True Positive  (lake predicted correctly)
      Red   (#FF1744) — False Positive (background predicted as lake)
      White (#FFFFFF) — False Negative (lake missed by model)
    """
    overlay = img_np.copy().astype(np.float32)
    gt   = gt_np.astype(bool)
    pred = pred_np.astype(bool)

    tp = gt &  pred
    fp = ~gt & pred
    fn = gt & ~pred

    alpha = 0.55
    overlay[tp] = (1 - alpha) * overlay[tp] + alpha * np.array([0,   229, 255], dtype=np.float32)
    overlay[fp] = (1 - alpha) * overlay[fp] + alpha * np.array([255, 23,  68],  dtype=np.float32)
    overlay[fn] = (1 - alpha) * overlay[fn] + alpha * np.array([255, 255, 255], dtype=np.float32)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_sample_figure(
    img_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    filename: str,
    save_path: Path,
    sample_idx: int,
    metrics: dict,
    threshold: float,
) -> None:
    """
    Render and save a 4-panel figure for one validation sample.

    Panels: Satellite Image | Ground-Truth Mask | Predicted Mask | Overlay
    """
    img_np  = _unnormalize(img_tensor)
    gt_np   = gt_tensor.squeeze().cpu().numpy()
    pred_np = pred_tensor.squeeze().cpu().numpy()
    overlay = make_overlay(img_np, gt_np, pred_np)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].imshow(img_np)
    axes[0].set_title("Satellite Image", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt_np, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Ground-Truth Mask", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(pred_np, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title(f"Predicted Mask (t={threshold})", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Prediction Overlay", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    legend_handles = [
        mpatches.Patch(color=(0 / 255, 229 / 255, 255 / 255), label="TP (lake, correct)"),
        mpatches.Patch(color=(255 / 255, 23 / 255, 68 / 255),  label="FP (false detection)"),
        mpatches.Patch(color=(1.0, 1.0, 1.0),                  label="FN (missed lake)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04),
    )

    iou_str = f"IoU={metrics.get('iou', 0):.3f}"
    f1_str  = f"F1={metrics.get('f1', 0):.3f}"
    fig.suptitle(
        f"Sample {sample_idx:03d} — {filename} — {f1_str}  {iou_str}",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_summary_grid(
    images: List[np.ndarray],
    gts: List[np.ndarray],
    preds: List[np.ndarray],
    save_path: Path,
    n_cols: int = 4,
) -> None:
    """
    Save a compact summary grid showing (satellite image | overlay) pairs for all samples.
    """
    n      = len(images)
    n_rows = (n + n_cols - 1) // n_cols  # ceil division

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols * 3.5, n_rows * 7))
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        row_img  = (i // n_cols) * 2
        row_pred = row_img + 1
        col      = i % n_cols

        overlay = make_overlay(images[i], gts[i], preds[i])
        axes[row_img,  col].imshow(images[i])
        axes[row_img,  col].axis("off")
        axes[row_img,  col].set_title(f"Sample {i + 1}", fontsize=8)
        axes[row_pred, col].imshow(overlay)
        axes[row_pred, col].axis("off")

    # Hide unused axes
    for i in range(n, n_rows * n_cols):
        row_img  = (i // n_cols) * 2
        row_pred = row_img + 1
        col      = i % n_cols
        axes[row_img,  col].axis("off")
        axes[row_pred, col].axis("off")

    legend_handles = [
        mpatches.Patch(color=(0 / 255, 229 / 255, 255 / 255), label="TP"),
        mpatches.Patch(color=(255 / 255, 23 / 255, 68 / 255),  label="FP"),
        mpatches.Patch(color=(1.0, 1.0, 1.0),                  label="FN"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "DeepLabV3+ Glacial Lake Segmentation — Summary Grid",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary grid  -> {save_path}")


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> None:

    # --- Device ---
    device = select_device(args.device)
    print(f"Device: {device}")

    # --- Reproducibility ---
    set_seed(args.seed)

    # --- Output directory ---
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results/predictions") / ts
    ensure_dirs(str(out_dir))
    print(f"Output directory: {out_dir.resolve()}")

    # --- Load model ---
    model, checkpoint = load_model(args.checkpoint, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # --- Validation dataset (no augmentation, is_train=False) ---
    val_ds = GLIDDataset(
        split_dir=args.val_dir,
        is_train=False,
        split="val",
    )
    n_total = len(val_ds)
    print(f"Validation dataset: {n_total} samples")

    # Random sample selection without replacement
    n_samples = min(args.n_samples, n_total)
    indices   = random.sample(range(n_total), n_samples)
    indices.sort()  # ascending order for reproducibility
    print(
        f"Selecting {n_samples} samples  "
        f"(indices: {indices[:5]}{'...' if n_samples > 5 else ''})\n"
    )

    # --- Aggregate metric tracker ---
    tracker = MetricTracker(threshold=args.threshold)

    # Collectors for summary grid
    all_images: List[np.ndarray] = []
    all_gts:    List[np.ndarray] = []
    all_preds:  List[np.ndarray] = []

    print(f"Running inference on {n_samples} samples ...\n")
    t_start = time.time()

    # Process in batches
    for batch_start in range(0, n_samples, args.batch_size):
        batch_indices = indices[batch_start : batch_start + args.batch_size]

        batch_imgs:   List[torch.Tensor] = []
        batch_masks:  List[torch.Tensor] = []
        batch_fnames: List[str]          = []

        for idx in batch_indices:
            img, mask = val_ds[idx]
            batch_imgs.append(img)
            batch_masks.append(mask)
            batch_fnames.append(val_ds.get_filename(idx))

        imgs_tensor  = torch.stack(batch_imgs).to(device)   # (B, 3, 512, 512)
        masks_tensor = torch.stack(batch_masks).to(device)  # (B, 1, 512, 512)

        # Inference — no AMP, no channels_last (both MPS-safe)
        with torch.inference_mode():
            logits = model(imgs_tensor)                      # (B, 1, 512, 512)
            probs  = torch.sigmoid(logits)
            preds  = (probs >= args.threshold).float()

        # Update global metrics
        tracker.update(preds.cpu(), masks_tensor.cpu())

        # Per-sample: save figure + collect arrays
        for local_i, (img, gt, pred, fname) in enumerate(
            zip(batch_imgs, batch_masks, preds.cpu(), batch_fnames)
        ):
            global_i = batch_start + local_i + 1

            tp, fp, fn, tn = compute_confusion_matrix_elements(
                pred.cpu(), gt.cpu(), threshold=args.threshold
            )
            sample_metrics = calculate_metrics(tp, fp, fn, tn)

            img_np  = _unnormalize(img.cpu())
            gt_np   = gt.squeeze().cpu().numpy()
            pred_np = pred.squeeze().numpy()

            all_images.append(img_np)
            all_gts.append(gt_np)
            all_preds.append(pred_np)

            out_name = f"sample_{global_i:03d}_{Path(fname).stem}.png"
            out_path = out_dir / out_name
            save_sample_figure(
                img_tensor=img.cpu(),
                gt_tensor=gt.cpu(),
                pred_tensor=pred.cpu(),
                filename=fname,
                save_path=out_path,
                sample_idx=global_i,
                metrics=sample_metrics,
                threshold=args.threshold,
            )
            print(
                f"  [{global_i:03d}/{n_samples:03d}] {fname:<20s}  "
                f"IoU={sample_metrics['iou']:.3f}  F1={sample_metrics['f1']:.3f}  "
                f"-> {out_name}"
            )

    elapsed = time.time() - t_start

    # --- Summary grid ---
    summary_path = out_dir / "summary_grid.png"
    save_summary_grid(all_images, all_gts, all_preds, summary_path)

    # --- Aggregate metrics ---
    agg = tracker.compute()
    print("\n" + "=" * 60)
    print(f"  INFERENCE COMPLETE — {n_samples} samples in {elapsed:.1f}s")
    print("=" * 60)
    print(f"  IoU            : {agg['iou']:.4f}  ({agg['iou']*100:.2f}%)")
    print(f"  F1 / Dice      : {agg['f1']:.4f}  ({agg['f1']*100:.2f}%)")
    print(f"  Precision      : {agg['precision']:.4f}  ({agg['precision']*100:.2f}%)")
    print(f"  Recall         : {agg['recall']:.4f}  ({agg['recall']*100:.2f}%)")
    print(f"  Accuracy       : {agg['accuracy']:.4f}  ({agg['accuracy']*100:.2f}%)")
    print("=" * 60)

    # --- Metrics text file ---
    metrics_path = out_dir / "metrics_summary.txt"
    chk_epoch = checkpoint.get("epoch", "unknown")
    chk_f1    = checkpoint.get("best_f1", checkpoint.get("best_metric_val", "unknown"))
    with open(metrics_path, "w") as f:
        f.write("DeepLabV3+ Glacial Lake Segmentation — Inference Metrics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Epoch      : {chk_epoch}\n")
        f.write(f"Best F1    : {chk_f1}\n")
        f.write(f"Samples    : {n_samples} / {n_total}\n")
        f.write(f"Threshold  : {args.threshold}\n")
        f.write(f"Device     : {device}\n")
        f.write(f"Elapsed    : {elapsed:.1f}s\n")
        f.write("-" * 60 + "\n")
        f.write(f"IoU        : {agg['iou']:.4f}\n")
        f.write(f"F1 / Dice  : {agg['f1']:.4f}\n")
        f.write(f"Precision  : {agg['precision']:.4f}\n")
        f.write(f"Recall     : {agg['recall']:.4f}\n")
        f.write(f"Accuracy   : {agg['accuracy']:.4f}\n")
    print(f"\nMetrics saved  -> {metrics_path}")
    print(f"All outputs in : {out_dir.resolve()}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
