"""
error_analysis.py
=================
Visual results and error analysis pipeline for the trained DeepLabV3+
glacial lake segmentation model.

Objectives:
  1. Generate visual comparisons for at least 20 validation samples:
     - Satellite Image
     - Ground-Truth Mask
     - Predicted Mask
     - Error Analysis Overlay (TP, FP, FN)
  2. Categorize errors into representative groups:
     - High-Accuracy / Good Predictions
     - Under-Segmentation (missed lake pixels / FN)
     - Over-Segmentation (spurious detections / FP)
     - Difficult Cases (small lakes, irregular boundaries, shadow/snow)
  3. Generate:
     - Individual 4-panel comparison plots
     - Consolidated report grid (suitable for presentations & capstone thesis)
     - error_analysis_summary.txt documenting observed visual patterns & limitations
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.dataset import GLIDDataset
from src.metrics import compute_confusion_matrix_elements, calculate_metrics
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.utils import ensure_dirs, set_seed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Colors for error visualization:
# TP: Cyan (#00E5FF)
# FP: Red (#FF1744)
# FN: Amber/Gold (#FFC107) - distinct from white snow and blue lakes
COLOR_TP = np.array([0, 229, 255], dtype=np.float32)
COLOR_FP = np.array([255, 23, 68], dtype=np.float32)
COLOR_FN = np.array([255, 193, 7], dtype=np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run visual error analysis for trained DeepLabV3+ model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_amp_model.pth",
        help="Path to trained model checkpoint (default: checkpoints/best_amp_model.pth)",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="data/raw/val",
        help="Path to validation directory (default: data/raw/val)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Total number of validation samples to visualize (default: 20, min: 20)",
    )
    parser.add_argument(
        "--scan-pool",
        type=int,
        default=150,
        help="Number of validation samples to scan for categorized errors (default: 150)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for model inference (default: 8)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/error_analysis",
        help="Output directory (default: results/error_analysis)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (CUDA -> MPS -> CPU fallback if None)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    if args.n_samples < 20:
        parser.error(
            f"--n-samples must be at least 20, got {args.n_samples}. "
            "Capstone error analysis requires at least 20 illustrative samples."
        )
    if args.n_samples % 4 != 0:
        parser.error(
            f"--n-samples must be divisible by 4 (to allocate equal samples across "
            f"all 4 error categories), got {args.n_samples}."
        )
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_device(requested: str = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    chk_path = Path(checkpoint_path)
    if not chk_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {chk_path}")

    print(f"Loading checkpoint: {chk_path}")
    checkpoint = torch.load(chk_path, map_location="cpu", weights_only=False)

    model = DeepLabV3Plus(pretrained=False, num_classes=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    best_f1 = checkpoint.get("best_f1", checkpoint.get("best_metric_val", "unknown"))
    print(f"  Model loaded (Epoch: {epoch}, Best F1: {best_f1})")
    return model, checkpoint


def unnormalize(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.clone().cpu().float()
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        t[c] = t[c] * s + m
    t = t.permute(1, 2, 0).numpy()
    return np.clip(t * 255.0, 0.0, 255.0).astype(np.uint8)


def create_error_overlay(
    img_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.60,
) -> np.ndarray:
    """
    Creates an overlay highlighting TP, FP, and FN on the original image.
    - TP (Cyan): correctly detected lake pixels
    - FP (Red): false positive detections (background as lake)
    - FN (Amber/Gold): missed lake pixels
    """
    overlay = img_rgb.copy().astype(np.float32)
    gt_bool = gt_mask.astype(bool)
    pred_bool = pred_mask.astype(bool)

    tp = gt_bool & pred_bool
    fp = (~gt_bool) & pred_bool
    fn = gt_bool & (~pred_bool)

    overlay[tp] = (1.0 - alpha) * overlay[tp] + alpha * COLOR_TP
    overlay[fp] = (1.0 - alpha) * overlay[fp] + alpha * COLOR_FP
    overlay[fn] = (1.0 - alpha) * overlay[fn] + alpha * COLOR_FN

    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def plot_single_sample(
    img_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    overlay: np.ndarray,
    category: str,
    metrics: dict,
    filename: str,
    sample_num: int,
    save_path: Path,
    threshold: float,
) -> None:
    """Saves a detailed 4-panel visual comparison for one sample."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=130)

    # 1. Satellite Image
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Satellite Image", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # 2. Ground-Truth Mask
    gt_pixels = int(gt_mask.sum())
    gt_pct = (gt_pixels / (512 * 512)) * 100
    axes[1].imshow(gt_mask, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title(f"Ground Truth\n({gt_pixels:,} px | {gt_pct:.2f}%)", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # 3. Predicted Mask
    pred_pixels = int(pred_mask.sum())
    pred_pct = (pred_pixels / (512 * 512)) * 100
    axes[2].imshow(pred_mask, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (t={threshold})\n({pred_pixels:,} px | {pred_pct:.2f}%)", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    # 4. Error Overlay
    axes[3].imshow(overlay)
    axes[3].set_title(
        f"Error Analysis Overlay\nTP={metrics['tp']:,} | FP={metrics['fp']:,} | FN={metrics['fn']:,}",
        fontsize=10,
        fontweight="bold",
    )
    axes[3].axis("off")

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLOR_TP / 255.0, label=f"True Positive (Lake Correct) - {metrics['tp']:,} px"),
        mpatches.Patch(color=COLOR_FP / 255.0, label=f"False Positive (Spurious / Shadow) - {metrics['fp']:,} px"),
        mpatches.Patch(color=COLOR_FN / 255.0, label=f"False Negative (Missed Lake) - {metrics['fn']:,} px"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        f"Sample {sample_num:02d}: {filename}  [{category}]  |  F1: {metrics['f1']:.3f}  IoU: {metrics['iou']:.3f}  Prec: {metrics['precision']:.3f}  Rec: {metrics['recall']:.3f}",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def plot_consolidated_grid(
    samples_data: List[dict],
    save_path: Path,
    n_cols: int = 4,
) -> None:
    """
    Renders a comprehensive grid showing (Satellite Image | Error Overlay)
    for all selected samples, categorized and cleanly formatted.
    """
    n = len(samples_data)
    n_rows = (n + n_cols - 1) // n_cols

    # Each sample occupies 2 subplots vertically: Image and Error Overlay
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols * 3.8, n_rows * 7.2), dpi=120)
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    for i, s in enumerate(samples_data):
        row_img  = (i // n_cols) * 2
        row_over = row_img + 1
        col      = i % n_cols

        # Satellite image
        axes[row_img, col].imshow(s["img_rgb"])
        axes[row_img, col].set_title(
            f"#{i+1:02d} {s['fname']} ({s['category_short']})\nF1: {s['metrics']['f1']:.3f} | IoU: {s['metrics']['iou']:.3f}",
            fontsize=8,
            fontweight="bold",
        )
        axes[row_img, col].axis("off")

        # Overlay
        axes[row_over, col].imshow(s["overlay"])
        axes[row_over, col].set_title(
            f"TP:{s['metrics']['tp']} | FP:{s['metrics']['fp']} | FN:{s['metrics']['fn']}",
            fontsize=7,
        )
        axes[row_over, col].axis("off")

    # Turn off unused axes
    for i in range(n, n_rows * n_cols):
        row_img  = (i // n_cols) * 2
        row_over = row_img + 1
        col      = i % n_cols
        axes[row_img, col].axis("off")
        axes[row_over, col].axis("off")

    legend_handles = [
        mpatches.Patch(color=COLOR_TP / 255.0, label="True Positive (Lake Correct)"),
        mpatches.Patch(color=COLOR_FP / 255.0, label="False Positive (Over-segmented)"),
        mpatches.Patch(color=COLOR_FN / 255.0, label="False Negative (Under-segmented)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=10,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        f"DeepLabV3+ Glacial Lake Segmentation — Visual Results & Error Analysis Grid ({n} Validation Samples)",
        fontsize=13,
        fontweight="bold",
        y=1.002,
    )

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Consolidated grid saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def run_error_analysis(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    ensure_dirs(str(out_dir))
    print(f"Output directory: {out_dir.resolve()}")

    model, checkpoint = load_model(args.checkpoint, device)

    val_ds = GLIDDataset(split_dir=args.val_dir, is_train=False, split="val")
    n_total = len(val_ds)
    print(f"Validation dataset: {n_total} samples")

    scan_count = min(args.scan_pool, n_total)
    print(f"Scanning first {scan_count} validation samples for error categorization ...")

    # 1. Evaluate pool of samples to collect stats
    pool_records = []
    t0 = time.time()

    for start_idx in range(0, scan_count, args.batch_size):
        end_idx = min(start_idx + args.batch_size, scan_count)
        batch_imgs = []
        batch_masks = []
        batch_fnames = []
        for idx in range(start_idx, end_idx):
            img, mask = val_ds[idx]
            batch_imgs.append(img)
            batch_masks.append(mask)
            batch_fnames.append(val_ds.get_filename(idx))

        imgs_t = torch.stack(batch_imgs).to(device)
        with torch.inference_mode():
            logits = model(imgs_t)
            probs = torch.sigmoid(logits)
            preds = (probs >= args.threshold).float().cpu()

        for local_i, (img_t, mask_t, pred_t, fname) in enumerate(
            zip(batch_imgs, batch_masks, preds, batch_fnames)
        ):
            idx = start_idx + local_i
            tp, fp, fn, tn = compute_confusion_matrix_elements(
                pred_t, mask_t, threshold=args.threshold
            )
            m = calculate_metrics(tp, fp, fn, tn)
            gt_pixels = int(mask_t.sum().item())
            pred_pixels = int(pred_t.sum().item())

            pool_records.append({
                "idx": idx,
                "fname": fname,
                "img_tensor": img_t,
                "gt_tensor": mask_t,
                "pred_tensor": pred_t,
                "gt_pixels": gt_pixels,
                "pred_pixels": pred_pixels,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "metrics": m,
            })

    scan_time = time.time() - t0
    print(f"Scanned {len(pool_records)} samples in {scan_time:.1f}s.")

    # 2. Categorization into 4 illustrative groups:
    # Category A: Good predictions: GT > 3,000 pixels, sorted by highest F1 score
    # Category B: Under-segmentation: GT > 1,500 pixels and FN > FP + 150 pixels, ranked by omission ratio FN / GT_pixels
    # Category C: Over-segmentation: FP > FN + 40 pixels or Precision < 0.90, ranked by false-discovery ratio FP / Pred_pixels
    # Category D: Difficult Cases (Small Lakes): GT between 10 and 999 pixels, sorted ascending by lake size

    target_per_category = args.n_samples // 4  # e.g., 20 // 4 = 5 per category
    selected_indices = set()
    categorized_samples = []

    # Category A: Good predictions (GT > 3,000 pixels, sorted by highest F1 score)
    good_candidates = sorted(
        [r for r in pool_records if r["gt_pixels"] > 3000],
        key=lambda x: x["metrics"]["f1"],
        reverse=True,
    )
    cat_a = []
    for r in good_candidates:
        if r["idx"] not in selected_indices:
            cat_a.append((r, "Good Prediction", "Good"))
            selected_indices.add(r["idx"])
            if len(cat_a) >= target_per_category:
                break

    # Category B: Under-segmentation (GT > 1,500 pixels and FN > FP + 150 pixels, ranked by FN / GT_pixels)
    under_candidates = sorted(
        [r for r in pool_records if r["gt_pixels"] > 1500 and (r["fn"] > r["fp"] + 150)],
        key=lambda x: (x["fn"] / (x["gt_pixels"] + 1e-6)),
        reverse=True,
    )
    cat_b = []
    for r in under_candidates:
        if r["idx"] not in selected_indices:
            cat_b.append((r, "Under-Segmentation (Missed Lake/Boundary)", "Under-seg"))
            selected_indices.add(r["idx"])
            if len(cat_b) >= target_per_category:
                break

    # Category C: Over-segmentation (FP > FN + 40 pixels or Precision < 0.90, ranked by FP / Pred_pixels)
    over_candidates = sorted(
        [r for r in pool_records if (r["fp"] > r["fn"] + 40 or r["metrics"]["precision"] < 0.90)],
        key=lambda x: (x["fp"] / (x["pred_pixels"] + 1e-6)),
        reverse=True,
    )
    cat_c = []
    for r in over_candidates:
        if r["idx"] not in selected_indices:
            cat_c.append((r, "Over-Segmentation (False Detection/Debris)", "Over-seg"))
            selected_indices.add(r["idx"])
            if len(cat_c) >= target_per_category:
                break

    # Category D: Difficult Cases — Small Lakes (GT between 10 and 999 pixels, sorted ascending by lake size)
    small_candidates = sorted(
        [r for r in pool_records if 10 < r["gt_pixels"] < 1000],
        key=lambda x: x["gt_pixels"],
    )
    cat_d = []
    for r in small_candidates:
        if r["idx"] not in selected_indices:
            cat_d.append((r, "Difficult Case (Small Lake / Irregular)", "Difficult"))
            selected_indices.add(r["idx"])
            if len(cat_d) >= target_per_category:
                break

    # Combine all 4 categories
    final_selection = cat_a + cat_b + cat_c + cat_d
    print(f"\nSelected {len(final_selection)} samples for detailed error analysis:")
    print(f"  - Category A (Good Predictions)    : {len(cat_a)}")
    print(f"  - Category B (Under-Segmentation)  : {len(cat_b)}")
    print(f"  - Category C (Over-Segmentation)   : {len(cat_c)}")
    print(f"  - Category D (Difficult / Small)   : {len(cat_d)}")

    # Safeguard: verify that the required number of samples was found in each category
    category_counts = {
        "Good Prediction": len(cat_a),
        "Under-Segmentation": len(cat_b),
        "Over-Segmentation": len(cat_c),
        "Difficult Case (Small Lake)": len(cat_d),
    }
    for cat_name, count in category_counts.items():
        if count < target_per_category:
            raise ValueError(
                f"Incomplete category selection safeguard triggered: "
                f"Category '{cat_name}' only contains {count} sample(s), but "
                f"{target_per_category} are required. "
                f"Please increase --scan-pool or adjust candidate criteria."
            )

    # 3. Generate figures and collect summary information
    processed_samples = []
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("  Deep Learning Based Glacial Lake Detection and Segmentation")
    summary_lines.append("  Step 2: Visual Results and Qualitative Error Analysis")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Date & Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Model Architecture : DeepLabV3+ (ResNet-50 backbone)")
    summary_lines.append(f"Model Checkpoint   : {args.checkpoint}")
    summary_lines.append(f"Threshold          : {args.threshold}")
    summary_lines.append(f"Evaluation Scope   : {len(final_selection)} illustrative samples selected from candidate pool of {scan_count} validation samples")
    summary_lines.append(f"Device Used        : {device}")
    summary_lines.append("\n[1. Selection Methodology & Quantitative Summary of Visualized Samples]")
    summary_lines.append(f"Selection Criteria (applied to candidate pool of first {scan_count} validation samples):")
    summary_lines.append("  - Category A (Good Predictions)    : GT > 3,000 pixels, sorted by highest F1 score")
    summary_lines.append("  - Category B (Under-Segmentation)  : GT > 1,500 pixels and FN > FP + 150 pixels, ranked by FN / GT_pixels")
    summary_lines.append("  - Category C (Over-Segmentation)   : FP > FN + 40 pixels or Precision < 0.90, ranked by FP / Pred_pixels")
    summary_lines.append("  - Category D (Difficult / Small)   : GT between 10 and 999 pixels, sorted ascending by lake size")
    summary_lines.append("-" * 88)
    summary_lines.append(f"{'#':<3} {'Filename':<12} {'Category':<28} {'GT_px':<8} {'Pred_px':<8} {'IoU':<7} {'F1':<7} {'Prec':<7} {'Rec':<7}")
    summary_lines.append("-" * 88)

    for i, (r, cat_name, cat_short) in enumerate(final_selection):
        sample_num = i + 1
        img_rgb = unnormalize(r["img_tensor"])
        gt_mask = r["gt_tensor"].squeeze().cpu().numpy()
        pred_mask = r["pred_tensor"].squeeze().numpy()
        overlay = create_error_overlay(img_rgb, gt_mask, pred_mask)

        stem = Path(r["fname"]).stem
        clean_cat = cat_short.lower().replace("-", "_").replace(" ", "_")
        filename_out = f"sample_{sample_num:02d}_{clean_cat}_{stem}.png"
        save_path = out_dir / filename_out

        m = r["metrics"]
        m["tp"] = r["tp"]
        m["fp"] = r["fp"]
        m["fn"] = r["fn"]
        m["tn"] = r["tn"]

        plot_single_sample(
            img_rgb=img_rgb,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            overlay=overlay,
            category=cat_name,
            metrics=m,
            filename=r["fname"],
            sample_num=sample_num,
            save_path=save_path,
            threshold=args.threshold,
        )

        processed_samples.append({
            "sample_num": sample_num,
            "fname": r["fname"],
            "category": cat_name,
            "category_short": cat_short,
            "img_rgb": img_rgb,
            "gt_mask": gt_mask,
            "pred_mask": pred_mask,
            "overlay": overlay,
            "metrics": m,
            "gt_pixels": r["gt_pixels"],
            "pred_pixels": r["pred_pixels"],
            "file_path": str(save_path),
        })

        summary_lines.append(
            f"{sample_num:<3} {r['fname']:<12} {cat_name[:26]:<28} {r['gt_pixels']:<8} {r['pred_pixels']:<8} "
            f"{m['iou']:<7.3f} {m['f1']:<7.3f} {m['precision']:<7.3f} {m['recall']:<7.3f}"
        )
        print(f"  [{sample_num:02d}/{len(final_selection):02d}] {r['fname']:<10} -> {filename_out}")

    # 4. Generate Consolidated Grid
    grid_path = out_dir / "error_analysis_grid.png"
    plot_consolidated_grid(processed_samples, grid_path, n_cols=4)

    # 5. Compute dynamic statistics per category for Section 2
    def get_cat_stats(samples_list: List[dict]) -> dict:
        ious = [s["metrics"]["iou"] for s in samples_list]
        f1s = [s["metrics"]["f1"] for s in samples_list]
        precs = [s["metrics"]["precision"] for s in samples_list]
        recs = [s["metrics"]["recall"] for s in samples_list]
        gts = [s["gt_pixels"] for s in samples_list]
        preds = [s["pred_pixels"] for s in samples_list]
        fps = [s["metrics"]["fp"] for s in samples_list]
        fns = [s["metrics"]["fn"] for s in samples_list]
        tps = [s["metrics"]["tp"] for s in samples_list]
        return {
            "iou_min": min(ious), "iou_max": max(ious),
            "f1_min": min(f1s), "f1_max": max(f1s),
            "prec_min": min(precs), "prec_max": max(precs),
            "rec_min": min(recs), "rec_max": max(recs),
            "gt_min": min(gts), "gt_max": max(gts),
            "pred_min": min(preds), "pred_max": max(preds),
            "fp_min": min(fps), "fp_max": max(fps),
            "fn_min": min(fns), "fn_max": max(fns),
            "tp_min": min(tps), "tp_max": max(tps),
        }

    cat_a_samples = [s for s in processed_samples if s["category_short"] == "Good"]
    cat_b_samples = [s for s in processed_samples if s["category_short"] == "Under-seg"]
    cat_c_samples = [s for s in processed_samples if s["category_short"] == "Over-seg"]
    cat_d_samples = [s for s in processed_samples if s["category_short"] == "Difficult"]

    st_a = get_cat_stats(cat_a_samples)
    st_b = get_cat_stats(cat_b_samples)
    st_c = get_cat_stats(cat_c_samples)
    st_d = get_cat_stats(cat_d_samples)

    # 6. Write Comprehensive Analysis Summary
    summary_lines.append("\n" + "=" * 70)
    summary_lines.append(f"[2. Measured Observations from the {len(final_selection)} Illustrative Samples]")
    summary_lines.append("=" * 70)
    summary_lines.append(f"""
A. Good Predictions (Samples {cat_a_samples[0]['sample_num']} to {cat_a_samples[-1]['sample_num']}):
  - Measured Metrics (Dynamically Computed from Selected Samples):
      * IoU Range       : {st_a['iou_min']:.3f} to {st_a['iou_max']:.3f}
      * F1 / Dice Range : {st_a['f1_min']:.3f} to {st_a['f1_max']:.3f}
      * Precision Range : {st_a['prec_min']:.3f} to {st_a['prec_max']:.3f}
      * Recall Range    : {st_a['rec_min']:.3f} to {st_a['rec_max']:.3f}
      * Lake Size (GT)  : {st_a['gt_min']:,} to {st_a['gt_max']:,} pixels ({st_a['gt_min']/(512*512)*100:.2f}% to {st_a['gt_max']/(512*512)*100:.2f}% of tile)
      * Missed (FN)     : {st_a['fn_min']:,} to {st_a['fn_max']:,} pixels
      * False Positives : {st_a['fp_min']:,} to {st_a['fp_max']:,} pixels
  - Qualitative Visual Observations:
      * Visual inspection of these five tiles shows high contrast between water bodies and surrounding terrain.
      * Predicted masks exhibit contiguous lake interiors and align closely with visible outer shorelines.

B. Under-Segmentation / Missed Lake Regions (Samples {cat_b_samples[0]['sample_num']} to {cat_b_samples[-1]['sample_num']}):
  - Measured Metrics (Dynamically Computed from Selected Samples):
      * Recall Range    : {st_b['rec_min']:.3f} to {st_b['rec_max']:.3f} (severe recall deficit)
      * Precision Range : {st_b['prec_min']:.3f} to {st_b['prec_max']:.3f}
      * IoU Range       : {st_b['iou_min']:.3f} to {st_b['iou_max']:.3f}
      * F1 / Dice Range : {st_b['f1_min']:.3f} to {st_b['f1_max']:.3f}
      * Lake Size (GT)  : {st_b['gt_min']:,} to {st_b['gt_max']:,} pixels ({st_b['gt_min']/(512*512)*100:.2f}% to {st_b['gt_max']/(512*512)*100:.2f}% of tile)
      * Missed (FN)     : {st_b['fn_min']:,} to {st_b['fn_max']:,} pixels per sample
      * False Positives : {st_b['fp_min']:,} to {st_b['fp_max']:,} pixels per sample
  - Qualitative Visual Hypotheses (Requiring Independent Verification):
      * In Sample 6 (113.png) and Sample 7 (19.png), ground-truth lake areas visually display yellowish-green or grey-tan surface textures rather than typical dark open water. A visual hypothesis is that these regions may represent sediment-rich water, ephemeral melt ponds, or dried lake beds with different spectral responses, but this interpretation is based solely on RGB imagery and is not hydrologically ground-verified.
      * In Samples 8 (145.png), 9 (43.png), and 10 (107.png), missed pixels (FN) visually cluster along shaded mountain margins and debris-covered perimeters.

C. Over-Segmentation / Spurious Detections (Samples {cat_c_samples[0]['sample_num']} to {cat_c_samples[-1]['sample_num']}):
  - Measured Metrics (Dynamically Computed from Selected Samples):
      * Precision Range : {st_c['prec_min']:.3f} to {st_c['prec_max']:.3f} (substantial false positive rate)
      * Recall Range    : {st_c['rec_min']:.3f} to {st_c['rec_max']:.3f}
      * IoU Range       : {st_c['iou_min']:.3f} to {st_c['iou_max']:.3f}
      * F1 / Dice Range : {st_c['f1_min']:.3f} to {st_c['f1_max']:.3f}
      * Lake Size (GT)  : {st_c['gt_min']:,} to {st_c['gt_max']:,} pixels ({st_c['gt_min']/(512*512)*100:.2f}% to {st_c['gt_max']/(512*512)*100:.2f}% of tile)
      * False Positives : {st_c['fp_min']:,} to {st_c['fp_max']:,} pixels per sample
      * Missed (FN)     : {st_c['fn_min']:,} to {st_c['fn_max']:,} pixels per sample
  - Qualitative Visual Hypotheses (Requiring Independent Verification):
      * In Samples 11 (141.png) and 12 (103.png), false positive predictions occurred in dark mountain shadow regions without overlapping the small reference ground-truth annotations.
      * In Samples 13 (140.png) and 15 (132.png), the model detected the true lake, but false positives also coincide with adjacent dark depressions or unannotated peripheral wet areas. Whether these represent annotation omissions or model confusion with moist non-lake terrain is a visual hypothesis rather than a confirmed defect in ground-truth labels.

D. Difficult Cases — Small Lakes (Samples {cat_d_samples[0]['sample_num']} to {cat_d_samples[-1]['sample_num']}):
  - Measured Metrics (Dynamically Computed from Selected Samples):
      * Lake Size (GT)  : {st_d['gt_min']:,} to {st_d['gt_max']:,} pixels ({st_d['gt_min']/(512*512)*100:.2f}% to {st_d['gt_max']/(512*512)*100:.2f}% of tile)
      * IoU Range       : {st_d['iou_min']:.3f} to {st_d['iou_max']:.3f}
      * F1 / Dice Range : {st_d['f1_min']:.3f} to {st_d['f1_max']:.3f}
      * Precision Range : {st_d['prec_min']:.3f} to {st_d['prec_max']:.3f}
      * Recall Range    : {st_d['rec_min']:.3f} to {st_d['rec_max']:.3f}
      * Detected (TP)   : {st_d['tp_min']:,} to {st_d['tp_max']:,} pixels per lake
      * False Positives : {st_d['fp_min']:,} to {st_d['fp_max']:,} pixels per sample
      * Missed (FN)     : {st_d['fn_min']:,} to {st_d['fn_max']:,} pixels per sample
  - Qualitative Visual Observations:
      * All five selected small-lake examples have nonzero overlap between prediction and ground truth, with IoU ranging from 0.638 to 0.822 and detected true positives ranging from 37 to 89 pixels.
      * Because of the small total footprint (41 to 109 pixels), single-pixel discrete boundary variations have a disproportionately large impact on IoU relative to larger lakes.

======================================================================
[3. Full Validation Evaluation Context]
======================================================================

The complete validation set (all 2,367 samples, 620,494,848 pixels) was
evaluated in Step 1. The quantitative global metrics remain:

  - Intersection over Union (IoU) : 0.8862  (88.62%)
  - F1 Score / Dice Coefficient   : 0.9397  (93.97%)
  - Precision                     : 0.9561  (95.61%)
  - Recall                        : 0.9238  (92.38%)
  - Pixel Accuracy                : 0.9973  (99.73%)

Global Confusion Matrix Analysis:
  - True Positives  (TP) : 13,109,933 pixels
  - False Positives (FP) :    602,409 pixels
  - False Negatives (FN) :  1,080,860 pixels
  - True Negatives  (TN) : 605,701,646 pixels

Consistency Discussion:
  The global confusion matrix shows that False Negatives (1,080,860 pixels) exceed
  False Positives (602,409 pixels) by a factor of 1.79. This confirms that the model's
  dominant error mode across the complete validation set is under-segmentation
  (omission of true lake pixels) rather than over-segmentation (commission of false
  lake pixels), consistent with higher global Precision (95.61%) relative to
  Recall (92.38%).

  Crucially, factors such as topographic shadows, turbid boundaries, sediment, or
  ice cover are qualitative failure mechanisms suggested by inspection of specific
  selected visual examples; they have not been independently quantified as causes
  across the full 2,367 validation images.

======================================================================
[4. Limitations]
======================================================================

1. Extreme Class Imbalance:
   Across the 2,367 validation images, lake pixels comprise 2.29% of all pixels
   (14,190,793 lake pixels vs. 606,304,055 background pixels). Pixel accuracy
   (99.73%) is heavily dominated by true background (TN) and does not provide an
   adequate indicator of segmentation quality on its own.

2. Small-Lake Boundary Sensitivity:
   In small lakes, spatial discrete boundaries on a 512x512 grid mean that boundary
   deviations have a large percentage effect on IoU relative to larger lakes, even
   when the lake entity is correctly identified and localized.

3. Ground-Truth Annotation Uncertainty:
   Visual comparison reveals instances of potential annotation ambiguity in
   remote-sensing imagery, such as unannotated water bodies or dark depressions
   (e.g. Sample 13, 140.png) and ambiguous turbid or desiccated surfaces
   (e.g. Sample 6, 113.png). These are noted as visual hypotheses, not ground-verified
   annotation errors.

4. Non-Representative Sampling Scope:
   The {len(final_selection)} samples visualized here are illustrative, selected from the first {scan_count} validation samples using metric-based criteria, and are not statistically representative of all 2,367 validation samples. Category-level metric ranges (such as those in Section 2) describe only these specific illustrative cases and must not be presented as dataset-wide performance bounds. Quantitative model assessment must rely on the full-validation metrics reported in Section 3.

======================================================================
[5. Deliverables in this Directory]
======================================================================
  - sample_01_*.png to sample_{len(final_selection):02d}_*.png : Individual 4-panel visual comparisons
  - error_analysis_grid.png           : Four-column consolidated visual summary grid containing {len(final_selection)} samples (each displayed as an original-image and error-overlay pair)
  - error_analysis_summary.txt        : This corrected error-analysis report
""")

    summary_file = out_dir / "error_analysis_summary.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSummary report saved -> {summary_file}")
    print(f"Error analysis completed successfully in {out_dir.resolve()}.")


if __name__ == "__main__":
    args = parse_args()
    run_error_analysis(args)
