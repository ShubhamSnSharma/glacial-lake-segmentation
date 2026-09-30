"""
evaluate.py
===========
Full-validation evaluation pipeline for trained glacial lake segmentation models.

Supports two architectures:
  - DeepLabV3+   (ResNet-50 + ASPP, ~40M params) — default, checkpoints/best_amp_model.pth
  - ResNet34-FCN (ResNet-34 + FCN decoder, ~24M params) — Step 3 baseline

The architecture is auto-detected from the checkpoint's 'model_name' key.
Use --model to override detection when the key is absent.

Processes EVERY sample in the validation split (default: data/raw/val), accumulates
pixel-level TP/FP/FN/TN counts across all images, and computes dataset-level metrics:
  - Intersection over Union (IoU / Jaccard)
  - F1 Score / Dice Coefficient
  - Precision
  - Recall
  - Pixel Accuracy

Metrics are derived from the global confusion matrix — NOT averaged per-image —
which is the correct approach for datasets with class-imbalanced patches.

Usage:
    # DeepLabV3+ — best trained checkpoint (default)
    python scripts/evaluate.py

    # ResNet34-FCN — Step 3 baseline
    python scripts/evaluate.py \
        --checkpoint resnet34_fcn/checkpoints/best_model.pth \
        --model      resnet34_fcn \
        --output-dir resnet34_fcn/results/evaluation

    # Explicit paths and options
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_amp_model.pth \
        --val-dir    data/raw/val \
        --batch-size 8 \
        --threshold  0.5 \
        --output-dir results/evaluation

    # CPU-only
    python scripts/evaluate.py --device cpu

Output (written to --output-dir):
    eval_metrics.json     — machine-readable metrics + configuration
    eval_summary.txt      — human-readable report for project submission

References:
    GLID dataset: https://doi.org/10.5281/zenodo.14838695
    DeepLabV3+  : Chen et al. (2018), arXiv:1802.02611
    ResNet-34   : He et al. (2016), arXiv:1512.03385
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Ensure src/ is importable regardless of where the script is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import GLIDDataset
from src.metrics import MetricTracker, calculate_metrics
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.models.resnet34_fcn import ResNet34FCN
from src.utils import ensure_dirs, set_seed

# Canonical architecture name aliases
_DEEPLABV3PLUS_ALIASES = frozenset({
    "deeplabv3plus", "deeplabv3+", "deeplabv3_plus", "deeplab", "advanced",
})
_RESNET34FCN_ALIASES = frozenset({
    "resnet34_fcn", "resnet34-fcn", "resnet34", "baseline",
})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full-validation evaluation of a trained glacial lake segmentation model "
            "(DeepLabV3+ or ResNet34-FCN) on all samples in the validation split."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_amp_model.pth",
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["deeplabv3plus", "resnet34_fcn"],
        help=(
            "Model architecture to evaluate. Auto-detected from the checkpoint's "
            "'model_name' key when not specified. Required only if the checkpoint "
            "does not contain a 'model_name' key."
        ),
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="data/raw/val",
        help="Path to the validation split root directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (tune down if MPS runs out of memory)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid probability threshold for classifying a pixel as lake",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Directory to save eval_metrics.json and eval_summary.txt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["mps", "cpu"],
        help="Inference device. Auto-detected (MPS -> CPU) when not specified.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (0 = main process; safest for MPS)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (for reproducibility of any stochastic operations)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device(requested: Optional[str]) -> torch.device:
    """Return MPS if available, otherwise CPU. Never requires CUDA."""
    if requested is not None:
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    device: torch.device,
    model_override: Optional[str] = None,
) -> tuple:
    """
    Build the correct architecture and restore weights from a checkpoint.

    Architecture resolution order:
      1. ``model_override`` argument (from --model CLI flag), if provided.
      2. ``checkpoint['model_name']`` key (present in Colab AMP and Trainer formats).
      3. ``checkpoint['cfg']['model']['name']`` (Local Trainer YAML config).
      4. Falls back to 'deeplabv3plus' for backwards compatibility with older checkpoints.

    Compatible with both:
      - Colab AMP format : keys include 'best_f1', 'model_name', 'amp'
      - Local Trainer    : keys include 'best_metric_val', 'cfg'

    Args:
        checkpoint_path: Path to .pth file.
        device: torch.device for inference.
        model_override: Optional architecture name to use instead of auto-detection.

    Returns:
        (model, checkpoint_dict, resolved_model_name)
    """
    chk_path = Path(checkpoint_path)
    if not chk_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: '{chk_path}'\n"
            "For DeepLabV3+: ensure best_amp_model.pth is in checkpoints/.\n"
            "For ResNet34-FCN: ensure best_model.pth is in resnet34_fcn/checkpoints/."
        )

    print(f"Loading checkpoint : {chk_path}")
    checkpoint = torch.load(chk_path, map_location="cpu", weights_only=False)

    # --- Resolve architecture ---
    if model_override is not None:
        resolved_name = model_override.lower()
        print(f"  Architecture : {resolved_name}  (forced via --model)")
    else:
        resolved_name = checkpoint.get(
            "model_name",
            checkpoint.get("cfg", {}).get("model", {}).get("name", "deeplabv3plus"),
        ).lower()
        print(f"  Architecture : {resolved_name}  (auto-detected from checkpoint)")

    # --- Instantiate correct architecture ---
    if resolved_name in _RESNET34FCN_ALIASES:
        model = ResNet34FCN(pretrained=False, num_classes=1)
        arch_label = "ResNet34-FCN (ResNet-34 backbone + FCN decoder)"
    elif resolved_name in _DEEPLABV3PLUS_ALIASES:
        model = DeepLabV3Plus(pretrained=False, num_classes=1)
        arch_label = "DeepLabV3+ (ResNet-50 backbone + ASPP)"
    else:
        raise ValueError(
            f"Unknown architecture '{resolved_name}'. "
            f"Supported: {sorted(_DEEPLABV3PLUS_ALIASES | _RESNET34FCN_ALIASES)}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    epoch   = checkpoint.get("epoch", "unknown")
    best_f1 = checkpoint.get("best_f1", checkpoint.get("best_metric_val", "unknown"))
    print(f"  Full name    : {arch_label}")
    print(f"  Epoch        : {epoch}")
    print(f"  Best F1      : {best_f1}")

    return model, checkpoint, resolved_name


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:

    run_start = datetime.now()
    set_seed(args.seed)

    # ---- Device ----
    device = select_device(args.device)
    print(f"\nDevice    : {device}")
    print(f"Threshold : {args.threshold}")
    print(f"Batch size: {args.batch_size}\n")

    # ---- Output directory ----
    out_dir = Path(args.output_dir)
    ensure_dirs(str(out_dir))

    # ---- Model ----
    model, checkpoint, resolved_arch = load_model(
        args.checkpoint, device, model_override=args.model
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters : {n_params:,}")

    # ---- Validation dataset ----
    val_ds = GLIDDataset(
        split_dir=args.val_dir,
        is_train=False,   # no augmentation
        split="val",
    )
    n_val = len(val_ds)
    print(f"Validation samples : {n_val}")

    # pin_memory is only safe on CUDA; skip for MPS/CPU
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,                          # deterministic order
        num_workers=args.num_workers,
        pin_memory=False,                       # MPS-safe
        drop_last=False,                        # evaluate EVERY sample
        persistent_workers=(args.num_workers > 0),
    )
    n_batches = len(val_loader)
    print(f"Batches            : {n_batches}  (batch_size={args.batch_size})\n")

    # ---- Metric accumulator ----
    tracker = MetricTracker(threshold=args.threshold)

    # ---- Timing ----
    t_eval_start     = time.perf_counter()
    batch_times      = []
    images_processed = 0

    # ---- Inference loop ----
    print("=" * 60)
    print(" Running full-validation inference ...")
    print("=" * 60)

    with torch.inference_mode():
        for batch_imgs, batch_masks in tqdm(
            val_loader,
            desc="Evaluating",
            unit="batch",
            ncols=72,
        ):
            t_batch_start = time.perf_counter()

            batch_imgs  = batch_imgs.to(device)   # (B, 3, 512, 512)
            batch_masks = batch_masks.to(device)   # (B, 1, 512, 512)

            # Forward pass — no AMP, no channels_last (MPS-safe)
            logits = model(batch_imgs)             # (B, 1, 512, 512)
            probs  = torch.sigmoid(logits)
            preds  = (probs >= args.threshold).float()

            # Accumulate confusion matrix on CPU to avoid MPS→CPU sync overhead
            tracker.update(preds.cpu(), batch_masks.cpu())

            batch_times.append(time.perf_counter() - t_batch_start)
            images_processed += batch_imgs.size(0)

    t_eval_end   = time.perf_counter()
    total_time   = t_eval_end - t_eval_start
    avg_img_time = total_time / max(images_processed, 1)

    # ---- Verify all samples were processed ----
    if images_processed != n_val:
        print(
            f"\n[WARNING] Processed {images_processed} images but dataset has {n_val}. "
            "Partial evaluation — results may be incomplete."
        )
    else:
        print(f"\n  All {images_processed} validation samples processed. ✓")

    # ---- Compute global dataset-level metrics ----
    # Uses accumulated global TP/FP/FN/TN — NOT an average of per-image metrics
    agg = tracker.compute()

    # Raw confusion matrix totals
    total_tp = tracker.total_tp
    total_fp = tracker.total_fp
    total_fn = tracker.total_fn
    total_tn = tracker.total_tn
    total_px = total_tp + total_fp + total_fn + total_tn

    # ---- Print results ----
    chk_epoch_disp = checkpoint.get("epoch", "unknown")
    arch_display = {
        **{k: "DeepLabV3+" for k in _DEEPLABV3PLUS_ALIASES},
        **{k: "ResNet34-FCN" for k in _RESNET34FCN_ALIASES},
    }.get(resolved_arch, resolved_arch)

    print()
    print("=" * 60)
    print(f"  FULL VALIDATION RESULTS — {arch_display} (Epoch {chk_epoch_disp})")
    print("=" * 60)
    print(f"  Samples evaluated  : {images_processed} / {n_val}")
    print(f"  Total pixels       : {total_px:,}")
    print(f"  Lake pixels (GT)   : {total_tp + total_fn:,}  ({(total_tp+total_fn)/total_px*100:.2f}%)")
    print(f"  Background (GT)    : {total_fp + total_tn:,}  ({(total_fp+total_tn)/total_px*100:.2f}%)")
    print("-" * 60)
    print(f"  Confusion Matrix:")
    print(f"    TP : {total_tp:>12,}")
    print(f"    FP : {total_fp:>12,}")
    print(f"    FN : {total_fn:>12,}")
    print(f"    TN : {total_tn:>12,}")
    print("-" * 60)
    print(f"  IoU            : {agg['iou']:.4f}  ({agg['iou']*100:.2f}%)")
    print(f"  F1 / Dice      : {agg['f1']:.4f}  ({agg['f1']*100:.2f}%)")
    print(f"  Precision      : {agg['precision']:.4f}  ({agg['precision']*100:.2f}%)")
    print(f"  Recall         : {agg['recall']:.4f}  ({agg['recall']*100:.2f}%)")
    print(f"  Pixel Accuracy : {agg['accuracy']:.4f}  ({agg['accuracy']*100:.2f}%)")
    print("-" * 60)
    print(f"  Total eval time    : {total_time:.1f}s")
    print(f"  Avg time / image   : {avg_img_time*1000:.1f} ms")
    print(f"  Avg time / batch   : {sum(batch_times)/len(batch_times)*1000:.1f} ms")
    print("=" * 60)

    # ---- Build output documents ----
    chk_epoch   = checkpoint.get("epoch", "unknown")
    chk_best_f1 = checkpoint.get("best_f1", checkpoint.get("best_metric_val", "unknown"))
    arch_full_label = {
        **{k: "DeepLabV3+ (ResNet-50 backbone + ASPP)" for k in _DEEPLABV3PLUS_ALIASES},
        **{k: "ResNet34-FCN (ResNet-34 backbone + FCN decoder)" for k in _RESNET34FCN_ALIASES},
    }.get(resolved_arch, resolved_arch)

    # -- JSON: machine-readable --
    results_json = {
        "evaluation_config": {
            "checkpoint":        str(Path(args.checkpoint).resolve()),
            "checkpoint_epoch":  chk_epoch,
            "checkpoint_best_f1": float(chk_best_f1) if isinstance(chk_best_f1, float) else chk_best_f1,
            "val_dir":           str(Path(args.val_dir).resolve()),
            "n_val_samples":     n_val,
            "images_processed":  images_processed,
            "batch_size":        args.batch_size,
            "threshold":         args.threshold,
            "device":            str(device),
            "num_workers":       args.num_workers,
            "seed":              args.seed,
            "run_timestamp":     run_start.isoformat(),
        },
        "confusion_matrix": {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "TN": total_tn,
            "total_pixels": total_px,
        },
        "metrics": {
            "iou":       round(agg["iou"],       6),
            "f1":        round(agg["f1"],        6),
            "precision": round(agg["precision"], 6),
            "recall":    round(agg["recall"],    6),
            "accuracy":  round(agg["accuracy"],  6),
        },
        "timing": {
            "total_eval_seconds":       round(total_time, 3),
            "avg_seconds_per_image":    round(avg_img_time, 6),
            "avg_ms_per_image":         round(avg_img_time * 1000, 3),
            "avg_ms_per_batch":         round(sum(batch_times) / len(batch_times) * 1000, 3),
            "n_batches":                n_batches,
        },
    }

    json_path = out_dir / "eval_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nJSON results saved : {json_path}")

    # -- TXT: human-readable summary --
    txt_path = out_dir / "eval_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  Deep Learning Based Glacial Lake Detection & Segmentation\n")
        f.write("  Full Validation Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("[Evaluation Configuration]\n")
        f.write(f"  Model architecture : {arch_full_label}\n")
        f.write(f"  Checkpoint         : {args.checkpoint}\n")
        f.write(f"  Checkpoint epoch   : {chk_epoch}\n")
        f.write(f"  Checkpoint best F1 : {chk_best_f1}\n")
        f.write(f"  Validation set     : {args.val_dir}\n")
        f.write(f"  Val samples (total): {n_val}\n")
        f.write(f"  Samples evaluated  : {images_processed}\n")
        f.write(f"  Batch size         : {args.batch_size}\n")
        f.write(f"  Threshold          : {args.threshold}\n")
        f.write(f"  Device             : {device}\n")
        f.write(f"  Run timestamp      : {run_start.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("[Pixel Statistics]\n")
        f.write(f"  Total pixels       : {total_px:,}\n")
        f.write(f"  Lake pixels (GT)   : {total_tp + total_fn:,}  ({(total_tp+total_fn)/total_px*100:.2f}%)\n")
        f.write(f"  Background (GT)    : {total_fp + total_tn:,}  ({(total_fp+total_tn)/total_px*100:.2f}%)\n\n")

        f.write("[Global Confusion Matrix]\n")
        f.write(f"  True  Positives (TP) : {total_tp:>12,}\n")
        f.write(f"  False Positives (FP) : {total_fp:>12,}\n")
        f.write(f"  False Negatives (FN) : {total_fn:>12,}\n")
        f.write(f"  True  Negatives (TN) : {total_tn:>12,}\n\n")

        f.write("[Segmentation Metrics — Global Pixel Level]\n")
        f.write(f"  IoU (Jaccard)    : {agg['iou']:.4f}  ({agg['iou']*100:.2f}%)\n")
        f.write(f"  F1 / Dice        : {agg['f1']:.4f}  ({agg['f1']*100:.2f}%)\n")
        f.write(f"  Precision        : {agg['precision']:.4f}  ({agg['precision']*100:.2f}%)\n")
        f.write(f"  Recall           : {agg['recall']:.4f}  ({agg['recall']*100:.2f}%)\n")
        f.write(f"  Pixel Accuracy   : {agg['accuracy']:.4f}  ({agg['accuracy']*100:.2f}%)\n\n")

        f.write("[Timing]\n")
        f.write(f"  Total evaluation time : {total_time:.1f} s\n")
        f.write(f"  Avg time per image    : {avg_img_time*1000:.1f} ms\n")
        f.write(f"  Avg time per batch    : {sum(batch_times)/len(batch_times)*1000:.1f} ms\n")
        f.write(f"  Batches processed     : {n_batches}\n\n")

        f.write("[Notes]\n")
        f.write("  - Metrics are computed from the global confusion matrix accumulated\n")
        f.write("    over all validation pixels, NOT as an average of per-image scores.\n")
        f.write("  - This is the standard methodology for semantic segmentation evaluation\n")
        f.write("    on datasets with highly imbalanced class distributions.\n")
        f.write("  - Epsilon = 1e-6 is used in denominators to handle edge cases.\n")
        f.write("  - No CUDA AMP or channels_last applied during this evaluation.\n")

    print(f"TXT summary saved  : {txt_path}")
    print(f"\nAll outputs in     : {out_dir.resolve()}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)


# ---------------------------------------------------------------------------
# Public API for compare_models.py
# ---------------------------------------------------------------------------

def load_eval_metrics(json_path: str) -> dict:
    """
    Load a previously saved eval_metrics.json produced by this script.

    Args:
        json_path: Path to eval_metrics.json.

    Returns:
        Parsed dict with keys 'evaluation_config', 'confusion_matrix', 'metrics', 'timing'.

    Raises:
        FileNotFoundError if the file does not exist.
    """
    p = Path(json_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Evaluation results not found at '{p}'.\n"
            "Run scripts/evaluate.py first to generate eval_metrics.json."
        )
    with open(p) as f:
        import json as _json
        return _json.load(f)
