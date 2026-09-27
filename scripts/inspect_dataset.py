"""
inspect_dataset.py
==================
Milestone 1 — Dataset inspection and verification script.

Runs the following checks and reports:

  1. Directory structure discovery (auto-detects image/mask subdirs)
  2. File count verification (expects 16,000 train / 2,367 val)
  3. Sample property checks (shape, dtype, mask value range)
  4. Mask value distribution (are masks strictly binary 0/255 on disk?)
  5. Class distribution (lake pixels vs background pixels)
  6. Per-channel normalization statistics (computed from raw pixel values)
  7. DataLoader batch shape smoke-test
  8. Sample visualization grid (saves to results/plots/)

Usage:
    python scripts/inspect_dataset.py --config configs/default.yaml

    # Faster run — skip normalization stats and use fewer samples
    python scripts/inspect_dataset.py --config configs/default.yaml \\
        --no-norm-stats --n-dist-samples 200 --n-viz 6

Output:
    results/logs/dataset_inspection.txt   — Full text report
    results/plots/sample_grid_train.png   — Grid of train image/mask pairs
    results/plots/sample_grid_val.png     — Grid of val image/mask pairs
    results/logs/dataset_stats.yaml       — Machine-readable stats summary
"""

import argparse
import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import yaml

# Make sure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.dataset import GLIDDataset, get_dataloaders_from_config, discover_split_paths
from src.utils import (
    load_config,
    set_seed,
    ensure_dirs,
    compute_class_distribution,
    compute_normalization_stats,
    verify_sample_properties,
    visualize_batch_grid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Logger:
    """Writes to both stdout and a log file simultaneously."""
    def __init__(self, filepath: str):
        ensure_dirs(str(Path(filepath).parent))
        self._file = open(filepath, "w", encoding="utf-8")

    def log(self, msg: str = "") -> None:
        print(msg)
        self._file.write(msg + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def section(logger: Logger, title: str) -> None:
    bar = "=" * 60
    logger.log(f"\n{bar}")
    logger.log(f"  {title}")
    logger.log(bar)


def check_raw_mask_values(dataset: GLIDDataset, n: int = 100, logger: Logger = None) -> dict:
    """
    Read masks directly from disk (bypassing binarization) and report unique values.
    GLID masks should only contain {0, 255}.
    """
    from PIL import Image
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    all_unique = set()

    for idx in indices:
        fname = dataset.image_files[idx]
        mask_pil = Image.open(dataset.mask_dir / fname).convert("L")
        arr = np.array(mask_pil)
        all_unique.update(np.unique(arr).tolist())

    msg = (
        f"Raw mask pixel values (on-disk, before binarization) — {n} samples: {sorted(all_unique)}\n"
        f"  Expected: {{0, 255}}"
    )
    if logger:
        logger.log(msg)
    else:
        print(msg)

    unexpected = all_unique - {0, 255}
    if unexpected:
        warn = f"  ⚠ WARNING: Unexpected values found: {unexpected}"
        if logger:
            logger.log(warn)
        else:
            print(warn)

    return {"raw_mask_values": sorted(all_unique), "unexpected_values": sorted(unexpected)}


# ---------------------------------------------------------------------------
# Main inspection routine
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    raw_dir  = Path(data_cfg["raw_dir"])
    train_dir = raw_dir / data_cfg.get("train_subdir", "GLID")
    val_dir   = raw_dir / data_cfg.get("val_subdir", "val")

    out_dir   = Path(cfg.get("output", {}).get("results_dir", "results"))
    log_path  = out_dir / "logs" / "dataset_inspection.txt"
    stats_path = out_dir / "logs" / "dataset_stats.yaml"
    plots_dir = out_dir / "plots"
    ensure_dirs(str(out_dir / "logs"), str(plots_dir))

    logger = Logger(str(log_path))
    logger.log("Glacial Lake Mapping — Dataset Inspection Report")
    logger.log(f"Timestamp  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Config file: {args.config}")

    all_stats = {}

    # -----------------------------------------------------------------------
    # 1. Directory structure
    # -----------------------------------------------------------------------
    section(logger, "1. Directory Structure Discovery")

    for split_name, split_path in [("TRAIN", train_dir), ("VAL", val_dir)]:
        logger.log(f"\n[{split_name}] Checking: {split_path}")
        if not split_path.is_dir():
            logger.log(f"  ✗ ERROR: Directory not found: '{split_path}'")
            logger.log(
                f"    → Have you downloaded and extracted the GLID dataset?\n"
                f"    → See data/README.md for download instructions.\n"
                f"    → Expected: GLID.rar → '{train_dir}'\n"
                f"               val.zip   → '{val_dir}'"
            )
            logger.close()
            sys.exit(1)

        try:
            image_dir, mask_dir = discover_split_paths(str(split_path))
            logger.log(f"  ✓ Images : {image_dir}")
            logger.log(f"  ✓ Masks  : {mask_dir}")
        except FileNotFoundError as e:
            logger.log(f"  ✗ ERROR: {e}")
            logger.close()
            sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. Build datasets (no augmentation for inspection)
    # -----------------------------------------------------------------------
    section(logger, "2. Loading Datasets")

    train_dataset = GLIDDataset(
        split_dir=str(train_dir),
        is_train=False,   # No augmentation for inspection
        split="train",
        image_subdir=data_cfg.get("image_subdir"),
        mask_subdir=data_cfg.get("mask_subdir"),
        normalize_mean=data_cfg.get("normalize_mean"),
        normalize_std=data_cfg.get("normalize_std"),
        image_size=data_cfg.get("image_size", 512),
    )
    val_dataset = GLIDDataset(
        split_dir=str(val_dir),
        is_train=False,
        image_subdir=data_cfg.get("image_subdir"),
        mask_subdir=data_cfg.get("mask_subdir"),
        normalize_mean=data_cfg.get("normalize_mean"),
        normalize_std=data_cfg.get("normalize_std"),
        image_size=data_cfg.get("image_size", 512),
    )

    logger.log(f"\n  {train_dataset}")
    logger.log(f"  {val_dataset}")

    # -----------------------------------------------------------------------
    # 3. File count
    # -----------------------------------------------------------------------
    section(logger, "3. File Count Verification")

    EXPECTED_TRAIN = 16000
    EXPECTED_VAL   = 2367

    n_train = len(train_dataset)
    n_val   = len(val_dataset)

    train_ok = n_train == EXPECTED_TRAIN
    val_ok   = n_val   == EXPECTED_VAL

    logger.log(f"  Train samples : {n_train:,}  (expected {EXPECTED_TRAIN:,})  {'✓' if train_ok else '⚠ MISMATCH'}")
    logger.log(f"  Val samples   : {n_val:,}    (expected {EXPECTED_VAL:,})    {'✓' if val_ok else '⚠ MISMATCH'}")
    logger.log(f"  Total samples : {n_train + n_val:,}")

    all_stats["file_counts"] = {
        "train": n_train, "val": n_val, "total": n_train + n_val
    }

    # -----------------------------------------------------------------------
    # 4. Raw mask value check (on-disk)
    # -----------------------------------------------------------------------
    section(logger, "4. Raw Mask Value Check (On-Disk)")
    logger.log("  Checking raw PNG mask values before binarization...")
    raw_stats = check_raw_mask_values(train_dataset, n=200, logger=logger)
    all_stats["raw_mask_values"] = raw_stats

    # -----------------------------------------------------------------------
    # 5. Sample property verification
    # -----------------------------------------------------------------------
    section(logger, "5. Tensor Property Verification")
    logger.log("  Verifying shapes, dtypes, and mask value range after loading...")
    try:
        prop_results = verify_sample_properties(train_dataset, n_samples=10)
        for k, v in prop_results.items():
            logger.log(f"  {k}: {v}")
        all_stats["tensor_properties"] = prop_results
    except AssertionError as e:
        logger.log(f"  ✗ ASSERTION FAILED: {e}")
        logger.close()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 6. Class distribution
    # -----------------------------------------------------------------------
    section(logger, "6. Class Distribution (Lake vs Background)")
    logger.log(f"  Scanning {args.n_dist_samples} train samples for class distribution...")

    dist_stats = compute_class_distribution(
        train_dataset, n_samples=args.n_dist_samples, verbose=False
    )
    logger.log(
        f"  Samples scanned  : {dist_stats['samples_scanned']:,}\n"
        f"  Total pixels     : {dist_stats['total_pixels']:,}\n"
        f"  Lake pixels      : {dist_stats['lake_pixels']:,}  ({dist_stats['lake_fraction'] * 100:.2f}%)\n"
        f"  Background pixels: {dist_stats['bg_pixels']:,}   ({dist_stats['bg_fraction'] * 100:.2f}%)\n"
        f"  Imbalance ratio  : {dist_stats['imbalance_ratio']:.1f}×  (bg:lake)"
    )
    all_stats["class_distribution"] = {
        k: (float(v) if isinstance(v, float) else int(v))
        for k, v in dist_stats.items()
    }

    # -----------------------------------------------------------------------
    # 7. Per-channel normalization statistics
    # -----------------------------------------------------------------------
    if not args.no_norm_stats:
        section(logger, "7. Per-Channel Normalization Statistics")
        logger.log(
            f"  Computing mean/std from raw pixel values — {args.n_norm_samples} train samples...\n"
            f"  (This may take a few minutes for large sample counts)"
        )
        norm_stats = compute_normalization_stats(
            train_dataset, n_samples=args.n_norm_samples, verbose=False
        )
        logger.log(
            f"  Dataset Mean (R, G, B): {[round(v, 4) for v in norm_stats['mean']]}\n"
            f"  Dataset Std  (R, G, B): {[round(v, 4) for v in norm_stats['std']]}\n"
            f"\n  ImageNet Mean (reference): [0.485, 0.456, 0.406]\n"
            f"  ImageNet Std  (reference): [0.229, 0.224, 0.225]"
        )
        all_stats["normalization_stats"] = norm_stats
    else:
        logger.log("\n  [7. Normalization stats skipped — use --no-norm-stats=False to enable]")

    # -----------------------------------------------------------------------
    # 8. DataLoader smoke test
    # -----------------------------------------------------------------------
    section(logger, "8. DataLoader Batch Smoke Test")

    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,       # 0 workers for inspection — avoids multiprocessing issues
        pin_memory=False,
    )
    batch_images, batch_masks = next(iter(loader))

    logger.log(
        f"  Batch image tensor shape : {tuple(batch_images.shape)}\n"
        f"  Batch mask tensor shape  : {tuple(batch_masks.shape)}\n"
        f"  Image dtype              : {batch_images.dtype}\n"
        f"  Mask dtype               : {batch_masks.dtype}\n"
        f"  Mask unique values in batch: {batch_masks.unique().tolist()}\n"
        f"  DataLoader ✓ — batches load correctly"
    )
    all_stats["dataloader_smoke_test"] = {
        "batch_image_shape": list(batch_images.shape),
        "batch_mask_shape":  list(batch_masks.shape),
        "passed": True,
    }

    # -----------------------------------------------------------------------
    # 9. Sample visualizations
    # -----------------------------------------------------------------------
    section(logger, "9. Sample Visualizations")

    # Sample randomly for visualization
    n_viz = min(args.n_viz, len(train_dataset), len(val_dataset))

    def sample_batch(dataset: GLIDDataset, n: int) -> tuple:
        indices = random.sample(range(len(dataset)), n)
        imgs, msks = [], []
        for idx in indices:
            img, msk = dataset[idx]
            imgs.append(img)
            msks.append(msk)
        return torch.stack(imgs), torch.stack(msks)

    logger.log(f"  Saving {n_viz} sample grids...")

    for split_name, ds, fname in [
        ("train", train_dataset, "sample_grid_train.png"),
        ("val",   val_dataset,   "sample_grid_val.png"),
    ]:
        imgs, msks = sample_batch(ds, n_viz)
        out_path = str(plots_dir / fname)
        fig = visualize_batch_grid(
            imgs, msks,
            n_samples=n_viz,
            save_path=out_path,
            normalize_mean=data_cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
            normalize_std=data_cfg.get("normalize_std", [0.229, 0.224, 0.225]),
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        logger.log(f"  Saved {split_name} grid → {out_path}")

    # -----------------------------------------------------------------------
    # Save machine-readable stats
    # -----------------------------------------------------------------------
    section(logger, "Summary")

    with open(stats_path, "w") as f:
        yaml.dump(all_stats, f, default_flow_style=False, sort_keys=False)
    logger.log(f"  Machine-readable stats saved → {stats_path}")
    logger.log(f"  Full text report saved       → {log_path}")
    logger.log(f"  Sample visualizations saved  → {plots_dir}/")
    logger.log(
        "\n  ✓ Milestone 1 dataset inspection complete.\n"
        "  → Review results above and the saved files before proceeding to Milestone 2."
    )
    logger.close()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLID dataset inspection — Milestone 1"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--n-dist-samples", type=int, default=500,
        help="Samples to scan for class distribution (default: 500)"
    )
    parser.add_argument(
        "--n-norm-samples", type=int, default=2000,
        help="Samples for computing per-channel mean/std (default: 2000)"
    )
    parser.add_argument(
        "--no-norm-stats", action="store_true",
        help="Skip per-channel normalization stats (faster run)"
    )
    parser.add_argument(
        "--n-viz", type=int, default=8,
        help="Number of sample images to visualize per split (default: 8)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
