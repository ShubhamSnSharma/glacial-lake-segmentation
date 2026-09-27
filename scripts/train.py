"""
train.py
========
Training entry point for the glacial lake segmentation model.

Usage:
    # Run a small smoke test on real data (verifies forward, loss, backward, step, val, checkpoint)
    python scripts/train.py --config configs/default.yaml --smoke-test

    # Run full training
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.dataset import get_dataloaders_from_config
from src.model import build_model
from src.trainer import Trainer
from src.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train semantic segmentation models for glacial lake mapping")
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint file (.pth) to resume training from"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Root directory for persistent outputs (checkpoints, results, logs, predictions)"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a 3-batch smoke test on real data to verify training pipeline"
    )
    parser.add_argument(
        "--smoke-test-batches", type=int, default=3,
        help="Number of batches to run during smoke test (default: 3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs specified in config"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size specified in config"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate specified in config"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model architecture ('resnet34_fcn' or 'deeplabv3plus')"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Override DataLoader num_workers"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use ('mps', 'cuda', or 'cpu'). Auto-detected if not specified."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.model is not None:
        cfg.setdefault("model", {})["name"] = args.model
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    elif args.smoke_test:
        cfg["data"]["num_workers"] = 0  # Instant loading for smoke testing on macOS

    if args.output_dir is not None:
        out_root = Path(args.output_dir)
        cfg.setdefault("output", {})
        cfg["output"]["checkpoint_dir"] = str(out_root / "checkpoints")
        cfg["output"]["results_dir"] = str(out_root / "results")
        cfg["output"]["plots_dir"] = str(out_root / "results" / "plots")
        cfg["output"]["logs_dir"] = str(out_root / "results" / "logs")
        cfg["output"]["predictions_dir"] = str(out_root / "results" / "predictions")

    set_seed(cfg.get("seed", 42))

    # Resolve target device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Build DataLoaders from real dataset
    print("Loading datasets...")
    dataloaders = get_dataloaders_from_config(cfg)
    train_loader, train_ds = dataloaders["train"]
    val_loader, val_ds = dataloaders["val"]

    # Build Model
    model_name = cfg.get("model", {}).get("name", "resnet34_fcn")
    print(f"Building model ({model_name})...")
    model = build_model(cfg)
    summary = model.get_parameter_summary()
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  Encoder parameters: {summary['encoder_parameters']:,}")
    print(f"  Decoder parameters: {summary['decoder_parameters']:,}")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.resume_checkpoint(args.resume)

    # Run training (or smoke test)
    if args.smoke_test:
        print(f"\n[SMOKE TEST MODE] Running {args.smoke_test_batches} train and val batches...")
        t_start = time.time()
        results = trainer.fit(smoke_test_batches=args.smoke_test_batches)
        elapsed = time.time() - t_start
        print(f"\n✓ Smoke test finished successfully in {elapsed:.2f}s.")
        print(f"  Checkpoint saved to: {trainer.checkpoint_dir / 'latest_model.pth'}")
        print(f"  Prediction sample saved to: {trainer.predictions_dir / 'epoch_01_predictions.png'}")
    else:
        results = trainer.fit()
        print(f"\nTraining complete. Best epoch: {results['best_epoch']} "
              f"(Best {cfg.get('output', {}).get('best_metric', 'f1')}: {results['best_metric_val']:.4f})")


if __name__ == "__main__":
    main()

