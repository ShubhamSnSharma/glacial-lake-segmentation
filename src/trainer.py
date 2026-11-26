"""
trainer.py
==========
Training and validation manager for the glacial lake segmentation model.

Features:
  - Mixed device support: MPS (Apple Silicon GPU), CUDA, CPU
  - End-to-end training and validation epoch loops with MetricTracker
  - Optimizer & Scheduler management (AdamW + Cosine Annealing with Warmup)
  - Gradient clipping for stable convergence
  - Checkpoint saving (best model based on F1/IoU, and latest model)
  - Visual prediction export: RGB Image, Ground Truth, Predicted Mask, Overlay
  - CSV training history logging
  - Smoke-test mode for verifying training pipelines on small batch subsets
"""

import csv
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import build_loss
from src.metrics import MetricTracker
from src.utils import ensure_dirs


class Trainer:
    """
    Manages the complete training, validation, checkpointing, and prediction cycle.

    Args:
        model:         Segmentation model (e.g. ResNet34FCN).
        train_loader:  PyTorch DataLoader for training.
        val_loader:    PyTorch DataLoader for validation.
        cfg:           Configuration dictionary from YAML.
        device:        Target torch.device (MPS, CUDA, or CPU).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = cfg
        self.train_cfg = cfg.get("training", {})
        self.eval_cfg = cfg.get("evaluation", {})
        self.out_cfg = cfg.get("output", {})

        # Setup Device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss Function
        self.criterion = build_loss(cfg).to(self.device)

        # Optimizer
        lr = float(self.train_cfg.get("learning_rate", 1e-4))
        weight_decay = float(self.train_cfg.get("weight_decay", 1e-4))
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Total epochs
        self.epochs = int(self.train_cfg.get("epochs", 50))
        warmup_epochs = int(self.train_cfg.get("warmup_epochs", 5))

        # Learning Rate Scheduler (Linear Warmup + Cosine Annealing)
        if warmup_epochs > 0 and self.epochs > warmup_epochs:
            warmup_scheduler = LinearLR(
                self.optimizer, start_factor=0.1, total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.epochs - warmup_epochs, eta_min=1e-6
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

        # Output Directories
        self.checkpoint_dir = Path(self.out_cfg.get("checkpoint_dir", "checkpoints"))
        self.results_dir = Path(self.out_cfg.get("results_dir", "results"))
        self.predictions_dir = Path(self.out_cfg.get("predictions_dir", "results/predictions"))
        self.logs_dir = Path(self.out_cfg.get("logs_dir", "results/logs"))
        ensure_dirs(
            str(self.checkpoint_dir),
            str(self.results_dir),
            str(self.predictions_dir),
            str(self.logs_dir),
        )

        # Tracking state
        self.best_metric_name = self.out_cfg.get("best_metric", "f1")
        self.best_metric_val = -1.0
        self.best_epoch = 0
        self.start_epoch = 1
        self.history = []

        # CSV log file
        self.csv_log_path = self.logs_dir / "training_history.csv"
        self._init_csv_log()

    def _init_csv_log(self) -> None:
        """Initialize the CSV log file header if not already present."""
        if self.csv_log_path.exists():
            return
        fieldnames = [
            "epoch",
            "train_loss",
            "train_iou",
            "train_f1",
            "train_precision",
            "train_recall",
            "val_loss",
            "val_iou",
            "val_f1",
            "val_precision",
            "val_recall",
            "val_accuracy",
            "learning_rate",
            "epoch_time",
        ]
        with open(self.csv_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)

    def resume_checkpoint(self, checkpoint_path: str) -> int:
        """
        Restore model, optimizer, scheduler, metrics, and history from a checkpoint file.

        Args:
            checkpoint_path: Path to .pth checkpoint file.

        Returns:
            The next epoch number to start training from.
        """
        chk_path = Path(checkpoint_path)
        if not chk_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"\nLoading checkpoint: {chk_path} ...")
        checkpoint = torch.load(chk_path, map_location=self.device, weights_only=False)

        # Restore model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer state
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler state
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore metrics & history
        saved_epoch = int(checkpoint.get("epoch", 0))
        self.start_epoch = saved_epoch + 1
        self.best_metric_val = float(checkpoint.get("best_metric_val", -1.0))
        self.best_epoch = int(checkpoint.get("best_epoch", 0))
        self.history = list(checkpoint.get("history", []))

        # Restore random generator state if available and compatible
        if "rng_state" in checkpoint and checkpoint["rng_state"] is not None:
            try:
                torch.set_rng_state(checkpoint["rng_state"])
            except Exception:
                pass

        print(f"✓ Resumed successfully from checkpoint.")
        print(f"  Last completed epoch: {saved_epoch}")
        print(f"  Next training epoch : {self.start_epoch}")
        print(f"  Best {self.best_metric_name} so far : {self.best_metric_val:.4f} (at epoch {self.best_epoch})")
        print(f"  History length      : {len(self.history)} epochs restored\n")

        return self.start_epoch

    def train_epoch(self, epoch: int, max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Execute one training epoch.

        Args:
            epoch: Current epoch number (1-indexed).
            max_batches: If set, limits iteration (for smoke testing).

        Returns:
            Dict containing train_loss, train_iou, train_f1, train_precision, train_recall.
        """
        self.model.train()
        tracker = MetricTracker(threshold=float(self.eval_cfg.get("threshold", 0.5)))
        pbar = tqdm(
            enumerate(self.train_loader),
            total=max_batches or len(self.train_loader),
            desc=f"Epoch {epoch:02d}/{self.epochs:02d} [Train]",
            leave=False,
        )

        for batch_idx, (images, masks) in pbar:
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss ({loss.item()}) encountered at batch {batch_idx}")

            loss.backward()

            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            batch_metrics = tracker.update(logits.detach(), masks, loss=loss.item())
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                f1=f"{batch_metrics['f1']:.3f}",
                iou=f"{batch_metrics['iou']:.3f}",
            )

        metrics = tracker.compute()
        return {
            "train_loss": metrics.get("loss", 0.0),
            "train_iou": metrics.get("iou", 0.0),
            "train_f1": metrics.get("f1", 0.0),
            "train_precision": metrics.get("precision", 0.0),
            "train_recall": metrics.get("recall", 0.0),
        }

    @torch.no_grad()
    def validate_epoch(self, epoch: int, max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Execute validation pass over the validation dataset.

        Args:
            epoch: Current epoch number.
            max_batches: If set, limits iteration (for smoke testing).

        Returns:
            Dict containing val_loss, val_iou, val_f1, val_precision, val_recall, val_accuracy.
        """
        self.model.eval()
        tracker = MetricTracker(threshold=float(self.eval_cfg.get("threshold", 0.5)))
        pbar = tqdm(
            enumerate(self.val_loader),
            total=max_batches or len(self.val_loader),
            desc=f"Epoch {epoch:02d}/{self.epochs:02d} [Val]  ",
            leave=False,
        )

        for batch_idx, (images, masks) in pbar:
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            batch_metrics = tracker.update(logits, masks, loss=loss.item())
            pbar.set_postfix(
                val_loss=f"{loss.item():.4f}",
                val_f1=f"{batch_metrics['f1']:.3f}",
                val_iou=f"{batch_metrics['iou']:.3f}",
            )

        metrics = tracker.compute()
        return {
            "val_loss": metrics.get("loss", 0.0),
            "val_iou": metrics.get("iou", 0.0),
            "val_f1": metrics.get("f1", 0.0),
            "val_precision": metrics.get("precision", 0.0),
            "val_recall": metrics.get("recall", 0.0),
            "val_accuracy": metrics.get("accuracy", 0.0),
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> str:
        """
        Save model, optimizer, scheduler, and training metadata to checkpoint file atomically.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_metric_val": self.best_metric_val,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "cfg": self.cfg,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        # Save latest atomically via temporary file and rename
        latest_path = self.checkpoint_dir / "latest_model.pth"
        latest_tmp = self.checkpoint_dir / "latest_model.pth.tmp"
        torch.save(checkpoint, latest_tmp)
        os.replace(latest_tmp, latest_path)
        print(f"Saved latest checkpoint: {latest_path}")

        # Save best atomically
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            best_tmp = self.checkpoint_dir / "best_model.pth.tmp"
            torch.save(checkpoint, best_tmp)
            os.replace(best_tmp, best_path)
            print(f"Saved best checkpoint: {best_path} ({self.best_metric_name}: {self.best_metric_val:.4f})")

        return str(latest_path)

    @torch.no_grad()
    def save_prediction_samples(self, epoch: int, n_samples: int = 6) -> str:
        """
        Render and save a grid of RGB image, Ground Truth mask, Predicted mask, and Overlay.
        """
        self.model.eval()
        threshold = float(self.eval_cfg.get("threshold", 0.5))

        images_list, gt_masks_list, pred_masks_list = [], [], []

        for images, masks in self.val_loader:
            images_dev = images.to(self.device)
            logits = self.model(images_dev)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            for i in range(images.size(0)):
                if len(images_list) >= n_samples:
                    break
                images_list.append(images[i].cpu())
                gt_masks_list.append(masks[i].cpu())
                pred_masks_list.append(preds[i].cpu())

            if len(images_list) >= n_samples:
                break

        n = len(images_list)
        if n == 0:
            return ""

        fig, axes = plt.subplots(4, n, figsize=(3 * n, 10))
        if n == 1:
            axes = axes[:, np.newaxis]

        # Normalization constants for unnormalizing
        mean = np.array(self.cfg.get("data", {}).get("normalize_mean", [0.485, 0.456, 0.406]))
        std = np.array(self.cfg.get("data", {}).get("normalize_std", [0.229, 0.224, 0.225]))

        for i in range(n):
            # 1. RGB Image
            img_np = images_list[i].permute(1, 2, 0).numpy() * std + mean
            img_np = np.clip(img_np, 0.0, 1.0)
            axes[0, i].imshow(img_np)
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_ylabel("Satellite RGB", fontsize=10, fontweight="bold")

            # 2. Ground Truth Mask
            gt_np = gt_masks_list[i].squeeze().numpy()
            axes[1, i].imshow(gt_np, cmap="Blues_r", vmin=0, vmax=1)
            axes[1, i].axis("off")
            if i == 0:
                axes[1, i].set_ylabel("Ground Truth", fontsize=10, fontweight="bold")

            # 3. Predicted Mask
            pred_np = pred_masks_list[i].squeeze().numpy()
            axes[2, i].imshow(pred_np, cmap="Blues_r", vmin=0, vmax=1)
            axes[2, i].axis("off")
            if i == 0:
                axes[2, i].set_ylabel("Predicted Mask", fontsize=10, fontweight="bold")

            # 4. Overlay (Cyan for lake, Red for false positive)
            overlay = img_np.copy()
            # Mark ground truth in green/cyan
            overlay[gt_np == 1, 1] = np.clip(overlay[gt_np == 1, 1] + 0.4, 0, 1)
            # Mark prediction in red
            overlay[pred_np == 1, 0] = np.clip(overlay[pred_np == 1, 0] + 0.5, 0, 1)
            axes[3, i].imshow(overlay)
            axes[3, i].axis("off")
            if i == 0:
                axes[3, i].set_ylabel("Overlay (Red=Pred, Green=GT)", fontsize=8, fontweight="bold")

        plt.suptitle(f"Validation Predictions — Epoch {epoch:02d}", fontsize=12, fontweight="bold")
        plt.tight_layout()

        save_path = self.predictions_dir / f"epoch_{epoch:02d}_predictions.png"
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)

    def fit(self, smoke_test_batches: Optional[int] = None) -> Dict:
        """
        Execute complete training cycle across all epochs.

        Args:
            smoke_test_batches: If set, runs only this many batches per epoch (e.g. 2 for smoke test).

        Returns:
            Dict of training history.
        """
        start_epoch = 1 if smoke_test_batches is not None else self.start_epoch
        total_epochs = 1 if smoke_test_batches is not None else self.epochs

        print(f"\n=======================================================")
        print(f"  Glacial Lake Segmentation — Training Started")
        print(f"=======================================================")
        print(f"  Device           : {self.device}")
        print(f"  Training samples : {len(self.train_loader.dataset):,}")
        print(f"  Validation samples: {len(self.val_loader.dataset):,}")
        print(f"  Batch size       : {self.train_loader.batch_size}")
        print(f"  Optimizer        : AdamW (lr={self.optimizer.param_groups[0]['lr']:.1e})")
        print(f"  Loss function    : CombinedBCEDiceLoss")
        print(f"  Epoch range      : {start_epoch} -> {total_epochs} "
              f"{'(Smoke Test Mode)' if smoke_test_batches else ''}")
        print(f"=======================================================\n")

        if start_epoch > total_epochs:
            print(f"Training already completed up to epoch {start_epoch - 1} (total epochs: {total_epochs}).")
            return {
                "best_epoch": self.best_epoch,
                "best_metric_val": self.best_metric_val,
                "history": self.history,
            }

        for epoch in range(start_epoch, total_epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch, max_batches=smoke_test_batches)
            val_metrics = self.validate_epoch(epoch, max_batches=smoke_test_batches)

            # Scheduler step
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            epoch_time = time.time() - t0

            # Check if best model
            target_metric = val_metrics.get(f"val_{self.best_metric_name}", val_metrics.get("val_f1", 0.0))
            is_best = target_metric > self.best_metric_val
            if is_best:
                self.best_metric_val = target_metric
                self.best_epoch = epoch

            # Record history before saving checkpoint so latest history is persisted
            record = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "learning_rate": current_lr,
                "epoch_time": round(epoch_time, 2),
            }
            self.history.append(record)

            # Save checkpoint (contains updated self.history)
            self.save_checkpoint(epoch, is_best=is_best)

            # Save predictions occasionally or during smoke test
            save_interval = int(self.out_cfg.get("save_predictions_every", 5))
            if (epoch % save_interval == 0) or is_best or (smoke_test_batches is not None):
                self.save_prediction_samples(epoch, n_samples=4)

            # Append to CSV log
            with open(self.csv_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    record["epoch"],
                    f"{record['train_loss']:.4f}",
                    f"{record['train_iou']:.4f}",
                    f"{record['train_f1']:.4f}",
                    f"{record.get('train_precision', 0.0):.4f}",
                    f"{record.get('train_recall', 0.0):.4f}",
                    f"{record['val_loss']:.4f}",
                    f"{record['val_iou']:.4f}",
                    f"{record['val_f1']:.4f}",
                    f"{record['val_precision']:.4f}",
                    f"{record['val_recall']:.4f}",
                    f"{record['val_accuracy']:.4f}",
                    f"{record['learning_rate']:.2e}",
                    record["epoch_time"],
                ])

            print(
                f"Epoch [{epoch:02d}/{total_epochs:02d}] ({epoch_time:.1f}s) — "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val F1: {val_metrics['val_f1']:.4f} | "
                f"Val IoU: {val_metrics['val_iou']:.4f} "
                f"{'[BEST ★]' if is_best else ''}"
            )

        return {
            "best_epoch": self.best_epoch,
            "best_metric_val": self.best_metric_val,
            "history": self.history,
        }

