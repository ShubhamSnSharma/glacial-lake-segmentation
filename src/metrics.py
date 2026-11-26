"""
metrics.py
==========
Evaluation metrics for glacial lake binary segmentation.

Metrics implemented:
  - IoU / Jaccard Index
  - F1-Score / Dice Coefficient
  - Precision
  - Recall
  - Pixel Accuracy
  - MetricTracker (accumulates confusion matrix across batches for unbiased epoch evaluation)
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


def compute_confusion_matrix_elements(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[int, int, int, int]:
    """
    Compute true positives, false positives, false negatives, and true negatives.

    Args:
        preds:   Probabilities or logits tensor.
        targets: Ground truth binary mask ({0, 1}).
        threshold: Threshold applied if preds is continuous.

    Returns:
        (tp, fp, fn, tn) as integers.
    """
    if preds.dtype == torch.float32 or preds.dtype == torch.float64:
        binary_preds = (preds >= threshold).long()
    else:
        binary_preds = (preds > 0).long()

    binary_targets = (targets > 0).long()

    tp = int(((binary_preds == 1) & (binary_targets == 1)).sum().item())
    fp = int(((binary_preds == 1) & (binary_targets == 0)).sum().item())
    fn = int(((binary_preds == 0) & (binary_targets == 1)).sum().item())
    tn = int(((binary_preds == 0) & (binary_targets == 0)).sum().item())

    return tp, fp, fn, tn


def calculate_metrics(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Calculate summary metrics from confusion matrix elements.

    Args:
        tp: True Positives count
        fp: False Positives count
        fn: False Negatives count
        tn: True Negatives count
        smooth: Epsilon to avoid division by zero

    Returns:
        Dict containing iou, f1, precision, recall, accuracy.
    """
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    f1 = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


class MetricTracker:
    """
    Accumulates true/false positives and negatives across validation batches.
    Computes global dataset-level metrics at epoch end.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        """Reset accumulated confusion matrix counters."""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        self.total_loss = 0.0
        self.batch_count = 0

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Update tracker with a batch of predictions and ground truths.

        Args:
            preds:   Raw logits or probabilities (B, 1, H, W).
            targets: Binary ground truth (B, 1, H, W).
            loss:    Optional scalar batch loss.

        Returns:
            Dict of batch-level metrics.
        """
        # If preds are raw logits, convert to probabilities
        if preds.min() < 0.0 or preds.max() > 1.0:
            probs = torch.sigmoid(preds)
        else:
            probs = preds

        tp, fp, fn, tn = compute_confusion_matrix_elements(probs, targets, threshold=self.threshold)
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_tn += tn

        if loss is not None:
            self.total_loss += float(loss)
            self.batch_count += 1

        return calculate_metrics(tp, fp, fn, tn)

    def compute(self) -> Dict[str, float]:
        """
        Compute global epoch-level metrics over all accumulated batches.

        Returns:
            Dict with 'iou', 'f1', 'precision', 'recall', 'accuracy', and 'mean_loss' (if tracked).
        """
        metrics = calculate_metrics(
            self.total_tp,
            self.total_fp,
            self.total_fn,
            self.total_tn,
        )
        if self.batch_count > 0:
            metrics["loss"] = self.total_loss / self.batch_count
        return metrics
