"""
losses.py
=========
Loss functions for binary glacial lake segmentation with extreme class imbalance.

Supports:
  - DiceLoss (soft differentiable Dice loss)
  - BinaryFocalLoss (focal loss for hard boundary mining)
  - CombinedBCEDiceLoss (standard combined loss recommended for remote sensing segmentation)
  - build_loss (config-driven loss factory)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Binary Dice Loss computed directly from unnormalized logits.

    Formula:
        Dice = (2 * sum(p * y) + smooth) / (sum(p) + sum(y) + smooth)
        Loss = 1.0 - Dice

    Args:
        smooth: Smoothing epsilon to prevent division by zero (default 1e-6).
        reduction: 'mean' across batch or 'sum'.
    """

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  Raw logits tensor of shape (B, 1, H, W) or (B, H, W).
            targets: Binary ground truth mask (same shape, values in {0, 1}).

        Returns:
            Scalar Dice loss tensor.
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions per batch item: (B, -1)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1).float()

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for focusing training on hard lake boundaries.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for the rare positive class (lake). Default 0.75.
        gamma: Focusing parameter for modulating loss from easy negatives. Default 2.0.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedBCEDiceLoss(nn.Module):
    """
    Composite loss combining Binary Cross-Entropy and Soft Dice Loss.

    Formula:
        Loss = bce_weight * BCEWithLogits + dice_weight * DiceLoss

    Args:
        bce_weight: Weight assigned to BCE loss (default 0.5).
        dice_weight: Weight assigned to Dice loss (default 0.5).
        pos_weight: Optional positive class weight for BCE to handle extreme imbalance.
        smooth: Epsilon for Dice loss.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        pos_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_tensor)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets.float())
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def build_loss(cfg: Optional[Dict] = None) -> nn.Module:
    """
    Factory function to instantiate loss function from configuration dict.

    Args:
        cfg: Configuration dictionary (training section).

    Returns:
        Instantiated nn.Module loss criterion.
    """
    if cfg is None:
        cfg = {}

    train_cfg = cfg.get("training", {})
    loss_name = train_cfg.get("loss", "bce_dice").lower()

    if loss_name in ("bce_dice", "combined", "bce_and_dice"):
        bce_w = train_cfg.get("bce_weight", 0.5)
        dice_w = train_cfg.get("dice_weight", 0.5)
        return CombinedBCEDiceLoss(bce_weight=bce_w, dice_weight=dice_w)
    elif loss_name == "dice":
        return DiceLoss()
    elif loss_name == "focal":
        alpha = train_cfg.get("focal_alpha", 0.75)
        gamma = train_cfg.get("focal_gamma", 2.0)
        return BinaryFocalLoss(alpha=alpha, gamma=gamma)
    elif loss_name in ("bce", "bce_with_logits"):
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss type: '{loss_name}'. Available: 'bce_dice', 'dice', 'focal', 'bce'")
