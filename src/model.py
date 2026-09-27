"""
model.py
========
Model builder and factory interface for the glacial lake mapping project.

Supports:
  - Baseline model: "resnet34_fcn" (ResNet-34 + progressive FCN decoder)
  - Advanced model: "deeplabv3plus" (ResNet-50 + ASPP + boundary decoder)

Exposes:
    build_model(cfg): Instantiates model from YAML configuration dictionary.
"""

from typing import Dict, Optional
import torch.nn as nn

from src.models.resnet34_fcn import ResNet34FCN
from src.models.deeplabv3_plus import DeepLabV3Plus


def build_model(cfg: Optional[Dict] = None) -> nn.Module:
    """
    Construct model instance according to configuration dictionary.

    Args:
        cfg: Configuration dictionary (loaded from configs/default.yaml).
             If None, defaults to the ResNet-34 FCN baseline model.

    Returns:
        Instantiated nn.Module (ResNet34FCN or DeepLabV3Plus).
    """
    if cfg is None:
        cfg = {}

    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", model_cfg.get("backbone", "resnet34_fcn")).lower()
    pretrained = model_cfg.get("pretrained", True)
    num_classes = model_cfg.get("num_classes", 1)
    freeze_encoder = model_cfg.get("freeze_encoder", False)

    # 1. Baseline Model: ResNet-34 + Progressive FCN Decoder
    if model_name in ("resnet34_fcn", "resnet34", "resnet34-fcn", "baseline"):
        model = ResNet34FCN(
            pretrained=pretrained,
            num_classes=num_classes,
            freeze_encoder=freeze_encoder,
        )

    # 2. Advanced Model: DeepLabV3+ with ResNet-50 & ASPP
    elif model_name in ("deeplabv3plus", "deeplabv3+", "deeplabv3_plus", "deeplab", "advanced"):
        dilations = model_cfg.get("aspp_dilations", [6, 12, 18])
        if isinstance(dilations, list):
            dilations = tuple(dilations)
        model = DeepLabV3Plus(
            pretrained=pretrained,
            num_classes=num_classes,
            dilations=dilations,
            freeze_encoder=freeze_encoder,
        )

    else:
        raise ValueError(
            f"Unsupported model architecture: '{model_name}'. "
            f"Supported options: 'resnet34_fcn' (baseline), 'deeplabv3plus' (advanced)."
        )

    return model
