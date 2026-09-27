"""
resnet34_fcn.py
===============
ResNet-34 encoder with an FCN-style decoder for glacial lake segmentation.

Methodology Attribution:
------------------------
1. FROM THE PAPER (Ma et al., 2025):
   - Backbone: ResNet-34 encoder.
   - Segmentation framework: Fully Convolutional Network (FCN) paradigm.
   - Input format: 512 × 512 × 3 RGB satellite imagery.
   - Target: Binary segmentation (1 = Glacial Lake, 0 = Background / Non-lake).

2. OUR IMPLEMENTATION CHOICES / DEVIATIONS:
   - Backbone Initialization: ImageNet-pretrained weights (torchvision ResNet34_Weights.DEFAULT)
     instead of GID (Gaofen Image Dataset) pretraining, because GID-pretrained weights
     are not publicly available.
   - Decoder Architecture: A lightweight progressive transposed convolution FCN head
     (512 -> 256 -> 128 -> 64 -> 32 -> 1) with BatchNorm and ReLU activations,
     which upsamples the 1/32 feature map smoothly back to 512×512 resolution.
     The paywalled paper refers to 'ResNet34-FT' with an FCN decoder without publishing
     the exact sub-layer specifications.
   - No U-Net skip connections, attention modules, or transformer layers are added,
     keeping the architecture lightweight, minimal, and true to the FCN concept.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34


class FCNDecoder(nn.Module):
    """
    Lightweight progressive transposed-convolution decoder.

    Takes the 1/32 downsampled feature map (512 channels, 16×16 for 512×512 input)
    and progressively upsamples it 32× back to the input spatial dimensions (512×512).

    Each stage consists of:
      ConvTranspose2d (stride 2, kernel size 4, padding 1) -> BatchNorm2d -> ReLU

    Stages:
      1. 512 -> 256 (16x16 -> 32x32)
      2. 256 -> 128 (32x32 -> 64x64)
      3. 128 -> 64  (64x64 -> 128x128)
      4. 64  -> 32  (128x128 -> 256x256)
      5. 32  -> 16  (256x256 -> 512x512)
      Final: Conv2d 16 -> num_classes (1x1 conv, raw logits)
    """

    def __init__(self, in_channels: int = 512, num_classes: int = 1) -> None:
        super().__init__()

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize decoder convolution layers with Kaiming normal."""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of decoder.

        Args:
            x: Feature map of shape (B, 512, H/32, W/32)

        Returns:
            Logits of shape (B, num_classes, H, W)
        """
        x = self.up1(x)  # -> (B, 256, H/16, W/16)
        x = self.up2(x)  # -> (B, 128, H/8, W/8)
        x = self.up3(x)  # -> (B, 64, H/4, W/4)
        x = self.up4(x)  # -> (B, 32, H/2, W/2)
        x = self.up5(x)  # -> (B, 16, H, W)
        logits = self.classifier(x)  # -> (B, num_classes, H, W)
        return logits


class ResNet34FCN(nn.Module):
    """
    ResNet-34 Encoder + FCN Decoder for Glacial Lake Segmentation.

    Args:
        pretrained: If True, loads ImageNet pretrained weights for ResNet-34.
        num_classes: Number of output classes (default 1 for binary segmentation).
        freeze_encoder: If True, freezes backbone weights (for probing / head pretraining).
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        # 1. Encoder: ResNet-34 backbone
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # 64 channels, stride 1 (relative to stem)
        self.layer2 = backbone.layer2  # 128 channels, stride 2
        self.layer3 = backbone.layer3  # 256 channels, stride 2
        self.layer4 = backbone.layer4  # 512 channels, stride 2

        # 2. Decoder: FCN head
        self.decoder = FCNDecoder(in_channels=512, num_classes=num_classes)

        if freeze_encoder:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder_parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all encoder parameters for end-to-end fine-tuning."""
        for param in self.encoder_parameters():
            param.requires_grad = True

    def encoder_parameters(self):
        """Yield iterator over all encoder parameters."""
        for p in self.stem.parameters():
            yield p
        for p in self.layer1.parameters():
            yield p
        for p in self.layer2.parameters():
            yield p
        for p in self.layer3.parameters():
            yield p
        for p in self.layer4.parameters():
            yield p

    def decoder_parameters(self):
        """Yield iterator over all decoder parameters."""
        return self.decoder.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input RGB image tensor of shape (B, 3, H, W), typically (B, 3, 512, 512).

        Returns:
            Raw unnormalized logits tensor of shape (B, 1, H, W).
            Apply torch.sigmoid(logits) to obtain predicted lake probabilities in [0, 1].
        """
        # Encoder forward
        x = self.stem(x)    # (B, 64, H/4, W/4)
        x = self.layer1(x)  # (B, 64, H/4, W/4)
        x = self.layer2(x)  # (B, 128, H/8, W/8)
        x = self.layer3(x)  # (B, 256, H/16, W/16)
        x = self.layer4(x)  # (B, 512, H/32, W/32)

        # Decoder forward
        logits = self.decoder(x)  # (B, 1, H, W)
        return logits

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Inference helper: Computes binary predicted mask (0 or 1).

        Args:
            x: Input tensor (B, 3, H, W)
            threshold: Probability threshold for lake class (default 0.5)

        Returns:
            Binary mask tensor (B, 1, H, W) with values {0.0, 1.0}
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
        return preds

    def get_parameter_summary(self) -> Dict[str, int]:
        """Return counts of trainable and total parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder_parameters())
        decoder_params = sum(p.numel() for p in self.decoder_parameters())
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "encoder_parameters": encoder_params,
            "decoder_parameters": decoder_params,
        }
