"""
deeplabv3_plus.py
=================
DeepLabV3+ semantic segmentation architecture for glacial lake mapping.

Methodology Attribution:
------------------------
1. PROJECT ROLE (Advanced Comparison Model):
   - Serves as the advanced semantic segmentation model to compare against
     the paper-based ResNet-34 FCN baseline.
   - Designed to capture multi-scale spatial context and crisp glacial lake
     boundaries through Atrous Spatial Pyramid Pooling (ASPP) and low-level
     encoder feature reuse.

2. METHODOLOGY & IMPLEMENTATION CHOICES:
   - Encoder Backbone: ResNet-50 with ImageNet-pretrained weights
     (torchvision ResNet50_Weights.DEFAULT).
   - ASPP Module: Multi-scale parallel dilated convolutions with dilation
     rates {6, 12, 18}, 1x1 standard convolution, and global image pooling.
   - Low-Level Feature Fusion: 1x1 projection of stride-4 features (Layer 1)
     fused with 4x upsampled high-level ASPP features.
   - Final Classifier: 3x3 conv refinement followed by 1x1 conv predicting
     raw unnormalized logits of shape (B, 1, 512, 512).
   - Pretrained weights: ImageNet-1k (our implementation choice).
   - Output: Raw logits (compatible with CombinedBCEDiceLoss).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ASPPConv(nn.Sequential):
    """Atrous (dilated) convolution branch for ASPP."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    """Global image context pooling branch for ASPP."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        out = super().forward(x)
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.

    Captures multi-scale contextual features using parallel convolutions
    with varying dilation rates.

    Args:
        in_channels: Number of input channels (e.g. 2048 from ResNet-50 layer4).
        out_channels: Intermediate channel dimension for each branch (default 256).
        dilations: Tuple of dilation rates for the 3x3 atrous convolutions.
    """

    def __init__(
        self,
        in_channels: int = 2048,
        out_channels: int = 256,
        dilations: Tuple[int, int, int] = (6, 12, 18),
    ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = ASPPConv(in_channels, out_channels, dilation=dilations[0])
        self.conv3 = ASPPConv(in_channels, out_channels, dilation=dilations[1])
        self.conv4 = ASPPConv(in_channels, out_channels, dilation=dilations[2])
        self.pooling = ASPPPooling(in_channels, out_channels)

        # 5 branches combined: 5 * out_channels -> out_channels
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.pooling(x)

        combined = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(combined)


class DeepLabV3PlusDecoder(nn.Module):
    """
    DeepLabV3+ Decoder module.

    Fuses high-level ASPP representations with 1x1-projected low-level
    backbone features to restore sharp spatial boundaries.

    Args:
        low_level_channels: Channel count of low-level features (e.g. 256 from layer1).
        aspp_channels: Channel count from ASPP module (default 256).
        low_level_proj_channels: Channels after low-level 1x1 projection (default 48).
        num_classes: Output channels (default 1 for binary segmentation).
    """

    def __init__(
        self,
        low_level_channels: int = 256,
        aspp_channels: int = 256,
        low_level_proj_channels: int = 48,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        # Low-level feature projection (1x1 conv)
        self.project_low_level = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_proj_channels),
            nn.ReLU(inplace=True),
        )

        # Refinement convolution after feature concatenation
        in_concat = aspp_channels + low_level_proj_channels  # 256 + 48 = 304
        self.refine = nn.Sequential(
            nn.Conv2d(in_concat, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Final pixel classifier
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        aspp_features: torch.Tensor,
        low_level_features: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            aspp_features: Output from ASPP of shape (B, 256, H/16 or H/32, W/16 or W/32).
            low_level_features: Low-level features from encoder layer1 of shape (B, 256, H/4, W/4).
            target_size: Original image dimensions (H, W), typically (512, 512).

        Returns:
            Logits of shape (B, num_classes, H, W).
        """
        # 1. Project low-level features
        low_proj = self.project_low_level(low_level_features)

        # 2. Upsample ASPP features to low-level spatial resolution (stride 4)
        aspp_up = F.interpolate(
            aspp_features,
            size=low_proj.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # 3. Concatenate and refine
        fused = torch.cat([aspp_up, low_proj], dim=1)  # (B, 304, H/4, W/4)
        refined = self.refine(fused)                   # (B, 256, H/4, W/4)

        # 4. Final classification
        logits_low = self.classifier(refined)          # (B, num_classes, H/4, W/4)

        # 5. Upsample to original input size (512x512)
        logits = F.interpolate(
            logits_low,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return logits


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ Semantic Segmentation Network.

    Encoder: ResNet-50 (with dilated convolutions in layer4 for output stride = 16)
    Context Module: ASPP with dilation rates {6, 12, 18}
    Decoder: Feature fusion of ASPP and low-level stride-4 features

    Args:
        pretrained: If True, loads ImageNet pretrained ResNet-50 weights.
        num_classes: Output channel count (default 1 for binary glacial lake mapping).
        dilations: Dilations for ASPP branches (default: (6, 12, 18)).
        freeze_encoder: If True, freezes encoder parameters.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 1,
        dilations: Tuple[int, int, int] = (6, 12, 18),
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        # 1. Backbone: ResNet-50 with replace_stride_with_dilation for output stride = 16
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(
            weights=weights,
            replace_stride_with_dilation=[False, False, True],  # layer4 dilation=2, stride=1 -> OS=16
        )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # 256 channels, stride 4 (Low-level features)
        self.layer2 = backbone.layer2  # 512 channels, stride 8
        self.layer3 = backbone.layer3  # 1024 channels, stride 16
        self.layer4 = backbone.layer4  # 2048 channels, stride 16 (dilated)

        # 2. ASPP Module
        self.aspp = ASPP(
            in_channels=2048,
            out_channels=256,
            dilations=dilations,
        )

        # 3. Decoder Head
        self.decoder = DeepLabV3PlusDecoder(
            low_level_channels=256,
            aspp_channels=256,
            low_level_proj_channels=48,
            num_classes=num_classes,
        )

        if freeze_encoder:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all encoder backbone parameters."""
        for param in self.encoder_parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all encoder backbone parameters for end-to-end training."""
        for param in self.encoder_parameters():
            param.requires_grad = True

    def encoder_parameters(self):
        """Yield an iterator over all backbone parameters."""
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
        """Yield an iterator over ASPP and decoder head parameters."""
        for p in self.aspp.parameters():
            yield p
        for p in self.decoder.parameters():
            yield p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input RGB tensor of shape (B, 3, H, W), typically (B, 3, 512, 512).

        Returns:
            Raw logits tensor of shape (B, 1, H, W).
        """
        target_size = (x.shape[2], x.shape[3])

        # Encoder forward
        x = self.stem(x)                  # (B, 64, H/4, W/4)
        low_level = self.layer1(x)        # (B, 256, H/4, W/4) - Low-level feature
        x = self.layer2(low_level)        # (B, 512, H/8, W/8)
        x = self.layer3(x)                # (B, 1024, H/16, W/16)
        x = self.layer4(x)                # (B, 2048, H/16, W/16) - High-level feature

        # ASPP forward
        aspp_out = self.aspp(x)           # (B, 256, H/16, W/16)

        # Decoder fusion & classification
        logits = self.decoder(aspp_out, low_level, target_size=target_size)
        return logits

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Inference helper: Computes binary predicted mask (0 or 1).

        Args:
            x: Input tensor (B, 3, H, W)
            threshold: Probability threshold for lake class (default 0.5)

        Returns:
            Binary mask tensor (B, 1, H, W) with values in {0.0, 1.0}
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
        aspp_params = sum(p.numel() for p in self.aspp.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "encoder_parameters": encoder_params,
            "aspp_parameters": aspp_params,
            "decoder_parameters": decoder_params,
            "head_parameters": aspp_params + decoder_params,
        }
