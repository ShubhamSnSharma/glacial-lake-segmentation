"""
attn_resunet.py
===============
Custom Attention-Gated ResNet-34 U-Net (Attn-ResUNet34 / Attn-GLNet) for
glacial lake detection and segmentation.

Architectural Overview:
-----------------------
1. Encoder:
   - ResNet-34 backbone initialized with torchvision ImageNet weights.
   - Extracts 5 feature levels:
       * e0 (stem pre-maxpool): (B,  64, 256, 256)
       * e1 (stem post-maxpool + layer1): (B,  64, 128, 128)
       * e2 (layer2):                      (B, 128,  64,  64)
       * e3 (layer3):                      (B, 256,  32,  32)
       * e4 (layer4 / bottleneck):         (B, 512,  16,  16)

2. Multi-Scale Attention Gates (Oktay et al., 2018):
   - 4 additive attention gates gating e3, e2, e1, and e0 using gating signals
     from corresponding upsampled decoder stages.
   - Suppresses background terrain noise (e.g. steep mountain shadows) while
     preserving boundary delineation of small/turbid glacial lakes.
   - Output projection bias initialized to +2.0 to ensure gates start open
     (alpha ~ 0.88), preventing early gradient starvation.

3. Symmetrical 5-Stage Decoder:
   - Stage 1: Upsamples e4 (512, 16x16) -> (256, 32x32), gates e3 -> (256, 32x32)
   - Stage 2: Upsamples d1 (256, 32x32) -> (128, 64x64), gates e2 -> (128, 64x64)
   - Stage 3: Upsamples d2 (128, 64x64) -> ( 64, 128x128), gates e1 -> ( 64, 128x128)
   - Stage 4: Upsamples d3 ( 64, 128x128) -> ( 32, 256x256), gates e0 -> ( 32, 256x256)
   - Stage 5: Upsamples d4 ( 32, 256x256) -> ( 16, 512x512)
   - Head: 1x1 conv projecting (16, 512x512) to (num_classes, 512x512) raw logits.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


class AttentionGate(nn.Module):
    """
    Additive Attention Gate module.

    Filters skip connection features (x) using coarse semantic gating signals (g)
    from deeper decoder layers. Suppresses uninformative background features while
    focusing on lake boundaries.

    Formula:
        phi = ReLU(BN(W_x * x) + BN(W_g * g))
        alpha = Sigmoid(psi * phi + b_psi)
        gated_x = x * alpha

    Args:
        in_channels_x: Number of channels in the skip connection tensor x.
        in_channels_g: Number of channels in the gating signal tensor g.
        inter_channels: Intermediate projection channels (default: in_channels_x // 2).
    """

    def __init__(
        self,
        in_channels_x: int,
        in_channels_g: int,
        inter_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if inter_channels is None:
            inter_channels = max(16, in_channels_x // 2)

        self.wx = nn.Sequential(
            nn.Conv2d(in_channels_x, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.wg = nn.Sequential(
            nn.Conv2d(in_channels_g, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        # Note: No BatchNorm before sigmoid so that b_psi directly controls the initial gate state
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights and set bias to +2.0 (open gate init)."""
        for seq in (self.wx, self.wg):
            conv = seq[0]
            bn = seq[1]
            if isinstance(conv, nn.Conv2d):
                nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(bn, nn.BatchNorm2d):
                nn.init.ones_(bn.weight)
                nn.init.zeros_(bn.bias)

        # Initialize psi with small weights and positive bias (+2.0).
        # This makes sigmoid(bias) ~ 0.88 at step 0, ensuring skip gradients flow
        # without being zero-suppressed during early training epochs.
        nn.init.normal_(self.psi.weight, mean=0.0, std=0.01)
        if self.psi.bias is not None:
            nn.init.constant_(self.psi.bias, 2.0)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the attention gate.

        Args:
            x: Skip feature tensor from encoder of shape (B, C_x, H_x, W_x).
            g: Gating feature tensor from decoder of shape (B, C_g, H_g, W_g).

        Returns:
            gated_x: Attention-weighted skip tensor of shape (B, C_x, H_x, W_x).
            alpha: Attention weight map in [0, 1] of shape (B, 1, H_x, W_x).
        """
        # Explicit spatial alignment: interpolate g if resolution does not match x
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        theta_x = self.wx(x)
        phi_g = self.wg(g)

        f = self.relu(theta_x + phi_g)
        psi = self.psi(f)
        alpha = self.sigmoid(psi)

        gated_x = x * alpha
        return gated_x, alpha


class DoubleConv(nn.Module):
    """
    Standard double convolution block:
    [Conv2d (3x3) -> BatchNorm -> ReLU] x 2
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
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
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling, optional attention gating of skip connections,
    and double convolution refinement.

    Args:
        in_channels: Channels entering from the deeper decoder stage.
        skip_channels: Channels from the corresponding encoder skip connection.
        out_channels: Channels output by this decoder stage.
        use_attention: Whether to apply AttentionGate to skip features (True for Attn-ResUNet,
                       False for standard ResUNet ablation).
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention

        # Upsample projection: Bilinear upsample + 3x3 conv to reduce channels smoothly
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

        if self.use_attention and skip_channels > 0:
            self.gate = AttentionGate(
                in_channels_x=skip_channels,
                in_channels_g=out_channels,
                inter_channels=max(16, skip_channels // 2),
            )
        else:
            self.gate = None

        concat_channels = out_channels + skip_channels if skip_channels > 0 else out_channels
        self.conv = DoubleConv(concat_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Feature tensor from previous deeper decoder stage.
            skip: Optional skip connection feature tensor from encoder.

        Returns:
            out: Refined decoder tensor.
            alpha: Attention map tensor (if gate is active), else None.
        """
        # Target spatial size is either the skip size or 2x of current feature map
        if skip is not None:
            target_size = skip.shape[2:]
        else:
            target_size = (x.shape[2] * 2, x.shape[3] * 2)

        x_up = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        g = self.up_conv(x_up)

        alpha = None
        if skip is not None:
            if self.gate is not None:
                gated_skip, alpha = self.gate(skip, g)
            else:
                gated_skip = skip
            out = torch.cat([g, gated_skip], dim=1)
        else:
            out = g

        out = self.conv(out)
        return out, alpha


class AttnResUNet34(nn.Module):
    """
    Attention-Gated ResNet-34 U-Net (Attn-ResUNet34 / Attn-GLNet).

    Combines an ImageNet-pretrained ResNet-34 encoder with a 5-stage attention-gated
    U-Net decoder for binary glacial lake segmentation.

    Args:
        pretrained: If True, initializes encoder with ResNet34_Weights.DEFAULT.
        num_classes: Output channel count (default 1 for binary lake mask logits).
        freeze_encoder: If True, freezes encoder parameters.
        use_attention: If True, activates Attention Gates on all skip connections.
                       If False, performs standard U-Net concatenation (clean ablation).
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 1,
        freeze_encoder: bool = False,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_attention = use_attention

        # -------------------------------------------------------------------
        # 1. Encoder: ResNet-34 Backbone
        # -------------------------------------------------------------------
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # 64 channels,  stride 4
        self.layer2 = backbone.layer2  # 128 channels, stride 8
        self.layer3 = backbone.layer3  # 256 channels, stride 16
        self.layer4 = backbone.layer4  # 512 channels, stride 32 (bottleneck)

        # -------------------------------------------------------------------
        # 2. Decoder: 5-Stage Attention-Gated Decoder
        # -------------------------------------------------------------------
        # Stage 1: 512 (16x16) -> 256 (32x32) + skip layer3 (256, 32x32)
        self.dec1 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
            use_attention=use_attention,
        )

        # Stage 2: 256 (32x32) -> 128 (64x64) + skip layer2 (128, 64x64)
        self.dec2 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
            use_attention=use_attention,
        )

        # Stage 3: 128 (64x64) -> 64 (128x128) + skip layer1 (64, 128x128)
        self.dec3 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
            use_attention=use_attention,
        )

        # Stage 4: 64 (128x128) -> 32 (256x256) + skip stem pre-pool (64, 256x256)
        self.dec4 = DecoderBlock(
            in_channels=64,
            skip_channels=64,
            out_channels=32,
            use_attention=use_attention,
        )

        # Stage 5: 32 (256x256) -> 16 (512x512) (no encoder skip, progressive restoration)
        self.dec5 = DecoderBlock(
            in_channels=32,
            skip_channels=0,
            out_channels=16,
            use_attention=False,
        )

        # Final 1x1 classifier producing raw logits
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_out", nonlinearity="relu")
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        if freeze_encoder:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder_parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all encoder parameters for fine-tuning."""
        for param in self.encoder_parameters():
            param.requires_grad = True

    def encoder_parameters(self):
        """Yield iterator over all encoder parameters."""
        for p in self.conv1.parameters():
            yield p
        for p in self.bn1.parameters():
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
        """Yield iterator over all decoder and classifier parameters."""
        for m in (self.dec1, self.dec2, self.dec3, self.dec4, self.dec5, self.classifier):
            for p in m.parameters():
                yield p

    def forward(
        self,
        x: torch.Tensor,
        return_attention_maps: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Input RGB image tensor of shape (B, 3, H, W), typically (B, 3, 512, 512).
            return_attention_maps: If True, returns a tuple of (logits, attention_dict).

        Returns:
            logits: Raw unnormalized logits of shape (B, num_classes, H, W).
            attention_maps (optional): Dict of attention maps from gates {"ag1", "ag2", "ag3", "ag4"}.
        """
        # 1. Encoder forward
        e0 = self.relu(self.bn1(self.conv1(x)))  # (B,  64, H/2,  W/2)  - 256x256 for 512x512
        e1 = self.maxpool(e0)                    # (B,  64, H/4,  W/4)  - 128x128
        e1 = self.layer1(e1)                     # (B,  64, H/4,  W/4)  - 128x128
        e2 = self.layer2(e1)                     # (B, 128, H/8,  W/8)  - 64x64
        e3 = self.layer3(e2)                     # (B, 256, H/16, W/16) - 32x32
        e4 = self.layer4(e3)                     # (B, 512, H/32, W/32) - 16x16

        # 2. Decoder forward with multi-scale attention gating
        d1, a4 = self.dec1(e4, skip=e3)          # -> (B, 256, H/16, W/16)
        d2, a3 = self.dec2(d1, skip=e2)          # -> (B, 128, H/8,  W/8)
        d3, a2 = self.dec3(d2, skip=e1)          # -> (B,  64, H/4,  W/4)
        d4, a1 = self.dec4(d3, skip=e0)          # -> (B,  32, H/2,  W/2)
        d5, _  = self.dec5(d4, skip=None)        # -> (B,  16, H,    W)

        logits = self.classifier(d5)             # -> (B, num_classes, H, W)

        if return_attention_maps:
            attn_dict = {"ag4": a4, "ag3": a3, "ag2": a2, "ag1": a1}
            return logits, attn_dict

        return logits

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Inference helper: Computes binary predicted mask (0.0 or 1.0).

        Args:
            x: Input tensor (B, 3, H, W).
            threshold: Sigmoid probability cutoff (default 0.5).

        Returns:
            Binary mask tensor (B, 1, H, W) with values {0.0, 1.0}.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
        return preds

    def get_parameter_summary(self) -> Dict[str, int]:
        """Return counts of trainable, encoder, decoder, and total parameters."""
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
