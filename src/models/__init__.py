"""
models package
==============
Segmentation model architectures for glacial lake mapping:
  - ResNet34FCN: Baseline model (ResNet-34 encoder + progressive FCN decoder)
  - DeepLabV3Plus: Advanced model (ResNet-50 encoder + ASPP + boundary decoder)
"""

from src.models.resnet34_fcn import ResNet34FCN, FCNDecoder
from src.models.deeplabv3_plus import DeepLabV3Plus, ASPP, DeepLabV3PlusDecoder

__all__ = [
    "ResNet34FCN",
    "FCNDecoder",
    "DeepLabV3Plus",
    "ASPP",
    "DeepLabV3PlusDecoder",
]
