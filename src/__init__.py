"""
glacial_lake_mapping
====================
Deep learning–based glacial lake detection and segmentation.

Based on:
    Ma D, Li J, Jiang L. 2025. Efficient glacial lake mapping by leveraging
    deep transfer learning and a new annotated glacial lake dataset.
    Journal of Hydrology 657: 133072.
    doi: 10.1016/j.jhydrol.2025.133072

IMPORTANT — Deviation from paper:
    The original paper uses a ResNet-34 pre-trained on the Gaofen Image Dataset
    (GID) as the encoder initialization. GID pretrained weights are not publicly
    available. This implementation uses ImageNet-pretrained ResNet-34 instead.

    Original paper:  GID-pretrained ResNet-34  →  GLID fine-tuning
    Our impl:        ImageNet-pretrained ResNet-34  →  GLID fine-tuning

    Our results are independent and should NOT be compared as direct
    reproductions of the paper's reported metrics.
"""
