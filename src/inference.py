"""
inference.py
============
Centralized, standalone inference and prediction engine for glacial lake segmentation.

Provides:
  - Unified model loading with architecture auto-detection (ResNet34FCN and DeepLabV3Plus).
  - In-memory image preprocessing (PIL Image, NumPy array, or file path).
  - Single-image inference returning probability maps and binary masks.
  - Ground-truth-optional visual overlays for standalone uploaded imagery.
  - Quantitative prediction summary statistics.

Designed for reuse across:
  - CLI scripts (scripts/predict.py)
  - Interactive web applications (Streamlit)
  - Batch / API inference workflows
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from src.models.deeplabv3_plus import DeepLabV3Plus
from src.models.resnet34_fcn import ResNet34FCN
from src.models.attn_resunet import AttnResUNet34


# ---------------------------------------------------------------------------
# Constants — Standard ImageNet statistics used across the project
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = (512, 512)
DEFAULT_THRESHOLD = 0.5

# Canonical architecture aliases
_DEEPLABV3PLUS_ALIASES = frozenset({
    "deeplabv3plus", "deeplabv3+", "deeplabv3_plus", "deeplab", "advanced",
})
_RESNET34FCN_ALIASES = frozenset({
    "resnet34_fcn", "resnet34-fcn", "resnet34", "baseline",
})
_ATTN_RESUNET_ALIASES = frozenset({
    "attn_resunet34", "attn_resunet", "att_resunet", "attn_glnet", "custom_attn",
})


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ModelMetadata:
    """Metadata describing a loaded model checkpoint."""
    model_name: str
    arch_type: str
    checkpoint_path: str
    epoch: Union[int, str]
    best_f1: Union[float, str]
    total_parameters: int
    trainable_parameters: int
    device: str
    default_threshold: float = DEFAULT_THRESHOLD


@dataclass
class PredictionResult:
    """Inference output bundle for a single image."""
    probability_map: np.ndarray      # (512, 512) float32 in [0.0, 1.0]
    binary_mask: np.ndarray          # (512, 512) uint8 in {0, 1}
    processed_image: np.ndarray      # (512, 512, 3) uint8 RGB
    overlay_image: np.ndarray        # (512, 512, 3) uint8 RGB
    statistics: Dict[str, Any]       # Lake pixel count, percentage, confidence
    threshold: float                 # Applied threshold


# ---------------------------------------------------------------------------
# Device Selection
# ---------------------------------------------------------------------------

def select_device(requested: Optional[str] = None) -> torch.device:
    """
    Select target device: CUDA if available, elif MPS if available, otherwise CPU.

    Args:
        requested: Optional device string ('cuda', 'mps', 'cpu').

    Returns:
        torch.device instance.
    """
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Unified Model Loading
# ---------------------------------------------------------------------------

def resolve_architecture_name(
    checkpoint: Dict[str, Any],
    model_override: Optional[str] = None,
) -> str:
    """
    Determine the model architecture from checkpoint metadata or override.

    Resolution order:
      1. model_override argument (if provided)
      2. checkpoint['model_name']
      3. checkpoint['cfg']['model']['name']
      4. Fallback to 'deeplabv3plus'
    """
    if model_override is not None:
        return model_override.lower()
    return checkpoint.get(
        "model_name",
        checkpoint.get("cfg", {}).get("model", {}).get("name", "deeplabv3plus"),
    ).lower()


def load_model(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    model_override: Optional[str] = None,
) -> Tuple[nn.Module, ModelMetadata]:
    """
    Build the architecture and safely restore trained weights from checkpoint.

    Auto-detects whether the checkpoint belongs to ResNet34FCN or DeepLabV3Plus.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Target execution device (auto-selected if None).
        model_override: Optional architecture override ('resnet34_fcn' or 'deeplabv3plus').

    Returns:
        (model, metadata) tuple with model in eval() mode on the specified device.
    """
    chk_path = Path(checkpoint_path)
    if not chk_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: '{chk_path}'")

    if device is None:
        target_device = select_device()
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    # Safe CPU load first
    checkpoint = torch.load(chk_path, map_location="cpu", weights_only=False)

    resolved_name = resolve_architecture_name(checkpoint, model_override)

    # Instantiate architecture
    if resolved_name in _RESNET34FCN_ALIASES:
        model = ResNet34FCN(pretrained=False, num_classes=1)
        canonical_arch = "ResNet34-FCN"
    elif resolved_name in _DEEPLABV3PLUS_ALIASES:
        model = DeepLabV3Plus(pretrained=False, num_classes=1)
        canonical_arch = "DeepLabV3+"
    elif resolved_name in _ATTN_RESUNET_ALIASES:
        model = AttnResUNet34(pretrained=False, num_classes=1)
        canonical_arch = "Attn-ResUNet34"
    else:
        raise ValueError(
            f"Unknown architecture '{resolved_name}'. "
            f"Supported: {sorted(_DEEPLABV3PLUS_ALIASES | _RESNET34FCN_ALIASES | _ATTN_RESUNET_ALIASES)}"
        )

    # Restore weights
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(target_device)
    model.eval()

    # Extract metadata
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    epoch = checkpoint.get("epoch", "unknown")
    best_f1 = checkpoint.get("best_f1", checkpoint.get("best_metric_val", "unknown"))

    metadata = ModelMetadata(
        model_name=resolved_name,
        arch_type=canonical_arch,
        checkpoint_path=str(chk_path),
        epoch=epoch,
        best_f1=best_f1,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        device=str(target_device),
        default_threshold=DEFAULT_THRESHOLD,
    )

    return model, metadata


# ---------------------------------------------------------------------------
# Image Preprocessing & Postprocessing
# ---------------------------------------------------------------------------

def preprocess_image(
    image_input: Union[Image.Image, np.ndarray, str, Path],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess an arbitrary input image into a standardized tensor for model inference.

    Args:
        image_input: PIL Image, NumPy array (HWC uint8 or float), or filesystem path.
        target_size: Target (width, height) spatial dimensions (default: (512, 512)).

    Returns:
        (tensor, rgb_numpy) where:
          - tensor: FloatTensor of shape (1, 3, H, W) normalized by ImageNet mean/std.
          - rgb_numpy: Original resized image as uint8 NumPy array of shape (H, W, 3).
    """
    # 1. Convert input to PIL Image in RGB mode
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found at '{img_path}'")
        pil_img = Image.open(img_path).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            # Grayscale (H, W) -> (H, W, 3)
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif image_input.ndim == 3:
            # HWC
            if image_input.shape[2] == 1:
                pil_img = Image.fromarray(image_input.squeeze(2)).convert("RGB")
            elif image_input.shape[2] == 4:
                # RGBA -> RGB
                pil_img = Image.fromarray(image_input).convert("RGB")
            elif image_input.shape[2] == 3:
                if image_input.dtype == np.uint8:
                    pil_img = Image.fromarray(image_input).convert("RGB")
                elif np.issubdtype(image_input.dtype, np.floating):
                    # Scaled [0.0, 1.0] -> [0, 255]
                    arr = np.clip(image_input * 255.0, 0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(arr).convert("RGB")
                else:
                    pil_img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
            else:
                raise ValueError(f"Unsupported channel dimension: {image_input.shape[2]}")
        else:
            raise ValueError(f"Expected 2D or 3D NumPy array, got shape {image_input.shape}")
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    # 2. Resize if necessary
    if pil_img.size != target_size:
        pil_img = pil_img.resize(target_size, Image.BILINEAR)

    # 3. Base RGB numpy array (512, 512, 3)
    rgb_numpy = np.array(pil_img, dtype=np.uint8)

    # 4. Standard PyTorch transforms (ToTensor + ImageNet Normalize)
    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    tensor = normalize(to_tensor(pil_img)).unsqueeze(0)  # -> (1, 3, 512, 512)
    return tensor, rgb_numpy


def unnormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization on a (3, H, W) or (1, 3, H, W) tensor.

    Returns:
        (H, W, 3) uint8 NumPy array in RGB.
    """
    t = tensor.clone().detach().cpu().float()
    if t.ndim == 4:
        t = t.squeeze(0)
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        t[c] = t[c] * s + m
    t = t.permute(1, 2, 0).numpy()
    return np.clip(t * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Statistics & Visual Overlays
# ---------------------------------------------------------------------------

def compute_prediction_stats(
    binary_mask: np.ndarray,
    probability_map: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute quantitative summary statistics for a predicted glacial lake mask.

    Args:
        binary_mask: (H, W) uint8 or boolean binary mask ({0, 1}).
        probability_map: Optional (H, W) float32 probability array in [0.0, 1.0].

    Returns:
        Dictionary containing lake pixel count, lake percentage, and confidence metrics.
    """
    mask_bool = binary_mask.astype(bool)
    total_pixels = int(binary_mask.size)
    lake_pixels = int(np.count_nonzero(mask_bool))
    lake_percentage = float((lake_pixels / total_pixels) * 100.0) if total_pixels > 0 else 0.0

    stats = {
        "lake_pixels": lake_pixels,
        "total_pixels": total_pixels,
        "lake_percentage": round(lake_percentage, 4),
        "has_lake_detected": bool(lake_pixels > 0),
    }

    if probability_map is not None:
        if lake_pixels > 0:
            lake_probs = probability_map[mask_bool]
            stats["mean_lake_confidence"] = float(np.mean(lake_probs))
            stats["max_lake_confidence"] = float(np.max(lake_probs))
            stats["min_lake_confidence"] = float(np.min(lake_probs))
        else:
            stats["mean_lake_confidence"] = 0.0
            stats["max_lake_confidence"] = float(np.max(probability_map))
            stats["min_lake_confidence"] = float(np.min(probability_map))
        stats["mean_scene_confidence"] = float(np.mean(probability_map))

    return stats


def create_prediction_overlay(
    image_np: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    alpha: float = 0.55,
    lake_color: Tuple[int, int, int] = (0, 229, 255),  # Cyan (#00E5FF)
) -> np.ndarray:
    """
    Create a high-contrast visual overlay of the predicted lake mask on the satellite image.

    Modes:
      1. Standalone / Unannotated Image (gt_mask is None):
         Highlights all predicted lake pixels with cyan (#00E5FF).
      2. Supervised Benchmark (gt_mask is provided):
         Uses standard 3-way color coding:
           - Cyan  (#00E5FF) : True Positive  (lake correctly detected)
           - Red   (#FF1744) : False Positive (background wrongly detected as lake)
           - White (#FFFFFF) : False Negative (lake missed by model)

    Args:
        image_np: (H, W, 3) uint8 RGB image array.
        pred_mask: (H, W) binary prediction mask ({0, 1} or bool).
        gt_mask: Optional (H, W) ground-truth binary mask.
        alpha: Transparency blending factor (0.0 to 1.0, default 0.55).
        lake_color: RGB tuple for positive prediction highlight (default cyan).

    Returns:
        (H, W, 3) uint8 RGB array suitable for PIL/Streamlit visualization.
    """
    overlay = image_np.copy().astype(np.float32)
    pred = pred_mask.astype(bool)

    if gt_mask is None:
        # Standalone prediction mode
        color_arr = np.array(lake_color, dtype=np.float32)
        overlay[pred] = (1.0 - alpha) * overlay[pred] + alpha * color_arr
    else:
        # Supervised comparative mode (TP / FP / FN)
        gt = gt_mask.astype(bool)
        tp = gt & pred
        fp = ~gt & pred
        fn = gt & ~pred

        cyan  = np.array([0.0,   229.0, 255.0], dtype=np.float32)
        red   = np.array([255.0, 23.0,  68.0],  dtype=np.float32)
        white = np.array([255.0, 255.0, 255.0], dtype=np.float32)

        overlay[tp] = (1.0 - alpha) * overlay[tp] + alpha * cyan
        overlay[fp] = (1.0 - alpha) * overlay[fp] + alpha * red
        overlay[fn] = (1.0 - alpha) * overlay[fn] + alpha * white

    return np.clip(overlay, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Primary Inference API
# ---------------------------------------------------------------------------

def predict_single_image(
    model: nn.Module,
    image_input: Union[Image.Image, np.ndarray, str, Path],
    device: Optional[Union[str, torch.device]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    gt_mask: Optional[Union[Image.Image, np.ndarray]] = None,
) -> PredictionResult:
    """
    Execute end-to-end inference on a single satellite image.

    Args:
        model: Loaded PyTorch segmentation model (ResNet34FCN or DeepLabV3Plus).
        image_input: PIL Image, NumPy array, or path to image file.
        device: Device to run inference on (inferred from model if None).
        threshold: Sigmoid probability threshold for lake classification (default 0.5).
        gt_mask: Optional ground truth mask for comparative overlay generation.

    Returns:
        PredictionResult dataclass with probability map, binary mask, overlay, and stats.
    """
    if device is None:
        # Infer device from model parameters
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = select_device()
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    model.eval()

    # 1. Preprocess
    tensor, rgb_numpy = preprocess_image(image_input, target_size=DEFAULT_IMAGE_SIZE)
    tensor = tensor.to(target_device)

    # 2. Inference
    with torch.inference_mode():
        logits = model(tensor)                         # -> (1, 1, 512, 512)
        probs = torch.sigmoid(logits)                  # -> (1, 1, 512, 512)
        preds = (probs >= threshold).to(torch.uint8)   # -> (1, 1, 512, 512)

    # 3. Format numpy outputs
    prob_map = probs.squeeze().cpu().numpy().astype(np.float32)  # (512, 512)
    binary_mask = preds.squeeze().cpu().numpy().astype(np.uint8)  # (512, 512)

    # 4. Optional GT mask preprocessing
    gt_np = None
    if gt_mask is not None:
        if isinstance(gt_mask, Image.Image):
            if gt_mask.size != DEFAULT_IMAGE_SIZE:
                gt_mask = gt_mask.resize(DEFAULT_IMAGE_SIZE, Image.NEAREST)
            gt_np = (np.array(gt_mask) > 127).astype(np.uint8)
        elif isinstance(gt_mask, np.ndarray):
            gt_np = (gt_mask > 0.5).astype(np.uint8)

    # 5. Visual overlay & statistics
    overlay = create_prediction_overlay(
        image_np=rgb_numpy,
        pred_mask=binary_mask,
        gt_mask=gt_np,
    )
    stats = compute_prediction_stats(
        binary_mask=binary_mask,
        probability_map=prob_map,
    )

    return PredictionResult(
        probability_map=prob_map,
        binary_mask=binary_mask,
        processed_image=rgb_numpy,
        overlay_image=overlay,
        statistics=stats,
        threshold=threshold,
    )


class GlacialLakePredictor:
    """
    Reusable stateful predictor wrapper for glacial lake segmentation.

    Maintains a loaded model in memory across multiple inference requests,
    ideal for interactive applications (e.g. Streamlit, REST API, or GUI).
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        model_override: Optional[str] = None,
        default_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.device = select_device(device) if isinstance(device, str) or device is None else device
        self.default_threshold = default_threshold
        self.model, self.metadata = load_model(
            checkpoint_path=checkpoint_path,
            device=self.device,
            model_override=model_override,
        )

    def predict(
        self,
        image_input: Union[Image.Image, np.ndarray, str, Path],
        threshold: Optional[float] = None,
        gt_mask: Optional[Union[Image.Image, np.ndarray]] = None,
    ) -> PredictionResult:
        """Run inference on an input image using the loaded model."""
        t = threshold if threshold is not None else self.default_threshold
        return predict_single_image(
            model=self.model,
            image_input=image_input,
            device=self.device,
            threshold=t,
            gt_mask=gt_mask,
        )

    def __repr__(self) -> str:
        return (
            f"GlacialLakePredictor("
            f"arch='{self.metadata.arch_type}', "
            f"params={self.metadata.total_parameters:,}, "
            f"device='{self.device}', "
            f"checkpoint='{self.metadata.checkpoint_path}')"
        )
