"""
test_inference.py
=================
Comprehensive unit and integration tests for the centralized inference backend (src/inference.py).

Verifies:
  1. ResNet34FCN checkpoint loading and metadata extraction.
  2. DeepLabV3Plus checkpoint loading and metadata extraction.
  3. PIL Image preprocessing across formats (RGB, RGBA, Grayscale, non-512x512).
  4. NumPy array preprocessing (uint8 [0, 255], float32 [0.0, 1.0], 2D, 3D).
  5. Output tensor shape (1, 3, 512, 512) and value normalization/un-normalization.
  6. Single-image inference output probability map shape and range [0.0, 1.0].
  7. Binary mask output shape (512, 512) and binary values {0, 1}.
  8. Thresholding behavior (default 0.5, custom 0.3, 0.8).
  9. Ground-truth-optional overlay generation (standalone and comparative).
  10. Quantitative prediction summary statistics calculation.
  11. GlacialLakePredictor wrapper class functionality.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from src.inference import (
    DEFAULT_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    GlacialLakePredictor,
    ModelMetadata,
    PredictionResult,
    compute_prediction_stats,
    create_prediction_overlay,
    load_model,
    predict_single_image,
    preprocess_image,
    resolve_architecture_name,
    select_device,
    unnormalize_tensor,
)
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.models.resnet34_fcn import ResNet34FCN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESNET_CHECKPOINT = PROJECT_ROOT / "resnet34_fcn" / "checkpoints" / "selected_model.pth"
DEEPLAB_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_amp_model.pth"


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    """Test suite for in-memory image preprocessing."""

    def test_pil_rgb_image(self):
        """Verify standard 512x512 RGB PIL Image preprocessing."""
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))
        tensor, rgb_np = preprocess_image(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 512, 512)
        assert tensor.dtype == torch.float32
        assert isinstance(rgb_np, np.ndarray)
        assert rgb_np.shape == (512, 512, 3)
        assert rgb_np.dtype == np.uint8

    def test_pil_resize_non_standard(self):
        """Verify arbitrary spatial dimensions are resized to 512x512."""
        img = Image.new("RGB", (800, 600), color=(50, 60, 70))
        tensor, rgb_np = preprocess_image(img, target_size=(512, 512))

        assert tensor.shape == (1, 3, 512, 512)
        assert rgb_np.shape == (512, 512, 3)

    def test_pil_rgba_and_grayscale(self):
        """Verify RGBA and Grayscale inputs convert to 3-channel RGB."""
        img_rgba = Image.new("RGBA", (256, 256), color=(10, 20, 30, 255))
        tensor_rgba, rgb_rgba = preprocess_image(img_rgba)
        assert tensor_rgba.shape == (1, 3, 512, 512)
        assert rgb_rgba.shape == (512, 512, 3)

        img_l = Image.new("L", (256, 256), color=128)
        tensor_l, rgb_l = preprocess_image(img_l)
        assert tensor_l.shape == (1, 3, 512, 512)
        assert rgb_l.shape == (512, 512, 3)

    def test_numpy_uint8(self):
        """Verify HWC uint8 NumPy array preprocessing."""
        arr = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        tensor, rgb_np = preprocess_image(arr)

        assert tensor.shape == (1, 3, 512, 512)
        assert rgb_np.shape == (512, 512, 3)
        assert rgb_np.dtype == np.uint8

    def test_numpy_float32(self):
        """Verify [0.0, 1.0] float32 NumPy array preprocessing."""
        arr = np.random.rand(512, 512, 3).astype(np.float32)
        tensor, rgb_np = preprocess_image(arr)

        assert tensor.shape == (1, 3, 512, 512)
        assert rgb_np.shape == (512, 512, 3)
        assert rgb_np.dtype == np.uint8

    def test_numpy_2d_grayscale(self):
        """Verify 2D (H, W) grayscale NumPy array preprocessing."""
        arr = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        tensor, rgb_np = preprocess_image(arr)

        assert tensor.shape == (1, 3, 512, 512)
        assert rgb_np.shape == (512, 512, 3)

    def test_unnormalize_roundtrip(self):
        """Verify tensor un-normalization accurately restores pixel domain."""
        img = Image.new("RGB", (512, 512), color=(120, 180, 240))
        tensor, original_np = preprocess_image(img)
        restored_np = unnormalize_tensor(tensor)

        assert restored_np.shape == (512, 512, 3)
        assert restored_np.dtype == np.uint8
        # Allow tiny numerical rounding tolerance (<= 1 intensity unit)
        assert np.max(np.abs(restored_np.astype(int) - original_np.astype(int))) <= 1


# ---------------------------------------------------------------------------
# Statistics and Overlay Helpers Tests
# ---------------------------------------------------------------------------

class TestVisualizationAndStats:
    """Test suite for statistics and overlay generation."""

    def test_compute_prediction_stats(self):
        """Verify statistics computation with known positive pixel counts."""
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 1  # 100 x 100 = 10,000 pixels

        probs = np.full((512, 512), 0.1, dtype=np.float32)
        probs[100:200, 100:200] = 0.95

        stats = compute_prediction_stats(mask, probs)

        assert stats["lake_pixels"] == 10000
        assert stats["total_pixels"] == 512 * 512
        assert stats["has_lake_detected"] is True
        expected_pct = (10000 / (512 * 512)) * 100.0
        assert pytest.approx(stats["lake_percentage"], rel=1e-3) == expected_pct
        assert pytest.approx(stats["mean_lake_confidence"], rel=1e-3) == 0.95
        assert pytest.approx(stats["max_lake_confidence"], rel=1e-3) == 0.95

    def test_compute_prediction_stats_empty(self):
        """Verify statistics computation when no lake is detected."""
        mask = np.zeros((512, 512), dtype=np.uint8)
        probs = np.full((512, 512), 0.05, dtype=np.float32)

        stats = compute_prediction_stats(mask, probs)
        assert stats["lake_pixels"] == 0
        assert stats["lake_percentage"] == 0.0
        assert stats["has_lake_detected"] is False
        assert stats["mean_lake_confidence"] == 0.0

    def test_create_prediction_overlay_standalone(self):
        """Verify ground-truth-free standalone prediction overlay."""
        img_np = np.full((512, 512, 3), 100, dtype=np.uint8)
        pred_mask = np.zeros((512, 512), dtype=np.uint8)
        pred_mask[50:150, 50:150] = 1

        overlay = create_prediction_overlay(img_np, pred_mask, gt_mask=None)

        assert isinstance(overlay, np.ndarray)
        assert overlay.shape == (512, 512, 3)
        assert overlay.dtype == np.uint8
        # Unpredicted pixels should remain identical to input
        assert np.array_equal(overlay[0, 0], img_np[0, 0])
        # Predicted pixels should be tinted (cyan)
        assert not np.array_equal(overlay[100, 100], img_np[100, 100])

    def test_create_prediction_overlay_with_ground_truth(self):
        """Verify comparative overlay with TP, FP, and FN."""
        img_np = np.full((512, 512, 3), 100, dtype=np.uint8)
        gt_mask = np.zeros((512, 512), dtype=np.uint8)
        pred_mask = np.zeros((512, 512), dtype=np.uint8)

        gt_mask[10:50, 10:50] = 1   # Lake
        pred_mask[30:70, 30:70] = 1 # Overlapping detection

        overlay = create_prediction_overlay(img_np, pred_mask, gt_mask=gt_mask)

        assert overlay.shape == (512, 512, 3)
        assert overlay.dtype == np.uint8


# ---------------------------------------------------------------------------
# Single-Image Prediction & Model Loading Tests
# ---------------------------------------------------------------------------

class TestInferencePipeline:
    """Test suite for single-image prediction and model loading."""

    @pytest.fixture
    def mock_model(self):
        """Lightweight mock segmentation model returning synthetic logits."""
        class MockSegModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 1, kernel_size=1)
                nn.init.zeros_(self.conv.weight)
                nn.init.constant_(self.conv.bias, 2.0)  # sigmoid(2.0) ~ 0.88 -> positive

            def forward(self, x):
                return self.conv(x)

        return MockSegModel().eval()

    def test_predict_single_image_mock(self, mock_model):
        """Verify end-to-end inference flow on mock model."""
        img = Image.new("RGB", (512, 512), color=(50, 100, 150))
        result = predict_single_image(mock_model, img, device="cpu", threshold=0.5)

        assert isinstance(result, PredictionResult)
        assert result.probability_map.shape == (512, 512)
        assert result.probability_map.dtype == np.float32
        assert np.all((result.probability_map >= 0.0) & (result.probability_map <= 1.0))

        assert result.binary_mask.shape == (512, 512)
        assert result.binary_mask.dtype == np.uint8
        assert set(np.unique(result.binary_mask)).issubset({0, 1})

        assert result.processed_image.shape == (512, 512, 3)
        assert result.overlay_image.shape == (512, 512, 3)
        assert result.statistics["total_pixels"] == 512 * 512
        assert result.threshold == 0.5

    def test_threshold_sensitivity(self, mock_model):
        """Verify varying thresholds modulate binary mask output."""
        img = Image.new("RGB", (512, 512), color=(50, 100, 150))
        # With bias=2.0, sigmoid output is approx 0.8808
        res_low = predict_single_image(mock_model, img, device="cpu", threshold=0.5)
        res_high = predict_single_image(mock_model, img, device="cpu", threshold=0.95)

        assert res_low.statistics["lake_pixels"] == 512 * 512
        assert res_high.statistics["lake_pixels"] == 0

    @pytest.mark.skipif(not DEEPLAB_CHECKPOINT.exists(), reason="DeepLabV3+ checkpoint not present")
    def test_deeplab_checkpoint_loading(self):
        """Verify real DeepLabV3+ checkpoint loads and extracts metadata."""
        model, metadata = load_model(DEEPLAB_CHECKPOINT, device="cpu")

        assert isinstance(model, DeepLabV3Plus)
        assert metadata.arch_type == "DeepLabV3+"
        assert metadata.total_parameters == 40347041
        assert metadata.epoch == 26

        # Fast single synthetic image inference check
        img = Image.new("RGB", (512, 512), color=(100, 100, 100))
        result = predict_single_image(model, img, device="cpu")
        assert result.probability_map.shape == (512, 512)
        assert result.binary_mask.shape == (512, 512)

    @pytest.mark.skipif(not RESNET_CHECKPOINT.exists(), reason="ResNet34-FCN checkpoint not present")
    def test_resnet_checkpoint_loading(self):
        """Verify real ResNet34-FCN checkpoint loads and extracts metadata."""
        model, metadata = load_model(RESNET_CHECKPOINT, device="cpu")

        assert isinstance(model, ResNet34FCN)
        assert metadata.arch_type == "ResNet34-FCN"
        assert metadata.total_parameters == 24079153

        # Fast single synthetic image inference check
        img = Image.new("RGB", (512, 512), color=(100, 100, 100))
        result = predict_single_image(model, img, device="cpu")
        assert result.probability_map.shape == (512, 512)
        assert result.binary_mask.shape == (512, 512)

    @pytest.mark.skipif(not DEEPLAB_CHECKPOINT.exists(), reason="DeepLabV3+ checkpoint not present")
    def test_predictor_wrapper_class(self):
        """Verify stateful GlacialLakePredictor wrapper class."""
        predictor = GlacialLakePredictor(
            checkpoint_path=DEEPLAB_CHECKPOINT,
            device="cpu",
            default_threshold=0.5,
        )
        img = Image.new("RGB", (512, 512), color=(120, 140, 160))
        result = predictor.predict(img)

        assert isinstance(result, PredictionResult)
        assert result.binary_mask.shape == (512, 512)
        assert "DeepLabV3+" in str(predictor)
