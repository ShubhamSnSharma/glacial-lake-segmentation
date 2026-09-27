"""
test_model.py
=============
Unit and integration tests for ResNet34FCN baseline, DeepLabV3Plus advanced model,
loss functions, and evaluation metrics.
"""

import unittest
import torch
import torch.nn as nn

from src.models.resnet34_fcn import ResNet34FCN, FCNDecoder
from src.models.deeplabv3_plus import DeepLabV3Plus, ASPP, DeepLabV3PlusDecoder
from src.model import build_model
from src.losses import DiceLoss, BinaryFocalLoss, CombinedBCEDiceLoss, build_loss
from src.metrics import MetricTracker, calculate_metrics, compute_confusion_matrix_elements


class TestModelArchitecture(unittest.TestCase):
    """Test suite for ResNet34FCN baseline model."""

    def setUp(self):
        # Use pretrained=False for fast offline testing
        self.model = ResNet34FCN(pretrained=False, num_classes=1)
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 512, 512)

    def test_forward_output_shape(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 1, 512, 512))
        self.assertEqual(output.dtype, torch.float32)

    def test_predict_mask(self):
        masks = self.model.predict_mask(self.input_tensor, threshold=0.5)
        self.assertEqual(masks.shape, (self.batch_size, 1, 512, 512))
        unique_vals = torch.unique(masks).tolist()
        for v in unique_vals:
            self.assertIn(v, [0.0, 1.0])

    def test_freeze_and_unfreeze_backbone(self):
        self.model.freeze_backbone()
        for p in self.model.encoder_parameters():
            self.assertFalse(p.requires_grad)
        for p in self.model.decoder_parameters():
            self.assertTrue(p.requires_grad)

        self.model.unfreeze_backbone()
        for p in self.model.encoder_parameters():
            self.assertTrue(p.requires_grad)

    def test_parameter_counts(self):
        summary = self.model.get_parameter_summary()
        self.assertIn("total_parameters", summary)
        self.assertIn("encoder_parameters", summary)
        self.assertIn("decoder_parameters", summary)
        self.assertGreater(summary["encoder_parameters"], 20_000_000)  # ResNet-34 is ~21M
        self.assertGreater(summary["decoder_parameters"], 1_000_000)

    def test_build_model_factory(self):
        cfg = {"model": {"backbone": "resnet34", "pretrained": False, "num_classes": 1}}
        model = build_model(cfg)
        self.assertIsInstance(model, ResNet34FCN)


class TestDeepLabV3PlusArchitecture(unittest.TestCase):
    """Test suite for DeepLabV3Plus advanced model."""

    def setUp(self):
        # Use pretrained=False for fast offline testing
        self.model = DeepLabV3Plus(pretrained=False, num_classes=1)
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 512, 512)

    def test_forward_output_shape(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 1, 512, 512))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_predict_mask(self):
        masks = self.model.predict_mask(self.input_tensor, threshold=0.5)
        self.assertEqual(masks.shape, (self.batch_size, 1, 512, 512))
        unique_vals = torch.unique(masks).tolist()
        for v in unique_vals:
            self.assertIn(v, [0.0, 1.0])

    def test_backward_gradient_flow(self):
        self.model.train()
        logits = self.model(self.input_tensor)
        targets = torch.randint(0, 2, (self.batch_size, 1, 512, 512)).float()
        criterion = CombinedBCEDiceLoss()
        loss = criterion(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        # Verify gradient existence in both backbone and decoder
        self.assertIsNotNone(self.model.stem[0].weight.grad)
        self.assertIsNotNone(self.model.decoder.classifier.weight.grad)

    def test_freeze_and_unfreeze_backbone(self):
        self.model.freeze_backbone()
        for p in self.model.encoder_parameters():
            self.assertFalse(p.requires_grad)
        for p in self.model.decoder_parameters():
            self.assertTrue(p.requires_grad)

        self.model.unfreeze_backbone()
        for p in self.model.encoder_parameters():
            self.assertTrue(p.requires_grad)

    def test_parameter_counts(self):
        summary = self.model.get_parameter_summary()
        self.assertIn("total_parameters", summary)
        self.assertIn("encoder_parameters", summary)
        self.assertIn("aspp_parameters", summary)
        self.assertIn("decoder_parameters", summary)
        self.assertGreater(summary["encoder_parameters"], 23_000_000)  # ResNet-50 is ~23.5M
        self.assertGreater(summary["head_parameters"], 5_000_000)      # ASPP + Decoder ~15M

    def test_build_model_factory(self):
        cfg = {"model": {"name": "deeplabv3plus", "pretrained": False, "num_classes": 1}}
        model = build_model(cfg)
        self.assertIsInstance(model, DeepLabV3Plus)


class TestLossFunctions(unittest.TestCase):
    """Test suite for segmentation losses."""

    def setUp(self):
        self.logits = torch.randn(2, 1, 128, 128, requires_grad=True)
        self.targets = torch.randint(0, 2, (2, 1, 128, 128)).float()

    def test_dice_loss(self):
        criterion = DiceLoss()
        loss = criterion(self.logits, self.targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(self.logits.grad)

    def test_combined_bce_dice_loss(self):
        criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        loss = criterion(self.logits, self.targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_focal_loss(self):
        criterion = BinaryFocalLoss()
        loss = criterion(self.logits, self.targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_build_loss_factory(self):
        loss_fn = build_loss({"training": {"loss": "bce_dice", "bce_weight": 0.4, "dice_weight": 0.6}})
        self.assertIsInstance(loss_fn, CombinedBCEDiceLoss)


class TestMetrics(unittest.TestCase):
    """Test suite for evaluation metrics."""

    def test_confusion_matrix_and_metrics(self):
        # Perfect prediction case
        targets = torch.tensor([[1, 0], [0, 1]])
        preds = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
        tp, fp, fn, tn = compute_confusion_matrix_elements(preds, targets, threshold=0.5)
        self.assertEqual((tp, fp, fn, tn), (2, 0, 0, 2))

        metrics = calculate_metrics(tp, fp, fn, tn)
        self.assertAlmostEqual(metrics["iou"], 1.0, places=4)
        self.assertAlmostEqual(metrics["f1"], 1.0, places=4)
        self.assertAlmostEqual(metrics["accuracy"], 1.0, places=4)

    def test_metric_tracker(self):
        tracker = MetricTracker(threshold=0.5)
        targets = torch.zeros(2, 1, 64, 64)
        targets[:, :, 10:20, 10:20] = 1.0
        preds = torch.zeros(2, 1, 64, 64)
        preds[:, :, 10:20, 10:20] = 10.0  # High logits for lake

        tracker.update(preds, targets, loss=0.15)
        summary = tracker.compute()
        self.assertIn("iou", summary)
        self.assertIn("f1", summary)
        self.assertIn("loss", summary)
        self.assertAlmostEqual(summary["loss"], 0.15, places=4)


if __name__ == "__main__":
    unittest.main()
