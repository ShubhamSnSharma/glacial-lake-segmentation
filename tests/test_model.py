"""
test_model.py
=============
Unit and integration tests for ResNet34FCN baseline, DeepLabV3Plus advanced model,
loss functions, and evaluation metrics.
"""

import os
import tempfile
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


class TestTrainerCheckpointResume(unittest.TestCase):
    """Test suite for checkpoint saving, atomic writes, and resume capability."""

    def setUp(self):
        import tempfile
        from torch.utils.data import TensorDataset, DataLoader
        from src.trainer import Trainer

        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_root = self.temp_dir.name

        self.cfg = {
            "training": {
                "epochs": 4,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "scheduler": "cosine",
                "warmup_epochs": 1,
                "loss": "bce_dice",
                "dice_weight": 0.5,
                "bce_weight": 0.5,
            },
            "evaluation": {"threshold": 0.5},
            "output": {
                "checkpoint_dir": f"{self.out_root}/checkpoints",
                "results_dir": f"{self.out_root}/results",
                "plots_dir": f"{self.out_root}/results/plots",
                "logs_dir": f"{self.out_root}/results/logs",
                "predictions_dir": f"{self.out_root}/results/predictions",
                "best_metric": "f1",
            },
            "data": {
                "batch_size": 2,
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
        }

        # Synthetic small dataset
        x = torch.randn(4, 3, 64, 64)
        y = torch.randint(0, 2, (4, 1, 64, 64)).float()
        ds = TensorDataset(x, y)
        self.train_loader = DataLoader(ds, batch_size=2)
        self.val_loader = DataLoader(ds, batch_size=2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_checkpoint_saving_and_content(self):
        from src.trainer import Trainer
        model = ResNet34FCN(pretrained=False, num_classes=1)
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            cfg=self.cfg,
            device=torch.device("cpu"),
        )
        trainer.best_metric_val = 0.85
        trainer.best_epoch = 2
        trainer.history = [{"epoch": 1, "val_f1": 0.70}, {"epoch": 2, "val_f1": 0.85}]

        chk_path = trainer.save_checkpoint(epoch=2, is_best=True)
        self.assertTrue(os.path.exists(chk_path))
        best_path = trainer.checkpoint_dir / "best_model.pth"
        self.assertTrue(best_path.exists())

        checkpoint = torch.load(chk_path, map_location="cpu", weights_only=False)
        self.assertIn("epoch", checkpoint)
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("optimizer_state_dict", checkpoint)
        self.assertIn("scheduler_state_dict", checkpoint)
        self.assertIn("best_metric_val", checkpoint)
        self.assertIn("best_epoch", checkpoint)
        self.assertIn("history", checkpoint)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["best_metric_val"], 0.85)
        self.assertEqual(len(checkpoint["history"]), 2)

    def test_resume_checkpoint_restores_state(self):
        from src.trainer import Trainer
        model1 = ResNet34FCN(pretrained=False, num_classes=1)
        trainer1 = Trainer(
            model=model1,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            cfg=self.cfg,
            device=torch.device("cpu"),
        )
        trainer1.best_metric_val = 0.78
        trainer1.best_epoch = 3
        trainer1.history = [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}]
        saved_path = trainer1.save_checkpoint(epoch=3, is_best=True)

        # Fresh trainer
        model2 = ResNet34FCN(pretrained=False, num_classes=1)
        trainer2 = Trainer(
            model=model2,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            cfg=self.cfg,
            device=torch.device("cpu"),
        )
        next_epoch = trainer2.resume_checkpoint(saved_path)

        self.assertEqual(next_epoch, 4)
        self.assertEqual(trainer2.start_epoch, 4)
        self.assertEqual(trainer2.best_metric_val, 0.78)
        self.assertEqual(trainer2.best_epoch, 3)
        self.assertEqual(len(trainer2.history), 3)

        # Verify model weights are restored identically
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_resume_training_continuation(self):
        from src.trainer import Trainer
        model1 = ResNet34FCN(pretrained=False, num_classes=1)
        cfg = dict(self.cfg)
        cfg["training"] = dict(self.cfg["training"])
        cfg["training"]["epochs"] = 3
        cfg["training"]["warmup_epochs"] = 1

        trainer1 = Trainer(
            model=model1,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            cfg=cfg,
            device=torch.device("cpu"),
        )
        # Train epoch 1 only
        train_m = trainer1.train_epoch(epoch=1)
        val_m = trainer1.validate_epoch(epoch=1)
        trainer1.history.append({"epoch": 1, **train_m, **val_m, "learning_rate": 1e-4, "epoch_time": 0.1})
        saved_path = trainer1.save_checkpoint(epoch=1, is_best=True)

        # Trainer 2 resumes and finishes training
        model2 = ResNet34FCN(pretrained=False, num_classes=1)
        trainer2 = Trainer(
            model=model2,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            cfg=cfg,
            device=torch.device("cpu"),
        )
        trainer2.resume_checkpoint(saved_path)
        self.assertEqual(trainer2.start_epoch, 2)

        results = trainer2.fit()
        # Epochs 2 and 3 should have been executed, bringing total history to 3
        self.assertEqual(len(results["history"]), 3)
        self.assertEqual([h["epoch"] for h in results["history"]], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()

