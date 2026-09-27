"""
test_dataset.py
===============
Unit and integration tests for GLIDDataset, augmentations, and data utilities.
Creates a temporary synthetic GLID dataset on disk to test all loaders without
requiring the full multi-GB download.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    from src.dataset import GLIDDataset, get_dataloader, get_dataloaders_from_config, discover_split_paths
    from src.utils import compute_class_distribution, compute_normalization_stats
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"ImportError in test_dataset: {e}")
    TORCH_AVAILABLE = False


class TestSyntheticGLID(unittest.TestCase):
    """Test suite using generated synthetic 512x512 image/mask pairs."""

    def setUp(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not installed in the active environment.")

        self.temp_dir = tempfile.mkdtemp()
        self.split_dir = Path(self.temp_dir) / "synthetic_val"
        self.img_dir = self.split_dir / "images"
        self.lbl_dir = self.split_dir / "labels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)

        # Generate 4 dummy sample pairs (512x512)
        self.num_samples = 4
        for i in range(1, self.num_samples + 1):
            # Synthetic RGB image
            img_arr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            Image.fromarray(img_arr).save(self.img_dir / f"sample_{i:04d}.png")

            # Synthetic mask: mostly 0 (background) with a square of 255 (lake)
            mask_arr = np.zeros((512, 512), dtype=np.uint8)
            mask_arr[100:200, 100:200] = 255
            Image.fromarray(mask_arr).save(self.lbl_dir / f"sample_{i:04d}.png")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_discover_split_paths(self):
        img_p, lbl_p = discover_split_paths(str(self.split_dir))
        self.assertEqual(img_p, self.img_dir)
        self.assertEqual(lbl_p, self.lbl_dir)

    def test_dataset_length_and_shapes(self):
        ds = GLIDDataset(split_dir=str(self.split_dir), is_train=False)
        self.assertEqual(len(ds), self.num_samples)

        img, mask = ds[0]
        self.assertEqual(img.shape, (3, 512, 512))
        self.assertEqual(mask.shape, (1, 512, 512))
        self.assertEqual(img.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.float32)

        # Verify mask binarization (0 and 1 only)
        unique_vals = torch.unique(mask).tolist()
        for v in unique_vals:
            self.assertIn(v, [0.0, 1.0])

    def test_dataloader_batch(self):
        ds = GLIDDataset(split_dir=str(self.split_dir), is_train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
        
        for imgs, masks in loader:
            self.assertEqual(imgs.shape, (2, 3, 512, 512))
            self.assertEqual(masks.shape, (2, 1, 512, 512))
            break

    def test_class_distribution_utility(self):
        ds = GLIDDataset(split_dir=str(self.split_dir), is_train=False)
        stats = compute_class_distribution(ds, n_samples=4, verbose=False)
        self.assertIn("lake_fraction", stats)
        self.assertGreater(stats["lake_fraction"], 0.0)
        self.assertLess(stats["lake_fraction"], 1.0)

    def test_normalization_stats_utility(self):
        ds = GLIDDataset(split_dir=str(self.split_dir), is_train=False)
        stats = compute_normalization_stats(ds, n_samples=4, verbose=False)
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertEqual(len(stats["mean"]), 3)
        self.assertEqual(len(stats["std"]), 3)

    def test_get_dataloader_factory(self):
        loader, ds = get_dataloader(
            split_dir=str(self.split_dir),
            batch_size=2,
            is_train=False,
            num_workers=0
        )
        self.assertEqual(len(ds), self.num_samples)
        batch_imgs, batch_masks = next(iter(loader))
        self.assertEqual(batch_imgs.shape, (2, 3, 512, 512))
        self.assertEqual(batch_masks.shape, (2, 1, 512, 512))


if __name__ == "__main__":
    unittest.main()
