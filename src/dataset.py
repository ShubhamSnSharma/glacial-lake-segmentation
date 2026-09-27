"""
dataset.py
==========
PyTorch Dataset and DataLoader factory for the GLID (Glacial Lake Image Dataset).

Dataset structure expected on disk (verified against Zenodo record 14838695):

    <split_dir>/
    ├── images/      # RGB PNG patches — 512×512×3, uint8
    └── labels/      # Binary mask PNGs — 512×512, uint8
                     #   glacial lake pixel value : 255
                     #   background pixel value   : 0

The mask values are normalized to {0, 1} inside __getitem__ before returning.

References:
    Ma D, Li J, Jiang L. 2025. Efficient glacial lake mapping by leveraging
    deep transfer learning and a new annotated glacial lake dataset.
    Journal of Hydrology 657: 133072. doi: 10.1016/j.jhydrol.2025.133072

    GLID dataset: https://doi.org/10.5281/zenodo.14838695
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Mask sub-directory candidates — will be probed in order
# ---------------------------------------------------------------------------
_MASK_SUBDIR_CANDIDATES: List[str] = [
    "labels", "label", "masks", "mask", "annotation", "annotations"
]
_IMAGE_SUBDIR_CANDIDATES: List[str] = [
    "images", "image", "img", "imgs"
]


def _find_subdir(root: Path, candidates: List[str]) -> Path:
    """Return the first existing candidate sub-directory, or raise."""
    for name in candidates:
        p = root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"Could not find any of {candidates} inside '{root}'. "
        f"Contents: {list(root.iterdir())}"
    )


def discover_split_paths(split_dir: str) -> Tuple[Path, Path]:
    """
    Auto-detect image and mask directories inside a split directory.

    Args:
        split_dir: Path to the split root (e.g. 'data/raw/GLID' or 'data/raw/val')

    Returns:
        (image_dir, mask_dir) as Path objects

    Raises:
        FileNotFoundError if expected subdirectories cannot be found.
    """
    root = Path(split_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Split directory not found: '{root}'")

    image_dir = _find_subdir(root, _IMAGE_SUBDIR_CANDIDATES)
    mask_dir = _find_subdir(root, _MASK_SUBDIR_CANDIDATES)
    return image_dir, mask_dir


# ---------------------------------------------------------------------------
# Synchronized augmentation helpers
# ---------------------------------------------------------------------------

class SyncedRandomAugment:
    """
    Applies the same random geometric transforms to both image and mask.

    Colour-altering transforms (jitter, normalize) are applied to the image
    only. Geometric transforms (flip, rotation) are applied identically to
    both using a shared random state so that spatial correspondence is
    preserved.

    Args:
        horizontal_flip: Randomly flip left-right with p=0.5
        vertical_flip:   Randomly flip top-bottom with p=0.5
        rotation_degrees: If > 0, randomly rotate by one of
                          {0, 90, 180, 270} degrees (square patches only).
    """

    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        rotation_degrees: int = 90,
    ) -> None:
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_degrees = rotation_degrees

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        # Horizontal flip
        if self.horizontal_flip and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Vertical flip
        if self.vertical_flip and random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Rotation — only right-angle multiples to avoid border artefacts
        if self.rotation_degrees > 0 and random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        return image, mask


# ---------------------------------------------------------------------------
# Core Dataset class
# ---------------------------------------------------------------------------

class GLIDDataset(Dataset):
    """
    PyTorch Dataset for the Glacial Lake Image Dataset (GLID).

    Each item returns:
        image  : FloatTensor of shape (3, H, W), values in [0, 1] then
                 normalized by ImageNet mean/std (if normalize=True)
        mask   : FloatTensor of shape (1, H, W), values in {0.0, 1.0}
                 (0 = background, 1 = glacial lake)

    Args:
        split_dir:  Path to the split root directory (e.g. 'data/raw/GLID')
        is_train:   If True, applies data augmentation.
        image_subdir: Name of the image sub-directory. Auto-detected if None.
        mask_subdir:  Name of the mask sub-directory. Auto-detected if None.
        normalize_mean: Per-channel mean for normalization. Defaults to ImageNet.
        normalize_std:  Per-channel std  for normalization. Defaults to ImageNet.
        augment_cfg:    Dict of augmentation flags passed to SyncedRandomAugment.
                        Ignored when is_train=False.
        image_size:     Resize images to this square size. Default 512 (no-op
                        since GLID images are already 512×512).
    """

    # ImageNet normalization constants (our deviation from the paper)
    IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD:  List[float] = [0.229, 0.224, 0.225]

    def __init__(
        self,
        split_dir: Optional[str] = None,
        is_train: bool = True,
        split: Optional[str] = None,
        image_subdir: Optional[str] = None,
        mask_subdir: Optional[str] = None,
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
        augment_cfg: Optional[Dict] = None,
        image_size: int = 512,
        root_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        target_dir = split_dir or root_dir
        if target_dir is None:
            raise ValueError("Must provide either 'split_dir' or 'root_dir'")

        split_root = Path(target_dir)
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Split directory not found: '{split_root}'. "
                "Check that the dataset has been downloaded and extracted. "
                "See data/README.md for instructions."
            )

        # Resolve image and mask directories
        if image_subdir:
            self.image_dir = split_root / image_subdir
        else:
            self.image_dir = _find_subdir(split_root, _IMAGE_SUBDIR_CANDIDATES)

        if mask_subdir:
            self.mask_dir = split_root / mask_subdir
        else:
            self.mask_dir = _find_subdir(split_root, _MASK_SUBDIR_CANDIDATES)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: '{self.image_dir}'")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: '{self.mask_dir}'")

        # Collect and sort filenames (sort by integer ID for reproducibility)
        self.image_files = self._collect_png_files(self.image_dir)
        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No PNG files found in '{self.image_dir}'. "
                "Verify that the archive was fully extracted."
            )

        # Verify matching mask files
        self._verify_mask_correspondence()

        # Transforms
        self.is_train = is_train
        self.split = split if split is not None else ("train" if is_train else "val")
        self.image_size = image_size
        mean = normalize_mean if normalize_mean else self.IMAGENET_MEAN
        std  = normalize_std  if normalize_std  else self.IMAGENET_STD

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=mean, std=std)

        augment_cfg = augment_cfg or {}
        self.augmenter = SyncedRandomAugment(
            horizontal_flip=augment_cfg.get("horizontal_flip", True),
            vertical_flip=augment_cfg.get("vertical_flip", True),
            rotation_degrees=augment_cfg.get("rotation_degrees", 90),
        ) if is_train else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_png_files(directory: Path) -> List[str]:
        """Return sorted list of PNG filenames (base names only)."""
        files = [f.name for f in directory.iterdir() if f.suffix.lower() == ".png"]
        # Sort numerically by stem (1.png, 2.png, ... 1000.png) if stems are ints
        try:
            files.sort(key=lambda x: int(Path(x).stem))
        except ValueError:
            files.sort()  # Fallback to lexicographic sort
        return files

    def _verify_mask_correspondence(self) -> None:
        """
        Verify that every image file has a corresponding mask file.
        Raises RuntimeError listing any missing masks.
        """
        missing = []
        for fname in self.image_files:
            if not (self.mask_dir / fname).exists():
                missing.append(fname)
        if missing:
            n = len(missing)
            examples = missing[:5]
            raise RuntimeError(
                f"{n} image(s) have no corresponding mask in '{self.mask_dir}'. "
                f"First {min(n, 5)} missing: {examples}"
            )

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.image_files[idx]

        # Load image (RGB) and mask (grayscale)
        image = Image.open(self.image_dir / fname).convert("RGB")
        mask  = Image.open(self.mask_dir  / fname).convert("L")

        # Optional resize (GLID images are already 512×512, this is a safety net)
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
            mask = mask.resize(
                (self.image_size, self.image_size), Image.NEAREST
            )

        # Synchronized augmentation (training only)
        if self.augmenter is not None:
            image, mask = self.augmenter(image, mask)

        # Convert image to tensor (→ float32, [0, 1]) and normalize
        image_tensor = self.normalize(self.to_tensor(image))

        # Convert mask to tensor: uint8 PIL → float32 tensor
        # Values are 0 or 255 in GLID; normalize to {0.0, 1.0}
        mask_np = np.array(mask, dtype=np.float32)
        mask_np = (mask_np > 127).astype(np.float32)   # Threshold at 127 for safety
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # → (1, H, W)

        return image_tensor, mask_tensor

    def get_filename(self, idx: int) -> str:
        """Return the base filename for sample at index idx."""
        return self.image_files[idx]

    def __repr__(self) -> str:
        return (
            f"GLIDDataset("
            f"n={len(self)}, "
            f"split={self.split}, "
            f"image_dir={self.image_dir}, "
            f"mask_dir={self.mask_dir}"
            f")"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    split_dir: str,
    is_train: bool,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_subdir: Optional[str] = None,
    mask_subdir: Optional[str] = None,
    normalize_mean: Optional[List[float]] = None,
    normalize_std: Optional[List[float]] = None,
    augment_cfg: Optional[Dict] = None,
    image_size: int = 512,
) -> Tuple[DataLoader, GLIDDataset]:
    """
    Build a GLIDDataset and wrap it in a DataLoader.

    Args:
        split_dir:  Path to the extracted split directory.
        is_train:   True = shuffled + augmented; False = deterministic, no augmentation.
        batch_size: Samples per batch.
        num_workers: Number of parallel data-loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        (remaining args forwarded to GLIDDataset)

    Returns:
        (loader, dataset) tuple — dataset is returned for introspection.
    """
    dataset = GLIDDataset(
        split_dir=split_dir,
        is_train=is_train,
        image_subdir=image_subdir,
        mask_subdir=mask_subdir,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        augment_cfg=augment_cfg,
        image_size=image_size,
    )

    # Pin memory is only supported on CUDA devices; avoid MPS warnings
    use_pin_memory = pin_memory and torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,          # Only shuffle training data
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=is_train,        # Drop incomplete last batch during training
        persistent_workers=(num_workers > 0),
    )
    return loader, dataset


# ---------------------------------------------------------------------------
# Config-aware factory
# ---------------------------------------------------------------------------

def get_dataloaders_from_config(cfg: Dict) -> Dict[str, Tuple[DataLoader, GLIDDataset]]:
    """
    Build train and val dataloaders from a parsed YAML config dict.

    Args:
        cfg: Full config dict (as loaded from configs/default.yaml)

    Returns:
        {'train': (loader, dataset), 'val': (loader, dataset)}

    Usage:
        import yaml
        with open('configs/default.yaml') as f:
            cfg = yaml.safe_load(f)
        loaders = get_dataloaders_from_config(cfg)
        train_loader, train_dataset = loaders['train']
        val_loader,   val_dataset   = loaders['val']
    """
    data_cfg = cfg["data"]
    raw_dir  = Path(data_cfg["raw_dir"])

    train_dir = raw_dir / data_cfg.get("train_subdir", "GLID")
    val_dir   = raw_dir / data_cfg.get("val_subdir", "val")

    augment_cfg = data_cfg.get("augmentation", {})

    shared_kwargs = dict(
        image_subdir=data_cfg.get("image_subdir"),
        mask_subdir=data_cfg.get("mask_subdir"),
        normalize_mean=data_cfg.get("normalize_mean"),
        normalize_std=data_cfg.get("normalize_std"),
        image_size=data_cfg.get("image_size", 512),
        batch_size=data_cfg.get("batch_size", 8),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )

    train_loader, train_dataset = get_dataloader(
        split_dir=str(train_dir),
        is_train=True,
        augment_cfg=augment_cfg,
        **shared_kwargs,
    )
    val_loader, val_dataset = get_dataloader(
        split_dir=str(val_dir),
        is_train=False,
        augment_cfg=None,
        **shared_kwargs,
    )

    return {
        "train": (train_loader, train_dataset),
        "val":   (val_loader, val_dataset),
    }
