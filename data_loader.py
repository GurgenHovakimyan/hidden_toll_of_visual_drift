"""
data_loader.py
==============
Single factory that builds train/validation ``DataLoader`` objects for either
CIFAR-10 or Tiny ImageNet.

Design goals
------------
* One entry point: :func:`get_dataloaders`.
* All dataset-specific differences (num_classes, image size, root) are read from
  :data:`config.DATASETS` — never hard-coded here.
* **Standard normalization only, NO data augmentation** on the train set, so the
  drift study starts from a pure, deterministic baseline.
"""

from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config


def _build_transform(image_size: int) -> transforms.Compose:
    """Deterministic transform: resize -> tensor -> normalize.

    No random cropping, flipping, or colour jitter — identical for train and
    validation so the baseline is reproducible.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ]
    )


def _load_cifar10(
    cfg: config.DatasetConfig, transform: transforms.Compose
) -> Tuple[datasets.VisionDataset, datasets.VisionDataset]:
    """Download (if needed) and return CIFAR-10 train/test datasets."""
    train_set = datasets.CIFAR10(
        root=cfg.root, train=True, download=True, transform=transform
    )
    val_set = datasets.CIFAR10(
        root=cfg.root, train=False, download=True, transform=transform
    )
    return train_set, val_set


def _load_tiny_imagenet(
    cfg: config.DatasetConfig, transform: transforms.Compose
) -> Tuple[datasets.VisionDataset, datasets.VisionDataset]:
    """Return Tiny ImageNet train/val datasets from the ImageFolder layout.

    Expects the standard ``tiny-imagenet-200/train`` and ``.../val`` folders
    (the val split must already be arranged into class sub-directories).
    """
    train_set = datasets.ImageFolder(
        root=f"{cfg.root}/train", transform=transform
    )
    val_set = datasets.ImageFolder(
        root=f"{cfg.root}/val", transform=transform
    )
    return train_set, val_set


# Dispatch table keeps the factory branch-free and easy to extend.
_LOADERS = {
    "cifar10": _load_cifar10,
    "tiny_imagenet": _load_tiny_imagenet,
}


def get_dataloaders(
    dataset_name: str,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> Tuple[DataLoader, DataLoader, config.DatasetConfig]:
    """Factory returning ``(train_loader, val_loader, dataset_config)``.

    Parameters
    ----------
    dataset_name:
        ``"cifar10"`` or ``"tiny_imagenet"`` (keys of :data:`config.DATASETS`).
    batch_size:
        Overrides :data:`config.BATCH_SIZE` when provided.
    num_workers:
        Overrides :data:`config.NUM_WORKERS` when provided.

    Returns
    -------
    tuple
        ``(train_loader, val_loader, DatasetConfig)``. The ``DatasetConfig`` is
        returned so callers can size model heads without re-reading config.
    """
    key = dataset_name.lower()
    if key not in config.DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose one of {list(config.DATASETS)}."
        )

    cfg = config.DATASETS[key]
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS

    transform = _build_transform(cfg.image_size)
    train_set, val_set = _LOADERS[key](cfg, transform)

    # persistent_workers avoids re-spawning workers every epoch (big speed-up
    # on Windows), and is only valid when num_workers > 0.
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, cfg
