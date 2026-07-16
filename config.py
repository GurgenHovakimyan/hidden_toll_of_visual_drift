"""
config.py
=========
Centralized configuration for the "Hidden Toll of Visual Drift" study.

Everything that is a *constant* for the experiments lives here so that the rest
of the codebase never hard-codes a value. Import `CONFIG` (or the individual
dataclasses) anywhere you need a setting.

DRY principle: change a hyper-parameter *once*, here, and it propagates to
`data_loader.py`, `models.py`, `trainer.py`, `evaluator.py` and `main.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED: int = 42

# --------------------------------------------------------------------------- #
# Normalization (mean = 0.5, std = 0.5 per channel, NO augmentation)
# --------------------------------------------------------------------------- #
NORM_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
NORM_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
BATCH_SIZE: int = 128
NUM_WORKERS: int = 4


@dataclass(frozen=True)
class DatasetConfig:
    """Static description of a dataset used in the study."""

    name: str                 # canonical key, e.g. "cifar10"
    num_classes: int          # size of the classification head
    image_size: int           # square spatial resolution fed to the network
    root: str                 # on-disk location for the raw data


# The two datasets analysed in the paper.
DATASETS: Dict[str, DatasetConfig] = {
    "cifar10": DatasetConfig(
        name="cifar10",
        num_classes=10,
        image_size=32,
        root="./data",
    ),
    "tiny_imagenet": DatasetConfig(
        name="tiny_imagenet",
        num_classes=200,
        image_size=64,
        root="./tiny-imagenet-200",
    ),
}

# --------------------------------------------------------------------------- #
# Models (all trained from scratch: pretrained=False)
# --------------------------------------------------------------------------- #
MODELS: List[str] = ["resnet18", "densenet121", "shufflenet_v2"]

# Initialise backbones from ImageNet-pretrained weights (faster convergence).
# The classification head is always re-initialised for the target num_classes.
PRETRAINED: bool = True

# --------------------------------------------------------------------------- #
# Optimization
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    """Optimizer / training-loop hyper-parameters (identical for every model)."""

    optimizer: str = "adam"
    learning_rate: float = 1e-3
    epochs: int = 30
    batch_size: int = BATCH_SIZE
    # Early stopping: stop if the monitored metric does not improve for
    # ``patience`` consecutive epochs. The *best* epoch's weights are kept.
    early_stop_patience: int = 7
    monitor: str = "val_loss"  # "val_loss" (minimise) or "val_acc" (maximise)


TRAIN: TrainConfig = TrainConfig()

# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
# The four pixel-level drift families studied.
DRIFT_TYPES: List[str] = ["noise", "blur", "brightness", "rotation"]

# 12 intensities spanning 0 (no drift) to 0.5 (strong drift). The lower decades
# are sampled logarithmically so the "onset" of drift is well resolved.
DRIFT_INTENSITIES: List[float] = [
    0.0,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]

# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
XAI_METHODS: List[str] = ["gradcam", "gradcampp"]

# Binary threshold used for IoU between heatmaps.
HEATMAP_BINARY_THRESHOLD: float = 0.5

# --------------------------------------------------------------------------- #
# Output paths
# --------------------------------------------------------------------------- #
@dataclass
class Paths:
    """Directory layout for every artefact produced by the pipeline."""

    root: str = "./outputs"
    checkpoints: str = field(init=False)
    learning_curves: str = field(init=False)
    heatmaps: str = field(init=False)
    results: str = field(init=False)

    def __post_init__(self) -> None:
        self.checkpoints = os.path.join(self.root, "checkpoints")
        self.learning_curves = os.path.join(self.root, "learning_curves")
        self.heatmaps = os.path.join(self.root, "heatmaps")
        self.results = os.path.join(self.root, "results")

    def create(self) -> None:
        """Create every output directory if it does not already exist."""
        for path in (
            self.root,
            self.checkpoints,
            self.learning_curves,
            self.heatmaps,
            self.results,
        ):
            os.makedirs(path, exist_ok=True)


PATHS: Paths = Paths()


# --------------------------------------------------------------------------- #
# Aggregated view (convenience single import)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Config:
    device: torch.device = DEVICE
    seed: int = SEED
    norm_mean: Tuple[float, float, float] = NORM_MEAN
    norm_std: Tuple[float, float, float] = NORM_STD
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    datasets: Dict[str, DatasetConfig] = field(default_factory=lambda: DATASETS)
    models: List[str] = field(default_factory=lambda: MODELS)
    pretrained: bool = PRETRAINED
    train: TrainConfig = field(default_factory=lambda: TRAIN)
    drift_types: List[str] = field(default_factory=lambda: DRIFT_TYPES)
    drift_intensities: List[float] = field(default_factory=lambda: DRIFT_INTENSITIES)
    xai_methods: List[str] = field(default_factory=lambda: XAI_METHODS)
    heatmap_binary_threshold: float = HEATMAP_BINARY_THRESHOLD
    paths: Paths = field(default_factory=lambda: PATHS)


CONFIG: _Config = _Config()


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) for reproducible runs."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
