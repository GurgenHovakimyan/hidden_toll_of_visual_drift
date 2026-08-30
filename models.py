"""
models.py
=========
Model factory for the three CNN architectures compared in the study:

    * ResNet-18
    * DenseNet-121
    * ShuffleNet V2 (x1.0)

All networks are instantiated from ImageNet-pretrained backbones
(``config.PRETRAINED``), and their classification head is resized dynamically to
the number of classes of the active dataset (10 for CIFAR-10, 200 for Tiny
ImageNet).

The module also exposes :func:`get_target_layers`, the single source of truth
for which convolutional layer Grad-CAM / Grad-CAM++ should hook into for each
architecture. Keeping this next to the model definitions avoids duplicating the
knowledge across `xai_utils.py` and the notebooks.
"""

from __future__ import annotations

from typing import List

import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    ResNet18_Weights,
    ShuffleNet_V2_X1_0_Weights,
    densenet121,
    resnet18,
    shufflenet_v2_x1_0,
)

import config

# Canonical model keys (kept in sync with config.MODELS).
_SUPPORTED = ("resnet18", "densenet121", "shufflenet_v2")


def get_model(
    name: str, num_classes: int, pretrained: bool | None = None
) -> nn.Module:
    """Instantiate a CNN with a head sized for ``num_classes``.

    Parameters
    ----------
    name:
        One of ``"resnet18"``, ``"densenet121"`` or ``"shufflenet_v2"``.
    num_classes:
        Number of output classes (10 for CIFAR-10, 200 for Tiny ImageNet).
    pretrained:
        If ``True`` load ImageNet-pretrained backbone weights (the classifier
        head is always re-initialised for ``num_classes``). Defaults to
        :data:`config.PRETRAINED` when ``None``.

    Returns
    -------
    torch.nn.Module
        The network with a freshly-initialised classifier of the right size.
    """
    key = name.lower()
    if pretrained is None:
        pretrained = config.PRETRAINED

    if key == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif key == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif key == "shufflenet_v2":
        weights = (
            ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model = shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported model '{name}'. Choose one of {list(_SUPPORTED)}."
        )

    return model


def get_target_layers(model: nn.Module, name: str) -> List[nn.Module]:
    """Return the layer(s) Grad-CAM should target for a given architecture.

    We hook the **penultimate** convolutional stage rather than the final one.
    On low-resolution inputs (32x32 CIFAR-10, 64x64 Tiny ImageNet) the last
    stage collapses to a 1x1 spatial map, which makes Grad-CAM degenerate
    (all-zero heatmaps). The penultimate stage retains a 2x2-4x4 map and yields
    informative saliency, matching the original study's ``layer3`` choice.

    Parameters
    ----------
    model:
        An instance created by :func:`get_model`.
    name:
        The architecture key used to create ``model``.

    Returns
    -------
    list[torch.nn.Module]
        A list (as expected by ``pytorch_grad_cam``) with the target layer.
    """
    key = name.lower()

    if key == "resnet18":
        return [model.layer3[-1]]

    if key == "densenet121":
        return [model.features.denseblock3]

    if key == "shufflenet_v2":
        return [model.stage3[-1]]

    raise ValueError(
        f"Unsupported model '{name}'. Choose one of {list(_SUPPORTED)}."
    )
