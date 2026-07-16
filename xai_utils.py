"""
xai_utils.py
============
Explainability heatmaps (Grad-CAM and Grad-CAM++) built on top of the
``pytorch-grad-cam`` library.

The single public function :func:`generate_heatmaps` takes a model, its target
layer(s) (see :func:`models.get_target_layers`) and a batch of inputs, and
returns per-image heatmaps normalized to ``[0, 1]`` for the model's **predicted**
class — exactly what `metrics.py` expects for SSIM / IoU / MSE comparisons.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Map the config method keys to the concrete CAM classes.
_CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcampp": GradCAMPlusPlus,
}


def generate_heatmaps(
    model: nn.Module,
    target_layers: List[nn.Module],
    input_tensor: torch.Tensor,
    method: str = "gradcam",
) -> np.ndarray:
    """Generate normalized CAM heatmaps for the predicted class of each input.

    Parameters
    ----------
    model:
        A trained network (in eval mode is recommended).
    target_layers:
        Layer(s) to hook, typically from :func:`models.get_target_layers`.
    input_tensor:
        Batch of normalized inputs ``(N, C, H, W)``.
    method:
        ``"gradcam"`` or ``"gradcampp"``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, H, W)`` with each heatmap already normalized to
        ``[0, 1]`` by ``pytorch-grad-cam``.
    """
    key = method.lower()
    if key not in _CAM_METHODS:
        raise ValueError(
            f"Unsupported XAI method '{method}'. "
            f"Choose one of {list(_CAM_METHODS)}."
        )

    # Predicted class per sample -> CAM targets.
    with torch.no_grad():
        logits = model(input_tensor)
        preds = torch.argmax(logits, dim=1)
    targets = [ClassifierOutputTarget(int(c)) for c in preds]

    cam_cls = _CAM_METHODS[key]
    # `pytorch-grad-cam` handles its own gradient context internally.
    with cam_cls(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    return grayscale_cam
