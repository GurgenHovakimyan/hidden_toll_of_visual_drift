"""
drift_utils.py
==============
Unified, batch-efficient pixel-level drift for the study.

The public entry point is :func:`apply_drift`, dispatched through a function
dictionary so adding a new drift family is a one-liner. All operators consume
and return a **normalized** batch tensor of shape ``(N, C, H, W)`` living in the
``[-1, 1]`` range (because the data pipeline normalizes with mean=std=0.5), and
every result is clamped back to that valid range.

Drift families
--------------
* ``noise``      : additive Gaussian noise  ``x + N(0, intensity^2)``.
* ``blur``       : Gaussian blur; intensity -> kernel size in {3, 5, 7, 9}.
* ``brightness`` : linear brightness scaling via torchvision.
* ``rotation``   : random rotation up to ``45 * intensity`` degrees.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torchvision.transforms.functional as TF

import config

# Valid range of a normalized tensor (mean = std = 0.5  ->  [-1, 1]).
_VMIN, _VMAX = -1.0, 1.0

# Kernel sizes used for the blur ladder (odd, increasing).
_BLUR_KERNELS = (3, 5, 7, 9)


# --------------------------------------------------------------------------- #
# Normalization helpers (mean = std = 0.5)
# --------------------------------------------------------------------------- #
def _to_unit(images: torch.Tensor) -> torch.Tensor:
    """Map normalized [-1, 1] tensor to [0, 1] for value-range-sensitive ops."""
    return images * 0.5 + 0.5


def _to_norm(images: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_to_unit`: map [0, 1] back to normalized [-1, 1]."""
    return images * 2.0 - 1.0


def _kernel_from_intensity(intensity: float) -> int:
    """Map a drift intensity in (0, 0.5] to an odd Gaussian-blur kernel size."""
    idx = min(int(intensity / config.DRIFT_INTENSITIES[-1] * len(_BLUR_KERNELS)),
              len(_BLUR_KERNELS) - 1)
    return _BLUR_KERNELS[idx]


# --------------------------------------------------------------------------- #
# Individual drift operators (each: (images, intensity) -> images)
# --------------------------------------------------------------------------- #
def _noise(images: torch.Tensor, intensity: float) -> torch.Tensor:
    """Additive Gaussian noise with standard deviation ``intensity``."""
    if intensity <= 0:
        return images
    noise = torch.randn_like(images) * intensity
    return images + noise


def _blur(images: torch.Tensor, intensity: float) -> torch.Tensor:
    """Gaussian blur whose kernel grows with intensity (batched)."""
    if intensity <= 0:
        return images
    kernel = _kernel_from_intensity(intensity)
    # sigma proportional to kernel following the usual (k/2 - 1) * 0.3 + 0.8 rule
    # is handled internally by torchvision when sigma is derived; we pass an
    # explicit sigma so the whole batch is blurred identically and efficiently.
    sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8
    return TF.gaussian_blur(images, kernel_size=kernel, sigma=sigma)


def _brightness(images: torch.Tensor, intensity: float) -> torch.Tensor:
    """Linear brightness scaling by a factor of ``1 + intensity``.

    Applied in [0, 1] space (as torchvision expects) then re-normalized.
    """
    if intensity <= 0:
        return images
    unit = _to_unit(images).clamp(0.0, 1.0)
    unit = TF.adjust_brightness(unit, brightness_factor=1.0 + intensity)
    return _to_norm(unit)


def _rotation(images: torch.Tensor, intensity: float) -> torch.Tensor:
    """Rotate the whole batch by a random angle in ``[-45i, 45i]`` degrees."""
    if intensity <= 0:
        return images
    max_angle = 45.0 * intensity
    angle = float(torch.empty(1).uniform_(-max_angle, max_angle).item())
    return TF.rotate(images, angle=angle)


# Function dictionary: the single dispatch point for all drift families.
_DRIFT_FUNCS: Dict[str, Callable[[torch.Tensor, float], torch.Tensor]] = {
    "noise": _noise,
    "blur": _blur,
    "brightness": _brightness,
    "rotation": _rotation,
}


def apply_drift(
    images: torch.Tensor, drift_type: str, intensity: float
) -> torch.Tensor:
    """Apply a pixel-level drift to a batch of normalized image tensors.

    Parameters
    ----------
    images:
        Normalized batch tensor ``(N, C, H, W)`` in ``[-1, 1]``.
    drift_type:
        One of ``"noise"``, ``"blur"``, ``"brightness"`` or ``"rotation"``.
    intensity:
        Drift strength in ``[0, 0.5]`` (0 is a no-op passthrough).

    Returns
    -------
    torch.Tensor
        The drifted batch, clamped back to the valid ``[-1, 1]`` range.
    """
    key = drift_type.lower()
    if key not in _DRIFT_FUNCS:
        raise ValueError(
            f"Unsupported drift '{drift_type}'. "
            f"Choose one of {list(_DRIFT_FUNCS)}."
        )

    drifted = _DRIFT_FUNCS[key](images, intensity)
    return torch.clamp(drifted, _VMIN, _VMAX)
