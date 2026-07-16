"""
metrics.py
==========
Reusable evaluation and statistics for the drift study.

Two groups of helpers:

1. **Heatmap similarity** — compare a batch of clean heatmaps against the
   corresponding drifted heatmaps:
       * SSIM (structural similarity, via scikit-image)
       * IoU  (binary overlap at threshold 0.5)
       * MSE  (mean squared error)

2. **Statistical tests** — wrappers over ``scipy.stats`` that take two flattened
   arrays (baseline vs drifted heatmap distributions) and return p-values for
   KS, Anderson-Darling, Welch's t-test and Wilcoxon rank-sum.

All functions are pure (no global state) so they compose cleanly inside the
unified evaluation loop.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import anderson_ksamp, ks_2samp, ranksums, ttest_ind
from skimage.metrics import structural_similarity as ssim

import config


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    """Top-1 accuracy as a fraction in ``[0, 1]``."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    return float((preds == labels).mean())


# --------------------------------------------------------------------------- #
# Per-heatmap similarity
# --------------------------------------------------------------------------- #
def compute_ssim(h1: np.ndarray, h2: np.ndarray) -> float:
    """Structural similarity between two single-channel heatmaps in [0, 1]."""
    return float(ssim(h1, h2, data_range=1.0))


def compute_iou(
    h1: np.ndarray, h2: np.ndarray, threshold: float | None = None
) -> float:
    """IoU of the binary masks obtained by thresholding each heatmap."""
    threshold = (
        config.HEATMAP_BINARY_THRESHOLD if threshold is None else threshold
    )
    m1 = h1 > threshold
    m2 = h2 > threshold
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(intersection / union) if union > 0 else 0.0


def compute_mse(h1: np.ndarray, h2: np.ndarray) -> float:
    """Mean squared error between two heatmaps."""
    return float(np.mean((h1 - h2) ** 2))


def compare_heatmaps(h1: np.ndarray, h2: np.ndarray) -> Dict[str, float]:
    """All three similarity metrics for a single heatmap pair."""
    return {
        "SSIM": compute_ssim(h1, h2),
        "IoU": compute_iou(h1, h2),
        "MSE": compute_mse(h1, h2),
    }


def compare_heatmap_batches(
    clean: np.ndarray, drifted: np.ndarray
) -> Dict[str, np.ndarray]:
    """Vectorised per-image similarity for two batches of shape ``(N, H, W)``.

    Returns
    -------
    dict
        ``{"SSIM": array(N,), "IoU": array(N,), "MSE": array(N,)}``.
    """
    if clean.shape != drifted.shape:
        raise ValueError(
            f"Batch shape mismatch: {clean.shape} vs {drifted.shape}."
        )
    n = clean.shape[0]
    ssim_scores = np.empty(n, dtype=np.float64)
    iou_scores = np.empty(n, dtype=np.float64)
    mse_scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        ssim_scores[i] = compute_ssim(clean[i], drifted[i])
        iou_scores[i] = compute_iou(clean[i], drifted[i])
        mse_scores[i] = compute_mse(clean[i], drifted[i])
    return {"SSIM": ssim_scores, "IoU": iou_scores, "MSE": mse_scores}


# --------------------------------------------------------------------------- #
# Statistical tests on flattened heatmap distributions
# --------------------------------------------------------------------------- #
def statistical_tests(
    baseline: np.ndarray, drifted: np.ndarray
) -> Dict[str, float]:
    """Return p-values comparing two flattened heatmap distributions.

    Parameters
    ----------
    baseline, drifted:
        1-D (or flattenable) arrays of heatmap values.

    Returns
    -------
    dict
        p-values keyed by ``"KS"``, ``"AD"``, ``"Welch_t"`` and ``"Wilcoxon"``.
        Note: Anderson-Darling exposes a *significance level* (clipped to
        [0.001, 0.25] by SciPy), returned here as its p-value analogue.
    """
    a = np.asarray(baseline).ravel()
    b = np.asarray(drifted).ravel()

    ks_p = float(ks_2samp(a, b).pvalue)
    welch_p = float(ttest_ind(a, b, equal_var=False).pvalue)
    wilcoxon_p = float(ranksums(a, b).pvalue)

    # anderson_ksamp raises if the samples are identical; guard defensively.
    try:
        ad_p = float(anderson_ksamp([a, b]).significance_level)
    except (ValueError, OverflowError):
        ad_p = float("nan")

    return {
        "KS": ks_p,
        "AD": ad_p,
        "Welch_t": welch_p,
        "Wilcoxon": wilcoxon_p,
    }
