"""
metrics.py
==========
Reusable evaluation and statistics for the drift study.

Two groups of helpers:

1. **Heatmap similarity** — compare a batch of clean heatmaps against the
   corresponding drifted heatmaps:
       * SSIM (structural similarity, via scikit-image)
       * IoU  (binary overlap, evaluated at several thresholds)
       * MSE  (mean squared error)
       * Pearson correlation and cosine similarity (continuous, mask-free
         alternatives to IoU requested in review)

2. **Statistical analysis** —
       * ``statistical_tests``: legacy *pixel-level, unpaired* distributional
         diagnostic (KS, Anderson-Darling, Welch t, Wilcoxon rank-sum).
       * ``paired_similarity_test``: primary *image-level, paired* test
         (Wilcoxon signed-rank + paired t) with a paired Cohen's d effect size
         and a bootstrap confidence interval on the mean similarity.

All functions are pure (no global state) so they compose cleanly inside the
unified evaluation loop.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import (
    anderson_ksamp,
    ks_2samp,
    pearsonr,
    ranksums,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)
from skimage.metrics import structural_similarity as ssim

import config

# Thresholds used for the IoU threshold-sensitivity analysis (reviewer R1).
IOU_THRESHOLDS: Sequence[float] = (0.3, 0.5, 0.7)


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


def compute_pearson(h1: np.ndarray, h2: np.ndarray) -> float:
    """Pearson correlation between two heatmaps treated as continuous fields.

    Continuous alternative to the thresholded IoU (reviewer R1): operates on the
    raw attribution values rather than a binary mask. Returns 0.0 when either
    map is constant (correlation undefined).
    """
    a = h1.ravel().astype(np.float64)
    b = h2.ravel().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(pearsonr(a, b)[0])


def compute_cosine(h1: np.ndarray, h2: np.ndarray) -> float:
    """Cosine similarity between two heatmaps flattened into vectors."""
    a = h1.ravel().astype(np.float64)
    b = h2.ravel().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def compute_iou_multi(
    h1: np.ndarray, h2: np.ndarray, thresholds: Sequence[float] = IOU_THRESHOLDS
) -> Dict[float, float]:
    """IoU at several thresholds for the threshold-sensitivity analysis."""
    return {t: compute_iou(h1, h2, threshold=t) for t in thresholds}


def compare_heatmaps(h1: np.ndarray, h2: np.ndarray) -> Dict[str, float]:
    """All similarity metrics for a single heatmap pair."""
    return {
        "SSIM": compute_ssim(h1, h2),
        "IoU": compute_iou(h1, h2),
        "MSE": compute_mse(h1, h2),
        "Pearson": compute_pearson(h1, h2),
        "Cosine": compute_cosine(h1, h2),
    }


def compare_heatmap_batches(
    clean: np.ndarray, drifted: np.ndarray
) -> Dict[str, np.ndarray]:
    """Vectorised per-image similarity for two batches of shape ``(N, H, W)``.

    Returns
    -------
    dict
        Per-image arrays of shape ``(N,)`` keyed by ``"SSIM"``, ``"IoU"``,
        ``"MSE"``, ``"Pearson"``, ``"Cosine"`` and one ``"IoU@t"`` entry per
        threshold in :data:`IOU_THRESHOLDS` (threshold-sensitivity analysis).
    """
    if clean.shape != drifted.shape:
        raise ValueError(
            f"Batch shape mismatch: {clean.shape} vs {drifted.shape}."
        )
    n = clean.shape[0]
    ssim_scores = np.empty(n, dtype=np.float64)
    iou_scores = np.empty(n, dtype=np.float64)
    mse_scores = np.empty(n, dtype=np.float64)
    pearson_scores = np.empty(n, dtype=np.float64)
    cosine_scores = np.empty(n, dtype=np.float64)
    iou_multi = {t: np.empty(n, dtype=np.float64) for t in IOU_THRESHOLDS}
    for i in range(n):
        ssim_scores[i] = compute_ssim(clean[i], drifted[i])
        iou_scores[i] = compute_iou(clean[i], drifted[i])
        mse_scores[i] = compute_mse(clean[i], drifted[i])
        pearson_scores[i] = compute_pearson(clean[i], drifted[i])
        cosine_scores[i] = compute_cosine(clean[i], drifted[i])
        for t in IOU_THRESHOLDS:
            iou_multi[t][i] = compute_iou(clean[i], drifted[i], threshold=t)
    out = {
        "SSIM": ssim_scores,
        "IoU": iou_scores,
        "MSE": mse_scores,
        "Pearson": pearson_scores,
        "Cosine": cosine_scores,
    }
    out.update({f"IoU@{t}": iou_multi[t] for t in IOU_THRESHOLDS})
    return out


# --------------------------------------------------------------------------- #
# Statistical tests on flattened heatmap distributions (pixel-level, unpaired)
# --------------------------------------------------------------------------- #
def statistical_tests(
    baseline: np.ndarray, drifted: np.ndarray
) -> Dict[str, float]:
    """Return p-values comparing two flattened heatmap distributions.

    NOTE ON THE STATISTICAL UNIT (reviewer R1): these four tests treat every
    *pixel* pooled across images as an independent observation and compare the
    clean vs drifted marginal distributions in an **unpaired** fashion. With
    ~10^5-10^6 spatially autocorrelated pixels the p-values are inevitably
    tiny, so this routine is retained only as a *distributional* diagnostic.
    The primary, correctly-powered analysis is the image-level *paired* test
    in :func:`paired_similarity_test` (unit = image, N = number of images).

    Parameters
    ----------
    baseline, drifted:
        1-D (or flattenable) arrays of heatmap values.

    Returns
    -------
    dict
        p-values keyed by ``"KS"``, ``"AD"``, ``"Welch_t"`` and ``"Wilcoxon"``.
        ``"Wilcoxon"`` here is the rank-sum (Mann-Whitney) test, i.e. unpaired.
        Anderson-Darling exposes a *significance level* (clipped to
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


# --------------------------------------------------------------------------- #
# Image-level paired analysis (the primary, correctly-powered design)
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> "tuple[float, float]":
    """Percentile bootstrap CI for the mean of a 1-D sample.

    The statistical unit is the *image*: ``values`` holds one number per image
    (e.g. the per-image SSIM between its clean and drifted heatmap).
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot_means = v[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def cohens_d_paired(diff: np.ndarray) -> float:
    """Paired Cohen's d = mean(diff) / sd(diff) for per-image differences."""
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size (non-parametric) for two independent samples.

    Ranges in [-1, 1]; |delta| ~ 0.11/0.28/0.43 = small/medium/large.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    # P(x>y) - P(x<y) via rank comparison (sub-sampled for very large inputs).
    cap = 4000
    if x.size > cap:
        x = np.random.default_rng(0).choice(x, cap, replace=False)
    if y.size > cap:
        y = np.random.default_rng(1).choice(y, cap, replace=False)
    diff = np.subtract.outer(x, y)
    return float((np.sign(diff).sum()) / (x.size * y.size))


def paired_similarity_test(
    sim_per_image: np.ndarray,
    reference: float = 1.0,
) -> Dict[str, float]:
    """Image-level paired test of heatmap change under drift.

    Each image is its own clean-vs-drifted pair, summarised by a per-image
    similarity ``sim_per_image`` (e.g. SSIM in [0, 1], where ``reference`` = the
    identical-map value). We test H0: the per-image dissimilarity
    ``d_i = reference - sim_i`` has zero median, using the **Wilcoxon
    signed-rank** test (paired), complemented by a paired t-test, a paired
    Cohen's d effect size, and a bootstrap CI for the mean similarity.

    Returns
    -------
    dict
        ``mean``, ``ci_low``, ``ci_high`` (mean similarity + 95% CI),
        ``wilcoxon_p`` (signed-rank), ``t_p`` (paired t), ``cohens_d`` (paired,
        on the dissimilarity), and ``n`` (number of images).
    """
    s = np.asarray(sim_per_image, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = int(s.size)
    mean = float(s.mean()) if n else float("nan")
    ci_low, ci_high = bootstrap_ci(s)
    diff = reference - s  # per-image dissimilarity

    # Wilcoxon signed-rank is undefined when every difference is zero.
    if n == 0 or np.allclose(diff, 0.0):
        wil_p, t_p, d = 1.0, 1.0, 0.0
    else:
        try:
            wil_p = float(wilcoxon(diff).pvalue)
        except ValueError:
            wil_p = 1.0
        t_p = float(ttest_rel(s, np.full_like(s, reference)).pvalue)
        d = cohens_d_paired(diff)

    return {
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilcoxon_p": wil_p,
        "t_p": t_p,
        "cohens_d": d,
        "n": n,
    }
