"""
evaluator.py
============
Unified concept-drift evaluation loop.

For a tractable subset of the test set, :func:`evaluate_concept_drift` sweeps
every ``(drift_type, intensity, xai_method)`` combination and, per combination:

1. applies the drift to the batch,
2. measures clean vs drifted accuracy,
3. generates Grad-CAM and Grad-CAM++ heatmaps for clean and drifted inputs,
4. computes SSIM / IoU / MSE between clean and drifted heatmaps,
5. runs KS / Anderson-Darling / Welch / Wilcoxon on the flattened heatmaps.

Clean heatmaps are computed **once per method per batch** and reused across all
drift types and intensities — the key optimization that keeps XAI generation
affordable. Results are returned as a tidy list of dicts (ready for a DataFrame).
"""

from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

import config
import metrics
from drift_utils import apply_drift
from models import get_target_layers
from xai_utils import generate_heatmaps


def _collect_subset(loader: DataLoader, num_batches: int, device: torch.device):
    """Materialise the first ``num_batches`` batches onto ``device``."""
    subset = []
    for i, (inputs, labels) in enumerate(loader):
        if i >= num_batches:
            break
        subset.append((inputs.to(device), labels.to(device)))
    return subset


def _predict(model: torch.nn.Module, inputs: torch.Tensor) -> np.ndarray:
    """Return predicted class indices for a batch as a NumPy array."""
    with torch.no_grad():
        preds = model(inputs).argmax(dim=1)
    return preds.cpu().numpy()


def evaluate_concept_drift(
    model: torch.nn.Module,
    test_loader: DataLoader,
    cfg: config._Config = config.CONFIG,
    model_name: str = "resnet18",
    dataset_name: str = "cifar10",
    num_subset_batches: int = 5,
) -> List[Dict]:
    """Sweep drift types x intensities x XAI methods and aggregate metrics.

    Parameters
    ----------
    model:
        A trained network (moved to ``cfg.device`` internally, set to eval).
    test_loader:
        Validation/test loader.
    cfg:
        Global config object.
    model_name:
        Architecture key, used to resolve Grad-CAM target layers.
    dataset_name:
        Dataset key, recorded in each result row.
    num_subset_batches:
        How many leading batches to evaluate (keeps XAI tractable).

    Returns
    -------
    list[dict]
        One row per ``(drift_type, intensity, xai_method)`` with aggregated
        accuracy, heatmap-similarity and statistical-test metrics.
    """
    device = cfg.device
    model = model.to(device).eval()
    target_layers = get_target_layers(model, model_name)

    subset = _collect_subset(test_loader, num_subset_batches, device)

    # --- Precompute clean predictions and clean heatmaps (once per method) --- #
    clean_preds = [_predict(model, inputs) for inputs, _ in subset]
    labels_np = [labels.cpu().numpy() for _, labels in subset]
    clean_acc = metrics.accuracy(
        np.concatenate(clean_preds), np.concatenate(labels_np)
    )

    clean_heatmaps: Dict[str, List[np.ndarray]] = {}
    for method in cfg.xai_methods:
        clean_heatmaps[method] = [
            generate_heatmaps(model, target_layers, inputs, method=method)
            for inputs, _ in subset
        ]

    results: List[Dict] = []

    for drift_type, intensity in itertools.product(
        cfg.drift_types, cfg.drift_intensities
    ):
        # Drift each subset batch once; reuse for every XAI method.
        drifted_inputs = [
            apply_drift(inputs, drift_type, intensity) for inputs, _ in subset
        ]
        drifted_preds = [_predict(model, d) for d in drifted_inputs]
        drifted_acc = metrics.accuracy(
            np.concatenate(drifted_preds), np.concatenate(labels_np)
        )

        per_method_ssim: Dict[str, np.ndarray] = {}
        for method in cfg.xai_methods:
            sim_batches: Dict[str, List[np.ndarray]] = {}
            clean_flat, drift_flat = [], []

            for b, (drifted_batch) in enumerate(drifted_inputs):
                clean_hm = clean_heatmaps[method][b]
                drift_hm = generate_heatmaps(
                    model, target_layers, drifted_batch, method=method
                )

                sim = metrics.compare_heatmap_batches(clean_hm, drift_hm)
                for k, v in sim.items():
                    sim_batches.setdefault(k, []).append(v)

                clean_flat.append(clean_hm.ravel())
                drift_flat.append(drift_hm.ravel())

            # Per-image arrays (statistical unit = image).
            sim_arrays = {
                k: np.concatenate(v) for k, v in sim_batches.items()
            }
            per_method_ssim[method] = sim_arrays["SSIM"]

            # Pixel-level, unpaired distributional diagnostic (legacy).
            clean_concat = np.concatenate(clean_flat)
            drift_concat = np.concatenate(drift_flat)
            stats = metrics.statistical_tests(clean_concat, drift_concat)

            # Image-level, paired analysis (primary) for SSIM and IoU.
            ssim_paired = metrics.paired_similarity_test(sim_arrays["SSIM"])
            iou_paired = metrics.paired_similarity_test(sim_arrays["IoU"])

            row = {
                "dataset": dataset_name,
                "model": model_name,
                "drift_type": drift_type,
                "intensity": intensity,
                "xai_method": method,
                "clean_acc": clean_acc,
                "drifted_acc": drifted_acc,
                "SSIM": float(np.mean(sim_arrays["SSIM"])),
                "IoU": float(np.mean(sim_arrays["IoU"])),
                "MSE": float(np.mean(sim_arrays["MSE"])),
                "Pearson": float(np.mean(sim_arrays["Pearson"])),
                "Cosine": float(np.mean(sim_arrays["Cosine"])),
                # IoU threshold-sensitivity analysis.
                "IoU@0.3": float(np.mean(sim_arrays["IoU@0.3"])),
                "IoU@0.5": float(np.mean(sim_arrays["IoU@0.5"])),
                "IoU@0.7": float(np.mean(sim_arrays["IoU@0.7"])),
                # Image-level paired SSIM: 95% CI, signed-rank p, effect size.
                "SSIM_ci_low": ssim_paired["ci_low"],
                "SSIM_ci_high": ssim_paired["ci_high"],
                "SSIM_wilcoxon_p": ssim_paired["wilcoxon_p"],
                "SSIM_paired_t_p": ssim_paired["t_p"],
                "SSIM_cohens_d": ssim_paired["cohens_d"],
                "IoU_ci_low": iou_paired["ci_low"],
                "IoU_ci_high": iou_paired["ci_high"],
                "IoU_wilcoxon_p": iou_paired["wilcoxon_p"],
                "IoU_cohens_d": iou_paired["cohens_d"],
                "n_images": ssim_paired["n"],
                # Legacy pixel-level, unpaired distributional tests.
                "KS_p": stats["KS"],
                "AD_p": stats["AD"],
                "Welch_p": stats["Welch_t"],
                "Wilcoxon_p": stats["Wilcoxon"],
            }
            results.append(row)

        # Direct paired Grad-CAM vs Grad-CAM++ comparison (reviewer R1):
        # do the two methods degrade differently under this drift?
        if "gradcam" in per_method_ssim and "gradcampp" in per_method_ssim:
            gc = per_method_ssim["gradcam"]
            gcpp = per_method_ssim["gradcampp"]
            n = min(gc.size, gcpp.size)
            delta = gcpp[:n] - gc[:n]  # per-image SSIM difference (++ minus base)
            if n == 0 or np.allclose(delta, 0.0):
                cmp_p, cmp_d = 1.0, 0.0
            else:
                try:
                    cmp_p = float(wilcoxon(delta).pvalue)
                except ValueError:
                    cmp_p = 1.0
                cmp_d = metrics.cohens_d_paired(delta)
            results.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "drift_type": drift_type,
                    "intensity": intensity,
                    "xai_method": "gradcam_vs_gradcampp",
                    "clean_acc": clean_acc,
                    "drifted_acc": drifted_acc,
                    "SSIM": float(np.mean(gcpp[:n]) - np.mean(gc[:n])),
                    "SSIM_wilcoxon_p": cmp_p,
                    "SSIM_cohens_d": cmp_d,
                    "n_images": int(n),
                }
            )

        print(
            f"[{dataset_name}/{model_name}] drift={drift_type} "
            f"intensity={intensity} drifted_acc={drifted_acc:.4f}"
        )

    return results
