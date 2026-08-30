"""
main.py
=======
Grand orchestrator for the "Hidden Toll of Visual Drift" study.

Nested sweep over datasets x models:

    for dataset in [cifar10, tiny_imagenet]:
        for model in [resnet18, densenet121, shufflenet_v2]:
            1. build data loaders + model
            2. train (saves checkpoint + learning-curve PNG)
            3. run the concept-drift evaluator
            4. persist per-model results incrementally

The run is **resumable**: re-launching after an interruption skips any model
whose per-model result CSV already exists, and reuses an existing checkpoint
(loading its weights instead of retraining). Partial results live in
``outputs/results/partial/`` and are concatenated into
``outputs/results/master_drift_results.csv`` at the end.

Run with:  ``python main.py``
"""

from __future__ import annotations

import glob
import os
import sys

# Make the repository root importable when run as `python scripts/run_experiments.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

from drift_study import config
from drift_study.data_loader import get_dataloaders
from drift_study.evaluator import evaluate_concept_drift
from drift_study.models import get_model
from drift_study.trainer import plot_learning_curves, train_model


def run_pipeline(cfg: config._Config = config.CONFIG) -> pd.DataFrame:
    """Execute the full dataset x model pipeline and return master results."""
    config.set_seed(cfg.seed)
    cfg.paths.create()

    partial_dir = os.path.join(cfg.paths.results, "partial")
    os.makedirs(partial_dir, exist_ok=True)

    datasets = ["cifar10", "tiny_imagenet"]

    for dataset_name in datasets:
        # Lazily build loaders only if this dataset still has work to do.
        pending = [
            m
            for m in cfg.models
            if not os.path.exists(
                os.path.join(partial_dir, f"{dataset_name}_{m}.csv")
            )
        ]
        if not pending:
            print(f"\n[skip] all models done for {dataset_name}.")
            continue

        train_loader, val_loader, ds_cfg = get_dataloaders(dataset_name)

        for model_name in cfg.models:
            run_name = f"{dataset_name}_{model_name}"
            partial_path = os.path.join(partial_dir, f"{run_name}.csv")

            # (a) already fully evaluated -> skip.
            if os.path.exists(partial_path):
                print(f"\n[skip] {run_name}: results already saved.")
                continue

            print("\n" + "=" * 70)
            print(f"RUN: {run_name}  (classes={ds_cfg.num_classes})")
            print("=" * 70)

            model = get_model(model_name, ds_cfg.num_classes)
            ckpt_path = os.path.join(cfg.paths.checkpoints, f"{run_name}.pth")

            # (b) reuse an existing checkpoint; otherwise train from scratch.
            if os.path.exists(ckpt_path):
                print(f"[resume] loading checkpoint -> {ckpt_path}")
                model.load_state_dict(
                    torch.load(ckpt_path, map_location=cfg.device)
                )
            else:
                history = train_model(
                    model, train_loader, val_loader, cfg, run_name=run_name
                )
                # (1) standard learning-curve image.
                curve_path = os.path.join(
                    cfg.paths.learning_curves,
                    f"{run_name}_learning_curve.png",
                )
                plot_learning_curves(
                    history, curve_path, title=f"{dataset_name} - {model_name}"
                )
                # (2) same curves with a red line marking the best epoch.
                curve_best_path = os.path.join(
                    cfg.paths.learning_curves,
                    f"{run_name}_learning_curve_best.png",
                )
                plot_learning_curves(
                    history,
                    curve_best_path,
                    title=f"{dataset_name} - {model_name} (best epoch)",
                    best_epoch=history.get("best_epoch"),
                )

            # (c) concept-drift evaluation -> persist immediately.
            rows = evaluate_concept_drift(
                model,
                val_loader,
                cfg,
                model_name=model_name,
                dataset_name=dataset_name,
            )
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            print(f"[saved] partial results -> {partial_path}")

    # Concatenate every per-model partial into the master CSV.
    partial_files = sorted(glob.glob(os.path.join(partial_dir, "*.csv")))
    master_df = pd.concat(
        (pd.read_csv(f) for f in partial_files), ignore_index=True
    )
    out_csv = os.path.join(cfg.paths.results, "master_drift_results.csv")
    master_df.to_csv(out_csv, index=False)
    print(f"\nSaved master results -> {out_csv}  ({len(master_df)} rows)")

    return master_df


if __name__ == "__main__":
    run_pipeline()
