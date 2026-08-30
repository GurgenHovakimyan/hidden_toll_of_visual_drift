"""Profile the drift-evaluation pipeline: wall time, peak GPU and CPU memory
per model-dataset, using the existing checkpoints. Writes a LaTeX-ready table.
"""

from __future__ import annotations

import os
import time

import torch

import config
from data_loader import get_dataloaders
from evaluator import evaluate_concept_drift
from models import get_model

try:
    import psutil

    _PROC = psutil.Process(os.getpid())

    def rss_gb():
        return _PROC.memory_info().rss / 1024 ** 3
except Exception:  # psutil not installed
    def rss_gb():
        return float("nan")


def main():
    cfg = config.CONFIG
    config.set_seed(cfg.seed)
    rows = []
    for dataset_name in ["cifar10", "tiny_imagenet"]:
        _, val_loader, ds_cfg = get_dataloaders(dataset_name)
        for model_name in cfg.models:
            model = get_model(model_name, ds_cfg.num_classes)
            ckpt = os.path.join(
                cfg.paths.checkpoints, f"{dataset_name}_{model_name}.pth"
            )
            model.load_state_dict(torch.load(ckpt, map_location=cfg.device))
            model = model.to(cfg.device).eval()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            cpu0 = rss_gb()
            t0 = time.perf_counter()
            evaluate_concept_drift(
                model, val_loader, cfg,
                model_name=model_name, dataset_name=dataset_name,
            )
            dt = (time.perf_counter() - t0) / 60.0
            gpu = (
                torch.cuda.max_memory_allocated() / 1024 ** 3
                if torch.cuda.is_available() else float("nan")
            )
            cpu = max(rss_gb() - cpu0, 0.0)
            rows.append((model_name, dataset_name, dt, gpu, cpu))
            print(f"{model_name:14s} {dataset_name:14s} time={dt:5.2f}min "
                  f"gpu={gpu:5.2f}GB cpu={cpu:5.2f}GB")
            del model

    disp = {"resnet18": "ResNet-18", "densenet121": "DenseNet-121",
            "shufflenet_v2": "ShuffleNet v2"}
    dsp = {"cifar10": "CIFAR-10", "tiny_imagenet": "Tiny ImageNet"}
    with open("_profile.txt", "w") as f:
        for m, d, t, g, c in rows:
            f.write(f"{disp[m]} & {dsp[d]} & {t:.2f} & {g:.2f} & {c:.2f} \\\\\n")
    print("wrote _profile.txt")


if __name__ == "__main__":
    main()
