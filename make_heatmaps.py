"""Regenerate the class-averaged Grad-CAM heatmap figures (Class{N}Heatmap.png).

Matches the manuscript layout: a 2x3 grid per class showing the class-averaged
heatmap for Train, Test (no drift), and Noise/Brightness/Rotation/Blur at the
strongest intensity (0.5), using the fixed penultimate-layer Grad-CAM and the
existing (ImageNet-pretrained) checkpoints on CIFAR-10.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from data_loader import get_dataloaders
from drift_utils import apply_drift
from models import get_model, get_target_layers
from xai_utils import generate_heatmaps

# (panel title, split, (drift_type, intensity) or None)
CONDITIONS = [
    ("Train Data Heatmaps", "train", None),
    ("Test Data (no drift)", "test", None),
    ("Noise Drift (0.5)", "test", ("noise", 0.5)),
    ("Brightness Drift (0.5)", "test", ("brightness", 0.5)),
    ("Rotation Drift (0.5)", "test", ("rotation", 0.5)),
    ("Blur Drift (0.5)", "test", ("blur", 0.5)),
]

# Same representative classes the manuscript figures use.
MODEL_CLASSES = {
    "resnet18": [8, 9],
    "densenet121": [0, 4],
    "shufflenet_v2": [2, 7],
}

NUM_BATCHES = 12  # subset used for class-averaging (12 * 128 = 1536 images)


def collect(loader, num_batches, device):
    subset = []
    for i, (x, y) in enumerate(loader):
        if i >= num_batches:
            break
        subset.append((x.to(device), y.to(device)))
    return subset


def class_average(model, target_layers, subset, drift, num_classes):
    sums = [None] * num_classes
    counts = [0] * num_classes
    for x, _ in subset:
        xin = apply_drift(x, drift[0], drift[1]) if drift else x
        with torch.no_grad():
            preds = model(xin).argmax(1).cpu().numpy()
        hm = generate_heatmaps(model, target_layers, xin, method="gradcam")
        for i, c in enumerate(preds):
            if sums[c] is None:
                sums[c] = np.zeros_like(hm[i], dtype=np.float64)
            sums[c] += hm[i]
            counts[c] += 1
    return [
        (sums[c] / counts[c]) if counts[c] > 0 else None
        for c in range(num_classes)
    ]


def main():
    cfg = config.CONFIG
    config.set_seed(cfg.seed)
    train_loader, val_loader, ds_cfg = get_dataloaders("cifar10")
    train_sub = collect(train_loader, NUM_BATCHES, cfg.device)
    test_sub = collect(val_loader, NUM_BATCHES, cfg.device)

    for model_name, classes in MODEL_CLASSES.items():
        model = get_model(model_name, ds_cfg.num_classes)
        ckpt = os.path.join(cfg.paths.checkpoints, f"cifar10_{model_name}.pth")
        model.load_state_dict(torch.load(ckpt, map_location=cfg.device))
        model = model.to(cfg.device).eval()
        target_layers = get_target_layers(model, model_name)

        cond_avgs = []
        for title, split, drift in CONDITIONS:
            sub = train_sub if split == "train" else test_sub
            cond_avgs.append(
                (title, class_average(model, target_layers, sub, drift, ds_cfg.num_classes))
            )

        for cls in classes:
            fig, axs = plt.subplots(2, 3, figsize=(8.5, 6))
            for ax, (title, avg) in zip(axs.flat, cond_avgs):
                h = avg[cls]
                if h is not None:
                    hn = (h - h.min()) / (h.max() - h.min() + 1e-8)
                    ax.imshow(hn, cmap="hot")
                ax.set_title(f"{title}\nClass {cls}", fontsize=10)
                ax.axis("off")
            fig.tight_layout()
            out = f"Class{cls}Heatmap.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {out}  (model={model_name})")


if __name__ == "__main__":
    main()
