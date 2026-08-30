"""
trainer.py
==========
Training loop and learning-curve reporting.

Two public functions:

* :func:`train_model`  — trains a network for ``config.TRAIN.epochs`` epochs,
  tracking train/val loss and train/val accuracy, saves the best/final weights
  to ``outputs/checkpoints/`` and returns a ``history`` dict.
* :func:`plot_learning_curves` — renders a 1x2 (loss | accuracy) figure and
  writes a high-resolution PNG to ``outputs/learning_curves/`` to document
  convergence for reviewers.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for saving PNGs
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from . import config


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one pass over ``loader``. Trains when ``optimizer`` is given.

    Returns
    -------
    tuple(float, float)
        ``(average_loss, accuracy)`` for the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: config._Config = config.CONFIG,
    run_name: str = "model",
) -> Dict[str, List[float]]:
    """Train ``model`` and return a history of the four tracked metrics.

    Parameters
    ----------
    model:
        Network to train (moved to ``cfg.device`` internally).
    train_loader, val_loader:
        Data loaders from :func:`data_loader.get_dataloaders`.
    cfg:
        Global config object (defaults to :data:`config.CONFIG`).
    run_name:
        Identifier used for the checkpoint filename, e.g. ``"cifar10_resnet18"``.

    Returns
    -------
    dict
        ``{"train_loss": [...], "val_loss": [...],
           "train_acc": [...], "val_acc": [...]}``.
    """
    device = cfg.device
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    cfg.paths.create()
    ckpt_path = os.path.join(cfg.paths.checkpoints, f"{run_name}.pth")

    # Early-stopping / best-epoch tracking.
    monitor = cfg.train.monitor
    minimise = monitor == "val_loss"
    best_metric = float("inf") if minimise else float("-inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(cfg.train.epochs):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"[{run_name}] Epoch {epoch + 1}/{cfg.train.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Did the monitored metric improve this epoch?
        current = val_loss if minimise else val_acc
        improved = current < best_metric if minimise else current > best_metric
        if improved:
            best_metric = current
            best_epoch = epoch + 1
            epochs_no_improve = 0
            # Save the BEST epoch's weights (not the last).
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.train.early_stop_patience:
                print(
                    f"[{run_name}] Early stopping at epoch {epoch + 1} "
                    f"(no {monitor} improvement for "
                    f"{cfg.train.early_stop_patience} epochs)."
                )
                break

    print(
        f"[{run_name}] Best epoch = {best_epoch} "
        f"({monitor}={best_metric:.4f}) -> {ckpt_path}"
    )
    # Record the best epoch so learning curves can highlight it.
    history["best_epoch"] = best_epoch
    # Restore the best-epoch weights so downstream evaluation uses them.
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    return history


def plot_learning_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Learning Curves",
    best_epoch: int | None = None,
) -> None:
    """Plot loss and accuracy curves side-by-side and save a high-res PNG.

    Parameters
    ----------
    history:
        The dict returned by :func:`train_model`.
    save_path:
        Full path (including ``.png``) to write the figure to.
    title:
        Figure super-title, typically ``"<dataset> - <model>"``.
    best_epoch:
        If given, a red vertical line is drawn at this epoch on both subplots
        to highlight the checkpointed best epoch.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=15)

    # Plot 1: loss
    ax_loss.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    ax_loss.plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    if best_epoch is not None:
        ax_loss.axvline(
            best_epoch, color="red", linestyle="--", linewidth=1.5,
            label=f"Best epoch ({best_epoch})",
        )
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Plot 2: accuracy
    ax_acc.plot(epochs, history["train_acc"], marker="o", label="Train Acc")
    ax_acc.plot(epochs, history["val_acc"], marker="o", label="Val Acc")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    if best_epoch is not None:
        ax_acc.axvline(
            best_epoch, color="red", linestyle="--", linewidth=1.5,
            label=f"Best epoch ({best_epoch})",
        )
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved learning curves -> {save_path}")
