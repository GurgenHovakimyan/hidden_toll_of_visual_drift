# The Hidden Toll of Visual Drift

A modular PyTorch research pipeline that quantifies how **pixel-level concept
drift** (noise, blur, brightness, rotation) degrades both **predictive
performance** and **explainability heatmaps** (Grad-CAM, Grad-CAM++) across three
CNN architectures and two datasets.

For every combination of *dataset × model × drift type × drift intensity × XAI
method*, the pipeline measures the drop in accuracy, the similarity between clean
and drifted heatmaps (SSIM, IoU, MSE), and the statistical divergence of the
heatmap distributions (Kolmogorov–Smirnov, Anderson–Darling, Welch's *t*-test,
Wilcoxon rank-sum).

---

## Research questions

1. How much does classification **accuracy** degrade as each drift type
   intensifies from 0 → 0.5?
2. How **stable are the explanations**? Do Grad-CAM / Grad-CAM++ heatmaps for a
   drifted image still resemble the clean-image heatmaps?
3. Do these effects differ across **architectures** (ResNet-18, DenseNet-121,
   ShuffleNet V2) and **datasets** (CIFAR-10 vs. Tiny ImageNet)?

---

## Repository structure

```
hidden_toll_of_visual_drift/
├── config.py          # Centralised constants: datasets, models, drift grid, training, paths
├── data_loader.py     # Single factory: standardized loaders for CIFAR-10 / Tiny ImageNet (no augmentation)
├── models.py          # Model factory (ResNet-18 / DenseNet-121 / ShuffleNet V2) + Grad-CAM target layers
├── drift_utils.py     # Unified apply_drift(): noise / blur / brightness / rotation on tensor batches
├── xai_utils.py       # generate_heatmaps(): Grad-CAM & Grad-CAM++ for the predicted class
├── metrics.py         # Accuracy, SSIM/IoU/MSE, and KS/AD/Welch/Wilcoxon statistical tests
├── trainer.py         # Training loop w/ early stopping + best-epoch checkpoint + learning-curve plots
├── evaluator.py       # Unified drift-sweep evaluation loop (the heart of the study)
├── main.py            # Orchestrator: nested dataset × model sweep, resumable, writes master CSV
├── run_pipeline.ps1   # One-command launcher (fresh run or resume)
├── requirements.txt   # Python dependencies
└── outputs/           # Generated artefacts (checkpoints, learning_curves, results)
```

The codebase follows a strict **DRY** design: every constant lives in
`config.py`, and a single evaluation loop iterates over all models, datasets and
drift types without duplicating logic.

---

## Methodology

| Aspect | Setting |
|---|---|
| **Datasets** | CIFAR-10 (10 classes, 32×32) · Tiny ImageNet (200 classes, 64×64) |
| **Models** | ResNet-18 · DenseNet-121 · ShuffleNet V2 (x1.0) |
| **Backbones** | ImageNet-pretrained; classification head re-initialised per dataset |
| **Normalisation** | mean = std = 0.5 per channel, **no data augmentation** (pure baseline) |
| **Optimiser** | Adam, lr = 1e-3, batch size = 128 |
| **Training** | up to 30 epochs, **early stopping** (patience 7 on val-loss), **best epoch** checkpointed |
| **Drift types** | `noise` (additive Gaussian) · `blur` (Gaussian) · `brightness` (ColorJitter) · `rotation` (≤ 45°) |
| **Drift grid** | 12 intensities: 0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5 |
| **XAI** | Grad-CAM and Grad-CAM++ on the last convolutional block |
| **Heatmap similarity** | SSIM, IoU (binary @ 0.5), MSE |
| **Statistical tests** | KS (`ks_2samp`), Anderson–Darling (`anderson_ksamp`), Welch's *t* (`ttest_ind`, unequal var), Wilcoxon rank-sum (`ranksums`) |

---

## Setup

```powershell
# 1. Create / activate an environment and install dependencies
pip install -r requirements.txt

# 2. Provide the datasets locally (NOT tracked in git):
#    ./data/cifar-10-batches-py/      (CIFAR-10; auto-downloads if missing)
#    ./tiny-imagenet-200/train/       (ImageFolder layout)
#    ./tiny-imagenet-200/val/         (arranged into class sub-folders)
```

### Running the pipeline

```powershell
# Fresh run — clears ./outputs and trains + evaluates all 6 models:
.\run_pipeline.ps1 -Fresh

# Resume after an interruption (skips finished models, reuses checkpoints):
.\run_pipeline.ps1
```

Or directly:

```powershell
python main.py
```

The run is **resumable**: each model's results are written to
`outputs/results/partial/` as it completes, existing checkpoints are reused
instead of retraining, and all partials are merged into the master CSV at the
end. This protects long runs against interruptions.

---

## Outputs

```
outputs/
├── checkpoints/                          # best-epoch weights, one .pth per model
├── learning_curves/
│   ├── <run>_learning_curve.png          # train/val loss & accuracy
│   └── <run>_learning_curve_best.png     # same, with a red line at the best epoch
└── results/
    ├── partial/<run>.csv                 # per-model drift results
    └── master_drift_results.csv          # merged table (576 rows)
```

The **master CSV** has one row per `dataset × model × drift_type × intensity ×
xai_method` with columns: `clean_acc, drifted_acc, SSIM, IoU, MSE, KS_p, AD_p,
Welch_p, Wilcoxon_p`.

---

## Results summary

**Baseline (clean) validation accuracy** — best-epoch models:

| Dataset | ResNet-18 | DenseNet-121 | ShuffleNet V2 |
|---|---|---|---|
| CIFAR-10 | 0.841 | **0.859** | 0.794 |
| Tiny ImageNet | 0.402 | **0.598** | 0.537 |

**Accuracy at maximum drift (intensity = 0.5)** — how far accuracy falls from the
clean baseline for each drift family:

| Dataset · Model | Noise | Blur | Brightness | Rotation |
|---|---|---|---|---|
| CIFAR-10 · ResNet-18 | 0.264 | 0.269 | 0.792 | 0.739 |
| CIFAR-10 · DenseNet-121 | 0.239 | 0.370 | 0.823 | 0.730 |
| CIFAR-10 · ShuffleNet V2 | 0.244 | 0.295 | 0.742 | 0.700 |
| Tiny ImageNet · ResNet-18 | 0.009 | 0.069 | 0.331 | 0.362 |
| Tiny ImageNet · DenseNet-121 | 0.031 | 0.111 | 0.525 | 0.464 |
| Tiny ImageNet · ShuffleNet V2 | 0.003 | 0.075 | 0.384 | 0.450 |

**Heatmap similarity at intensity = 0.5** (averaged over drift types, Tiny
ImageNet):

| Model | Method | SSIM | IoU | MSE |
|---|---|---|---|---|
| ResNet-18 | Grad-CAM | 0.492 | 0.530 | 0.139 |
| ResNet-18 | Grad-CAM++ | 0.578 | 0.595 | 0.107 |
| DenseNet-121 | Grad-CAM | 0.439 | 0.483 | 0.155 |
| DenseNet-121 | Grad-CAM++ | 0.500 | 0.527 | 0.132 |
| ShuffleNet V2 | Grad-CAM | 0.550 | 0.130 | 0.141 |
| ShuffleNet V2 | Grad-CAM++ | 0.613 | 0.614 | 0.092 |

### Key findings

- **Noise is the most destructive drift**: at intensity 0.5 it collapses accuracy
  to near-chance, especially on Tiny ImageNet (ResNet-18 → 0.009, ShuffleNet →
  0.003). **Blur** is the next most harmful.
- **Brightness and rotation are comparatively benign** — models retain most of
  their accuracy even at the strongest setting.
- **Explanations drift too**: on Tiny ImageNet, heatmap SSIM/IoU drop
  substantially under strong drift, i.e. the model not only misclassifies but
  also "looks" at different regions. Grad-CAM++ is consistently **more stable**
  (higher SSIM/IoU, lower MSE) than Grad-CAM.
- **Tiny ImageNet degrades far more than CIFAR-10** for the same drift intensity,
  reflecting its harder 200-class, higher-resolution setting.

### Note / limitation

For **CIFAR-10** the last-block Grad-CAM maps are effectively degenerate
(SSIM ≈ 1.0, IoU ≈ 0, MSE ≈ 0) because a 32×32 input collapses to a ~1×1 spatial
feature map at the final convolutional block, so there is almost no spatial
structure to compare. The heatmap-similarity analysis is therefore only
meaningful on **Tiny ImageNet** (64×64). Choosing an earlier target layer would
be required to study CAM stability on CIFAR-10.

---

## Requirements

`torch`, `torchvision`, `grad-cam`, `scikit-image`, `scikit-learn`, `scipy`,
`numpy`, `pandas`, `matplotlib` (see `requirements.txt`). A CUDA GPU is strongly
recommended.

## License

See [LICENSE](LICENSE).
