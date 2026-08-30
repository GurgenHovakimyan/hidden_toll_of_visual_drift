"""drift_study — library for the "Hidden Toll of Visual Drift" study.

Modules:
    config       Centralized experiment configuration.
    data_loader  CIFAR-10 / Tiny ImageNet data loaders.
    drift_utils  Synthetic covariate-shift (drift) operators.
    models       Model factory and Grad-CAM target layers.
    trainer      Training loop and learning-curve plotting.
    xai_utils    Grad-CAM / Grad-CAM++ heatmap generation.
    metrics      Heatmap similarity metrics and statistical tests.
    evaluator    End-to-end concept-drift evaluation.
"""
