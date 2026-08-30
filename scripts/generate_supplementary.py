"""
generate_supplementary.py
=========================
Render the reviewer-requested supplementary LaTeX tables directly from
``outputs/results/master_drift_results.csv`` (produced by ``scripts/run_experiments.py``).

It emits ``supplementary_tables.tex`` containing, for every dataset/model:

* the full accuracy / SSIM (+95% CI) / IoU / Pearson / cosine table across the
  twelve drift intensities (extends the ResNet-18/CIFAR-10 tables in the paper
  to all architectures and to Tiny ImageNet);
* a Grad-CAM vs Grad-CAM++ paired comparison (Wilcoxon signed-rank p-value and
  paired Cohen's d) at the strongest intensity of each drift type;
* an IoU threshold-sensitivity table (IoU at 0.3 / 0.5 / 0.7).

The script degrades gracefully: any column that is absent in the CSV is simply
skipped, so it works with both the legacy and the enhanced result schema.

Usage:  python scripts/generate_supplementary.py
"""

from __future__ import annotations

import os

import pandas as pd

RESULTS_CSV = os.path.join("outputs", "results", "master_drift_results.csv")
OUT_TEX = "supplementary_tables.tex"

PRIMARY_METHOD = "gradcam"
COMPARE_TAG = "gradcam_vs_gradcampp"


def _fmt(x, pct=False, dec=2):
    if pd.isna(x):
        return "--"
    return f"{x * 100:.{dec}f}" if pct else f"{x:.{dec}f}"


def _has(df: pd.DataFrame, *cols: str) -> bool:
    return all(c in df.columns for c in cols)


def full_results_table(df: pd.DataFrame, dataset: str, model: str) -> str:
    sub = df[
        (df["dataset"] == dataset)
        & (df["model"] == model)
        & (df["xai_method"] == PRIMARY_METHOD)
    ].sort_values(["drift_type", "intensity"])
    if sub.empty:
        return ""

    has_ci = _has(sub, "SSIM_ci_low", "SSIM_ci_high")
    has_cont = _has(sub, "Pearson", "Cosine")
    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        f"\\caption{{Full drift results for {model} on {dataset} "
        "(Grad-CAM). SSIM shown with bootstrap 95\\% CI.}",
        f"\\label{{tab:supp_{dataset}_{model}}}",
    ]
    ncol = 4 + (1 if has_cont else 0) + (1 if has_cont else 0)
    header = "Drift & $\\gamma$ & Acc.(\\%) & SSIM(\\%)"
    if has_ci:
        header += " [95\\% CI]"
    header += " & IoU(\\%)"
    if has_cont:
        header += " & Pearson & Cosine"
    lines.append("\\begin{tabular}{|" + "c|" * (ncol + 1) + "}")
    lines.append("\\hline")
    lines.append(header + " \\\\")
    lines.append("\\hline")
    for _, r in sub.iterrows():
        ssim = _fmt(r.get("SSIM"), pct=True)
        if has_ci and not pd.isna(r.get("SSIM_ci_low")):
            ssim += f" [{_fmt(r['SSIM_ci_low'], pct=True)}, {_fmt(r['SSIM_ci_high'], pct=True)}]"
        row = (
            f"{r['drift_type']} & {r['intensity']:g} & "
            f"{_fmt(r.get('drifted_acc'), pct=True)} & {ssim} & "
            f"{_fmt(r.get('IoU'), pct=True)}"
        )
        if has_cont:
            row += f" & {_fmt(r.get('Pearson'))} & {_fmt(r.get('Cosine'))}"
        lines.append(row + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def gradcam_compare_table(df: pd.DataFrame) -> str:
    sub = df[df["xai_method"] == COMPARE_TAG]
    if sub.empty or not _has(sub, "SSIM_wilcoxon_p", "SSIM_cohens_d"):
        return ""
    strongest = sub.sort_values("intensity").groupby(
        ["dataset", "model", "drift_type"], as_index=False
    ).last()
    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Grad-CAM vs Grad-CAM++ paired comparison at the strongest "
        "intensity per drift type: Wilcoxon signed-rank $p$ on the per-image "
        "SSIM difference and paired Cohen's $d$.}",
        "\\label{tab:supp_gradcam_compare}",
        "\\begin{tabular}{|c|c|c|c|c|c|}",
        "\\hline",
        "Dataset & Model & Drift & $\\gamma$ & Wilcoxon $p$ & Cohen's $d$ \\\\",
        "\\hline",
    ]
    for _, r in strongest.iterrows():
        lines.append(
            f"{r['dataset']} & {r['model']} & {r['drift_type']} & "
            f"{r['intensity']:g} & {r['SSIM_wilcoxon_p']:.2e} & "
            f"{_fmt(r['SSIM_cohens_d'])} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def iou_threshold_table(df: pd.DataFrame) -> str:
    cols = ["IoU@0.3", "IoU@0.5", "IoU@0.7"]
    sub = df[df["xai_method"] == PRIMARY_METHOD]
    if sub.empty or not _has(sub, *cols):
        return ""
    strongest = sub.sort_values("intensity").groupby(
        ["dataset", "model", "drift_type"], as_index=False
    ).last()
    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{IoU threshold-sensitivity at the strongest intensity per "
        "drift type: overlap recomputed at thresholds 0.3, 0.5 and 0.7.}",
        "\\label{tab:supp_iou_threshold}",
        "\\begin{tabular}{|c|c|c|c|c|c|}",
        "\\hline",
        "Dataset & Model & Drift & IoU@0.3 & IoU@0.5 & IoU@0.7 \\\\",
        "\\hline",
    ]
    for _, r in strongest.iterrows():
        lines.append(
            f"{r['dataset']} & {r['model']} & {r['drift_type']} & "
            f"{_fmt(r['IoU@0.3'], pct=True)} & {_fmt(r['IoU@0.5'], pct=True)} & "
            f"{_fmt(r['IoU@0.7'], pct=True)} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    if not os.path.exists(RESULTS_CSV):
        raise SystemExit(
            f"Results file not found: {RESULTS_CSV}. Run scripts/run_experiments.py first."
        )
    df = pd.read_csv(RESULTS_CSV)

    blocks = ["% Auto-generated by generate_supplementary.py. Do not edit by hand.", ""]
    for dataset in sorted(df["dataset"].unique()):
        for model in sorted(df[df["dataset"] == dataset]["model"].unique()):
            blocks.append(full_results_table(df, dataset, model))
    blocks.append(gradcam_compare_table(df))
    blocks.append(iou_threshold_table(df))

    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(b for b in blocks if b))
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
