<#
    run_pipeline.ps1
    ================
    Convenience launcher for the "Hidden Toll of Visual Drift" pipeline.

    USAGE (from any location):

      # Morning fresh start (clears old outputs, then runs everything):
      .\run_pipeline.ps1 -Fresh

      # Resume after an interruption / shutdown (keeps finished work):
      .\run_pipeline.ps1

    What it does
    ------------
    * Activates the `sadr` conda environment for the run.
    * -Fresh  : deletes ./outputs and the old log so the run starts clean
                (use this once in the morning since the training settings
                changed to 30 epochs + best-epoch checkpointing).
    * default : leaves ./outputs in place so main.py RESUMES — it skips any
                model whose results are already saved and reuses checkpoints.
    * Streams all output to .\outputs_run.log (and the console).

    Training behaviour (configured in config.py / trainer.py)
    ---------------------------------------------------------
    * epochs capped at 30, Adam lr=1e-3, pretrained backbones.
    * Early stopping: patience 7 on val_loss.
    * The BEST epoch (lowest val_loss) is saved as the checkpoint and used
      for the drift/XAI evaluation - not the last (overfit) epoch.
#>

param(
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"

# Move to the folder this script lives in.
Set-Location -Path $PSScriptRoot

# Stop any stray python processes from a previous run.
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

if ($Fresh) {
    Write-Host "[run_pipeline] -Fresh: clearing previous outputs..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".\outputs" -ErrorAction SilentlyContinue
    Remove-Item -Force ".\outputs_run.log" -ErrorAction SilentlyContinue
}
else {
    Write-Host "[run_pipeline] Resume mode: keeping existing outputs." -ForegroundColor Cyan
}

Write-Host "[run_pipeline] Launching main.py in conda env 'sadr'..." -ForegroundColor Green
conda run -n sadr --no-capture-output python -u main.py *> ".\outputs_run.log"

Write-Host "[run_pipeline] Finished. See .\outputs_run.log and .\outputs\results\master_drift_results.csv" -ForegroundColor Green
