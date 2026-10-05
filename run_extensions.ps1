# run_extensions.ps1  —— Execute all 4 extension experiments in order
# Usage: cd to project root and run: .\extensions\run_extensions.ps1

$proj = (Get-Item "C:\Users\11404\Desktop\*MD9120\DOA-CNN-TCA-ResNeXt")[0].FullName
Set-Location $proj
$env:PYTHONPATH = $proj

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  DOA-CNN Extension Experiments Pipeline" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# --- EX4: CRB Analysis (~1 min) ---
Write-Host "`n[EX4] CRB vs RMSE Analysis..." -ForegroundColor Yellow
python extensions/ex4_crb_analysis.py
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] EX4 failed" -ForegroundColor Red; exit 1 }

# --- EX2: Threshold Optimization (~5 min) ---
Write-Host "`n[EX2] Threshold Optimization..." -ForegroundColor Yellow
python extensions/ex2_threshold_opt.py
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] EX2 failed" -ForegroundColor Red; exit 1 }

# --- EX3: Array Perturbation (~1 hr) ---
Write-Host "`n[EX3] Array Perturbation Robustness..." -ForegroundColor Yellow
python extensions/ex3_perturbation.py
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] EX3 failed" -ForegroundColor Red; exit 1 }

Write-Host "`n==============================================" -ForegroundColor Green
Write-Host "  EX2 + EX3 + EX4 Complete!" -ForegroundColor Green
Write-Host "  Figures: results/extensions/" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run Focal Loss training (overnight, ~20h):" -ForegroundColor Magenta
Write-Host "  python extensions/ex1_focal_loss/train_focal.py" -ForegroundColor Magenta
