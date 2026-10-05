# run_eval_all.ps1   —— Evaluate all 4 trained models and generate Tables + Figures
$proj = (Get-Item "C:\Users\11404\Desktop\*MD9120\DOA-CNN-TCA-ResNeXt")[0].FullName
Set-Location $proj
$env:PYTHONPATH = $proj
$python = "python"

$models = @(
    @{ cfg = "configs/raw_t16.yaml"; ckpt = "results/checkpoints/raw_t16_best.pth" },
    @{ cfg = "configs/raw_t32.yaml"; ckpt = "results/checkpoints/raw_t32_best.pth" },
    @{ cfg = "configs/cov_t16.yaml"; ckpt = "results/checkpoints/cov_t16_best.pth" },
    @{ cfg = "configs/cov_t32.yaml"; ckpt = "results/checkpoints/cov_t32_best.pth" }
)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  DOA-CNN-TCA  Full Evaluation Pipeline" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

foreach ($m in $models) {
    Write-Host "`n--- Evaluating $($m.cfg) ---" -ForegroundColor Yellow
    & $python eval/metrics.py --config $m.cfg --checkpoint $m.ckpt --results results
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] metrics.py failed for $($m.cfg)" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n--- Generating Tables (CSV + LaTeX) ---" -ForegroundColor Yellow
& $python eval/metrics.py --config configs/raw_t16.yaml --results results --all
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] table generation failed" -ForegroundColor Red; exit 1 }

Write-Host "`n--- Generating Figures (Fig 4-8) ---" -ForegroundColor Yellow
& $python eval/visualize.py --all --results results
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] visualize.py failed" -ForegroundColor Red; exit 1 }

Write-Host "`n============================================" -ForegroundColor Green
Write-Host "  All evaluation complete!" -ForegroundColor Green
Write-Host "  Tables : results/tables/" -ForegroundColor Green
Write-Host "  Figures: results/figures/" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
