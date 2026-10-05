# train_all.ps1
# Train all 4 DOA models in sequence
# Run from project root: .\train_all.ps1
# Estimated time: ~12-16 hours total on RTX 4090

$ErrorActionPreference = "Stop"
$ProjectRoot = "C:\Users\11404\Desktop\MD9120\DOA-CNN-TCA-ResNeXt"
Set-Location $ProjectRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " DOA-CNN-TCA-ResNeXt -- Training All 4 Models" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path "results/checkpoints" | Out-Null
New-Item -ItemType Directory -Force -Path "results/logs" | Out-Null

# Start TensorBoard in background (visit http://localhost:6006)
Write-Host "Starting TensorBoard at http://localhost:6006 ..." -ForegroundColor Cyan
Start-Process -NoNewWindow -FilePath "tensorboard" -ArgumentList "--logdir results/logs"

# ---- Train 4 models --------------------------------------------------------
$configs = @("raw_t16", "raw_t32", "cov_t16", "cov_t32")
$i = 1
foreach ($cfg in $configs) {
    Write-Host ""
    Write-Host "[$i/4] Training $cfg ..." -ForegroundColor Yellow
    $start = Get-Date
    python train/trainer.py --config configs/$cfg.yaml
    $elapsed = (Get-Date) - $start
    Write-Host "  Done in $([math]::Round($elapsed.TotalMinutes, 1)) min" -ForegroundColor Green
    $i++
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " All 4 models trained!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Get-ChildItem results/checkpoints/*_best.pth | ForEach-Object {
    $mb = [math]::Round($_.Length / 1MB, 1)
    Write-Host "  $($_.Name)  ${mb} MB"
}
