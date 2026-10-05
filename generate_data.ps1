# generate_data.ps1
# Full 8M dataset generation for DOA-CNN-TCA-ResNeXt
# Run from project root: .\generate_data.ps1
# Estimated time: 3-5 hours (CPU multiprocessing)

$ErrorActionPreference = "Stop"
$ProjectRoot = "C:\Users\11404\Desktop\MD9120\DOA-CNN-TCA-ResNeXt"
Set-Location $ProjectRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " DOA-CNN-TCA-ResNeXt -- Full Dataset Generation (8M total)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Create data directory
New-Item -ItemType Directory -Force -Path "data" | Out-Null

# ---- Raw T=16 (2M samples) ------------------------------------------------
Write-Host "[1/8] Raw T=16 Train (1.6M)..." -ForegroundColor Yellow
python datasets/generate_raw.py --T 16 --samples 1600000 --snr 10 --out data/raw_t16_train.h5

Write-Host "[2/8] Raw T=16 Val (200K)..." -ForegroundColor Yellow
python datasets/generate_raw.py --T 16 --samples 200000 --snr 10 --out data/raw_t16_val.h5

Write-Host "[3/8] Raw T=16 Test (200K)..." -ForegroundColor Yellow
python datasets/generate_raw.py --T 16 --samples 200000 --snr 10 --out data/raw_t16_test.h5

# ---- Raw T=32 (2M samples) ------------------------------------------------
Write-Host "[4/8] Raw T=32 Train (1.6M)..." -ForegroundColor Yellow
python datasets/generate_raw.py --T 32 --samples 1600000 --snr 10 --out data/raw_t32_train.h5

Write-Host "[5/8] Raw T=32 Val (200K)..." -ForegroundColor Yellow
python datasets/generate_raw.py --T 32 --samples 200000 --snr 10 --out data/raw_t32_val.h5

Write-Host "[6/8] Raw T=32 Test (200K)..." -ForegroundColor Yellow
python datasets/generate_raw.py --T 32 --samples 200000 --snr 10 --out data/raw_t32_test.h5

# ---- Cov T=16 (2M samples) ------------------------------------------------
Write-Host "[7/8] Cov T=16 Train (1.6M)..." -ForegroundColor Yellow
python datasets/generate_cov.py --T 16 --samples 1600000 --snr 10 --out data/cov_t16_train.h5

Write-Host "      Cov T=16 Val (200K)..." -ForegroundColor Yellow
python datasets/generate_cov.py --T 16 --samples 200000 --snr 10 --out data/cov_t16_val.h5

Write-Host "      Cov T=16 Test (200K)..." -ForegroundColor Yellow
python datasets/generate_cov.py --T 16 --samples 200000 --snr 10 --out data/cov_t16_test.h5

# ---- Cov T=32 (2M samples) ------------------------------------------------
Write-Host "[8/8] Cov T=32 Train (1.6M)..." -ForegroundColor Yellow
python datasets/generate_cov.py --T 32 --samples 1600000 --snr 10 --out data/cov_t32_train.h5

Write-Host "      Cov T=32 Val (200K)..." -ForegroundColor Yellow
python datasets/generate_cov.py --T 32 --samples 200000 --snr 10 --out data/cov_t32_val.h5

Write-Host "      Cov T=32 Test (200K)..." -ForegroundColor Yellow
python datasets/generate_cov.py --T 32 --samples 200000 --snr 10 --out data/cov_t32_test.h5

# ---- Summary ---------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Dataset generation complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Get-ChildItem data/*.h5 | ForEach-Object {
    $mb = [math]::Round($_.Length / 1MB, 1)
    Write-Host "  $($_.Name)  ${mb} MB"
}
