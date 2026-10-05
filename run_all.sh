#!/usr/bin/env bash
# run_all.sh
# One-shot pipeline: generate data -> train 4 models -> evaluate -> plot figures
# Usage: bash run_all.sh
# Prerequisites: conda/venv activated with requirements.txt installed

set -e  # exit on first error
set -u  # treat unset variables as errors

echo "============================================================"
echo " DOA-CNN-TCA-ResNeXt  --  Full Pipeline"
echo "============================================================"

# ── 0. Create output dirs ─────────────────────────────────────────────────────
mkdir -p data results/checkpoints results/logs results/tables results/figures

# ── 1. Verify array geometry ──────────────────────────────────────────────────
echo ""
echo "[Step 1/8] Verifying TCA array geometry..."
python -c "from datasets.array_geometry import get_tca_positions; \
           p=get_tca_positions(); \
           print(f'  TCA positions: {p.tolist()}'); \
           assert len(p)==12, f'Expected 12 sensors, got {len(p)}'; \
           print('  OK: 12 sensors confirmed')"

# ── 2. Generate datasets (8M total) ───────────────────────────────────────────
echo ""
echo "[Step 2/8] Generating raw signal datasets (T=16 and T=32)..."
echo "  This may take 3-5 hours on CPU. Progress shown per worker."

python datasets/generate_raw.py --T 16 --samples 1600000 --snr 10 --out data/raw_t16_train.h5
python datasets/generate_raw.py --T 16 --samples  200000 --snr 10 --out data/raw_t16_val.h5
python datasets/generate_raw.py --T 16 --samples  200000 --snr 10 --out data/raw_t16_test.h5

python datasets/generate_raw.py --T 32 --samples 1600000 --snr 10 --out data/raw_t32_train.h5
python datasets/generate_raw.py --T 32 --samples  200000 --snr 10 --out data/raw_t32_val.h5
python datasets/generate_raw.py --T 32 --samples  200000 --snr 10 --out data/raw_t32_test.h5

echo ""
echo "[Step 3/8] Generating covariance matrix datasets (T=16 and T=32)..."
python datasets/generate_cov.py --T 16 --samples 1600000 --snr 10 --out data/cov_t16_train.h5
python datasets/generate_cov.py --T 16 --samples  200000 --snr 10 --out data/cov_t16_val.h5
python datasets/generate_cov.py --T 16 --samples  200000 --snr 10 --out data/cov_t16_test.h5

python datasets/generate_cov.py --T 32 --samples 1600000 --snr 10 --out data/cov_t32_train.h5
python datasets/generate_cov.py --T 32 --samples  200000 --snr 10 --out data/cov_t32_val.h5
python datasets/generate_cov.py --T 32 --samples  200000 --snr 10 --out data/cov_t32_test.h5

echo ""
echo "  All HDF5 datasets generated."

# ── 3. Train 4 models ─────────────────────────────────────────────────────────
echo ""
echo "[Step 4/8] Training Raw+T16 model..."
python train/trainer.py --config configs/raw_t16.yaml

echo ""
echo "[Step 5/8] Training Raw+T32 model..."
python train/trainer.py --config configs/raw_t32.yaml

echo ""
echo "[Step 6/8] Training Cov+T16 model..."
python train/trainer.py --config configs/cov_t16.yaml

echo ""
echo "[Step 7/8] Training Cov+T32 model..."
python train/trainer.py --config configs/cov_t32.yaml

echo ""
echo "  All 4 models trained. Checkpoints in results/checkpoints/"

# ── 4. Evaluate all models ────────────────────────────────────────────────────
echo ""
echo "[Step 8a/8] Evaluating models and generating Tables 2-5..."
for cfg in raw_t16 raw_t32 cov_t16 cov_t32; do
    python eval/metrics.py --config configs/${cfg}.yaml
done
python eval/metrics.py --config configs/raw_t16.yaml --all   # generate all LaTeX tables

# ── 5. Classical algorithm benchmark ──────────────────────────────────────────
echo ""
echo "[Step 8b/8] Running MUSIC + ESPRIT benchmark (Table 6)..."
python eval/compare_classical.py --snr_range 0 20 --step 2 --T 16 --samples 500

# ── 6. Generate all figures ───────────────────────────────────────────────────
echo ""
echo "[Step 8c/8] Generating all figures (Fig.4 - Fig.8)..."
python eval/visualize.py --all --results results/

echo ""
echo "============================================================"
echo " Pipeline complete!  Results:"
echo "  Tables  -> results/tables/"
echo "  Figures -> results/figures/"
echo "  Models  -> results/checkpoints/*_best.pth"
echo "  Logs    -> results/logs/  (view: tensorboard --logdir results/logs)"
echo "============================================================"
