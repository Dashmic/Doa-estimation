# 操作指南 — DOA-CNN-TCA-ResNeXt

## 0. 安装依赖

```bash
cd DOA-CNN-TCA-ResNeXt
pip install -r requirements.txt
```

---

## 1. ⚠️ 先确认 TCA 传感器位置（必做）

对照论文 **Fig.2**，检查 `datasets/array_geometry.py` 中的位置：

```python
# 当前12个位置（d = λ/2 单位）
[0, 5, 6, 10, 12, 15, 18, 20, 24, 25, 30, 35]
```

如果不对，只修改 `get_tca_positions()` 函数，其余代码不用改。

---

## 2. 快速冒烟测试（先跑小数据确认无报错）

```bash
# 小量数据测试
python datasets/generate_raw.py --T 16 --samples 10000 --out data/raw_t16_train.h5
python datasets/generate_raw.py --T 16 --samples  2000 --out data/raw_t16_val.h5
python datasets/generate_raw.py --T 16 --samples  2000 --out data/raw_t16_test.h5

# 测试模型 shape
python models/resnext_doa.py

# 跑几个 epoch 确认训练正常
python train/trainer.py --config configs/raw_t16.yaml
```

---

## 3. 生成完整 8M 数据集（耗时约 3-5 小时）

```bash
# Raw signal (T=16)
python datasets/generate_raw.py --T 16 --samples 1600000 --snr 10 --out data/raw_t16_train.h5
python datasets/generate_raw.py --T 16 --samples  200000 --snr 10 --out data/raw_t16_val.h5
python datasets/generate_raw.py --T 16 --samples  200000 --snr 10 --out data/raw_t16_test.h5

# Raw signal (T=32)
python datasets/generate_raw.py --T 32 --samples 1600000 --snr 10 --out data/raw_t32_train.h5
python datasets/generate_raw.py --T 32 --samples  200000 --snr 10 --out data/raw_t32_val.h5
python datasets/generate_raw.py --T 32 --samples  200000 --snr 10 --out data/raw_t32_test.h5

# Covariance (T=16)
python datasets/generate_cov.py --T 16 --samples 1600000 --snr 10 --out data/cov_t16_train.h5
python datasets/generate_cov.py --T 16 --samples  200000 --snr 10 --out data/cov_t16_val.h5
python datasets/generate_cov.py --T 16 --samples  200000 --snr 10 --out data/cov_t16_test.h5

# Covariance (T=32)
python datasets/generate_cov.py --T 32 --samples 1600000 --snr 10 --out data/cov_t32_train.h5
python datasets/generate_cov.py --T 32 --samples  200000 --snr 10 --out data/cov_t32_val.h5
python datasets/generate_cov.py --T 32 --samples  200000 --snr 10 --out data/cov_t32_test.h5
```

---

## 4. 训练 4 个模型（每个约 3-4 小时，RTX 4090）

```bash
python train/trainer.py --config configs/raw_t16.yaml
python train/trainer.py --config configs/raw_t32.yaml
python train/trainer.py --config configs/cov_t16.yaml
python train/trainer.py --config configs/cov_t32.yaml

# 监控训练曲线（另开终端）
tensorboard --logdir results/logs
```

断点续训：
```bash
python train/trainer.py --config configs/raw_t16.yaml --resume results/checkpoints/raw_t16_best.pth
```

---

## 5. 评估 & 生成 Tables

```bash
# 逐一评估（生成 JSON + Table CSV/LaTeX）
python eval/metrics.py --config configs/raw_t16.yaml
python eval/metrics.py --config configs/raw_t32.yaml
python eval/metrics.py --config configs/cov_t16.yaml
python eval/metrics.py --config configs/cov_t32.yaml

# 生成所有 LaTeX 表格
python eval/metrics.py --config configs/raw_t16.yaml --all
```

---

## 6. 经典算法对比（MUSIC + ESPRIT）

```bash
python eval/compare_classical.py --snr_range 0 20 --step 2 --T 16 --samples 500
```

生成 `results/tables/table6_comparison.csv` 和 `.tex`

---

## 7. 生成所有 Figures

```bash
python eval/visualize.py --all --results results/
# 输出到 results/figures/fig4~fig8.png
```

---

## 8. 一键运行（全部流程）

```bash
bash run_all.sh
```

---

## 9. Presentation 演示

```bash
cd notebooks
python demo.py
# 或者在 Jupyter 里：jupyter notebook
```

---

## 输出文件位置

| 类型 | 路径 |
|------|------|
| 模型权重 | `results/checkpoints/*_best.pth` |
| Table 2-5 CSV | `results/tables/table_*.csv` |
| Table 2-5 LaTeX | `results/tables/table_*.tex` |
| Table 6 | `results/tables/table6_comparison.*` |
| Fig.4-8 PNG | `results/figures/fig*.png` |
| TensorBoard | `results/logs/` |

---

## 常见问题

**Q: 数据生成太慢？**  
A: `--workers` 参数控制并行进程数（默认 cpu_count-1），RTX 4090 机器 CPU 核心多，一般够用。

**Q: CUDA out of memory？**  
A: configs 里把 `batch_size: 128` 改为 `64`。

**Q: 论文指标对不上？**  
A: 先确认 `array_geometry.py` 的传感器位置与论文 Fig.2 一致，再检查 SNR 定义（当前用信号功率/噪声功率定义）。
