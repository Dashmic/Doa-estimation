# FYP 任务进度追踪 | v0.1.2
# DOA-CNN-TCA-ResNeXt | EIE4127

## ✅ 冒烟测试 — 9/9 PASS (2026-03-24 19:04)
> 环境：Python 3.10.6 | PyTorch 2.11.0+cu126 | CUDA True | RTX 4090 D

| 测试 | 结果 | 说明 |
|------|------|------|
| T1 TCA 阵列几何 | ✅ | 位置 [0,5,6,10,12,15,20,25,36,42,48,54] 正确 |
| T2 Raw 数据生成 | ✅ | (500,2,12,16) HDF5 正常 |
| T3 Cov 数据生成 | ✅ | (500,2,12,12) HDF5 正常 |
| T4 DataLoader | ✅ | batch (32,2,12,16)，avg 9.5 active classes |
| T5 模型 raw 输入 | ✅ | 23.2M 参数，输出 in [0,1] |
| T6 模型 cov 输入 | ✅ | (2,2,12,12) → (2,121) 正确 |
| T7 训练 3 epochs | ✅ | val_loss=0.2546，val_acc=0.94，checkpoint 279MB |
| T8 MUSIC | ✅ | MeanErr=0.00deg（SNR=10dB） |
| T9 ESPRIT | ✅ | 非均匀阵列精度有限，对比实验中会体现 |

---

## Phase 1 — 数据生成 (Day 1-2)
- [x] 项目目录结构
- [x] requirements.txt + 4个 yaml configs
- [x] datasets/array_geometry.py  —— TCA 12传感器位置
- [x] datasets/generate_raw.py    —— Raw signal HDF5 生成
- [x] datasets/generate_cov.py    —— Covariance HDF5 生成
- [x] datasets/data_loader.py     —— PyTorch Dataset

## Phase 2 — 模型 (Day 3-4)
- [x] models/resnext_doa.py  —— 修改版 ResNeXt-50 (2ch → FC121 + Sigmoid)
- [ ] **[待执行]** 验证 shape 正确后开始训练

## Phase 3 — 训练 (Day 4-6)
- [x] train/trainer.py  —— BCELoss + AdamW + 早停 + TensorBoard
- [ ] **[待执行]** 训练 Raw+T16  → results/checkpoints/raw_t16_best.pth
- [ ] **[待执行]** 训练 Raw+T32  → results/checkpoints/raw_t32_best.pth
- [ ] **[待执行]** 训练 Cov+T16  → results/checkpoints/cov_t16_best.pth
- [ ] **[待执行]** 训练 Cov+T32  → results/checkpoints/cov_t32_best.pth

## Phase 4 — 评估 (Day 7-8)
- [x] eval/metrics.py  —— 4指标 + SNR sweep + Table 2-5 CSV/LaTeX
- [ ] **[待执行]** 生成 Table 2-5（要求与论文误差 < 0.5%）

## Phase 5 — 扩展实验 (Day 9-10)
- [x] eval/compare_classical.py  —— MUSIC + ESPRIT + Table 6
- [ ] **[待执行]** 运行 SNR=0-20dB benchmark，生成 Table 6

## Phase 6 — 可视化 & 交付 (Day 11-12)
- [x] eval/visualize.py      —— 自动生成 Fig.4-8 (300 DPI)
- [x] notebooks/demo.py      —— Presentation 演示脚本
- [x] run_all.sh             —— 一键流水线
- [x] README.md              —— 双语（中英文）
- [ ] **[待执行]** 完成训练后运行 run_all.sh 生成全套图表
- [ ] **[待执行]** 上传 .pth 到 Google Drive，更新 README 链接

## 可选加分项
- [ ] 阵列位置扰动实验 (Array Position Perturbation)
- [ ] 宽带信号实验 (Wideband Signal)
