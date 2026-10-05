# 项目理论文档 | Project Theory & Mathematics
## DOA-CNN-TCA-ResNeXt | EIE4127 FYP

> **作者：** PolyU EIE4127 FYP  
> **参考论文：** "A Unified Approach for Target Direction Finding Based on CNNs", IEEE MLSP 2020  
> **DOI：** 10.1109/MLSP49062.2020.9231787

---

## 目录

1. [问题定义](#1-问题定义)
2. [阵列信号模型](#2-阵列信号模型)
3. [稀疏互质阵列 TCA](#3-稀疏互质阵列-tca)
4. [协方差矩阵与差分阵列](#4-协方差矩阵与差分阵列)
5. [经典算法：MUSIC](#5-经典算法music)
6. [经典算法：ESPRIT](#6-经典算法esprit)
7. [深度学习方法：CNN DOA 估计](#7-深度学习方法cnn-doa-估计)
8. [ResNeXt-50 架构](#8-resnext-50-架构)
9. [多标签分类与损失函数](#9-多标签分类与损失函数)
10. [评估指标推导](#10-评估指标推导)
11. [任务分解与实验路线图](#11-任务分解与实验路线图)
12. [Mathematica 演算建议](#12-mathematica-演算建议)

---

## 1. 问题定义

**波达方向（DOA）估计**：给定 $P$ 个传感器组成的阵列，有 $K$ 个窄带信号源从不同方向 $\{\theta_1, \theta_2, \ldots, \theta_K\}$ 入射，估计所有信源的方向角。

**核心挑战：**
- 欠定问题：$K > P$（信源数 > 传感器数），经典方法失效
- 噪声干扰
- 实时性要求

**本项目框架：**
- 角度范围：$\theta \in [-60°, +60°]$，步长 $1°$，共 **121 个候选角度**
- 信源数：$K \in \{1, 2, \ldots, 16\}$（随机，可超过传感器数）
- 处理方式：**多标签分类**（每个候选角度独立预测激活概率）

---

## 2. 阵列信号模型

### 2.1 接收信号模型

设阵列有 $P$ 个传感器，位置向量为 $\mathbf{p} = [p_0, p_1, \ldots, p_{P-1}]^T$（以半波长 $d = \lambda/2$ 为单位）。

$K$ 个窄带信号源的接收信号快拍（snapshot）为：

$$\mathbf{x}(t) = \mathbf{A}(\boldsymbol{\theta})\, \mathbf{s}(t) + \mathbf{n}(t), \quad t = 1, 2, \ldots, T$$

其中：
- $\mathbf{x}(t) \in \mathbb{C}^{P \times 1}$：第 $t$ 个快拍的接收信号向量
- $\mathbf{A}(\boldsymbol{\theta}) = [\mathbf{a}(\theta_1), \ldots, \mathbf{a}(\theta_K)] \in \mathbb{C}^{P \times K}$：导向矩阵
- $\mathbf{s}(t) \in \mathbb{C}^{K \times 1}$：信源信号向量
- $\mathbf{n}(t) \in \mathbb{C}^{P \times 1}$：加性高斯白噪声
- $T$：快拍数（本项目 $T \in \{16, 32\}$）

### 2.2 导向向量

第 $k$ 个信源的导向向量（steering vector）：

$$\mathbf{a}(\theta_k) = \left[e^{j2\pi p_0 \sin\theta_k / \lambda}, e^{j2\pi p_1 \sin\theta_k / \lambda}, \ldots, e^{j2\pi p_{P-1} \sin\theta_k / \lambda}\right]^T$$

由于 $p_i$ 以 $d = \lambda/2$ 为单位，令 $u_i = p_i \cdot d / \lambda = p_i / 2$，则：

$$a_i(\theta_k) = e^{j\pi p_i \sin\theta_k}$$

直观理解：导向向量描述了从方向 $\theta_k$ 到达的平面波在各传感器上产生的**相位延迟**。

### 2.3 统计假设

- 信源信号：$\mathbf{s}(t) \sim \mathcal{CN}(\mathbf{0}, \mathbf{R}_s)$，$\mathbf{R}_s = \text{diag}(\sigma_1^2, \ldots, \sigma_K^2)$（不相关信源）
- 噪声：$\mathbf{n}(t) \sim \mathcal{CN}(\mathbf{0}, \sigma_n^2 \mathbf{I})$（空间白噪声）
- 信源与噪声独立

### 2.4 SNR 定义

$$\text{SNR} = 10\log_{10}\left(\frac{\sigma_s^2}{\sigma_n^2}\right) \text{ dB}$$

代码中的信噪比实现：
$$\sigma_n = \sqrt{\frac{\overline{P_{signal}}}{\text{SNR}_{linear} \cdot 2}}$$

其中 $\overline{P_{signal}} = \frac{1}{PT}\sum_{t,i} |x_i^{clean}(t)|^2$

---

## 3. 稀疏互质阵列 TCA

### 3.1 互质阵列基本定义

两个正整数 $M, N$ 满足 $\gcd(M, N) = 1$（互质）。

传统互质阵列（Conventional Coprime Array）由两个子阵组成：

$$\mathcal{P}_{conv} = \{nMd \mid 0 \leq n \leq N-1\} \cup \{mNd \mid 0 \leq m \leq 2M-1\}$$

总传感器数：$S_{conv} = 2M + N - 1$

缺点：存在冗余传感器，影响孔径效率。

### 3.2 TCA 定义（论文 Eq.3 & Fig.2）

**稀疏互质阵列（Thinned Coprime Array, TCA）** 移除冗余传感器，由三个子阵组成：

$$X = X_1 \cup X_2 \cup X_3$$

$$\begin{cases}
X_1 = \{nMd \mid 0 \leq n \leq N-1\} & \text{(}N\text{ 个传感器)} \\
X_2 = \{mNd \mid 1 \leq m \leq \lfloor M/2 \rfloor\} & \text{(} \lfloor M/2 \rfloor \text{ 个传感器)} \\
X_3 = \{(m+M+1)Nd \mid 0 \leq m \leq M-2\} & \text{(} M-1 \text{ 个传感器)}
\end{cases}$$

总传感器数（论文 Eq.4）：

$$\boxed{S = M + N + \left\lfloor \frac{M}{2} \right\rfloor - 1}$$

### 3.3 M=5, N=6 的具体计算

| 子阵 | 计算 | 位置（×d） |
|------|------|-----------|
| $X_1$ | $n \times 5, n=0,...,5$ | {0, 5, 10, 15, 20, 25} |
| $X_2$ | $m \times 6, m=1,2$ | {6, 12} |
| $X_3$ | $(m+6) \times 6, m=0,...,3$ | {36, 42, 48, 54} |
| **TCA** | 并集 | **{0, 5, 6, 10, 12, 15, 20, 25, 36, 42, 48, 54}** |

验证：$S = 5 + 6 + \lfloor 5/2 \rfloor - 1 = 5 + 6 + 2 - 1 = \mathbf{12}$ ✓

**阵列孔径：** $54d$（远大于 12 个均匀阵元的 $11d$）

### 3.4 差分阵列（Virtual Array）

协方差矩阵向量化后可得到虚拟差分阵列（Difference Co-array）：

$$\mathcal{D} = \{p_i - p_j \mid p_i, p_j \in \text{TCA}\}$$

差分阵列的连续孔径决定了系统能分辨的最大信源数（自由度 DOF）。TCA 的连续差分阵列孔径 $\gg S$，实现欠定估计（$K > P$）。

---

## 4. 协方差矩阵与差分阵列

### 4.1 理论协方差矩阵

$$\mathbf{R}_{xx} = \mathbb{E}[\mathbf{x}(t)\mathbf{x}^H(t)] = \mathbf{A}\mathbf{R}_s\mathbf{A}^H + \sigma_n^2\mathbf{I}$$

其中 $\mathbf{R}_s = \text{diag}(\sigma_1^2, \ldots, \sigma_K^2)$

### 4.2 样本协方差矩阵（MLE 估计）

$$\hat{\mathbf{R}}_{xx} = \frac{1}{T}\sum_{t=1}^T \mathbf{x}(t)\mathbf{x}^H(t) = \frac{1}{T}\mathbf{X}\mathbf{X}^H$$

其中 $\mathbf{X} = [\mathbf{x}(1), \ldots, \mathbf{x}(T)] \in \mathbb{C}^{P \times T}$

**性质：**
- Hermitian 正定：$\hat{\mathbf{R}} = \hat{\mathbf{R}}^H$，$\hat{\mathbf{R}} \succ 0$
- 渐近一致估计：$\hat{\mathbf{R}} \xrightarrow{T \to \infty} \mathbf{R}_{xx}$

### 4.3 CNN 的两种输入格式

**格式1 - Raw Signal：** 直接使用 $T$ 个快拍

$$\mathbf{X}_{in}^{raw} = \begin{bmatrix} \text{Re}(\mathbf{X}) \\ \text{Im}(\mathbf{X}) \end{bmatrix} \in \mathbb{R}^{2 \times P \times T}$$

**格式2 - Covariance Matrix：**

$$\mathbf{X}_{in}^{cov} = \begin{bmatrix} \text{Re}(\hat{\mathbf{R}}) \\ \text{Im}(\hat{\mathbf{R}}) \end{bmatrix} \in \mathbb{R}^{2 \times P \times P}$$

---

## 5. 经典算法：MUSIC

### 5.1 特征分解

对理论协方差矩阵进行特征值分解：

$$\mathbf{R}_{xx} = \mathbf{U}_s \boldsymbol{\Lambda}_s \mathbf{U}_s^H + \sigma_n^2 \mathbf{U}_n \mathbf{U}_n^H$$

其中：
- $\mathbf{U}_s \in \mathbb{C}^{P \times K}$：信号子空间（对应 $K$ 个大特征值）
- $\mathbf{U}_n \in \mathbb{C}^{P \times (P-K)}$：噪声子空间（对应 $P-K$ 个小特征值 $\approx \sigma_n^2$）

**关键性质：** 信号子空间与噪声子空间正交

$$\mathbf{a}(\theta_k) \perp \mathbf{U}_n, \quad k = 1, \ldots, K$$

即 $\mathbf{U}_n^H \mathbf{a}(\theta_k) = \mathbf{0}$

### 5.2 MUSIC 谱

$$P_{MUSIC}(\theta) = \frac{1}{\mathbf{a}^H(\theta)\mathbf{U}_n\mathbf{U}_n^H\mathbf{a}(\theta)}$$

真实 DOA 处，$\mathbf{a}(\theta_k) \perp \mathbf{U}_n$，分母趋向 0，谱值趋向无穷（谱峰）。

### 5.3 DOA 估计步骤

1. 计算样本协方差 $\hat{\mathbf{R}}$
2. 特征值分解，确定信号子空间维度 $K$
3. 取最小 $P-K$ 个特征向量构成 $\hat{\mathbf{U}}_n$
4. 在 $\theta \in [-90°, 90°]$ 上计算 $P_{MUSIC}(\theta)$
5. 找 $K$ 个峰值位置 → DOA 估计

**分辨率：** 理论分辨率 $\approx 0.9\lambda / D$，其中 $D$ 为阵列孔径

### 5.4 MUSIC 的局限性

- 需要精确已知信源数 $K$
- 当 $K \geq P$ 时（欠定情况）失效（但使用差分阵列后可改善）
- 相干信源导致协方差矩阵秩亏

---

## 6. 经典算法：ESPRIT

### 6.1 旋转不变性原理

ESPRIT 要求阵列具有**旋转不变性**（两个相同的子阵，间距固定为 $\Delta$）：

$$\mathbf{A}_2 = \mathbf{A}_1 \mathbf{\Phi}$$

其中 $\mathbf{\Phi} = \text{diag}(e^{j\pi\Delta\sin\theta_1}, \ldots, e^{j\pi\Delta\sin\theta_K})$

### 6.2 ESPRIT 求解

从信号子空间中提取两个子矩阵 $\mathbf{E}_1, \mathbf{E}_2$（对应两个子阵）：

$$\mathbf{E}_2 = \mathbf{E}_1 \mathbf{\Psi}$$

通过最小二乘解 $\mathbf{\Psi}$：

$$\mathbf{\Psi} = (\mathbf{E}_1^H\mathbf{E}_1)^{-1}\mathbf{E}_1^H\mathbf{E}_2$$

$\mathbf{\Psi}$ 的特征值 $\lambda_k = e^{j\pi\Delta\sin\theta_k}$，因此：

$$\hat{\theta}_k = \arcsin\left(\frac{\angle \lambda_k}{\pi \Delta}\right)$$

### 6.3 TCA 上的 ESPRIT 限制

TCA 是非均匀阵列，不满足严格的旋转不变性。实际使用时需进行**虚拟阵列插值**或在差分阵列的均匀部分上应用 ESPRIT，精度受限。

---

## 7. 深度学习方法：CNN DOA 估计

### 7.1 问题重构

**核心思想：** 将 DOA 估计转化为多标签分类问题。

定义 121 维标签向量（DOA 格网 $-60°$ 到 $+60°$，步长 $1°$）：

$$\mathbf{y} = [y_{-60}, y_{-59}, \ldots, y_{60}]^T \in \{0, 1\}^{121}$$

$$y_i = \begin{cases}1 & \text{若 } \theta_i \text{ 处有信源} \\ 0 & \text{否则}\end{cases}$$

### 7.2 CNN 的优势

| 特性 | 经典方法 | CNN |
|------|---------|-----|
| 欠定问题（$K > P$）| ❌ 失效 | ✅ 处理 |
| 不需知道 $K$ | ❌ 需要 | ✅ 自动 |
| 噪声鲁棒性 | 中等 | 高 |
| 相干信源 | 困难 | ✅ 处理 |
| 实时推理 | 慢（谱搜索）| ✅ 毫秒级 |

### 7.3 输入预处理

**Raw 模式：**
$$f_{raw}: \mathbb{R}^{2 \times 12 \times T} \to [0,1]^{121}$$

**Cov 模式：**
$$f_{cov}: \mathbb{R}^{2 \times 12 \times 12} \to [0,1]^{121}$$

实数化：将复数矩阵分拆为实部和虚部两个通道，使 CNN 能够处理复数信号。

### 7.4 理论联系

CNN 本质上在学习信号子空间的隐式表示：
- 协方差矩阵输入 → CNN 提取类似特征分解的特征
- Raw 信号输入 → CNN 同时学习协方差估计和子空间分解

---

## 8. ResNeXt-50 架构

### 8.1 残差连接（ResNet）

$$\mathbf{h}(\mathbf{x}) = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

残差学习：网络只需学习残差 $\mathcal{F} = \mathbf{h} - \mathbf{x}$，缓解梯度消失。

### 8.2 分组卷积（ResNeXt 的 Cardinality）

ResNeXt 在 ResNet 基础上引入 **Cardinality**（分组数 $C$）：

标准卷积：$C=1$，通道全连接

分组卷积（$C=32$）：将通道分成 32 组，每组独立卷积，再拼接：

$$\mathbf{y} = \sum_{i=1}^{C} \mathcal{T}_i(\mathbf{x}^{(i)})$$

参数量相比标准卷积减少 $C$ 倍，但性能相当或更好。

### 8.3 本项目的修改（对应论文 Table 1）

| 层 | 原始 ResNeXt-50 | 本项目修改 |
|---|----------------|-----------|
| Conv1 输入通道 | 3（RGB） | **2**（实部+虚部） |
| FC 输出维度 | 1000（ImageNet） | **121**（DOA类别） |
| 最终激活 | Softmax | **Sigmoid**（多标签） |
| Cardinality | 32 | 32（不变） |
| 总参数 | ~25M | ~**23.2M** |

输入适配：对于 Raw 输入 $(2, 12, T)$，$T=16$ 或 $32$，Conv1 接受 $2$ 通道宽 $12$ 高 $T$ 的图像。

---

## 9. 多标签分类与损失函数

### 9.1 Sigmoid 激活

对每个输出节点 $i$（对应角度 $\theta_i$）独立：

$$\hat{y}_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}} \in (0, 1)$$

表示"角度 $\theta_i$ 有信源"的概率。与 Softmax 不同，各节点概率之和不必为 1。

### 9.2 二元交叉熵损失（BCELoss）

$$\mathcal{L}_{BCE} = -\frac{1}{B \cdot 121}\sum_{b=1}^B\sum_{i=1}^{121} \left[y_{b,i}\log\hat{y}_{b,i} + (1-y_{b,i})\log(1-\hat{y}_{b,i})\right]$$

其中 $B$ 为批次大小，$y_{b,i} \in \{0,1\}$ 为真实标签。

**梯度：**
$$\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$$

### 9.3 推断时的阈值

测试阶段，若 $\hat{y}_i > \tau$（阈值，通常 $\tau=0.5$），则预测角度 $\theta_i$ 有信源：

$$\tilde{y}_i = \mathbf{1}[\hat{y}_i > \tau]$$

### 9.4 AdamW 优化器

Adam 的权重衰减修正版：

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\nabla_\theta \mathcal{L}$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)(\nabla_\theta \mathcal{L})^2$$
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \eta \lambda \theta_t$$

最后一项 $-\eta\lambda\theta_t$ 为解耦权重衰减（区别于 L2 正则化中的 Adam）。

本项目参数：$\eta=10^{-3}$，$\lambda=10^{-2}$，$\beta_1=0.9$，$\beta_2=0.999$

---

## 10. 评估指标推导

对于 121 维多标签预测，逐元素统计（element-wise）：

| | 预测正 ($\tilde{y}=1$) | 预测负 ($\tilde{y}=0$) |
|--|---|---|
| **真实正** ($y=1$) | TP | FN |
| **真实负** ($y=0$) | FP | TN |

**注：** 总量 $= B \times 121$（批次 × 类别数）

$$\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$（预测为有信源中，真正有信源的比例）

$$\text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$$（真实信源被检测到的比例）

$$\text{Specificity} = \frac{TN}{TN + FP}$$（真实无信源中被正确排除的比例）

**在 DOA 估计中的直观含义：**
- **Recall 高** → 信源漏检少（重要！）
- **Precision 高** → 虚警少
- **Specificity 高** → 非信源角度很少被误判

---

## 11. 任务分解与实验路线图

### Phase 1：基础复现（已完成 ✅）

```
论文核心内容再现：
├── TCA 几何（Fig.2）→ array_geometry.py
├── 信号生成（Eq.1-2）→ generate_raw.py, generate_cov.py
├── ResNeXt-50 修改（Table 1）→ resnext_doa.py
└── BCELoss + AdamW 训练 → trainer.py

预期结果：
├── Table 2: Raw+T16 的 Acc/Prec/Rec/Spec vs SNR
├── Table 3: Raw+T32 --
├── Table 4: Cov+T16 --
└── Table 5: Cov+T32 --
```

### Phase 2：扩展实验（本项目亮点）

```
Extension 1：经典算法对比（Table 6）
├── MUSIC：协方差 + 特征分解 + 谱搜索
├── ESPRIT：旋转不变子空间估计
└── 对比维度：SNR（0-20dB）× 方法 × 指标

Extension 2（可选）：阵列扰动鲁棒性
├── 对传感器位置加 ±Δp 高斯扰动
├── Δp ~ N(0, ε²d²)，ε ∈ {1%, 5%, 10%}
└── 对比 CNN vs MUSIC 的性能退化曲线

Extension 3（可选）：宽带信号
├── 信源带宽 B% × 中心频率
├── 测试原始模型在 out-of-distribution 下的性能
└── 分析频率失配（frequency mismatch）影响
```

### Phase 3：可视化（最终交付）

```
Figure 4: 训练/验证 Loss 曲线（4个模型）
Figure 5: 性能 vs T（T=16 vs T=32 对比）
Figure 6: CNN vs MUSIC vs ESPRIT SNR 曲线
Figure 7: TCA 阵列示意图
Figure 8: MUSIC 空间谱（含真实 DOA 标注）
```

---

## 12. Mathematica 演算建议

以下公式适合在 Mathematica 中进行符号推导和可视化：

### 12.1 导向向量与导向矩阵

```mathematica
(* TCA 位置 *)
positions = {0, 5, 6, 10, 12, 15, 20, 25, 36, 42, 48, 54};

(* 导向向量 *)
a[theta_] := Exp[I * Pi * positions * Sin[theta * Pi/180]]

(* 导向矩阵可视化 *)
A = Table[a[theta], {theta, -60, 60, 1}];
MatrixPlot[Abs[Transpose[A]], ColorFunction -> "Rainbow"]
```

### 12.2 MUSIC 谱计算

```mathematica
(* 样本协方差矩阵特征分解 *)
{evals, evecs} = Eigensystem[R];
SortBy[Transpose[{evals, evecs}], First]...

(* MUSIC 伪谱 *)
Pmusic[theta_, Un_] :=
    1 / Abs[ConjugateTranspose[a[theta]] . Un . ConjugateTranspose[Un] . a[theta]]
```

### 12.3 TCA 差分阵列

```mathematica
(* 差分阵列 *)
diffArray = Union[Flatten[Table[positions[[i]] - positions[[j]],
    {i, 1, Length[positions]}, {j, 1, Length[positions]}]]];

(* 连续孔径 *)
consecutive = Max[Select[diffArray, # >= 0 &] /. ...
```

### 12.4 SNR 鲁棒性分析

```mathematica
(* 理论 RMSE 下界（Cramér-Rao Bound）*)
CRB[snr_, T_, theta_] :=
    6 / (T * snr * (2*Pi)^2 * Cos[theta]^2 * Sum[...])
```

### 12.5 信号子空间维度判断（MDL/AIC 准则）

```mathematica
(* MDL 准则估计信源数 *)
MDL[lam_, n_] := Table[
    - (n - k) * Sum[Log[lam[[i]]], {i, k+1, Length[lam]}]
    + (n - k) * Log[GeometricMean[lam[[k+1;;]]]/Mean[lam[[k+1;;]]]]
    + k/2 * (2*n - k) * Log[T] / 2,
    {k, 0, Length[lam]-1}]
```

---

## 参考文献

1. Pal, P. & Vaidyanathan, P. (2010). Nested arrays: A novel approach to array processing with enhanced degrees of freedom. *IEEE Trans. Signal Process.*

2. Liu, C.L. & Vaidyanathan, P. (2017). Coprime arrays and samplers for space-time adaptive processing. *IEEE Trans. Signal Process.*

3. **[核心论文]** Ma, Zi-Yue et al. (2020). A Unified Approach for Target Direction Finding Based on Convolutional Neural Networks. *IEEE MLSP 2020.* DOI: 10.1109/MLSP49062.2020.9231787

4. Xie, S. et al. (2017). Aggregated Residual Transformations for Deep Neural Networks (ResNeXt). *CVPR 2017.*

5. Schmidt, R. (1986). Multiple emitter location and signal parameter estimation. *IEEE Trans. Antennas Propag.*

6. Roy, R. & Kailath, T. (1989). ESPRIT—Estimation of signal parameters via rotational invariance techniques. *IEEE Trans. Acoust. Speech Signal Process.*
