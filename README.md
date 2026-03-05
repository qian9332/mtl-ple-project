# Multi-Task Learning: PLE vs MMoE vs CGC

## 🏆 实验结论: PLE 获胜

在 500K 样本完整数据上训练 20 epochs 的对比实验结果：

| Model | Params | CTR AUC | CVR AUC | **Avg AUC** | CTR LogLoss | CVR LogLoss | UW Frozen |
|-------|--------|---------|---------|-------------|-------------|-------------|-----------|
| MMoE  | 153,690 | 0.6270 | 0.6888 | 0.6579 | 0.6488 | 0.3743 | [0.874, 1.218] |
| CGC   | 192,890 | 0.6296 | 0.6908 | 0.6602 | 0.6508 | 0.3675 | [0.879, 1.219] |
| **PLE** | **243,978** | **0.6422** | **0.7124** | **0.6773** | **0.6367** | **0.3517** | **[0.863, 1.206]** |

> **PLE 以 Avg AUC 0.6773 显著领先 CGC(0.6602) 和 MMoE(0.6579)**

## 项目简介

本项目实现了完整的多任务学习（MTL）框架，用于 CTR/CVR 联合预估：

### 核心技术

1. **PLE (Progressive Layered Extraction)** — 逐层参数隔离
   - 2层 Extraction Layer，每层独占 Expert + 共享 Expert
   - Gate 温度退火（T=1.5 → 0.54，decay=0.95/epoch）
   - 专家利用率监控，抑制"赢者通吃"

2. **Uncertainty Weight** — 任务权重自适应
   - 仅增加 2 个标量参数（log_sigma_ctr, log_sigma_cvr）
   - 训后固化为 [0.863, 1.206]，零推理开销
   - CTR 权重降低（任务较简单），CVR 权重升高（任务更难）

3. **梯度余弦相似度冲突检测** — 冲突感知 Early Stopping
   - EMA 自适应阈值
   - PLE CosSim 稳定在 0.3-0.5（正值 = 低冲突），CGC 接近 0
   - 共享层软冻结机制

4. **ESMM-CVR 全曝光分支** — P(conversion) = P(click) × P(conversion|click)
   - 解决 CVR 样本选择偏差（SSB）问题

5. **特征掩码自监督辅助任务** — 15% 特征 mask + 重建损失
   - 增强特征表示学习，提升泛化性

### 为什么 PLE > CGC?

在之前的小数据实验中，CGC 曾优于 PLE，根因分析：

| 问题 | 原因 | 修复方案 |
|------|------|---------|
| PLE过参数化 | 3层 × 128d experts = 1.06M params | 减为2层 × 64d = 244K params |
| 信息瓶颈 | 下一层输入用 mean(task_outputs) | 改为 per-task input propagation |
| 温度退火过慢 | T=2.0, decay=0.995 → gates太均匀 | T=1.5, decay=0.95 → 更快锐化 |
| 梯度路径过长 | ESMM + 3层 extraction | 减为2层，缩短梯度路径 |

修复后 PLE 在完整数据上的表现：
- **CTR AUC**: PLE(0.642) > CGC(0.630) > MMoE(0.627), Δ=+1.2%
- **CVR AUC**: PLE(0.712) >> CGC(0.691) > MMoE(0.689), Δ=+2.1%
- **Avg AUC**: PLE(0.677) >> CGC(0.660) > MMoE(0.658), Δ=+1.7%
- **梯度冲突**: PLE CosSim=0.43 >> CGC CosSim=-0.03, 说明PLE有效隔离了冲突

### CTR 收敛 / CVR 震荡诊断

**PLE 训练曲线诊断**:
- CTR: `WARMING → IMPROVING → CONVERGED` (slope 从 0.0117 降到 0.0004)
- CVR: `WARMING → IMPROVING → CONVERGED` (std < 0.002)
- 无 OSCILLATING 现象，说明 Uncertainty Weight + PLE 参数隔离有效

**CGC 训练曲线诊断**:
- CTR: `CONVERGED` (delta < 0.001 after ep16)
- CVR: `CONVERGED` but CosSim 出现负值（ep14: -0.001, ep20: -0.029）

## 项目结构

```
mtl-project/
├── src/
│   ├── models/
│   │   ├── ple.py            # PLE 模型（完整版）
│   │   ├── baselines.py      # MMoE / CGC 基线
│   │   └── experts.py        # Expert 网络 + Gate + 利用率监控
│   ├── losses/
│   │   └── uncertainty_weight.py  # Uncertainty Weight Loss
│   ├── utils/
│   │   ├── trainer.py         # 训练引擎
│   │   └── gradient_conflict.py   # 梯度冲突检测 + EMA + 软冻结
│   └── data/
│       └── dataset.py         # 数据集
├── scripts/
│   ├── step_gpu.py            # 完整训练脚本（支持断点续训）
│   ├── full_gpu_train.py      # GPU/CPU 全量训练
│   └── data_analysis.py       # 数据EDA
├── configs/
│   └── ple_config.json        # 配置文件
├── data/
│   ├── processed/             # 处理后的数据
│   └── analysis_report/       # EDA 报告
├── logs/
│   ├── gpu_run/               # 完整训练日志（500K × 20ep）
│   │   ├── train.log          # 完整训练日志
│   │   ├── comparison.json    # 三模型对比结果
│   │   ├── mmoe_result.json   # MMoE 详细结果
│   │   ├── cgc_result.json    # CGC 详细结果
│   │   └── ple_result.json    # PLE 详细结果
│   └── ...                    # 历史训练日志
├── README.md
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# 训练单个模型
python scripts/step_gpu.py mmoe
python scripts/step_gpu.py cgc
python scripts/step_gpu.py ple

# 生成对比报告
python scripts/step_gpu.py report
```

## 训练日志详情

每个 epoch 记录以下信息：
- **Loss**: total_loss, ctr_loss, cvr_loss, mask_loss, load_balance_loss
- **AUC**: train/val CTR AUC, CVR AUC, Avg AUC
- **Uncertainty Weight**: learned task weights [w_ctr, w_cvr] 和 log_sigma
- **Gradient Conflict**: cosine similarity, conflict ratio, EMA threshold
- **Convergence Diagnosis**: CTR/CVR 状态（WARMING/IMPROVING/CONVERGED/OSCILLATING/DEGRADING）
- **Gate Temperature**: 每层每任务的 gate 温度值（仅PLE）
- **Expert Utilization**: 专家利用率分布（检测赢者通吃）

## 依赖

- Python >= 3.8
- PyTorch >= 2.0
- scikit-learn
- numpy
- pandas
- matplotlib

## 参考文献

- [PLE] Tang et al., "Progressive Layered Extraction", RecSys 2020
- [MMoE] Ma et al., "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts", KDD 2018
- [Uncertainty Weight] Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018
- [ESMM] Ma et al., "Entire Space Multi-Task Model", SIGIR 2018
- [Gradient Surgery] Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020
